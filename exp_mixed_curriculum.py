"""exp_mixed_curriculum.py — Mixed curriculum v14l: velocity-gated fe suppression.

v14l = v14j + replace w_f_pos_only with w_f_stable (velocity-gated) in episode A.

v14k/v14j problem (88-89% ceiling):
  w_f_pos_only fires near-pi regardless of velocity. Suppresses f_extra during:
    (a) HOLD: near-pi, slow — CORRECT, f_extra not needed, Q@top handles it
    (b) SWING-UP final push: near-pi, FAST (q1d~5 rad/s) — WRONG, kills final swing
  Result: arr stuck 197-218 because final push torque is weakened.
  v14k eval gate improved post but also hurt arr (218 vs 165), net ~zero gain.

v14l fix — use w_f_stable (velocity-gated) in episode A:
  w_f_stable = w * near_goal * low_vel * f_extra^2
  low_vel = exp(-(q1d^2 + q2d^2) / 2)
  Hold (q1d~0): low_vel~1 → suppression fires → fe@top decays → stable hold
  Swing-up final (q1d~5): low_vel~exp(-12.5)~0 → NO suppression → full f_extra
  f_net can provide strong final push through pi → arr drops toward 163 target.
  Eval reverts to ungated rollout (natural velocity-gated suppression handles it).

v14j reference: ep40 achieved 88.6% f01, arr=197, post=98.2%.
"""

import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

# ── Config ────────────────────────────────────────────────────────────────
X0          = [0.0, 0.0, 0.0, 0.0]
X_GOAL      = [math.pi, 0.0, 0.0, 0.0]
DT          = 0.05
HORIZON     = 10
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
STATE_DIM   = 4
CONTROL_DIM = 2

HIDDEN_DIM      = 128
GATE_RANGE_Q    = 0.99
GATE_RANGE_R    = 0.20
F_EXTRA_BOUND   = 2.5   # v14b hit ceiling (8.944) at ep9 → frozen L_bot=2.089 for ep9-13.
                        # 2.5 → clip=11.18. fe@top stays small (detach_f=True in top).
                        # v9b QP instability was fe@top=13.3 (no detach); safe here.
F_KICKSTART_AMP = 1.0
Q_BIAS_Q1       = -3.0
# SeparatedLinearizationNetwork always uses sin/cos input — no USE_SINCOS flag needed.

# Top-episode near_pi sharpness: target = near_pi^power × STABLE + (1-near_pi^power) × PUMP
# power=4: near_pi^4=0.0625 at π/2 → target=0.103 (8× weaker than power=1's 0.755).
# Kept even with separated networks: top's q_profile still directly trains q_net.
Q_NEAR_PI_POWER = 4

META_EPOCHS      = 200
N_BOTTOM_PER_TOP = 3      # bottom gradient steps per top step
N_BOTTOM         = 170    # swing-up episode length
N_TOP            = 100    # hold episode length
LR               = 1e-3
WEIGHT_DECAY     = 1e-4

# Q-profile: state-conditional target for w_q_profile in both episodes.
W_Q_PROFILE = 100.0
PUMP    = [0.01, 0.01, 1.0, 1.0]  # at bottom states: small Q → f_extra pumps
STABLE  = [1.5,  1.5,  1.0, 1.0]  # at top states (top episode): strong Q → hold
# Bottom uses PUMP at ALL states (q_stable=PUMP, so state_phase doesn't matter).
# Top uses PUMP→STABLE (state_phase=True): PUMP at non-π states, STABLE at π.

# Top-specific: hold loss applied on ALL steps of top episode
W_STABLE_PHASE     = 3.0
STABLE_PHASE_STEPS = N_TOP

# Top-specific: direct f_extra suppression near π.
# Set to 0.0 — even 0.5 overwhelms the bottom energy-tracking gradient
# (w_f_pos_only gradient ~500× larger than bottom tracking signal).
# State-conditional f_extra will emerge naturally via trunk learning.
W_F_POS_ONLY_TOP = 0.0

# Bottom-specific: velocity-gated fe suppression in episode A (v14l).
# w_f_stable = near_goal × low_vel × f_extra² where low_vel=exp(-(q1d²+q2d²)/2).
# Fires only when near-pi AND slow (hold phase). Near zero during swing-up (q1d~5 rad/s).
# Allows full f_extra during final swing push → lower arr. Decays fe during hold → high post.
W_F_STABLE_BOT = 1.0    # was w_f_pos_only=0.05 (v14k) — no velocity discrimination
# Bottom q_profile: weak PUMP target everywhere. Prevents Q@mid tracking Q@top.
# Equilibrium: 3×10×(Q-0.01) = 100×(Q-1.5) → Q@top≈1.156; Q@mid≈0.081.
W_Q_PROFILE_BOT   = 10.0
N_Q_PROFILE_STEPS = 5    # short: 15 QP solves/meta-epoch, negligible overhead

# Top-specific: hard-zero f_extra near top AND completely decouple f_head
# from top-episode QP gradient.
F_GATE_THRESH_TOP  = 0.8
DETACH_F_EXTRA_TOP = True   # f_head gets ZERO QP gradient from top episode

# Top-start perturbation ranges
TOP_PERT_Q1  = 0.30
TOP_PERT_Q1D = 0.60
TOP_PERT_Q2  = 0.20
TOP_PERT_Q2D = 0.50

EVAL_EVERY = 10
SAVE_EVERY = 50
SAVE_DIR   = "saved_models"
LOG_FILE   = "/tmp/mixed_curriculum_v14l.log"


# ── Helpers ───────────────────────────────────────────────────────────────
def apply_q1_bias(model, bias_val):
    """Init Q[q1] and Q[q1d] gates low to avoid Q-only trap at start."""
    q_final = [m for m in model.q_head.modules()
               if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(model.horizon - 1):
            q_final.bias[k * model.state_dim + 0] = bias_val
            q_final.bias[k * model.state_dim + 1] = bias_val


def make_energy_demo(n, device):
    """Cosine-eased q1 ramp 0→π for bottom energy-tracking episodes."""
    demo = torch.zeros((n, 4), dtype=torch.float64, device=device)
    for i in range(n):
        alpha = i / max(n - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


def make_hold_demo(n, device):
    """Flat demo at q1=π for top cos_q1-tracking episodes."""
    demo = torch.zeros((n, 4), dtype=torch.float64, device=device)
    demo[:, 0] = math.pi
    return demo


def sample_top(device):
    """Random near-top start for hold episodes."""
    return torch.tensor([
        math.pi + random.uniform(-TOP_PERT_Q1,  TOP_PERT_Q1),
        random.uniform(-TOP_PERT_Q1D, TOP_PERT_Q1D),
        random.uniform(-TOP_PERT_Q2,  TOP_PERT_Q2),
        random.uniform(-TOP_PERT_Q2D, TOP_PERT_Q2D),
    ], dtype=torch.float64, device=device)


def probe_network(model, mpc, device):
    """Quick forward-only probe: Q[q1] gate and f_extra norm at key states."""
    model.eval()
    results = {}
    with torch.no_grad():
        for name, q1 in [("bot", 0.0), ("mid", math.pi / 2), ("top", math.pi)]:
            s    = torch.tensor([q1, 0.0, 0.0, 0.0], dtype=torch.float64, device=device)
            hist = s.unsqueeze(0).expand(5, -1).contiguous()
            gQ, _, fe, _, _, _ = model(hist, mpc.q_base_diag, mpc.r_base_diag)
            results[name] = {
                "Q_q1":   float(gQ[:, 0].mean()),
                "fe_norm": float(fe.norm()),
            }
    model.train()
    return results


def eval2k(model, mpc, x0, x_goal, steps=2000):
    """Full 2000-step rollout: fraction of time with wrap_dist < 0.10."""
    model.eval()
    x_t, _ = train_module.rollout(
        lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps,
    )
    traj  = x_t.cpu().numpy()
    wraps = np.array([
        math.sqrt(
            math.atan2(math.sin(s[0] - math.pi), math.cos(s[0] - math.pi)) ** 2
            + s[1] ** 2 + s[2] ** 2 + s[3] ** 2
        )
        for s in traj
    ])
    arr  = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    f01  = float((wraps < 0.10).mean())
    model.train()
    return f01, arr, post


def save_best(model_class_kwargs, best_state, meta, best_f01, save_dir, tag=""):
    name = f"stageF_mixed_v14l{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ep{meta}"
    m = network_module.SeparatedLinearizationNetwork(**model_class_kwargs).double()
    m.load_state_dict(best_state)
    network_module.ModelManager(base_dir=save_dir).save_training_session(
        model=m, loss_history=[],
        training_params={
            "experiment": "mixed_curriculum_v14l",
            "meta_epoch": meta,
            "best_f01":   best_f01,
            "w_q_profile": W_Q_PROFILE,
            "w_q_profile_bot": W_Q_PROFILE_BOT,
            "w_f_stable_bot": W_F_STABLE_BOT,
            "n_q_profile_steps": N_Q_PROFILE_STEPS,
            "pump":  PUMP,
            "stable": STABLE,
            "n_bottom": N_BOTTOM,
            "n_top":    N_TOP,
            "n_bottom_per_top": N_BOTTOM_PER_TOP,
            "f_gate_thresh_top":  F_GATE_THRESH_TOP,
            "detach_f_extra_top": DETACH_F_EXTRA_TOP,
            "w_f_pos_only_top":   W_F_POS_ONLY_TOP,
        },
        session_name=name,
    )
    return name


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    log = open(LOG_FILE, "w", buffering=1)

    def out(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     dtype=torch.float64, device=device)
    x_goal = torch.tensor(X_GOAL, dtype=torch.float64, device=device)

    out("=" * 80)
    out("  EXP: MIXED CURRICULUM v14l — velocity-gated fe suppression in episode A")
    out(f"  device: {device}")
    out(f"  Architecture: SEPARATED f_net (f+r heads) | q_net (q head only)")
    out(f"  Input: sin/cos goal-centred [sin(q1-π),cos(q1-π),dq1/8,sin(q2),cos(q2),dq2/8]")
    out(f"  Bottom A: {N_BOTTOM} steps × {N_BOTTOM_PER_TOP}/meta | energy | "
        f"w_f_stable={W_F_STABLE_BOT} | velocity-gated: suppresses hold fe, not swing-up")
    out(f"  Bottom B: {N_Q_PROFILE_STEPS} steps × {N_BOTTOM_PER_TOP}/meta | "
        f"optimizer_q | PUMP everywhere | w_q_profile_bot={W_Q_PROFILE_BOT}")
    out(f"  Top:    {N_TOP} steps × 1/meta | cos_q1 | w_q_profile={W_Q_PROFILE} | "
        f"near_pi_power={Q_NEAR_PI_POWER} | detach_f=True | w_stable={W_STABLE_PHASE}")
    out(f"  Q-profile (top): PUMP={PUMP} → STABLE={STABLE}  (power={Q_NEAR_PI_POWER})")
    out(f"  Near_pi@π/2: {0.5**Q_NEAR_PI_POWER:.4f} → target≈{0.5**Q_NEAR_PI_POWER*STABLE[0]+(1-0.5**Q_NEAR_PI_POWER)*PUMP[0]:.3f}")
    out(f"  Expected: Q@top→1.156, Q@mid→0.081, Q@bot→0.01 (equil. w/ bottom PUMP)")
    out(f"  META_EPOCHS={META_EPOCHS}  LR={LR}  hidden={HIDDEN_DIM}")
    out("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, dtype=torch.float64, device=device)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, dtype=torch.float64, device=device)

    demo_bottom = make_energy_demo(N_BOTTOM, device)
    demo_top    = make_hold_demo(N_TOP, device)

    model_kwargs = dict(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=F_KICKSTART_AMP,
    )
    model = network_module.SeparatedLinearizationNetwork(**model_kwargs).to(device).double()
    apply_q1_bias(model, Q_BIAS_Q1)

    # Separate optimizers: f_net trains in bottom, q_net trains in top.
    # Shared AdamW decimated q_net's effective LR: 3 zero-grad bottom updates
    # diluted q_net's Adam moments by ~4×, causing Q@top to grow at only 0.006/epoch.
    optimizer_f = torch.optim.AdamW(model.f_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer_q = torch.optim.AdamW(model.q_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Initial probe
    p = probe_network(model, mpc, device)
    out(f"\n  Init: Q[q1] bot={p['bot']['Q_q1']:.3f}  mid={p['mid']['Q_q1']:.3f}  "
        f"top={p['top']['Q_q1']:.3f}")
    out(f"        fe   bot={p['bot']['fe_norm']:.3f}  mid={p['mid']['fe_norm']:.3f}  "
        f"top={p['top']['fe_norm']:.3f}")
    out(f"\n  Starting ... (first CVXPY compile ~25 min)\n")
    hdr = (f"  {'Meta':>5}  {'L_bot':>8}  {'L_top':>8}  "
           f"{'Q@bot':>6}  {'Q@mid':>6}  {'Q@top':>6}  "
           f"{'fe@bot':>7}  {'fe@top':>7}  {'f01':>6}  {'arr':>5}  {'post':>6}")
    out(hdr)
    out("  " + "-" * len(hdr.rstrip()))

    best_f01   = 0.0
    best_state = None
    t0 = time.time()

    for meta in range(META_EPOCHS):
        # ── Bottom episodes: swing-up from x0=[0,0,0,0] ───────────────
        # v14l bottom loop — velocity-gated w_f_stable in episode A:
        #   A. Energy tracking + w_f_pos_only: suppression only fires when pendulum
        #      reaches π in the rollout. Self-regulating: no fe suppression when stuck.
        #   B. Short q_profile (optimizer_q, PUMP): Q@mid control.
        L_bot_last = float("nan")
        for _ in range(N_BOTTOM_PER_TOP):
            # A. Energy tracking + velocity-gated fe suppression
            loss_b, _ = train_module.train_linearization_network(
                lin_net=model, mpc=mpc,
                x0=x0, x_goal=x_goal, demo=demo_bottom,
                num_steps=N_BOTTOM, num_epochs=1, lr=LR,
                track_mode="energy",
                detach_gates_Q_for_qp=True,
                w_f_stable=W_F_STABLE_BOT,
                external_optimizer=optimizer_f,
                restore_best=False,
            )
            L_bot_last = loss_b[0] if loss_b else float("nan")

            # B. Short q_profile — trains q_net with PUMP target at all states
            train_module.train_linearization_network(
                lin_net=model, mpc=mpc,
                x0=x0, x_goal=x_goal, demo=demo_bottom,
                num_steps=N_Q_PROFILE_STEPS, num_epochs=1, lr=LR,
                track_mode="energy",
                detach_gates_Q_for_qp=True,
                w_q_profile=W_Q_PROFILE_BOT,
                q_profile_pump=PUMP,
                q_profile_stable=PUMP,
                q_profile_state_phase=True,
                external_optimizer=optimizer_q,
                restore_best=False,
            )

        # ── Top episode: hold from near-top random start ───────────────
        # ONLY q_head trains via QP gradient (detach_f_extra_for_qp=True).
        # f_head gets ZERO gradient from top (detach + gate). q_net trains freely.
        x0_top = sample_top(device)
        loss_t, _ = train_module.train_linearization_network(
            lin_net=model, mpc=mpc,
            x0=x0_top, x_goal=x_goal, demo=demo_top,
            num_steps=N_TOP, num_epochs=1, lr=LR,
            track_mode="cos_q1",
            w_q_profile=W_Q_PROFILE,
            q_profile_pump=PUMP,
            q_profile_stable=STABLE,
            q_profile_state_phase=True,
            q_profile_near_pi_power=Q_NEAR_PI_POWER,
            w_stable_phase=W_STABLE_PHASE,
            stable_phase_steps=STABLE_PHASE_STEPS,
            w_f_pos_only=W_F_POS_ONLY_TOP,
            f_gate_thresh=F_GATE_THRESH_TOP,
            detach_f_extra_for_qp=DETACH_F_EXTRA_TOP,
            external_optimizer=optimizer_q,
            restore_best=False,
        )
        L_top = loss_t[0] if loss_t else float("nan")

        # ── Probe network state ────────────────────────────────────────
        p = probe_network(model, mpc, device)

        f01_str = arr_str = post_str = "—"
        mark = ""
        if (meta + 1) % EVAL_EVERY == 0 or meta == 0:
            f01, arr, post = eval2k(model, mpc, x0, x_goal)
            f01_str  = f"{f01:.1%}"
            arr_str  = str(arr) if arr is not None else "None"
            post_str = f"{post:.1%}" if post is not None else "N/A"
            if f01 > best_f01:
                best_f01   = f01
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                mark = " ★"

        line = (f"  [{meta+1:>3}]  {L_bot_last:>8.3f}  {L_top:>8.3f}  "
                f"  {p['bot']['Q_q1']:.3f}  {p['mid']['Q_q1']:.3f}  {p['top']['Q_q1']:.3f}  "
                f"  {p['bot']['fe_norm']:.3f}   {p['top']['fe_norm']:.3f}  "
                f"  {f01_str}  {arr_str}  {post_str}{mark}")
        out(line)

        # Periodic checkpoint save
        if (meta + 1) % SAVE_EVERY == 0 and best_state is not None:
            name = save_best(model_kwargs, best_state, meta + 1, best_f01, SAVE_DIR)
            out(f"  → Saved: {name}")

    # Final save
    elapsed = time.time() - t0
    if best_state is not None:
        name = save_best(model_kwargs, best_state, META_EPOCHS, best_f01, SAVE_DIR, tag="_FINAL")
        out(f"\n  FINAL best f01={best_f01:.1%}  saved: {name}")
    out(f"  Total time: {elapsed/60:.1f} min")
    log.close()


if __name__ == "__main__":
    main()
