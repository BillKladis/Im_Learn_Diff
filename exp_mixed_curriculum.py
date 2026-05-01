"""exp_mixed_curriculum.py — Mixed curriculum v7: swing-up + hold, fully learned.

v7 key insight: detach_f_extra_for_qp=True in the TOP episode.

ROOT CAUSE IDENTIFIED (via gradient diagnostic):
  Bottom-only training: fe@bot INCREASES correctly (5.46→5.51 over 9 steps).
  Bootstrapping is working — energy tracking gradient consistently pushes f_head
  toward better pumping.

  v6 failure: even with f_gate_thresh=0.8, the pendulum falls out of the gate
  zone after ~12 steps. The remaining ~88 top-episode steps have gate inactive
  (near_pi < 0.8), so gradient DOES flow to f_head from these recovery states.
  These gradients pull fe@bot DOWN (opposite direction to bottom episode).

FIX (detach_f_extra_for_qp=True in top episode only):
  Added to Simulate.py: `detach_f_extra_for_qp` parameter.
  When True: f_extra = f_extra.detach() before entering QP.
  The QP loss gradient CANNOT propagate back through f_extra to f_head —
  only the Q/R paths (trunk → Q-head → Q → QP → loss) carry gradient.
  Result:
    → f_head receives ZERO gradient from the top episode (ALL states, not just near π)
    → Adam momentum for f_head: built purely from bottom-episode gradients
    → Q/trunk/encoder still trained from both episodes (complementary signals)
    → Top episode forces Q[q1]→STABLE for hold via Q-only path

PHILOSOPHY:
  No frozen modules. No hand-tuned gate formulas. No delta_Q corrections.
  Mixed curriculum: N_BOTTOM_PER_TOP bottom episodes + 1 top episode per
  meta-epoch, all through train_linearization_network() with shared AdamW.

  BOTTOM episode (x0=[0,0,0,0]):
    track_mode="energy" + state-conditional Q-profile (PUMP→STABLE)
    → f_head learns resonant energy pumping (clean gradient, no top interference)
    → diagnostic: bottom-only training shows fe@bot growing 5.46→5.51+ in 9 steps

  TOP episode (x0 near π, random perturbation):
    track_mode="cos_q1" + detach_f_extra_for_qp=True + f_gate_thresh + w_stable_phase
    → f_extra detached from QP gradient: f_head gets ZERO gradient from top episode
    → Q[q1] gradient path intact: Q-head learns STABLE profile from hold loss
    → f_gate_thresh=0.8: hard-zeros f_extra near top (so QP doesn't get bad FF)
    → Q differentiation forced: hold works only via Q, not f_extra cheating
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
F_EXTRA_BOUND   = 3.0
F_KICKSTART_AMP = 1.0
Q_BIAS_Q1       = -3.0

META_EPOCHS      = 200
N_BOTTOM_PER_TOP = 3      # bottom gradient steps per top step
N_BOTTOM         = 170    # match exp_no_demo.py proven step count
N_TOP            = 100    # hold episode length
LR               = 1e-3
WEIGHT_DECAY     = 1e-4

# Q-profile (state-conditional, shared by both episodes)
W_Q_PROFILE = 100.0
PUMP   = [0.01, 0.01, 1.0, 1.0]   # low Q[q1/q1d] at bottom → f_extra pumps
STABLE = [1.5,  1.5,  1.0, 1.0]   # high Q[q1/q1d] at top  → QP stabilises

# Top-specific: hold loss on ALL steps
W_STABLE_PHASE     = 3.0
STABLE_PHASE_STEPS = N_TOP   # fire from step 0

# Top-specific: hard-zero f_extra near top AND completely decouple f_head
# from top-episode gradient via detach_f_extra_for_qp.
# f_gate_thresh: zeroes f_extra when near_pi > threshold (good for QP behavior).
# detach_f_extra_for_qp: severs f_extra→QP gradient chain entirely, so f_head
# gets ZERO gradient from ANY top-episode state (not just near-top).
F_GATE_THRESH_TOP       = 0.8   # zero f_extra near top (near_pi > 0.8)
DETACH_F_EXTRA_TOP      = True  # cut f_head gradient path entirely in top episode

# Top-start perturbation ranges
TOP_PERT_Q1  = 0.30
TOP_PERT_Q1D = 0.60
TOP_PERT_Q2  = 0.20
TOP_PERT_Q2D = 0.50

EVAL_EVERY = 10
SAVE_EVERY = 50
SAVE_DIR   = "saved_models"
LOG_FILE   = "/tmp/mixed_curriculum.log"


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
    name = f"stageF_mixed_v7{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ep{meta}"
    m = network_module.LinearizationNetwork(**model_class_kwargs).double()
    m.load_state_dict(best_state)
    network_module.ModelManager(base_dir=save_dir).save_training_session(
        model=m, loss_history=[],
        training_params={
            "experiment": "mixed_curriculum_v7",
            "meta_epoch": meta,
            "best_f01":   best_f01,
            "w_q_profile": W_Q_PROFILE,
            "pump":  PUMP,
            "stable": STABLE,
            "n_bottom": N_BOTTOM,
            "n_top":    N_TOP,
            "n_bottom_per_top": N_BOTTOM_PER_TOP,
            "f_gate_thresh_top": F_GATE_THRESH_TOP,
            "detach_f_extra_top": DETACH_F_EXTRA_TOP,
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

    out("=" * 76)
    out("  EXP: MIXED CURRICULUM v7 — detach_f_extra_for_qp in top episode")
    out(f"  device: {device}")
    out(f"  Bottom: {N_BOTTOM}steps × {N_BOTTOM_PER_TOP}/meta | energy | w_q_profile={W_Q_PROFILE}")
    out(f"  Top: {N_TOP}steps × 1/meta | cos_q1 | w_stable={W_STABLE_PHASE} | f_gate={F_GATE_THRESH_TOP}")
    out(f"  Q-profile: PUMP={PUMP} → STABLE={STABLE}  (state-conditional)")
    out(f"  detach_f_extra_for_qp=True in top: f_head gets ZERO gradient from top episode")
    out(f"  f_gate_thresh_top={F_GATE_THRESH_TOP}: also hard-zeros f_extra near π during rollout")
    out(f"  META_EPOCHS={META_EPOCHS}  LR={LR}  EVAL_EVERY={EVAL_EVERY}")
    out("=" * 76)

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
    model = network_module.LinearizationNetwork(**model_kwargs).to(device).double()
    apply_q1_bias(model, Q_BIAS_Q1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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
        # f_head gradient is 100% from these bottom episodes (since top
        # episode uses f_gate_thresh which detaches gradient to f_head).
        # N_BOTTOM_PER_TOP gradient steps amplify the swing-up gradient.
        L_bot_last = float("nan")
        for _ in range(N_BOTTOM_PER_TOP):
            loss_b, _ = train_module.train_linearization_network(
                lin_net=model, mpc=mpc,
                x0=x0, x_goal=x_goal, demo=demo_bottom,
                num_steps=N_BOTTOM, num_epochs=1, lr=LR,
                track_mode="energy",
                w_q_profile=W_Q_PROFILE,
                q_profile_pump=PUMP,
                q_profile_stable=STABLE,
                q_profile_state_phase=True,
                external_optimizer=optimizer,
                restore_best=False,
            )
            L_bot_last = loss_b[0] if loss_b else float("nan")

        # ── Top episode: hold from near-top random start ───────────────
        # detach_f_extra_for_qp=True: completely severs gradient from QP back
        # to f_head. f_head receives ZERO gradient from this episode regardless
        # of where the pendulum is (near top or fallen to bottom).
        # f_gate_thresh=F_GATE_THRESH_TOP: additionally hard-zeros f_extra near π
        # so the QP doesn't receive destabilizing feedforward near the goal.
        # Only Q (via trunk → Q-head) gets gradient → forced Q differentiation.
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
            w_stable_phase=W_STABLE_PHASE,
            stable_phase_steps=STABLE_PHASE_STEPS,
            f_gate_thresh=F_GATE_THRESH_TOP,
            detach_f_extra_for_qp=DETACH_F_EXTRA_TOP,
            external_optimizer=optimizer,
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
