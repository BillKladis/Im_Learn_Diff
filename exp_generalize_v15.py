"""exp_generalize_v15.py — Generalization: perturbed starts, fine-tune from v14j.

Goal: teach the controller to swing up and hold from a range of initial positions,
not just the single hanging position [0, 0, 0, 0].

v15 = v14m curriculum + perturbed bottom starts + load from v14j checkpoint.

Strategy:
  - Fine-tune from v14j ep40 (88.6% f01, arr=197) which already swings up reliably.
  - Gradually expand starting perturbations from [0,0,0,0] over PERT_RAMP_EPOCHS.
  - Bottom episode x0 sampled from U(-pert_range, pert_range) for q1, q2 (at rest).
  - make_energy_demo() updated to ramp from q1_start → π (not always from 0).
  - Multi-start eval: mean f01 over N_EVAL_STARTS sampled starts + original [0,0,0,0].
  - Lower LR (5e-4) for fine-tuning to avoid catastrophic forgetting.

Curriculum unchanged from v14m:
  A. Clean energy tracking with perturbed x0 (no fe suppression).
  B-fe. Near-top short rollout + w_f_pos_only=0.2 (meta >= SUPPRESS_START).
  B-q. Short q_profile PUMP (optimizer_q).
  Top. Hold episode with cos_q1 + q_profile STABLE (optimizer_q).
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
F_EXTRA_BOUND   = 2.5
F_KICKSTART_AMP = 1.0
Q_BIAS_Q1       = -3.0

Q_NEAR_PI_POWER = 4

META_EPOCHS      = 150
N_BOTTOM_PER_TOP = 3
N_BOTTOM         = 170
N_TOP            = 100
LR               = 5e-4    # lower than v14m (1e-3) — fine-tuning from checkpoint
WEIGHT_DECAY     = 1e-4

W_Q_PROFILE    = 100.0
PUMP    = [0.01, 0.01, 1.0, 1.0]
STABLE  = [1.5,  1.5,  1.0, 1.0]

W_STABLE_PHASE     = 3.0
STABLE_PHASE_STEPS = N_TOP

W_F_POS_ONLY_TOP = 0.0

W_F_POS_ONLY_FE = 1.0    # stronger than v14m=0.2 — must counter perturbed bottom gradient
N_FE_STEPS      = 20     # more steps — 20×3=60 B-fe steps/meta to maintain fe@top suppression
SUPPRESS_START  = 0      # start from ep1 — checkpoint already has fe@top suppressed, must maintain

W_Q_PROFILE_BOT   = 10.0
N_Q_PROFILE_STEPS = 5

F_GATE_THRESH_TOP  = 0.8
DETACH_F_EXTRA_TOP = True

# Top-start perturbation ranges (hold episodes — same as v14m)
TOP_PERT_Q1  = 0.30
TOP_PERT_Q1D = 0.60
TOP_PERT_Q2  = 0.20
TOP_PERT_Q2D = 0.50

# Bottom-start perturbation (generalization — gradually expanded).
# Starts at zero pert, ramps linearly to MAX over PERT_RAMP_EPOCHS meta-epochs.
# q1/q2 only (at rest: q1d=q2d=0). Small velocity perturbation added after q1 ramp.
MAX_BOT_PERT_Q1  = 0.50   # ±29° on inner pendulum starting angle
MAX_BOT_PERT_Q2  = 0.40   # ±23° on outer pendulum starting angle
MAX_BOT_PERT_QD  = 0.30   # ±0.3 rad/s initial velocity (small)
PERT_RAMP_EPOCHS = 60     # reach full perturbation range by this meta-epoch

# Multi-start eval config
N_EVAL_STARTS   = 5       # number of random perturbed starts in eval (+ fixed x0)
EVAL_EVERY      = 10
SAVE_EVERY      = 50
SAVE_DIR        = "saved_models"
LOG_FILE        = "/tmp/generalize_v15.log"

# Checkpoint to fine-tune from — best v14m result (89.9% f01, arr=168)
LOAD_CHECKPOINT = "saved_models/stageF_mixed_v14m_20260503_102608_ep50/stageF_mixed_v14m_20260503_102608_ep50.pth"


# ── Helpers ───────────────────────────────────────────────────────────────
def make_energy_demo(n, device, q1_start=0.0):
    """Cosine-eased q1 ramp q1_start→π for energy-tracking episodes."""
    demo = torch.zeros((n, 4), dtype=torch.float64, device=device)
    span = math.pi - q1_start
    for i in range(n):
        alpha = i / max(n - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = q1_start + span * t
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


def sample_bottom(device, pert_scale):
    """Random perturbed hanging start. pert_scale in [0, 1] controls range."""
    q1  = random.uniform(-MAX_BOT_PERT_Q1 * pert_scale, MAX_BOT_PERT_Q1 * pert_scale)
    q2  = random.uniform(-MAX_BOT_PERT_Q2 * pert_scale, MAX_BOT_PERT_Q2 * pert_scale)
    q1d = random.uniform(-MAX_BOT_PERT_QD * pert_scale, MAX_BOT_PERT_QD * pert_scale)
    q2d = random.uniform(-MAX_BOT_PERT_QD * pert_scale, MAX_BOT_PERT_QD * pert_scale)
    return torch.tensor([q1, q1d, q2, q2d], dtype=torch.float64, device=device)


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
                "Q_q1":    float(gQ[:, 0].mean()),
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


def eval_multi(model, mpc, x0_fixed, x_goal, device, n_perturbed=N_EVAL_STARTS, steps=2000):
    """Eval over fixed start + n_perturbed random perturbed starts. Returns mean f01."""
    starts = [x0_fixed]
    for _ in range(n_perturbed):
        starts.append(sample_bottom(device, pert_scale=1.0))
    results = []
    for x0 in starts:
        f01, arr, post = eval2k(model, mpc, x0, x_goal, steps)
        results.append((f01, arr, post))
    f01s = [r[0] for r in results]
    mean_f01 = float(np.mean(f01s))
    min_f01  = float(np.min(f01s))
    fixed_f01, fixed_arr, fixed_post = results[0]
    return mean_f01, min_f01, fixed_f01, fixed_arr, fixed_post


def save_checkpoint(model_class_kwargs, state_dict, meta, f01_label, save_dir, tag=""):
    """Save a given state_dict (best or current) to disk."""
    name = f"stageF_gen_v15{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ep{meta}"
    m = network_module.SeparatedLinearizationNetwork(**model_class_kwargs).double()
    m.load_state_dict(state_dict)
    network_module.ModelManager(base_dir=save_dir).save_training_session(
        model=m, loss_history=[],
        training_params={
            "experiment": "generalize_v15",
            "meta_epoch": meta,
            "f01_label":  f01_label,
            "load_checkpoint": LOAD_CHECKPOINT,
            "max_bot_pert_q1": MAX_BOT_PERT_Q1,
            "max_bot_pert_q2": MAX_BOT_PERT_Q2,
            "pert_ramp_epochs": PERT_RAMP_EPOCHS,
            "w_f_pos_only_fe": W_F_POS_ONLY_FE,
            "suppress_start": SUPPRESS_START,
            "pump": PUMP, "stable": STABLE,
        },
        session_name=name,
    )
    return name


def save_best(model_class_kwargs, best_state, meta, best_f01, save_dir, tag=""):
    return save_checkpoint(model_class_kwargs, best_state, meta, f"{best_f01:.1%}", save_dir, tag)


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
    out("  EXP: GENERALIZE v15 — perturbed starts, fine-tune from v14j")
    out(f"  device: {device}")
    out(f"  Checkpoint: {LOAD_CHECKPOINT}")
    out(f"  Bottom A: {N_BOTTOM} steps × {N_BOTTOM_PER_TOP}/meta | energy | perturbed x0")
    out(f"  Perturbation: q1±{MAX_BOT_PERT_Q1:.2f} q2±{MAX_BOT_PERT_Q2:.2f} "
        f"qd±{MAX_BOT_PERT_QD:.2f} | ramp over {PERT_RAMP_EPOCHS} epochs")
    out(f"  Bottom B-fe: {N_FE_STEPS} steps × {N_BOTTOM_PER_TOP}/meta | near-top | "
        f"w_f_pos_only={W_F_POS_ONLY_FE} | starts at meta={SUPPRESS_START}")
    out(f"  Bottom B-q: {N_Q_PROFILE_STEPS} steps × {N_BOTTOM_PER_TOP}/meta | "
        f"PUMP everywhere | w_q_profile_bot={W_Q_PROFILE_BOT}")
    out(f"  Top: {N_TOP} steps × 1/meta | cos_q1 | w_q_profile={W_Q_PROFILE} | "
        f"detach_f=True | w_stable={W_STABLE_PHASE}")
    out(f"  Eval: fixed [0,0,0,0] + {N_EVAL_STARTS} random perturbed starts (mean f01)")
    out(f"  META_EPOCHS={META_EPOCHS}  LR={LR}  hidden={HIDDEN_DIM}")
    out("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, dtype=torch.float64, device=device)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, dtype=torch.float64, device=device)

    demo_top = make_hold_demo(N_TOP, device)

    model_kwargs = dict(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=F_KICKSTART_AMP,
    )
    model = network_module.SeparatedLinearizationNetwork(**model_kwargs).to(device).double()

    # Load v14j checkpoint
    ckpt = torch.load(LOAD_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    out(f"\n  Loaded checkpoint. Verifying...")

    optimizer_f = torch.optim.AdamW(model.f_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer_q = torch.optim.AdamW(model.q_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    p = probe_network(model, mpc, device)
    out(f"  Init: Q[q1] bot={p['bot']['Q_q1']:.3f}  mid={p['mid']['Q_q1']:.3f}  "
        f"top={p['top']['Q_q1']:.3f}")
    out(f"        fe   bot={p['bot']['fe_norm']:.3f}  mid={p['mid']['fe_norm']:.3f}  "
        f"top={p['top']['fe_norm']:.3f}")

    # Verify initial performance from fixed x0
    f01_init, arr_init, post_init = eval2k(model, mpc, x0, x_goal)
    out(f"  Baseline (x0=[0,0,0,0]): f01={f01_init:.1%}  arr={arr_init}  "
        f"post={post_init:.1%}")
    out(f"\n  Starting training ... (CVXPY already compiled)\n")

    hdr = (f"  {'Meta':>5}  {'L_bot':>8}  {'L_top':>8}  "
           f"{'Q@bot':>6}  {'Q@mid':>6}  {'Q@top':>6}  "
           f"{'fe@bot':>7}  {'fe@top':>7}  {'pert':>5}  "
           f"{'f01_mn':>7}  {'f01_fx':>7}  {'arr':>5}  {'post':>6}")
    out(hdr)
    out("  " + "-" * len(hdr.rstrip()))

    best_f01        = f01_init    # best fixed-start f01 (for compatibility)
    best_state      = {k: v.clone() for k, v in model.state_dict().items()}
    best_mean_f01   = 0.0         # best mean f01 at full perturbation range
    best_mean_state = None
    t0 = time.time()

    for meta in range(META_EPOCHS):
        # Perturbation scale: linearly ramps from 0 → 1 over PERT_RAMP_EPOCHS
        pert_scale = min(1.0, (meta + 1) / PERT_RAMP_EPOCHS)

        # ── Bottom episodes ────────────────────────────────────────────
        L_bot_last = float("nan")
        for _ in range(N_BOTTOM_PER_TOP):
            # A. Perturbed energy tracking — no suppression
            x0_bot = sample_bottom(device, pert_scale)
            demo_bottom = make_energy_demo(N_BOTTOM, device, q1_start=float(x0_bot[0]))
            loss_b, _ = train_module.train_linearization_network(
                lin_net=model, mpc=mpc,
                x0=x0_bot, x_goal=x_goal, demo=demo_bottom,
                num_steps=N_BOTTOM, num_epochs=1, lr=LR,
                track_mode="energy",
                detach_gates_Q_for_qp=True,
                external_optimizer=optimizer_f,
                restore_best=False,
            )
            L_bot_last = loss_b[0] if loss_b else float("nan")

            # B-fe. Delayed near-top fe suppression
            if meta >= SUPPRESS_START:
                x0_fe = sample_top(device)
                train_module.train_linearization_network(
                    lin_net=model, mpc=mpc,
                    x0=x0_fe, x_goal=x_goal, demo=demo_top,
                    num_steps=N_FE_STEPS, num_epochs=1, lr=LR,
                    track_mode="cos_q1",
                    detach_gates_Q_for_qp=True,
                    detach_f_extra_for_qp=True,
                    w_f_pos_only=W_F_POS_ONLY_FE,
                    external_optimizer=optimizer_f,
                    restore_best=False,
                )

            # B-q. Short q_profile PUMP — trains q_net
            x0_bot_q = sample_bottom(device, pert_scale)
            demo_bot_q = make_energy_demo(N_Q_PROFILE_STEPS, device, q1_start=float(x0_bot_q[0]))
            train_module.train_linearization_network(
                lin_net=model, mpc=mpc,
                x0=x0_bot_q, x_goal=x_goal, demo=demo_bot_q,
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

        # ── Top episode ────────────────────────────────────────────────
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

        # ── Probe and eval ─────────────────────────────────────────────
        p = probe_network(model, mpc, device)

        f01_mn_str = f01_fx_str = arr_str = post_str = "—"
        mark = ""
        if (meta + 1) % EVAL_EVERY == 0 or meta == 0:
            mean_f01, min_f01, fixed_f01, fixed_arr, fixed_post = eval_multi(
                model, mpc, x0, x_goal, device
            )
            f01_mn_str = f"{mean_f01:.1%}"
            f01_fx_str = f"{fixed_f01:.1%}"
            arr_str    = str(fixed_arr) if fixed_arr is not None else "None"
            post_str   = f"{fixed_post:.1%}" if fixed_post is not None else "N/A"
            # Best tracked on fixed start f01 (comparable to v14 series)
            if fixed_f01 > best_f01:
                best_f01   = fixed_f01
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                mark = " ★"
            # Track best MEAN f01 once at full perturbation range
            if pert_scale >= 1.0 and mean_f01 > best_mean_f01:
                best_mean_f01   = mean_f01
                best_mean_state = {k: v.clone() for k, v in model.state_dict().items()}

        line = (f"  [{meta+1:>3}]  {L_bot_last:>8.3f}  {L_top:>8.3f}  "
                f"  {p['bot']['Q_q1']:.3f}  {p['mid']['Q_q1']:.3f}  {p['top']['Q_q1']:.3f}  "
                f"  {p['bot']['fe_norm']:.3f}   {p['top']['fe_norm']:.3f}  "
                f"  {pert_scale:.2f}  "
                f"  {f01_mn_str}  {f01_fx_str}  {arr_str}  {post_str}{mark}")
        out(line)

        if (meta + 1) % SAVE_EVERY == 0:
            # Save current state (not best_state) — generalization training improves over time
            curr_state = {k: v.clone() for k, v in model.state_dict().items()}
            name = save_checkpoint(model_kwargs, curr_state, meta + 1,
                                   f"curr_mean={mean_f01:.1%}_fix={fixed_f01:.1%}", SAVE_DIR)
            out(f"  → Saved (current): {name}")
            if best_mean_state is not None:
                name_m = save_checkpoint(model_kwargs, best_mean_state, meta + 1,
                                         f"best_mean={best_mean_f01:.1%}", SAVE_DIR, tag="_BESTMEAN")
                out(f"  → Saved (best_mean): {name_m}")

    elapsed = time.time() - t0
    # Save final current state AND best-mean state
    curr_state = {k: v.clone() for k, v in model.state_dict().items()}
    name = save_checkpoint(model_kwargs, curr_state, META_EPOCHS,
                           f"final_curr", SAVE_DIR, tag="_FINAL")
    out(f"\n  FINAL curr state saved: {name}")
    if best_mean_state is not None:
        name_m = save_checkpoint(model_kwargs, best_mean_state, META_EPOCHS,
                                 f"best_mean={best_mean_f01:.1%}", SAVE_DIR, tag="_BESTMEAN")
        out(f"  FINAL best_mean={best_mean_f01:.1%}  saved: {name_m}")
    out(f"  Total time: {elapsed/60:.1f} min")
    log.close()


if __name__ == "__main__":
    main()
