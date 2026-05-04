"""exp_noise_robust_v16.py — Robustify v14m against actuator and sensor noise.

Fine-tune the best single-start model (v14m, 89.9% f01) with noise augmentation
so it maintains performance under the control noise that halved f01 in eval.

Noise schedule (progressive over NOISE_RAMP_EPOCHS):
  Phase 0 (warm-up, epochs 1-NOISE_RAMP_EPOCHS):  noise linearly ramps 0 → target
  Phase 1 (steady, after NOISE_RAMP_EPOCHS):       noise held at target

Target noise:  σ_ctrl = 0.05 per joint  (half of the σ=0.10 that devastated eval)
               σ_obs  = [0.02, 0.05, 0.02, 0.05]  (angles/velocities separately)

Curriculum: identical to v14m (4-episode meta-step: A + B-fe + B-q + Top).
No generalization perturbations — just the canonical x0=[0,0,0,0] start.
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

# ── Config ─────────────────────────────────────────────────────────────────
X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
DT        = 0.05
HORIZON   = 10
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
STATE_DIM   = 4
CONTROL_DIM = 2
HIDDEN_DIM  = 128
GATE_RANGE_Q    = 0.99
GATE_RANGE_R    = 0.20
F_EXTRA_BOUND   = 2.5
F_KICKSTART_AMP = 1.0

META_EPOCHS      = 100
N_BOTTOM_PER_TOP = 3
N_BOTTOM  = 170
N_TOP     = 100
N_FE_STEPS = 5
N_Q_STEPS  = 5
LR           = 2e-4    # fine-tuning: lower than v14m (1e-3)
WEIGHT_DECAY = 1e-4

W_Q_PROFILE  = 100.0
PUMP    = [0.01, 0.01, 1.0, 1.0]
STABLE  = [1.5,  1.5,  1.0, 1.0]

W_STABLE_PHASE     = 3.0
STABLE_PHASE_STEPS = N_TOP
W_F_POS_ONLY_FE = 0.2
F_GATE_THRESH_TOP  = 0.8
DETACH_F_EXTRA_TOP = True
W_Q_PROFILE_BOT    = 10.0
Q_NEAR_PI_POWER    = 4

# Noise targets (applied to training rollouts)
CTRL_SIGMA_TARGET = [0.05, 0.05]           # per joint torque noise
OBS_SIGMA_TARGET  = [0.02, 0.05, 0.02, 0.05]  # [q1, q1d, q2, q2d]

# Ramp: linearly increase noise from 0 to target over this many meta-epochs
NOISE_RAMP_EPOCHS = 20

# Top-episode perturbation (same as v14m)
TOP_PERT_Q1  = 0.30
TOP_PERT_Q1D = 0.60
TOP_PERT_Q2  = 0.20
TOP_PERT_Q2D = 0.50

EVAL_EVERY = 10
SAVE_EVERY = 50
SAVE_DIR   = "saved_models"
LOG_FILE   = "/tmp/noise_robust_v16.log"

LOAD_CHECKPOINT = (
    "saved_models/stageF_mixed_v14m_20260503_102608_ep50/"
    "stageF_mixed_v14m_20260503_102608_ep50.pth"
)


# ── Helpers ────────────────────────────────────────────────────────────────
def make_energy_demo(n, device, q1_start=0.0):
    demo = torch.zeros((n, 4), dtype=torch.float64, device=device)
    span = math.pi - q1_start
    for i in range(n):
        alpha = i / max(n - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = q1_start + span * t
    return demo


def make_hold_demo(n, device):
    demo = torch.zeros((n, 4), dtype=torch.float64, device=device)
    demo[:, 0] = math.pi
    return demo


def sample_top(device):
    return torch.tensor([
        math.pi + random.uniform(-TOP_PERT_Q1,  TOP_PERT_Q1),
        random.uniform(-TOP_PERT_Q1D, TOP_PERT_Q1D),
        random.uniform(-TOP_PERT_Q2,  TOP_PERT_Q2),
        random.uniform(-TOP_PERT_Q2D, TOP_PERT_Q2D),
    ], dtype=torch.float64, device=device)


def noise_at(meta, target, ramp_epochs):
    """Scale noise linearly from 0 to target over ramp_epochs."""
    scale = min(1.0, (meta + 1) / ramp_epochs)
    return [s * scale for s in target]


def eval2k(model, mpc, x0, x_goal, steps=2000):
    model.eval()
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
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


def save_checkpoint(model_kwargs, state_dict, meta, label, save_dir, tag=""):
    name = f"stageF_noiserobust_v16{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ep{meta}"
    m = network_module.SeparatedLinearizationNetwork(**model_kwargs).double()
    m.load_state_dict(state_dict)
    network_module.ModelManager(base_dir=save_dir).save_training_session(
        model=m, loss_history=[],
        training_params={
            "experiment": "noise_robust_v16",
            "meta_epoch": meta,
            "label": label,
            "load_checkpoint": LOAD_CHECKPOINT,
            "ctrl_sigma_target": CTRL_SIGMA_TARGET,
            "obs_sigma_target": OBS_SIGMA_TARGET,
            "noise_ramp_epochs": NOISE_RAMP_EPOCHS,
        },
        session_name=name,
    )
    return name


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    log = open(LOG_FILE, "w", buffering=1)

    def out(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     dtype=torch.float64, device=device)
    x_goal = torch.tensor(X_GOAL, dtype=torch.float64, device=device)

    out("=" * 80)
    out("  EXP: NOISE ROBUST v16 — fine-tune v14m with ctrl + obs noise augmentation")
    out(f"  device: {device}")
    out(f"  Checkpoint: {LOAD_CHECKPOINT}")
    out(f"  Target ctrl σ: {CTRL_SIGMA_TARGET}  obs σ: {OBS_SIGMA_TARGET}")
    out(f"  Noise ramp: 0 → target over {NOISE_RAMP_EPOCHS} meta-epochs")
    out(f"  META_EPOCHS={META_EPOCHS}  LR={LR}  N_BOTTOM={N_BOTTOM}  N_TOP={N_TOP}")
    out("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, dtype=torch.float64, device=device)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, dtype=torch.float64, device=device)

    demo_top = make_hold_demo(N_TOP, device)
    demo_bottom = make_energy_demo(N_BOTTOM, device, q1_start=0.0)

    model_kwargs = dict(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=F_KICKSTART_AMP,
    )
    model = network_module.SeparatedLinearizationNetwork(**model_kwargs).to(device).double()

    ckpt = torch.load(LOAD_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    optimizer_f = torch.optim.AdamW(model.f_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer_q = torch.optim.AdamW(model.q_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    f01_init, arr_init, post_init = eval2k(model, mpc, x0, x_goal)
    out(f"\n  Baseline (clean): f01={f01_init:.1%}  arr={arr_init}  post={post_init:.1%}")
    out(f"\n  Starting training ...\n")

    hdr = (f"  {'Meta':>5}  {'L_bot':>8}  {'L_top':>8}  "
           f"{'σ_c':>5}  {'f01':>7}  {'arr':>5}  {'post':>6}  mark")
    out(hdr)
    out("  " + "─" * len(hdr.rstrip()))

    best_f01   = f01_init
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    t0 = time.time()

    for meta in range(META_EPOCHS):
        ctrl_sig = noise_at(meta, CTRL_SIGMA_TARGET, NOISE_RAMP_EPOCHS)
        obs_sig  = noise_at(meta, OBS_SIGMA_TARGET,  NOISE_RAMP_EPOCHS)

        L_bot_last = float("nan")
        for _ in range(N_BOTTOM_PER_TOP):
            # A. Swing-up energy tracking with noise
            loss_b, _ = train_module.train_linearization_network(
                lin_net=model, mpc=mpc,
                x0=x0, x_goal=x_goal, demo=demo_bottom,
                num_steps=N_BOTTOM, num_epochs=1, lr=LR,
                track_mode="energy",
                detach_gates_Q_for_qp=True,
                detach_f_extra_for_qp=True,
                external_optimizer=optimizer_f,
                restore_best=False,
                train_noise_sigma=obs_sig,
                train_ctrl_sigma=ctrl_sig,
            )
            L_bot_last = loss_b[0] if loss_b else float("nan")

            # B-fe. Near-top fe suppression (no ctrl noise — short, near goal)
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
                train_noise_sigma=obs_sig,
            )

            # B-q. Short PUMP for q_net
            demo_bot_q = make_energy_demo(N_Q_STEPS, device)
            train_module.train_linearization_network(
                lin_net=model, mpc=mpc,
                x0=x0, x_goal=x_goal, demo=demo_bot_q,
                num_steps=N_Q_STEPS, num_epochs=1, lr=LR,
                track_mode="energy",
                detach_gates_Q_for_qp=True,
                w_q_profile=W_Q_PROFILE_BOT,
                q_profile_pump=PUMP,
                q_profile_stable=PUMP,
                q_profile_state_phase=True,
                external_optimizer=optimizer_q,
                restore_best=False,
                train_noise_sigma=obs_sig,
            )

        # Top episode: hold with noise (trains robustness of hold)
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
            w_f_pos_only=0.0,
            f_gate_thresh=F_GATE_THRESH_TOP,
            detach_f_extra_for_qp=DETACH_F_EXTRA_TOP,
            external_optimizer=optimizer_q,
            restore_best=False,
            train_noise_sigma=obs_sig,
            train_ctrl_sigma=ctrl_sig,
        )
        L_top = loss_t[0] if loss_t else float("nan")

        # Eval
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

        σ_c = ctrl_sig[0]
        out(f"  [{meta+1:>3}]  {L_bot_last:>8.3f}  {L_top:>8.3f}  "
            f"  {σ_c:.3f}  {f01_str:>7}  {arr_str:>5}  {post_str:>6}{mark}")

        if (meta + 1) % SAVE_EVERY == 0:
            curr = {k: v.clone() for k, v in model.state_dict().items()}
            name = save_checkpoint(model_kwargs, curr, meta + 1,
                                   f"curr_f01={f01_str}", SAVE_DIR)
            out(f"  → Saved (current): {name}")
            name_b = save_checkpoint(model_kwargs, best_state, meta + 1,
                                     f"best_f01={best_f01:.1%}", SAVE_DIR, tag="_BEST")
            out(f"  → Saved (best):    {name_b}")

    elapsed = time.time() - t0
    curr = {k: v.clone() for k, v in model.state_dict().items()}
    name = save_checkpoint(model_kwargs, curr, META_EPOCHS,
                           "final_curr", SAVE_DIR, tag="_FINAL")
    out(f"\n  FINAL curr saved: {name}")
    name_b = save_checkpoint(model_kwargs, best_state, META_EPOCHS,
                             f"best_f01={best_f01:.1%}", SAVE_DIR, tag="_BEST")
    out(f"  FINAL best saved: {name_b}")
    out(f"  Total time: {elapsed/60:.1f} min")
    log.close()


if __name__ == "__main__":
    main()
