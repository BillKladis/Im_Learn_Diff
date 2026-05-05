"""exp_hardware_v2_continue.py — Continue v2 noise-robust training from ep80 diag.

Process died at ep89. This resumes from ep80 and runs 20 more epochs to
reach ep100 FINAL with full noise schedule (σ=0.005 stage throughout).
"""

import glob
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

X0     = [0.0, 0.0, 0.0, 0.0]
X_GOAL = [math.pi, 0.0, 0.0, 0.0]
DT     = 0.05
HORIZON = 10

STATE_DIM   = 4
CONTROL_DIM = 2
HIDDEN_DIM      = 128
GATE_RANGE_Q    = 0.99
GATE_RANGE_R    = 0.20
F_EXTRA_BOUND   = 1.5
F_KICKSTART_AMP = 0.01

META_EPOCHS_CONTINUE = 20   # ep80→ep100
START_EPOCH          = 80
NOISE_SIGMA          = [0.005, 0.10, 0.005, 0.10]  # hardest stage

N_BOTTOM_PER_TOP = 3
N_BOTTOM         = 25
N_TOP            = 100
LR               = 5e-4
WEIGHT_DECAY     = 1e-4

W_Q_PROFILE  = 100.0
PUMP         = [1.0, 1.0, 1.0, 1.0]
STABLE       = [2.0, 1.0, 2.0, 1.0]
W_F_END_REG      = 1.0
F_END_REG_STEPS  = 10
Q_NEAR_PI_POWER  = 4
W_STABLE_PHASE     = 3.0
STABLE_PHASE_STEPS = N_TOP
W_F_POS_ONLY_TOP = 0.3
F_GATE_THRESH_TOP  = 0.8
DETACH_F_EXTRA_TOP = True
W_F_POS_ONLY_FE = 0.5
N_FE_STEPS      = 5
W_Q_PROFILE_BOT   = 10.0
N_Q_PROFILE_STEPS = 5
TOP_PERT_Q1  = 0.30
TOP_PERT_Q1D = 0.30
TOP_PERT_Q2  = 0.20
TOP_PERT_Q2D = 0.30

SAVE_DIR = "saved_models"
LOG_FILE = "/tmp/hw_v2_continue.log"


def make_energy_demo(n, device):
    demo = torch.zeros((n, 4), dtype=torch.float64, device=device)
    for i in range(n):
        alpha = i / max(n - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


def make_hold_demo(n, device):
    demo = torch.zeros((n, 4), dtype=torch.float64, device=device)
    demo[:, 0] = math.pi
    return demo


def sample_top(device):
    return torch.tensor([
        math.pi + random.uniform(-TOP_PERT_Q1, TOP_PERT_Q1),
        random.uniform(-TOP_PERT_Q1D, TOP_PERT_Q1D),
        random.uniform(-TOP_PERT_Q2, TOP_PERT_Q2),
        random.uniform(-TOP_PERT_Q2D, TOP_PERT_Q2D),
    ], dtype=torch.float64, device=device)


def eval2k(model, mpc, x0, x_goal):
    model.eval()
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0,
                                   x_goal=x_goal, num_steps=2000)
    traj  = x_t.cpu().numpy()
    wraps = np.array([
        math.sqrt(
            math.atan2(math.sin(s[0] - math.pi), math.cos(s[0] - math.pi))**2
            + s[1]**2 + s[2]**2 + s[3]**2
        ) for s in traj
    ])
    arr  = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    f01  = float((wraps < 0.10).mean())
    model.train()
    return f01, arr, post


def save_checkpoint(model_kwargs, state_dict, meta, label, tag=""):
    name = f"hw_v2_nr{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ep{meta}"
    m = network_module.SeparatedLinearizationNetwork(**model_kwargs).double()
    m.load_state_dict(state_dict)
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=m, loss_history=[],
        training_params={"experiment": "hw_v2_continue", "meta_epoch": meta, "label": label},
        session_name=name,
    )
    return name


def main():
    log = open(LOG_FILE, "w", buffering=1)
    def out(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0, dtype=torch.float64, device=device)
    x_goal = torch.tensor(X_GOAL, dtype=torch.float64, device=device)

    out(f"  Resuming v2 from ep80, running {META_EPOCHS_CONTINUE} more epochs")
    out(f"  σ={NOISE_SIGMA}  device={device}")

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, dtype=torch.float64, device=device)

    demo_bottom = make_energy_demo(N_BOTTOM, device)
    demo_top    = make_hold_demo(N_TOP, device)

    model_kwargs = dict(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=F_KICKSTART_AMP,
    )

    ckpt = max(glob.glob("saved_models/hw_v2_nr_diag_20260505_095639_ep80/*.pth"),
               key=os.path.getmtime)
    out(f"  Loading: {ckpt}")
    data = torch.load(ckpt, map_location=device, weights_only=False)
    sd   = data.get("model_state_dict", data)
    model = network_module.SeparatedLinearizationNetwork(**model_kwargs).to(device).double()
    model.load_state_dict(sd)

    optimizer_f = torch.optim.AdamW(model.f_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer_q = torch.optim.AdamW(model.q_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    sigma = NOISE_SIGMA
    best_f01 = 0.0
    best_state = None
    t0 = time.time()

    out(f"\n  {'Ep':>5}  {'L_bot':>8}  {'L_top':>8}  {'f01':>7}  {'arr':>5}  {'post':>6}")
    out("  " + "-"*55)

    for i in range(META_EPOCHS_CONTINUE):
        meta = START_EPOCH + i

        for _ in range(N_BOTTOM_PER_TOP):
            train_module.train_linearization_network(
                lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, demo=demo_bottom,
                num_steps=N_BOTTOM, num_epochs=1, lr=LR, track_mode="energy",
                detach_gates_Q_for_qp=True, w_f_end_reg=W_F_END_REG,
                f_end_reg_steps=F_END_REG_STEPS, train_noise_sigma=sigma,
                external_optimizer=optimizer_f, restore_best=False,
            )
            x0_fe = sample_top(device)
            train_module.train_linearization_network(
                lin_net=model, mpc=mpc, x0=x0_fe, x_goal=x_goal, demo=demo_top,
                num_steps=N_FE_STEPS, num_epochs=1, lr=LR, track_mode="cos_q1",
                detach_gates_Q_for_qp=True, detach_f_extra_for_qp=True,
                w_f_pos_only=W_F_POS_ONLY_FE, external_optimizer=optimizer_f,
                restore_best=False,
            )
            train_module.train_linearization_network(
                lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, demo=demo_bottom,
                num_steps=N_Q_PROFILE_STEPS, num_epochs=1, lr=LR, track_mode="energy",
                detach_gates_Q_for_qp=True, w_q_profile=W_Q_PROFILE_BOT,
                q_profile_pump=PUMP, q_profile_stable=PUMP, q_profile_state_phase=True,
                train_noise_sigma=sigma, external_optimizer=optimizer_q, restore_best=False,
            )

        x0_top = sample_top(device)
        loss_t, _ = train_module.train_linearization_network(
            lin_net=model, mpc=mpc, x0=x0_top, x_goal=x_goal, demo=demo_top,
            num_steps=N_TOP, num_epochs=1, lr=LR, track_mode="cos_q1",
            w_q_profile=W_Q_PROFILE, q_profile_pump=PUMP, q_profile_stable=STABLE,
            q_profile_state_phase=True, q_profile_near_pi_power=Q_NEAR_PI_POWER,
            w_stable_phase=W_STABLE_PHASE, stable_phase_steps=STABLE_PHASE_STEPS,
            w_f_pos_only=W_F_POS_ONLY_TOP, f_gate_thresh=F_GATE_THRESH_TOP,
            detach_f_extra_for_qp=DETACH_F_EXTRA_TOP,
            external_optimizer=optimizer_q, restore_best=False,
        )
        L_top = loss_t[0] if loss_t else float("nan")

        f01, arr, post = eval2k(model, mpc, x0, x_goal)
        f01_str  = f"{f01:.1%}"
        arr_str  = str(arr) if arr is not None else "None"
        post_str = f"{post:.1%}" if post is not None else "N/A"
        mark = ""
        if f01 > best_f01:
            best_f01   = f01
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            mark = " ★"

        L_bot = float("nan")
        out(f"  [{meta+1:>3}]  {L_bot:>8.3f}  {L_top:>8.3f}  {f01_str:>7}  {arr_str:>5}  {post_str:>6}{mark}")

    elapsed = time.time() - t0
    if best_state is not None:
        name = save_checkpoint(model_kwargs, best_state, START_EPOCH + META_EPOCHS_CONTINUE,
                               f"best_f01={best_f01:.1%}", tag="_FINAL")
        out(f"\n  FINAL ep100  best f01={best_f01:.1%}  saved: {name}")
    out(f"  Total time: {elapsed/60:.1f} min")
    log.close()


if __name__ == "__main__":
    main()
