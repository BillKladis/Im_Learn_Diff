"""exp_noise_train.py — Train no-demo with NOISE INJECTION on observations.

Same successful no-demo config (synthetic energy ramp + state-phase
Q-profile + end-phase Q-high) but with Gaussian noise added to the
state_history during training.  Goal: trained model robust to MUCH
higher noise levels than the clean-trained model.

Train at σ_q=0.10, σ_qd=0.30 (heavier than expected deployment noise
to add safety margin).
"""

import math
import os
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

X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
EPOCHS    = 100
LR        = 1e-3
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM, CONTROL_DIM = 4, 2
SAVE_DIR    = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

# NOISE INJECTION DURING TRAINING (heavy enough for robustness margin)
TRAIN_NOISE_SIGMA = [0.10, 0.30, 0.10, 0.30]


def apply_q1_kickstart(lin_net, state_dim, horizon, bias_val):
    q_final = [m for m in lin_net.q_head.modules()
               if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            q_final.bias[k * state_dim + 0] = bias_val
            q_final.bias[k * state_dim + 1] = bias_val


def make_synthetic_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


class PrintMonitor:
    def __init__(self, num_epochs):
        self.num_epochs    = num_epochs
        self._header_shown = False
    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>9}  {'GoalDist':>9}  "
              f"{'QDev':>7}  {'fNorm':>7}  {'fτ1[0]':>8}  {'Time':>6}")
        print("─" * 100)
        self._header_shown = True
    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        if epoch == 0 or (epoch+1) % 2 == 0 or epoch == num_epochs-1:
            print(f"{epoch+1:>4}/{num_epochs:<4}"
                  f"  {loss:>10.3f}"
                  f"  {info.get('loss_track',        float('nan')):>9.3f}"
                  f"  {info.get('pure_end_error',    float('nan')):>9.4f}"
                  f"  {info.get('mean_Q_gate_dev',   float('nan')):>7.4f}"
                  f"  {info.get('mean_f_extra_norm', float('nan')):>7.3f}"
                  f"  {info.get('mean_f_tau1_first', float('nan')):>8.3f}"
                  f"  {info.get('epoch_time',        float('nan')):>5.2f}s",
                  flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo = make_synthetic_demo(NUM_STEPS, device)

    print("=" * 76)
    print("  EXP: train with NOISE INJECTION on observations")
    print(f"  q_base_diag = {Q_BASE_DIAG}  (DEFAULT)")
    print(f"  TRAIN_NOISE_SIGMA = {TRAIN_NOISE_SIGMA}")
    print(f"  Target: model robust at all tested noise levels (incl. xhigh)")
    print(f"  EPOCHS = {EPOCHS}  LR = {LR}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=0.99, gate_range_r=0.20,
        f_extra_bound=3.0, f_kickstart_amp=0.0,
    ).to(device).double()
    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, -3.0)

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(num_epochs=EPOCHS)

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=False, track_mode="energy", w_terminal_anchor=0.0,
        w_q_profile=100.0,
        q_profile_pump=[0.01, 0.01, 1.0, 1.0],
        q_profile_stable=[1.0, 1.0, 1.0, 1.0],
        q_profile_state_phase=True,
        w_end_q_high=80.0,
        end_phase_steps=20,
        train_noise_sigma=TRAIN_NOISE_SIGMA,   # KEY: inject noise during training
    )
    elapsed = time.time() - t0

    x_final, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    dist_clean = float(np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL)))
    print(f"\n  CLEAN rollout goal_dist = {dist_clean:.4f}  (best ckpt restored)")

    session_name = f"stageD_noisetrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "no_demo_with_noise_injection",
            "q_base_diag": Q_BASE_DIAG,
            "train_noise_sigma": TRAIN_NOISE_SIGMA,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"  Saved → saved_models/{session_name}/")
    print(f"  Total training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
