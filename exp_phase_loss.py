"""
exp_phase_loss.py — Phase-conditional outer loss with default q_base_diag.

User's idea: directly expose gates_Q[:,0] and f_extra magnitudes to the
outer loss.  For the first 60-70% of the trajectory ("pumping phase"),
penalise large q1 gate and reward large |f_extra|.  For the last
30-40% ("stabilisation phase"), no extra penalty so the network can
use Q-gates normally for stabilisation.

This directly counteracts the gradient trap by giving the optimizer
a strong, phase-aware signal that q1 should be near zero during pump.

Configuration:
  q_base_diag = [12, 5, 50, 40]   (DEFAULT, NOT zeroed)
  gate_range_q = 0.99
  q1 kickstart bias = -4.0  (init gate ≈ 0.011)
  f kickstart amp = 1.0     (sinusoidal pumping pattern)
  TRACK_MODE = "energy"
  w_q1_phase_pen = 5.0
  w_f_phase_reward = 1.0
  phase_split_frac = 0.7    (steps 0-118 are pump, 119-170 are stabilise)
  EPOCHS = 150              (longer to allow convergence)
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

DEMO_CSV  = "run_20260428_001459_rollout_final.csv"
X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
EPOCHS    = 150
LR        = 5e-4
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM   = 4
CONTROL_DIM = 2
SAVE_DIR    = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]   # DEFAULT, NOT zeroed
GATE_RANGE_Q   = 0.99
GATE_RANGE_R   = 0.20
F_EXTRA_BOUND  = 3.0
F_KICKSTART    = 1.0
Q_GATE_KICKSTART_BIAS = -4.0

# Phase-conditional outer loss
W_Q1_PHASE_PEN   = 5.0    # Penalty on q1 gate during pump phase
W_F_PHASE_REWARD = 1.0    # Reward (penalty for shortfall) on |f_extra| during pump
PHASE_SPLIT_FRAC = 0.7    # First 70% is pump, last 30% is stabilisation

TRACK_MODE = "energy"


def apply_q1_kickstart(lin_net, state_dim, horizon, bias_val):
    q_final = [m for m in lin_net.q_head.modules()
               if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            q_final.bias[k * state_dim + 0] = bias_val


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
        if epoch == 0 or (epoch+1) % 5 == 0 or epoch == num_epochs-1:
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
    demo   = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)

    print("=" * 76)
    print("  EXP: Phase-conditional outer loss + default q_base_diag")
    print(f"  q_base_diag = {Q_BASE_DIAG}")
    print(f"  Phase split: pump=0-{int(PHASE_SPLIT_FRAC * NUM_STEPS)}  "
          f"stabilise={int(PHASE_SPLIT_FRAC * NUM_STEPS)}-{NUM_STEPS}")
    print(f"  w_q1_phase_pen={W_Q1_PHASE_PEN}  w_f_phase_reward={W_F_PHASE_REWARD}")
    print(f"  Epochs={EPOCHS}  LR={LR}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=F_KICKSTART,
    ).to(device).double()

    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, Q_GATE_KICKSTART_BIAS)

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(num_epochs=EPOCHS)

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=False, track_mode=TRACK_MODE, w_terminal_anchor=0.0,
        w_q1_phase_pen=W_Q1_PHASE_PEN,
        w_f_phase_reward=W_F_PHASE_REWARD,
        phase_split_frac=PHASE_SPLIT_FRAC,
    )
    elapsed = time.time() - t0

    x_final, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    dist_final = np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL))
    result = "SUCCESS" if dist_final < 1.0 else "FAIL"

    print(f"\n  goal_dist={dist_final:.4f}  epochs={len(loss_history)}  "
          f"time={elapsed:.0f}s  {result}")

    session_name = f"stageD_phase_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "phase_conditional_outer_loss",
            "q_base_diag": Q_BASE_DIAG,
            "w_q1_phase_pen": W_Q1_PHASE_PEN,
            "w_f_phase_reward": W_F_PHASE_REWARD,
            "phase_split_frac": PHASE_SPLIT_FRAC,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
