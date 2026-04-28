"""
exp_q1_gate_reg.py — Q1-gate regularization with full default q_base_diag.

HYPOTHESIS:
  The outer tracking loss always pushes the q1 gate UP (more pull toward
  π = more apparent progress). Adding an explicit regularization term
  `w_q1_gate_reg * relu(gates_Q[:,0].mean() - target)^2` directly
  counteracts this, keeping q1 gate near target=0.01 regardless of
  tracking gradient direction.

  With q1 gate pinned near 0.01, effective q1 cost = 0.12, which is below
  the energy-tracking threshold (~2.0).  f_extra can then learn the pumping
  pattern as in the ZERO_Q1_COSTS baseline.

SWEEP: test w_q1_gate_reg in {2.0, 5.0, 10.0, 20.0} to find the right
balance between tracking loss and gate regularization.
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
EPOCHS    = 80
LR        = 1e-3
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM   = 4
CONTROL_DIM = 2
SAVE_DIR    = "saved_models"

# DEFAULT q_base_diag — NOT zeroed
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

GATE_RANGE_Q   = 0.99
GATE_RANGE_R   = 0.20
F_EXTRA_BOUND  = 3.0
F_KICKSTART    = 1.0   # sinusoidal f kickstart
Q_GATE_KICKSTART_BIAS = -4.0   # q1 gate starts at ~0.011
Q1_GATE_TARGET = 0.01  # regularization target
TRACK_MODE     = "energy"
W_TERMINAL     = 0.0


def apply_q1_kickstart(lin_net, state_dim, horizon, bias_val):
    q_final = [m for m in lin_net.q_head.modules()
               if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            q_final.bias[k * state_dim + 0] = bias_val


class PrintMonitor:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self._header_shown = False

    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>9}  {'GoalDist':>9}  "
              f"{'QDev':>7}  {'fNorm':>7}  {'LR':>9}  {'Time':>6}")
        print("─" * 100)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        if epoch == 0 or (epoch+1) % 1 == 0 or epoch == num_epochs-1:
            print(f"{epoch+1:>4}/{num_epochs:<4}"
                  f"  {loss:>10.3f}"
                  f"  {info.get('loss_track',        float('nan')):>9.3f}"
                  f"  {info.get('pure_end_error',    float('nan')):>9.4f}"
                  f"  {info.get('mean_Q_gate_dev',   float('nan')):>7.4f}"
                  f"  {info.get('mean_f_extra_norm', float('nan')):>7.3f}"
                  f"  {info.get('learning_rate',     float('nan')):>9.2e}"
                  f"  {info.get('epoch_time',        float('nan')):>5.2f}s",
                  flush=True)


def run(w_reg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)

    print(f"\n{'─' * 70}")
    print(f"  w_q1_gate_reg={w_reg}  target={Q1_GATE_TARGET}")
    print(f"  q_base_diag={Q_BASE_DIAG}  gate_range_q={GATE_RANGE_Q}")
    print(f"{'─' * 70}")

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
        grad_debug=False, track_mode=TRACK_MODE, w_terminal_anchor=W_TERMINAL,
        w_q1_gate_reg=w_reg, q1_gate_reg_target=Q1_GATE_TARGET,
    )
    elapsed = time.time() - t0

    x_final, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    dist_final = np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL))
    result = "SUCCESS" if dist_final < 1.0 else "FAIL"

    print(f"\n  w_reg={w_reg}  goal_dist={dist_final:.4f}  "
          f"epochs={len(loss_history)}  time={elapsed:.0f}s  {result}")

    session_name = f"stageD_q1reg_{w_reg:.0f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "q1_gate_regularization",
            "w_q1_gate_reg": w_reg,
            "q1_gate_reg_target": Q1_GATE_TARGET,
            "q_base_diag": Q_BASE_DIAG,
            "track_mode": TRACK_MODE,
        },
        session_name=session_name, recorder=recorder,
    )
    return dist_final, result


if __name__ == "__main__":
    results = []
    for w in [2.0, 5.0, 10.0, 20.0]:
        d, r = run(w)
        results.append((w, d, r))
        print()

    print("\n=== Q1-GATE REGULARIZATION SUMMARY ===")
    print("  q_base_diag = [12, 5, 50, 40]  (DEFAULT, unchanged)")
    print(f"  {'w_q1_gate_reg':>14}  {'goal_dist':>10}  {'result':>8}")
    for w, d, r in results:
        print(f"  {w:>14.1f}  {d:>10.4f}  {r:>8}")
