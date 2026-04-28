"""
exp_small_q1.py — Ablation: how small does q1 cost need to be for
energy-tracking imitation to still achieve swing-up?

Config: q_base_diag = [0.5, 0.5, 50, 40]
  - q1 cost = 0.5  (vs default 12, or 0 in working baseline)
  - f_extra bound ±3 can overcome: 0.5 * π ≈ 1.57 N·m per step
  - gate_range_q = 0.95 (standard)
  - f_kickstart = 0 (same as working baseline)
  - energy tracking

If this succeeds, increase to [2, 0.5, 50, 40] to find threshold.
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
TRACK_MODE = "energy"
STATE_DIM   = 4
CONTROL_DIM = 2
SAVE_DIR    = "saved_models"

# ── The variable under test ───────────────────────────────────────────────
Q1_COST   = 0.5    # test values: 0.5, 1.0, 2.0, 4.0
Q1D_COST  = 0.5
# Default for q2/q2d: 50/40 (unchanged)

GATE_RANGE_Q   = 0.95
GATE_RANGE_R   = 0.20
F_EXTRA_BOUND  = 3.0
F_KICKSTART    = 0.0
W_TERMINAL     = 0.0

class PrintMonitor:
    def __init__(self, print_every, num_epochs):
        self.print_every   = print_every
        self.num_epochs    = num_epochs
        self._header_shown = False

    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>9}  {'GoalDist':>9}  "
              f"{'QDev':>7}  {'fNorm':>7}  {'fτ1[0]':>8}  {'LR':>9}  {'Time':>6}")
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
                  f"  {info.get('mean_f_tau1_first', float('nan')):>8.3f}"
                  f"  {info.get('learning_rate',     float('nan')):>9.2e}"
                  f"  {info.get('epoch_time',        float('nan')):>5.2f}s",
                  flush=True)


def run(q1_cost, q1d_cost):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)

    print("=" * 70)
    print(f"  exp_small_q1:  q_base_diag = [{q1_cost}, {q1d_cost}, 50, 40]")
    print(f"  energy tracking  |  gate_range_q={GATE_RANGE_Q}  |  f_kickstart=0")
    print("=" * 70)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(
        [q1_cost, q1d_cost, 50.0, 40.0], device=device, dtype=torch.float64,
    )

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=F_KICKSTART,
    ).to(device).double()

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(print_every=1, num_epochs=EPOCHS)

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=False, track_mode=TRACK_MODE,
        w_terminal_anchor=W_TERMINAL,
    )
    total_time = time.time() - t0

    x_final, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS)
    dist_final = np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL))

    print(f"\n  q1_cost={q1_cost}  epochs={len(loss_history)}  goal_dist={dist_final:.4f}  "
          f"time={total_time:.0f}s  "
          f"{'SUCCESS' if dist_final < 1.0 else 'FAIL'}")

    session_name = f"stageD_sq1_{q1_cost:.1f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "q1_cost": q1_cost, "q1d_cost": q1d_cost,
            "experiment": "small_q1_ablation", "track_mode": TRACK_MODE,
            "epochs": EPOCHS, "lr": LR,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"  Saved → saved_models/{session_name}/")
    return dist_final


if __name__ == "__main__":
    # Test multiple q1 cost levels sequentially.
    results = []
    for q1c in [0.5, 1.0, 2.0]:
        d = run(q1c, q1c * 0.5)   # q1d_cost = half of q1_cost
        results.append((q1c, d))
        print()

    print("\n=== SMALL Q1 ABLATION SUMMARY ===")
    print(f"{'q1_cost':>10}  {'goal_dist':>10}  {'result':>8}")
    for q1c, d in results:
        print(f"{q1c:>10.1f}  {d:>10.4f}  {'SUCCESS' if d < 1.0 else 'FAIL'}")
