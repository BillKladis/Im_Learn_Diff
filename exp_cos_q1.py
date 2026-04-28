"""
exp_cos_q1.py — Test cos_q1 wrapped-angle tracking with default
q_base_diag = [12, 5, 50, 40] and wide Q-gates with q1 kickstart.

HYPOTHESIS:
  cos_q1 loss = (cos(q1)-cos(q1_demo))^2 + (sin(q1)-sin(q1_demo))^2
              + 0.1*(q1d-q1d_demo)^2/64
  is UNIQUE — spinning-fast-at-bottom produces the same energy as
  upright but a very different cos/sin.  This forces the gradient to
  distinguish between the two, potentially enabling the Q-gate to
  suppress q1 cost while f_extra pumps energy.

The experiment keeps the architecture identical (gate_range_q=0.99,
q1 kickstart bias=-4) but replaces energy tracking with cos_q1 tracking.
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
TRACK_MODE = "cos_q1"   # THE KEY CHANGE
STATE_DIM   = 4
CONTROL_DIM = 2
SAVE_DIR    = "saved_models"

# Default q_base_diag (NOT zeroed) — NN must learn to handle it
ZERO_Q1_COSTS = False

GATE_RANGE_Q   = 0.99    # wide gates
GATE_RANGE_R   = 0.20
F_EXTRA_BOUND  = 3.0
F_KICKSTART    = 1.0     # sinusoidal f kickstart
W_TERMINAL     = 0.0
Q_GATE_KICKSTART_BIAS = -4.0  # q1 gate → init ≈ 0.011


def apply_q1_gate_kickstart(lin_net, state_dim, horizon, bias_val):
    q_final = [m for m in lin_net.q_head.modules()
               if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            q_final.bias[k * state_dim + 0] = bias_val
    init_gate = 1.0 + GATE_RANGE_Q * math.tanh(bias_val)
    print(f"  Q-gate kickstart: bias={bias_val}  init_q1_gate≈{init_gate:.4f}")


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)

    print("=" * 76)
    print("  EXP: cos_q1 tracking + default q_base_diag + wide gates")
    print("  q_base_diag = [12, 5, 50, 40]  track_mode = cos_q1")
    print(f"  gate_range_q={GATE_RANGE_Q}  q1_kickstart={Q_GATE_KICKSTART_BIAS}  "
          f"f_kickstart={F_KICKSTART}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    # q_base_diag stays default [12, 5, 50, 40]

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=F_KICKSTART,
    ).to(device).double()

    apply_q1_gate_kickstart(lin_net, STATE_DIM, HORIZON, Q_GATE_KICKSTART_BIAS)

    # Gradient smoke test
    grad_report = train_module.gradient_flow_smoke_test(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal,
        demo=demo, num_steps=5, track_mode=TRACK_MODE,
    )
    mods = grad_report["module_norms"]
    print(f"\n  Smoke: loss={grad_report['smoke_loss']:.6f}  "
          f"tot={grad_report['total_norm']:.3e}  "
          f"q={mods['q_head']:.3e}  f={mods['f_head']:.3e}")

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(print_every=1, num_epochs=EPOCHS)

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=True, grad_debug_every=5,
        track_mode=TRACK_MODE,
        w_terminal_anchor=W_TERMINAL,
    )
    total_time = time.time() - t0

    x_final, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS)
    dist_final = np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL))

    print(f"\n  Training: {len(loss_history)} epochs  {total_time:.0f}s")
    print(f"  goal_dist: {dist_final:.4f}  {'SUCCESS <1.0' if dist_final < 1.0 else 'FAIL'}")

    session_name = f"stageD_cosq1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "cos_q1_tracking_default_q_base",
            "track_mode": TRACK_MODE,
            "gate_range_q": GATE_RANGE_Q,
            "q_gate_kickstart": Q_GATE_KICKSTART_BIAS,
            "f_kickstart": F_KICKSTART,
            "zero_q1_costs": False,
            "epochs": EPOCHS, "lr": LR,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"  Saved → saved_models/{session_name}/")

    print("\n  === COS_Q1 EXPERIMENT SUMMARY ===")
    print(f"  track_mode      : cos_q1")
    print(f"  q_base_diag     : [12, 5, 50, 40] (DEFAULT)")
    print(f"  gate_range_q    : {GATE_RANGE_Q}")
    print(f"  q1_kickstart    : {Q_GATE_KICKSTART_BIAS}")
    print(f"  epochs (actual) : {len(loss_history)}/{EPOCHS}")
    print(f"  goal_dist       : {dist_final:.4f}  {'SUCCESS' if dist_final < 1.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
