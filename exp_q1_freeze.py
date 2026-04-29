"""
exp_q1_freeze.py — Freeze q_head entirely; only train f_head (+ trunk/encoder).

With q_head frozen and q1 kickstart at bias=-4:
  - q1 gate stays at ~0.011 forever (no drift)
  - q1_dot, q2, q2_dot gates stay at 1.0 forever
  - Effective q1_cost = 12 × 0.011 = 0.132  (well below ~2.0 threshold)
  - f_head learns pumping pattern (same as ZERO_Q1_COSTS baseline)

If this works (goal_dist < 1.0): the culprit is Q-gate drift during joint
training, NOT the nonzero q1 base cost.  Solution: two-phase training or
explicit Q-gate profile regularization.

If this fails: the effective q1_cost=0.132 still causes some QP
interference that the ZERO_Q1_COSTS case avoids.
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

# DEFAULT q_base_diag (unchanged)
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

GATE_RANGE_Q   = 0.99
GATE_RANGE_R   = 0.20
F_EXTRA_BOUND  = 3.0
F_KICKSTART    = 1.0
Q_GATE_KICKSTART_BIAS = -4.0  # q1 gate → init ≈ 0.011; FROZEN at this value
TRACK_MODE     = "energy"


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
              f"{'QDev':>7}  {'fNorm':>7}  {'fτ1[0]':>8}  {'Time':>6}")
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
                  f"  {info.get('epoch_time',        float('nan')):>5.2f}s",
                  flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)

    print("=" * 76)
    print("  EXP: FREEZE q_head — only train f_head + trunk + encoder")
    print("  q_base_diag = [12, 5, 50, 40]  (DEFAULT)")
    print(f"  q1 gate FIXED at {1 + GATE_RANGE_Q * math.tanh(Q_GATE_KICKSTART_BIAS):.4f}  "
          f"→ effective q1_cost = {12 * (1 + GATE_RANGE_Q * math.tanh(Q_GATE_KICKSTART_BIAS)):.4f}")
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

    # ── FREEZE q_head ─────────────────────────────────────────────────────
    for param in lin_net.q_head.parameters():
        param.requires_grad = False
    n_trainable = sum(p.numel() for p in lin_net.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in lin_net.q_head.parameters())
    print(f"\n  Frozen q_head params : {n_frozen:,}")
    print(f"  Trainable params     : {n_trainable:,}")

    # Verify initial q1 gate
    with torch.no_grad():
        dummy_hist = torch.stack([x0.clone() for _ in range(5)], dim=0)
        gQ_init, _, _, _, _, _ = lin_net(dummy_hist, mpc.q_base_diag, mpc.r_base_diag)
    print(f"  Initial q1 gate (step 0): {gQ_init[0, 0].item():.5f}")
    print(f"  Initial q2 gate (step 0): {gQ_init[0, 2].item():.5f}")

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(num_epochs=EPOCHS)

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=False, track_mode=TRACK_MODE, w_terminal_anchor=0.0,
    )
    elapsed = time.time() - t0

    # Verify q1 gate is still at kickstart after training
    with torch.no_grad():
        gQ_final, _, _, _, _, _ = lin_net(dummy_hist, mpc.q_base_diag, mpc.r_base_diag)
    print(f"\n  Final q1 gate (step 0): {gQ_final[0, 0].item():.5f}  (unchanged by freeze)")

    x_final, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    dist_final = np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL))
    result = "SUCCESS" if dist_final < 1.0 else "FAIL"

    print(f"\n  goal_dist={dist_final:.4f}  epochs={len(loss_history)}  "
          f"time={elapsed:.0f}s  {result}")

    session_name = f"stageD_q1freeze_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "q1_head_frozen",
            "q_base_diag": Q_BASE_DIAG,
            "q1_gate_fixed": float(gQ_final[0, 0].item()),
            "effective_q1_cost": 12.0 * float(gQ_final[0, 0].item()),
            "track_mode": TRACK_MODE,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"  Saved → saved_models/{session_name}/")

    print("\n  === FREEZE EXPERIMENT SUMMARY ===")
    print(f"  q_base_diag       : [12, 5, 50, 40] (DEFAULT)")
    print(f"  q1 gate frozen at : {gQ_final[0, 0].item():.5f}")
    print(f"  effective q1_cost : {12 * gQ_final[0, 0].item():.4f}")
    print(f"  goal_dist         : {dist_final:.4f}  {result}")
    print(f"  epochs            : {len(loss_history)}/{EPOCHS}")


if __name__ == "__main__":
    main()
