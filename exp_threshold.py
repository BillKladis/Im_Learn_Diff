"""
exp_threshold.py — Find exact q1_cost threshold where energy tracking
breaks, and test whether large f_extra_bound (15) can handle default
q_base_diag = [12, 5, 50, 40].

Experiments:
  A) Sweep q1_cost: 3.0, 5.0, 8.0 — find where success → fail
  B) f_extra_bound = 15 with default q_base_diag = [12, 5, 50, 40]
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


class QuietMonitor:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
    def log_epoch(self, epoch, num_epochs, loss, info):
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch == num_epochs-1:
            gd = info.get('pure_end_error', float('nan'))
            print(f"  [{epoch+1:>2}/{num_epochs}] loss={loss:.4f} goal_dist={gd:.4f}",
                  flush=True)


def run_experiment(label, q_base_diag, f_extra_bound, track_mode="energy",
                   gate_range_q=0.95, f_kickstart=0.0, q1_kickstart_bias=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(q_base_diag, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=gate_range_q, gate_range_r=0.20,
        f_extra_bound=f_extra_bound, f_kickstart_amp=f_kickstart,
    ).to(device).double()

    if q1_kickstart_bias is not None:
        q_final = [m for m in lin_net.q_head.modules()
                   if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            for k in range(HORIZON - 1):
                q_final.bias[k * STATE_DIM + 0] = q1_kickstart_bias

    recorder = network_module.NetworkOutputRecorder()
    monitor  = QuietMonitor(num_epochs=EPOCHS)

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=False, track_mode=track_mode, w_terminal_anchor=0.0,
    )

    x_final, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    dist_final = np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL))
    elapsed = time.time() - t0

    result = "SUCCESS" if dist_final < 1.0 else "FAIL"
    print(f"\n  {label:40s}  goal_dist={dist_final:.4f}  epochs={len(loss_history):>3}  "
          f"time={elapsed:.0f}s  {result}")

    session_name = f"stageD_thresh_{label.replace(' ', '_')[:20]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "label": label, "q_base_diag": q_base_diag,
            "f_extra_bound": f_extra_bound, "track_mode": track_mode,
        },
        session_name=session_name, recorder=recorder,
    )
    return dist_final, result


if __name__ == "__main__":
    results = []

    # ── Part A: find q1_cost threshold (energy tracking) ───────────────
    print("\n=== PART A: q1_cost threshold sweep ===")
    for q1c in [3.0, 5.0, 8.0]:
        print(f"\n--- q1_cost={q1c} ---")
        d, r = run_experiment(
            label=f"q1cost={q1c} energy_track",
            q_base_diag=[q1c, q1c * 0.5, 50.0, 40.0],
            f_extra_bound=3.0,
            track_mode="energy",
        )
        results.append((f"q1_cost={q1c}", d, r))

    # ── Part B: large f_extra_bound (=15) with default q_base_diag ─────
    print("\n=== PART B: large f_extra_bound with default q_base_diag ===")
    print("\n--- f_extra_bound=15, q_base=[12,5,50,40], energy track ---")
    d, r = run_experiment(
        label="q1=12 fbound=15 energy",
        q_base_diag=[12.0, 5.0, 50.0, 40.0],
        f_extra_bound=15.0,
        track_mode="energy",
        f_kickstart=1.0,   # scale kickstart with bound
    )
    results.append(("q1=12 fbound=15", d, r))

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n\n=== THRESHOLD EXPERIMENT SUMMARY ===")
    print(f"{'Experiment':42s}  {'goal_dist':>10}  {'result':>8}")
    print("-" * 70)
    # Known from earlier experiments:
    for label, dist, res in [
        ("q1_cost=0  (baseline ZERO_Q1_COSTS)", 0.18, "SUCCESS"),
        ("q1_cost=0.5 energy_track", 0.087, "SUCCESS"),
        ("q1_cost=1.0 energy_track", 0.126, "SUCCESS"),
        ("q1_cost=2.0 energy_track", 0.281, "SUCCESS"),
    ]:
        print(f"  {label:42s}  {dist:>10.4f}  {res:>8}")
    for label, dist, res in results:
        print(f"  {label:42s}  {dist:>10.4f}  {res:>8}")
    print(f"  {'q1_cost=12 fbound=3 (failed Q-mod)':42s}  {'2.86':>10}  {'FAIL':>8}")
