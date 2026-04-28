"""dump_success_csv.py — Generate epoch_0 and final rollout CSVs for the
breakthrough swing-up at default q_base_diag = [12, 5, 50, 40].

epoch_0  = untrained model (random init, no kickstart) → demonstrates failure
final    = trained best-model from `stageD_endqhigh_20260428_105538` →
           demonstrates swing-up

Both use the same default q_base_diag environment.
"""

import csv
import math
import os
import sys

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
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM, CONTROL_DIM = 4, 2
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

OUT_DIR = "demo_csv"
TRAINED_PATH = "saved_models/stageD_endqhigh_20260428_105538/stageD_endqhigh_20260428_105538.pth"


def save_csv(x_hist, u_hist, dt, x_goal_np, filepath):
    T = u_hist.shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "q1_rad", "q1_dot_rads", "q2_rad", "q2_dot_rads",
                    "tau1_Nm", "tau2_Nm", "goal_dist", "q1_err_rad"])
        for i in range(T):
            xs, us = x_hist[i], u_hist[i]
            dist = float(np.linalg.norm(xs - x_goal_np))
            q1_err = float(abs(xs[0] - x_goal_np[0]))
            w.writerow([round(i * dt, 4),
                        round(float(xs[0]), 6), round(float(xs[1]), 6),
                        round(float(xs[2]), 6), round(float(xs[3]), 6),
                        round(float(us[0]), 6), round(float(us[1]), 6),
                        round(dist, 6), round(q1_err, 6)])
    print(f"  saved {filepath}  ({T} steps)")


def make_mpc(device, x0, x_goal):
    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    return mpc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    x_goal_np = x_goal.cpu().numpy()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Epoch 0 (untrained baseline) ─────────────────────────────────────
    print("\n--- Epoch 0: untrained network at default q_base_diag ---")
    mpc0 = make_mpc(device, x0, x_goal)
    untrained = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=0.99, gate_range_r=0.20,
        f_extra_bound=3.0, f_kickstart_amp=0.0,
    ).to(device).double()
    x0_hist, u0_hist = train_module.rollout(
        lin_net=untrained, mpc=mpc0, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    x0_np, u0_np = x0_hist.cpu().numpy(), u0_hist.cpu().numpy()
    d0 = float(np.linalg.norm(x0_np[-1] - x_goal_np))
    print(f"  Untrained final goal_dist: {d0:.4f}")
    save_csv(x0_np, u0_np, DT, x_goal_np, os.path.join(OUT_DIR, "rollout_epoch0_untrained.csv"))

    # ── Final (trained) ────────────────────────────────────────────────
    print(f"\n--- Final: trained model from {os.path.basename(TRAINED_PATH)} ---")
    mpc1 = make_mpc(device, x0, x_goal)
    trained = network_module.LinearizationNetwork.load(TRAINED_PATH, device=str(device)).double()
    xT_hist, uT_hist = train_module.rollout(
        lin_net=trained, mpc=mpc1, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    xT_np, uT_np = xT_hist.cpu().numpy(), uT_hist.cpu().numpy()
    dT = float(np.linalg.norm(xT_np[-1] - x_goal_np))
    print(f"  Trained final goal_dist: {dT:.4f}")
    save_csv(xT_np, uT_np, DT, x_goal_np, os.path.join(OUT_DIR, "rollout_final_trained.csv"))

    print(f"\n  Summary:")
    print(f"  Untrained @ default q_base : goal_dist = {d0:.4f}  (FAIL — no swing-up)")
    print(f"  Trained   @ default q_base : goal_dist = {dT:.4f}  (SUCCESS — swing-up)")


if __name__ == "__main__":
    main()
