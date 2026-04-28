"""verify_stable.py — Load latest stabilize model and evaluate stability.

Runs rollouts at multiple step counts (170/250/400/600) and prints both
raw and wrapped goal distance.  Also prints the trajectory tail to confirm
the pendulum holds the upright (not just reaches it).
"""

import math
import os
import sys
import glob

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

X_GOAL = [math.pi, 0.0, 0.0, 0.0]
DT = 0.05
HORIZON = 10
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]


def wrapped_goal_dist(x_state, x_goal):
    q1_err = math.atan2(math.sin(x_state[0] - x_goal[0]),
                        math.cos(x_state[0] - x_goal[0]))
    return math.sqrt(q1_err**2 + x_state[1]**2 + x_state[2]**2 + x_state[3]**2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.zeros(4, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    # Find latest stabilize model
    candidates = sorted(glob.glob("saved_models/stageD_stabilize_*"), reverse=True)
    if not candidates:
        print("No stageD_stabilize_* found, trying stageD_nodemo_*")
        candidates = sorted(glob.glob("saved_models/stageD_nodemo_*"), reverse=True)
    model_dir = candidates[0]
    pth_files = glob.glob(f"{model_dir}/*.pth")
    model_path = pth_files[0]
    print(f"Loading: {model_path}")

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(model_path, device=str(device)).double()

    print("\n=== Stability evaluation ===")
    print(f"  {'Steps':>6}  {'Time(s)':>8}  {'raw':>8}  {'wrapped':>8}  {'q1°':>8}  {'q1d':>7}  {'Status':>10}")
    print("  " + "─" * 70)

    for n in [170, 220, 300, 400, 600]:
        x_traj, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n,
        )
        last = x_traj[-1].cpu().numpy()
        raw  = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrap = wrapped_goal_dist(last, X_GOAL)
        status = "STABLE" if wrap < 0.3 else ("CLOSE" if wrap < 1.0 else "FAIL")
        print(f"  {n:>6}  {n*DT:>8.1f}  {raw:>8.3f}  {wrap:>8.3f}  "
              f"{math.degrees(last[0]):>8.2f}  {last[1]:>7.3f}  {status:>10}")

    # Show last 10 steps for the longest rollout
    print(f"\n  Trajectory tail (last 20 of 600 steps):")
    print(f"  {'step':>5}  {'q1°':>8}  {'q1d':>7}  {'q2°':>8}  {'q2d':>7}  {'wrap':>7}")
    x_traj, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=600,
    )
    traj_np = x_traj.cpu().numpy()
    for i in range(580, 600):
        s = traj_np[i]
        wrp = wrapped_goal_dist(s, X_GOAL)
        print(f"  {i:>5}  {math.degrees(s[0]):>8.2f}  {s[1]:>7.3f}  "
              f"{math.degrees(s[2]):>8.2f}  {s[3]:>7.3f}  {wrp:>7.3f}")


if __name__ == "__main__":
    main()
