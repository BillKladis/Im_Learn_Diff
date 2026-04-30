"""exp_dq_scan2d.py — 2-D scan over (dq0, dq1) to find optimal Q-restoration.

MOTIVATION:
  q1restore_test Phase 1 showed dq0=0.25,dq1=0 → 36.8% (vs 26.2% baseline).
  optinit eval showed dq0=0.987,dq1=0.987 → 26.0% (dq1 is HARMFUL).
  This script scans the 2-D (dq0, dq1) grid to find the joint optimum,
  focusing on small dq1 values (0 to 0.3) and the best dq0 range.

USAGE:
  python exp_dq_scan2d.py                      # default grid
  python exp_dq_scan2d.py --dq0 0.987 --dq1 0  # single point
  python exp_dq_scan2d.py --phase1              # dq1=0 only, wide dq0 range
"""

import argparse, math, os, sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

POSONLY_FINAL = "saved_models/stageD_posonly_ft_20260430_083618/stageD_posonly_ft_20260430_083618.pth"
X0          = [0.0, 0.0, 0.0, 0.0]
X_GOAL      = [math.pi, 0.0, 0.0, 0.0]
DT          = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]; THRESH = 0.8


class FixedBoost(nn.Module):
    def __init__(self, lin_net, dq0, dq1, dq2=0.0, dq3=0.0, thresh=THRESH):
        super().__init__()
        self.lin_net = lin_net; self.thresh = thresh
        self.x_goal_q1 = math.pi
        self.f_extra_bound = lin_net.f_extra_bound; self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim; self.control_dim = lin_net.control_dim
        dQ = torch.zeros((lin_net.horizon - 1, lin_net.state_dim), dtype=torch.float64)
        dQ[:, 0] = dq0; dQ[:, 1] = dq1; dQ[:, 2] = dq2; dQ[:, 3] = dq3
        self.register_buffer('delta_Q', dQ)

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        fe = fe * (1.0 - gate)
        gQ = gQ + gate * self.delta_Q
        return gQ, gR, fe, qd, rd, gQf


def eval2k(model, mpc, x0, x_goal, steps=2000):
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([math.sqrt(
        math.atan2(math.sin(s[0]-math.pi), math.cos(s[0]-math.pi))**2
        + s[1]**2 + s[2]**2 + s[3]**2) for s in traj])
    arr = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    return float((wraps < 0.10).mean()), float((wraps < 0.30).mean()), arr, post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dq0", type=float, default=None)
    parser.add_argument("--dq1", type=float, default=None)
    parser.add_argument("--phase1", action="store_true", help="dq1=0 only scan")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.eval()

    # Build config list
    if args.dq0 is not None and args.dq1 is not None:
        configs = [(args.dq0, args.dq1, f"dq0={args.dq0:.3f},dq1={args.dq1:.3f}")]
    elif args.phase1:
        dq0_vals = [0.0, 0.25, 0.50, 0.75, 0.987, 1.25, 1.50, 1.75]
        configs = [(d, 0.0, f"dq0={d:.3f},dq1=0") for d in dq0_vals]
    else:
        # 2-D grid: dq0 ∈ [0.25, 0.50, 0.75, 0.987, 1.25, 1.50], dq1 ∈ [0, 0.10, 0.25, 0.50]
        dq0_vals = [0.25, 0.50, 0.75, 0.987, 1.25, 1.50]
        dq1_vals = [0.0, 0.10, 0.25, 0.50]
        configs = [(d0, d1, f"dq0={d0:.3f},dq1={d1:.3f}") for d0 in dq0_vals for d1 in dq1_vals]

    print(f"  DQ 2-D SCAN  [{datetime.now().strftime('%H:%M')}]  {len(configs)} configs")
    print(f"  {'config':>28}  {'f<0.10':>8}  {'f<0.30':>8}  {'arr':>5}  {'post':>8}")
    print(f"  {'-'*65}")

    best = 0.0
    for dq0, dq1, label in configs:
        model = FixedBoost(lin_net, dq0=dq0, dq1=dq1, thresh=THRESH)
        f01, f03, arr, post = eval2k(model, mpc, x0, x_goal)
        mark = " ★" if f01 > best else ""
        if f01 > best: best = f01
        print(f"  {label:>28}  {f01:>8.1%}  {f03:>8.1%}  {str(arr):>5}  "
              f"{(f'{post:.1%}' if post else 'N/A'):>8}{mark}", flush=True)

    print(f"\n  Best: {best:.1%}  (baseline: 26.2%)")
    if best > 0.262:
        print(f"  ★★★ IMPROVEMENT: {best:.1%} > 26.2% ★★★")


if __name__ == "__main__":
    main()
