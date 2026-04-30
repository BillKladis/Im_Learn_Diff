"""exp_q1restore_test.py — Direct test of q1/q1d Q-weight restoration.

KEY FINDING:
  The baseline network outputs gates_Q[:, 0] ≈ 0.013 and gates_Q[:, 1] ≈ 0.013
  at the top state [π, 0, 0, 0]. This gives:
    Effective Q[q1]  = 12.0 × 0.013 = 0.156  (vs base 12.0 — 98.7% suppressed!)
    Effective Q[q1d] = 5.0  × 0.013 = 0.064  (vs base 5.0  — 98.7% suppressed!)
  The MPC near the top is essentially IGNORING q1 and q1d deviations.

  q2 and q2d are fine: gates_Q[:, 2] ≈ 1.001, gates_Q[:, 3] ≈ 0.999

HYPOTHESIS:
  Restoring Q[q1] and Q[q1d] to their base values by setting:
    delta_Q[:, 0] = 0.987  (q1 correction)
    delta_Q[:, 1] = 0.987  (q1d correction)
    delta_Q[:, 2] = 0.0    (q2 unchanged)
    delta_Q[:, 3] = 0.0    (q2d unchanged)
  should GREATLY improve hold quality by making the MPC properly track q1.

This script tests a range of (dq1, dq1d) combinations to find the optimal.
"""

import math, os, sys, time
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
X0 = [0.0, 0.0, 0.0, 0.0]; X_GOAL = [math.pi, 0.0, 0.0, 0.0]
DT = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]; THRESH = 0.8


class FixedHoldBoost(nn.Module):
    """Fixed (non-trainable) holdboost for direct evaluation."""
    def __init__(self, lin_net, dq0, dq1, dq2=0.0, dq3=0.0, thresh=0.8):
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


def eval_steps(model, mpc, x0, x_goal, steps):
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([math.sqrt(math.atan2(math.sin(s[0]-math.pi), math.cos(s[0]-math.pi))**2
                                + s[1]**2 + s[2]**2 + s[3]**2) for s in traj])
    arr = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    return float((wraps < 0.10).mean()), float((wraps < 0.30).mean()), arr, post


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    print("=" * 70)
    print(f"  Q1/Q1D WEIGHT RESTORATION TEST  [{datetime.now().strftime('%H:%M')}]")
    print(f"  Network gates_Q[q1]=0.013, gates_Q[q1d]=0.013 at top")
    print(f"  Testing delta_Q to restore these to useful values")
    print("=" * 70)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.eval()

    # Phase 1: test different levels of q1-only restoration
    configs_q1_only = [
        (0.0, 0.0, "baseline"),
        (0.25, 0.0, "q1=0.25,q1d=0"),
        (0.50, 0.0, "q1=0.50,q1d=0"),
        (0.75, 0.0, "q1=0.75,q1d=0"),
        (0.987, 0.0, "q1=0.987,q1d=0"),
        (1.50, 0.0, "q1=1.50,q1d=0"),
    ]

    print(f"\n  PHASE 1: q1-only (q1d=0):")
    print(f"  {'config':>25}  {'f<0.10':>8}  {'f<0.30':>8}  {'arr':>5}  {'post':>8}")
    best = 0.0
    for dq0, dq1, label in configs_q1_only:
        model = FixedHoldBoost(lin_net, dq0=dq0, dq1=dq1, thresh=THRESH)
        f01, f03, arr, post = eval_steps(model, mpc, x0, x_goal, steps=2000)
        mark = " ★★" if f01 > best else ""
        if f01 > best: best = f01
        print(f"  {label:>25}  {f01:>8.1%}  {f03:>8.1%}  {str(arr):>5}  "
              f"{(f'{post:.1%}' if post else 'N/A'):>8}{mark}", flush=True)

    # Phase 2: test q1+q1d combined restoration
    configs_combined = [
        (0.987, 0.987, "q1+q1d=0.987"),
        (0.987, 0.5, "q1=0.987,q1d=0.5"),
        (0.5, 0.987, "q1=0.5,q1d=0.987"),
        (0.5, 0.5, "q1+q1d=0.5"),
        (1.5, 1.5, "q1+q1d=1.5"),
        (0.987, 0.987, "combined_full (check)"),
    ]

    print(f"\n  PHASE 2: q1+q1d combined:")
    print(f"  {'config':>25}  {'f<0.10':>8}  {'f<0.30':>8}  {'arr':>5}  {'post':>8}")
    for dq0, dq1, label in configs_combined:
        model = FixedHoldBoost(lin_net, dq0=dq0, dq1=dq1, thresh=THRESH)
        f01, f03, arr, post = eval_steps(model, mpc, x0, x_goal, steps=2000)
        mark = " ★★" if f01 > best else ""
        if f01 > best: best = f01
        print(f"  {label:>25}  {f01:>8.1%}  {f03:>8.1%}  {str(arr):>5}  "
              f"{(f'{post:.1%}' if post else 'N/A'):>8}{mark}", flush=True)

    print(f"\n  Baseline (no boost): 26.2%")
    print(f"  Best found: {best:.1%}")
    if best > 0.262:
        print(f"  ★★★ IMPROVEMENT: {best:.1%} > 26.2% ★★★")
    else:
        print(f"  No improvement — q1 restoration doesn't help with Q_BASE formula")


if __name__ == "__main__":
    main()
