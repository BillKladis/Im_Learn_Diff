"""exp_holdonly_quick.py — Fast hold-quality test starting from near the upright.

PURPOSE:
  Fast directional check: given a delta_Q value, how well can the controller
  hold the double pendulum near [pi, 0, 0, 0] when starting from there?

  Bypasses swing-up (starts from near-top x0) so results arrive in ~5 min
  instead of ~30 min for the full 2000-step test.

  Use to quickly validate a hypothesis before committing to the full test.

USAGE:
  python exp_holdonly_quick.py                 # sweeps common delta_Q values
  python exp_holdonly_quick.py 0.987 0.987     # test specific (dq0, dq1)
"""

import math, os, sys
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
X_GOAL = [math.pi, 0.0, 0.0, 0.0]
DT = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]; THRESH = 0.8
HOLD_STEPS = 600  # hold from various near-top starts


class FixedHoldBoost(nn.Module):
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


def eval_hold(model, mpc, x0, x_goal, steps):
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([math.sqrt(
        math.atan2(math.sin(s[0]-math.pi), math.cos(s[0]-math.pi))**2
        + s[1]**2 + s[2]**2 + s[3]**2) for s in traj])
    return float((wraps < 0.10).mean()), float((wraps < 0.30).mean())


def main():
    dq0_arg = float(sys.argv[1]) if len(sys.argv) > 1 else None
    dq1_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x_goal, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.eval()

    # Test from multiple perturbed starts, report average
    starts = [
        [math.pi + 0.05, 0.0, 0.0, 0.0],
        [math.pi - 0.05, 0.0, 0.0, 0.0],
        [math.pi + 0.10, 0.2, 0.0, 0.0],
        [math.pi - 0.10, -0.2, 0.0, 0.0],
        [math.pi, 0.3, 0.1, 0.1],
    ]

    if dq0_arg is not None:
        configs = [(dq0_arg, dq1_arg, f"dq0={dq0_arg:.3f},dq1={dq1_arg:.3f}")]
    else:
        configs = [
            (0.0, 0.0, "baseline"),
            (0.25, 0.0, "q1=0.25"),
            (0.50, 0.0, "q1=0.50"),
            (0.75, 0.0, "q1=0.75"),
            (0.987, 0.0, "q1=0.987"),
            (1.50, 0.0, "q1=1.50"),
            (0.987, 0.987, "q1+q1d=0.987"),
        ]

    print(f"  HOLD-ONLY QUICK TEST  [{datetime.now().strftime('%H:%M')}]  {HOLD_STEPS} steps/start, {len(starts)} starts")
    print(f"  {'config':>22}  {'f<0.10':>8}  {'f<0.30':>8}")
    print(f"  {'-'*50}")

    for dq0, dq1, label in configs:
        model = FixedHoldBoost(lin_net, dq0=dq0, dq1=dq1, thresh=THRESH)
        results = []
        for s in starts:
            x0 = torch.tensor(s, device=device, dtype=torch.float64)
            f01, f03 = eval_hold(model, mpc, x0, x_goal, HOLD_STEPS)
            results.append((f01, f03))
        avg_f01 = sum(r[0] for r in results) / len(results)
        avg_f03 = sum(r[1] for r in results) / len(results)
        print(f"  {label:>22}  {avg_f01:>8.1%}  {avg_f03:>8.1%}", flush=True)


if __name__ == "__main__":
    main()
