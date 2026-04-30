"""exp_dq_scale_sweep.py — Quick forward-only sweep of delta_Q scale factors.

Tests the 82.9% checkpoint's delta_Q scaled by various factors to see if
higher/lower Q boost near the top improves or hurts hold quality.
No backprop, just evaluation → fast after CVXPY compile.

USAGE:
  python exp_dq_scale_sweep.py
  python exp_dq_scale_sweep.py --steps 2000     # longer eval
  python exp_dq_scale_sweep.py --scales "0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0"
"""

import argparse, math, os, sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

POSONLY_FINAL = "saved_models/stageD_posonly_ft_20260430_083618/stageD_posonly_ft_20260430_083618.pth"
BEST_CKPT     = "saved_models/stageD_optinit_holdboost_dq0.99x0.99_20260430_165519/stageD_optinit_holdboost_dq0.99x0.99_20260430_165519.pth"
X0            = [0.0, 0.0, 0.0, 0.0]
X_GOAL        = [math.pi, 0.0, 0.0, 0.0]
DT            = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]; THRESH = 0.8


class HoldBoostWrapper(nn.Module):
    def __init__(self, lin_net, thresh, dQ_init, dR_init=None, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net = lin_net; self.thresh = thresh; self.x_goal_q1 = x_goal_q1
        self.f_extra_bound = lin_net.f_extra_bound; self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim; self.control_dim = lin_net.control_dim
        self.delta_Q = nn.Parameter(dQ_init.clone().double())
        r_shape = (lin_net.horizon, lin_net.control_dim)
        self.delta_R = nn.Parameter(
            dR_init.clone().double() if dR_init is not None
            else torch.zeros(r_shape, dtype=torch.float64)
        )

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        fe = fe * (1.0 - gate)
        gQ = gQ + gate * self.delta_Q
        gR = gR + gate * self.delta_R
        return gQ, gR, fe, qd, rd, gQf


def eval_trajectory(model, mpc, x0, x_goal, steps):
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([math.sqrt(
        math.atan2(math.sin(s[0]-math.pi), math.cos(s[0]-math.pi))**2
        + s[1]**2 + s[2]**2 + s[3]**2) for s in traj])
    arr = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    return float((wraps < 0.10).mean()), float((wraps < 0.30).mean()), arr, post


def load_best(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    tp = ckpt['metadata'].get('training_params', {})
    dQ = tp.get('best_delta_Q')
    dR = tp.get('best_delta_R')
    return (torch.tensor(dQ, dtype=torch.float64) if dQ else None,
            torch.tensor(dR, dtype=torch.float64) if dR else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=BEST_CKPT)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--scales", type=str,
                        default="0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0")
    args = parser.parse_args()
    scales = [float(s) for s in args.scales.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.eval()

    dQ, dR = load_best(args.ckpt)
    boost = HoldBoostWrapper(lin_net, thresh=THRESH, dQ_init=dQ, dR_init=dR)
    base_dQ = dQ.clone()
    base_dR = dR.clone()

    print(f"  DQ SCALE SWEEP  steps={args.steps}")
    print(f"  Base dQ mean: {base_dQ.mean(0).tolist()}")
    print(f"  {'scale':>8}  {'f<0.10':>8}  {'f<0.30':>8}  {'arr':>5}  {'post':>7}  {'eff Q[q1]':>10}")
    print(f"  {'-'*60}")

    for scale in scales:
        boost.delta_Q.data = base_dQ * scale
        boost.delta_R.data = base_dR * scale
        f01, f03, arr, post = eval_trajectory(boost, mpc, x0, x_goal, args.steps)
        # Effective Q[q1] at top (gate=1): q_base[q1] * (gates_Q[q1]_at_top + delta_Q[q1])
        # gates_Q[q1]_at_top ≈ 0.013, delta_Q[q1] = scale * base_dQ[q1]
        eff_q1 = Q_BASE_DIAG[0] * (0.013 + scale * float(base_dQ.mean(0)[0]))
        mark = " ★" if scale == 1.0 else ""
        print(f"  {scale:>8.3f}  {f01:>7.1%}  {f03:>7.1%}  {str(arr):>5}  "
              f"{f'{post:.1%}' if post else 'N/A':>7}  {eff_q1:>10.2f}{mark}", flush=True)


if __name__ == "__main__":
    main()
