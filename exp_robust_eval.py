"""exp_robust_eval.py — Robustness evaluation of the best HoldBoost model.

Tests the best delta_Q (from exp_optinit_holdboost or exp_boost_continue)
across multiple starting conditions to check if improvement is robust or
specific to the single [0,0,0,0] starting point.

USAGE:
  python exp_robust_eval.py                         # eval best checkpoint
  python exp_robust_eval.py --ckpt PATH             # explicit checkpoint
  python exp_robust_eval.py --steps 1000            # shorter eval
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
BEST_CKPT     = "saved_models/stageD_optinit_holdboost_dq0.99x0.99_20260430_165519/stageD_optinit_holdboost_dq0.99x0.99_20260430_165519.pth"
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(
        x0=x_goal, x_goal=x_goal, N=HORIZON, device=device
    )
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.eval()

    dQ, dR = load_best(args.ckpt)
    boost = HoldBoostWrapper(lin_net, thresh=THRESH, dQ_init=dQ, dR_init=dR)

    # ZeroFNet baseline (delta_Q=0, f_extra zeroed near top)
    class _Zero(nn.Module):
        def __init__(self, ln, thresh):
            super().__init__()
            self.lin_net = ln; self.thresh = thresh
            self.f_extra_bound = ln.f_extra_bound; self.horizon = ln.horizon
            self.state_dim = ln.state_dim; self.control_dim = ln.control_dim
        def forward(self, x_seq, q_base_diag=None, r_base_diag=None):
            gQ, gR, fe, qd, rd, gQf = self.lin_net(x_seq, q_base_diag, r_base_diag)
            q1 = x_seq[-1, 0]
            near_pi = (1.0 + torch.cos(q1 - math.pi)) / 2.0
            gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0., 1.)
            fe = fe * (1.0 - gate)
            return gQ, gR, fe, qd, rd, gQf
    zero_model = _Zero(lin_net, THRESH)

    starting_conditions = [
        ([0.0, 0.0, 0.0, 0.0],     "x0=[0,0,0,0] (standard)"),
        ([0.1, 0.0, 0.0, 0.0],     "x0=[0.1,0,0,0]"),
        ([-0.1, 0.0, 0.0, 0.0],    "x0=[-0.1,0,0,0]"),
        ([0.5, 0.0, 0.0, 0.0],     "x0=[0.5,0,0,0] (faster)"),
        ([-0.5, 0.0, 0.0, 0.0],    "x0=[-0.5,0,0,0]"),
        ([0.0, 0.5, 0.0, 0.0],     "x0=[0,0.5,0,0] (q1d bias)"),
        ([0.0, -0.5, 0.0, 0.0],    "x0=[0,-0.5,0,0]"),
        ([0.0, 0.0, 0.3, 0.0],     "x0=[0,0,0.3,0] (q2 offset)"),
    ]

    print(f"  ROBUST EVAL [{datetime.now().strftime('%H:%M')}]  steps={args.steps}")
    print(f"  Comparing: boost (82.9% baseline) vs ZeroFNet (26.2%)")
    print(f"  {'Starting condition':>30}  {'boost f<0.10':>12}  {'zero f<0.10':>12}  {'arr':>5}")
    print(f"  {'-'*75}")

    boost_results, zero_results = [], []
    for x0_list, label in starting_conditions:
        x0 = torch.tensor(x0_list, device=device, dtype=torch.float64)
        f01b, f03b, arrb, postb = eval_trajectory(boost, mpc, x0, x_goal, args.steps)
        f01z, f03z, arrz, postz = eval_trajectory(zero_model, mpc, x0, x_goal, args.steps)
        boost_results.append(f01b)
        zero_results.append(f01z)
        delta = f01b - f01z
        print(f"  {label:>30}  {f01b:>11.1%}  {f01z:>11.1%}  {str(arrb):>5}  Δ={delta:+.1%}", flush=True)

    print(f"\n  Average: boost={sum(boost_results)/len(boost_results):.1%}  "
          f"zero={sum(zero_results)/len(zero_results):.1%}")
    print(f"  ZeroFNet baseline: 26.2%  |  boost trained: 82.9%")


if __name__ == "__main__":
    main()
