"""exp_qboost_sweep.py — Quick sweep of scalar Q/R boosts near top.

Instead of gradient-based training, directly test: what additive correction
to Q (or R) near top maximizes hold quality?

Sweep: gates_Q += gate * scalar_boost (same boost for all horizon steps/dims)
Evaluates each value of scalar_boost for 2000 steps and reports frac<0.10.

This answers: "Is there ANY Q/R modification that improves over ZeroFNet 26.2%?"
If yes: use that value in holdboost. If no: ZeroFNet is the optimum.
"""

import math
import os
import sys
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
DT          = 0.05
HORIZON     = 10
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
THRESH      = 0.8   # ZeroFNet gate threshold


class QBoostWrapper(nn.Module):
    """ZeroFNet + additive constant boosts to Q and/or R near top."""
    def __init__(self, lin_net, thresh, q_boost, r_boost, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net    = lin_net
        self.thresh     = thresh
        self.x_goal_q1  = x_goal_q1
        self.q_boost    = q_boost   # scalar: added to all Q gate elements near top
        self.r_boost    = r_boost   # scalar: added to all R gate elements near top
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon       = lin_net.horizon
        self.state_dim     = lin_net.state_dim
        self.control_dim   = lin_net.control_dim

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf = self.lin_net(
            x_sequence, q_base_diag, r_base_diag
        )
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        f_extra = f_extra * (1.0 - gate)
        gates_Q = gates_Q + gate * self.q_boost
        gates_R = gates_R + gate * self.r_boost
        return gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf


def eval_rollout(model, mpc, x0, x_goal, steps=2000):
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([
        math.sqrt(
            math.atan2(math.sin(s[0] - math.pi), math.cos(s[0] - math.pi)) ** 2
            + s[1] ** 2 + s[2] ** 2 + s[3] ** 2
        )
        for s in traj
    ])
    arr_idx = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post_01 = None
    if arr_idx is not None:
        post = wraps[arr_idx:]
        post_01 = float((post < 0.10).mean())
    return {
        "arr_idx": arr_idx,
        "frac_01": float((wraps < 0.10).mean()),
        "frac_03": float((wraps < 0.30).mean()),
        "post_arr_01": post_01,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    print("=" * 80)
    print("  Q/R BOOST SWEEP — Direct evaluation (no training)")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  ZeroFNet baseline: 26.2% frac<0.10 (thresh=0.80, 2000-step)")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.eval()

    # Sweep Q boost (with R_boost=0)
    q_boosts = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    print("\n  Q BOOST SWEEP (R_boost=0):")
    print(f"  {'q_boost':>8}  {'frac<0.10':>10}  {'frac<0.30':>10}  {'arr':>6}  {'post<0.10':>10}")
    print("  " + "─" * 60)
    best_q, best_q_frac = 0.0, 0.0
    for qb in q_boosts:
        model = QBoostWrapper(lin_net, THRESH, q_boost=qb, r_boost=0.0)
        r = eval_rollout(model, mpc, x0, x_goal, steps=2000)
        post = f"{r['post_arr_01']:.1%}" if r['post_arr_01'] is not None else "N/A"
        mark = " ← BEST" if r['frac_01'] > best_q_frac else ""
        print(f"  {qb:>8.2f}  {r['frac_01']:>10.1%}  {r['frac_03']:>10.1%}  "
              f"{str(r['arr_idx']):>6}  {post:>10}{mark}")
        if r['frac_01'] > best_q_frac:
            best_q_frac = r['frac_01']
            best_q = qb

    print(f"\n  Best Q boost: {best_q:.2f} → {best_q_frac:.1%} frac<0.10")

    # Sweep R boost (with Q_boost=best_q)
    r_boosts = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
    print(f"\n  R BOOST SWEEP (Q_boost={best_q:.2f}):")
    print(f"  {'r_boost':>8}  {'frac<0.10':>10}  {'frac<0.30':>10}  {'arr':>6}  {'post<0.10':>10}")
    print("  " + "─" * 60)
    best_r, best_r_frac = 0.0, best_q_frac
    for rb in r_boosts:
        model = QBoostWrapper(lin_net, THRESH, q_boost=best_q, r_boost=rb)
        r = eval_rollout(model, mpc, x0, x_goal, steps=2000)
        post = f"{r['post_arr_01']:.1%}" if r['post_arr_01'] is not None else "N/A"
        mark = " ← BEST" if r['frac_01'] > best_r_frac else ""
        print(f"  {rb:>8.2f}  {r['frac_01']:>10.1%}  {r['frac_03']:>10.1%}  "
              f"{str(r['arr_idx']):>6}  {post:>10}{mark}")
        if r['frac_01'] > best_r_frac:
            best_r_frac = r['frac_01']
            best_r = rb

    print(f"\n  Best (Q_boost={best_q:.2f}, R_boost={best_r:.2f}) → {best_r_frac:.1%}")
    print(f"\n  Baseline (no boost): 26.2%")
    if best_r_frac > 0.262:
        print(f"  ★ IMPROVEMENT FOUND: {best_r_frac:.1%} > 26.2%")
    else:
        print(f"  No scalar Q/R boost improves over ZeroFNet baseline.")
        print(f"  ZeroFNet 26.2% appears to be near-optimal for this architecture.")

    # Fine-grained sweep around best Q if improvement found
    if best_q != 0.0:
        fine_q = [best_q + d for d in [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]]
        print(f"\n  FINE Q SWEEP around {best_q:.2f}:")
        for qb in fine_q:
            model = QBoostWrapper(lin_net, THRESH, q_boost=qb, r_boost=best_r)
            r = eval_rollout(model, mpc, x0, x_goal, steps=2000)
            post = f"{r['post_arr_01']:.1%}" if r['post_arr_01'] is not None else "N/A"
            print(f"  {qb:>8.3f}  {r['frac_01']:>10.1%}  {r['frac_03']:>10.1%}  arr={r['arr_idx']}  post={post}")


if __name__ == "__main__":
    main()
