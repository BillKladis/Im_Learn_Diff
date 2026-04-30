"""exp_perdim_sweep.py — Per-dimension Q/R boost sweep from delta_Q=0 baseline.

MOTIVATION:
  exp_qboost_targeted showed uniform q_boost=0.05 → 24.5% (WORSE than 26.2%).
  Uniform Q boost hurts because increasing q2/q2d weight interferes with q1 control.

  This script tests EACH DIMENSION independently to find which dims help or hurt.
  Both positive AND negative values (reduction might help for some dims).

DESIGN:
  For each dimension d in {q1, q1d, q2, q2d}:
    For boost ∈ {-0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5}:
      delta_Q[:, d] = boost, all other dims = 0
      Evaluate 2000-step frac<0.10

  Also test R boost independently (same design).

  Hypothesis: boosting q1/q1d helps, boosting q2/q2d hurts.
  Expected optimal: delta_Q[:, 0] > 0, delta_Q[:, 2] ≈ 0 or slightly negative.
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
X0          = [0.0, 0.0, 0.0, 0.0]; X_GOAL = [math.pi, 0.0, 0.0, 0.0]
DT          = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]; THRESH = 0.8


class HoldBoostWrapper(nn.Module):
    def __init__(self, lin_net, thresh, delta_Q=None, delta_R=None, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net = lin_net; self.thresh = thresh; self.x_goal_q1 = x_goal_q1
        self.f_extra_bound = lin_net.f_extra_bound; self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim; self.control_dim = lin_net.control_dim
        q_shape = (lin_net.horizon - 1, lin_net.state_dim)
        r_shape = (lin_net.horizon,     lin_net.control_dim)
        if delta_Q is None:
            self.register_buffer('delta_Q', torch.zeros(q_shape, dtype=torch.float64))
        else:
            self.register_buffer('delta_Q', delta_Q)
        if delta_R is None:
            self.register_buffer('delta_R', torch.zeros(r_shape, dtype=torch.float64))
        else:
            self.register_buffer('delta_R', delta_R)

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf = self.lin_net(
            x_sequence, q_base_diag, r_base_diag
        )
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        f_extra = f_extra * (1.0 - gate)
        gates_Q = gates_Q + gate * self.delta_Q
        gates_R = gates_R + gate * self.delta_R
        return gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf


def eval2k(model, mpc, x0, x_goal, steps=1000):
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
    print(f"  PER-DIM Q/R BOOST SWEEP  [{datetime.now().strftime('%H:%M')}]")
    print(f"  Baseline: 26.2% (2000-step)  q_boost=0.05→24.5% (scalar hurt)")
    print(f"  Testing each dim independently from 0 baseline (1000-step evals)")
    print("=" * 70)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.eval()

    boosts_q = [-0.30, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    dim_names = ['q1', 'q1d', 'q2', 'q2d']
    Q_BASE = Q_BASE_DIAG

    best_overall = 0.262
    best_config = None

    for dim, name in enumerate(dim_names):
        print(f"\n  Q DIM={name} (base={Q_BASE[dim]}), all other dims = 0:")
        print(f"  {'boost':>8}  {'frac<0.10':>10}  {'frac<0.30':>10}  {'arr':>6}  {'post':>8}")
        print("  " + "─" * 50)
        best_this_dim = 0.262
        for b in boosts_q:
            dQ = torch.zeros((HORIZON-1, 4), dtype=torch.float64, device=device)
            dQ[:, dim] = b
            model = HoldBoostWrapper(lin_net, THRESH, delta_Q=dQ)
            f01, f03, arr, post = eval2k(model, mpc, x0, x_goal, steps=1000)
            mark = ""
            if f01 > best_overall:
                mark = " ★★"
                best_overall = f01
                best_config = (dim, name, b, f01)
            elif f01 > best_this_dim:
                mark = " ★"
                best_this_dim = f01
            print(f"  {b:>8.2f}  {f01:>10.1%}  {f03:>10.1%}  {str(arr):>6}  "
                  f"{(f'{post:.1%}' if post else 'N/A'):>8}{mark}", flush=True)

    # Also test R boost with Q=0 (control cost change)
    boosts_r = [-0.50, -0.30, -0.10, -0.05, 0.0, 0.05, 0.10, 0.30]
    for rdim, rname in enumerate(['u1', 'u2']):
        print(f"\n  R DIM={rname}, all other dims = 0:")
        print(f"  {'boost':>8}  {'frac<0.10':>10}  {'frac<0.30':>10}  {'arr':>6}  {'post':>8}")
        print("  " + "─" * 50)
        for b in boosts_r:
            dR = torch.zeros((HORIZON, 2), dtype=torch.float64, device=device)
            dR[:, rdim] = b
            model = HoldBoostWrapper(lin_net, THRESH, delta_R=dR)
            f01, f03, arr, post = eval2k(model, mpc, x0, x_goal, steps=1000)
            mark = " ★★" if f01 > best_overall else ""
            if f01 > best_overall:
                best_overall = f01
                best_config = ('R', rname, b, f01)
            print(f"  {b:>8.2f}  {f01:>10.1%}  {f03:>10.1%}  {str(arr):>6}  "
                  f"{(f'{post:.1%}' if post else 'N/A'):>8}{mark}", flush=True)

    # Test best Q dims combined (if any improved)
    print(f"\n  SUMMARY:")
    print(f"  Baseline: 26.2%")
    print(f"  Best single-dim: {best_overall:.1%}", end="")
    if best_config is not None:
        print(f"  (dim={best_config[1]}, boost={best_config[2]:.2f})")
    else:
        print(f"  (no improvement found)")

    if best_overall > 0.262:
        print(f"  ★ IMPROVEMENT: {best_overall:.1%} > 26.2%!")
        print(f"  → Use this as delta_Q initialization for holdboost training")
    else:
        print(f"  No single-dim Q/R boost improves baseline.")
        print(f"  The optimal correction must be multi-dim (gradient training needed).")


if __name__ == "__main__":
    main()
