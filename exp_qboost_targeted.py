"""exp_qboost_targeted.py — Targeted evaluation of positive scalar Q boosts near top.

Previous qboost_sweep showed q_boost=-0.5→0%, -0.3→0.6%. Positive boosts not tested.
This script sweeps q_boost ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5} with 2000-step eval.
Evaluates both scalar Q boost AND per-state-dim Q boost (q1 vs q1d vs q2 vs q2d separately).
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


class QBoostWrapper(nn.Module):
    def __init__(self, lin_net, thresh, delta_Q, delta_R=0.0, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net = lin_net; self.thresh = thresh; self.x_goal_q1 = x_goal_q1
        self.delta_Q = delta_Q; self.delta_R = delta_R
        self.f_extra_bound = lin_net.f_extra_bound; self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim; self.control_dim = lin_net.control_dim

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, f_extra, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        f_extra = f_extra * (1.0 - gate)
        gQ = gQ + gate * self.delta_Q
        gR = gR + gate * self.delta_R
        return gQ, gR, f_extra, qd, rd, gQf


def eval2k(model, mpc, x0, x_goal):
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=2000)
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
    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.eval()

    print("=" * 70)
    print(f"  TARGETED Q BOOST EVAL  [{datetime.now().strftime('%H:%M')}]")
    print(f"  Baseline: 26.2% (ZeroFNet thresh=0.80, no boost)")
    print("=" * 70)

    # ── 1. Scalar positive boosts ────────────────────────────────────────────
    print("\n  SCALAR Q BOOST (same for all dims/steps):")
    print(f"  {'q_boost':>8}  {'frac<0.10':>10}  {'frac<0.30':>10}  {'arr':>6}  {'post':>8}")
    print("  " + "─" * 50)
    best_f01, best_qb = 0.262, 0.0
    t0 = time.time()
    for qb in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50]:
        dQ = torch.full((HORIZON-1, 4), qb, dtype=torch.float64, device=device)
        model = QBoostWrapper(lin_net, THRESH, delta_Q=dQ)
        f01, f03, arr, post = eval2k(model, mpc, x0, x_goal)
        mark = " ★" if f01 > best_f01 else ""
        print(f"  {qb:>8.2f}  {f01:>10.1%}  {f03:>10.1%}  {str(arr):>6}  "
              f"{(f'{post:.1%}' if post else 'N/A'):>8}{mark}", flush=True)
        if f01 > best_f01:
            best_f01 = f01; best_qb = qb

    print(f"\n  Best scalar Q boost: {best_qb:.2f} → {best_f01:.1%}  (elapsed {time.time()-t0:.0f}s)")

    # ── 2. Per-dimension Q boosts at best scalar ─────────────────────────────
    if best_qb > 0.0:
        print(f"\n  PER-DIM Q BOOST (one dim at a time, others at {best_qb:.2f}):")
        print(f"  {'boost_dim':>12}  {'frac<0.10':>10}  {'arr':>6}  {'post':>8}")
        print("  " + "─" * 45)
        best_pd_f01, best_pd = best_f01, [best_qb, best_qb, best_qb, best_qb]
        dim_names = ['q1', 'q1d', 'q2', 'q2d']
        for dim, name in enumerate(dim_names):
            for mult in [0.0, 0.5, best_qb, 2.0*best_qb]:
                dQ = torch.full((HORIZON-1, 4), best_qb, dtype=torch.float64, device=device)
                dQ[:, dim] = mult
                model = QBoostWrapper(lin_net, THRESH, delta_Q=dQ)
                f01, _, arr, post = eval2k(model, mpc, x0, x_goal)
                mark = " ★" if f01 > best_pd_f01 else ""
                print(f"  {name+'='+str(mult):>12}  {f01:>10.1%}  {str(arr):>6}  "
                      f"{(f'{post:.1%}' if post else 'N/A'):>8}{mark}", flush=True)
                if f01 > best_pd_f01:
                    best_pd_f01 = f01

    # ── 3. R boost sweep ─────────────────────────────────────────────────────
    print(f"\n  SCALAR R BOOST (Q_boost={best_qb:.2f}):")
    print(f"  {'r_boost':>8}  {'frac<0.10':>10}  {'arr':>6}  {'post':>8}")
    print("  " + "─" * 40)
    for rb in [-0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
        dQ = torch.full((HORIZON-1, 4), best_qb, dtype=torch.float64, device=device)
        dR = torch.full((HORIZON, 2), rb, dtype=torch.float64, device=device)
        model = QBoostWrapper(lin_net, THRESH, delta_Q=dQ, delta_R=dR)
        f01, _, arr, post = eval2k(model, mpc, x0, x_goal)
        mark = " ★" if f01 > best_f01 else ""
        print(f"  {rb:>8.2f}  {f01:>10.1%}  {str(arr):>6}  "
              f"{(f'{post:.1%}' if post else 'N/A'):>8}{mark}", flush=True)

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")
    print(f"  Best result vs baseline: {best_f01:.1%} vs 26.2%")
    if best_f01 > 0.262:
        print(f"  ★★ IMPROVEMENT FOUND! Optimal Q boost = {best_qb:.2f} ★★")
    else:
        print(f"  No improvement. ZeroFNet 26.2% appears near-optimal.")


if __name__ == "__main__":
    main()
