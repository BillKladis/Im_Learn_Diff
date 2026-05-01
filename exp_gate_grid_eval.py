"""exp_gate_grid_eval.py — Grid evaluation of (w, b) gate parameters.

Tests specific LinearRampGate configs in ONE process (CVXPY compiled once).
Extends the threshold sweep by allowing independent w (slope) and b (intercept).

The threshold sweep tests the "natural" diagonal where width = 1-thresh:
  thresh=0.80 → w=5, b=-4 (width=0.20)
  thresh=0.85 → w=6.67, b=-5.67 (width=0.15)

This script tests OFF-DIAGONAL configs like:
  thresh=0.80, width=0.20 (w=5, b=-4) — same as wrapper
  thresh=0.78, width=0.125 (w=8, b=-6.24) — steeper slope (full at 0.905)
  thresh=0.78, width=0.20 (w=5, b=-3.9) — same slope, lower thresh
  thresh=0.85, width=0.20 (w=5, b=-4.25) — higher thresh, same slope
"""

import math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

POSONLY_FINAL = "saved_models/stageD_posonly_ft_20260430_083618/stageD_posonly_ft_20260430_083618.pth"
SCALE4_CKPT   = "saved_models/stageD_scale4.0x_dQ_20260430_192447/stageD_scale4.0x_dQ_20260430_192447.pth"
X0     = [0.0, 0.0, 0.0, 0.0]
X_GOAL = [math.pi, 0.0, 0.0, 0.0]
DT     = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
STATE_HIST = 5


class LinearRampGate(torch.nn.Module):
    def __init__(self, lin_net, dQ_ref, dR_ref, w=5.0, b=-4.0):
        super().__init__()
        self.lin_net = lin_net
        self.register_buffer('dQ_ref', dQ_ref.clone())
        self.register_buffer('dR_ref', dR_ref.clone())
        self.w = w; self.b = b
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim
        self.control_dim = lin_net.control_dim

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - math.pi)) / 2.0
        alpha = (self.w * near_pi + self.b).clamp(0.0, 1.0)
        gQ = gQ + alpha * self.dQ_ref
        gR = gR + alpha * self.dR_ref
        fe = fe * (1.0 - alpha)
        return gQ, gR, fe, qd, rd, gQf


def eval2k(model, mpc, x0, x_goal, steps=2000):
    with torch.no_grad():
        x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([math.sqrt(
        math.atan2(math.sin(s[0]-math.pi), math.cos(s[0]-math.pi))**2
        + s[1]**2 + s[2]**2 + s[3]**2) for s in traj])
    arr = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    return float((wraps < 0.10).mean()), arr, post


def main():
    device = torch.device("cpu")
    x0_eval = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal  = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0_eval, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device='cpu').double()
    lin_net.requires_grad_(False)

    ckpt = torch.load(SCALE4_CKPT, map_location='cpu', weights_only=False)
    tp = ckpt['metadata']['training_params']
    dQ_ref = torch.tensor(tp['best_delta_Q'], dtype=torch.float64)
    dR_ref = torch.tensor(tp['best_delta_R'], dtype=torch.float64)

    # Grid: (w, b) combinations
    # Format: (w, b, description)
    configs = [
        # Threshold sweep diagonal (natural pairs where width = 1-thresh)
        (5.000, -4.000, "thresh=0.800 width=0.200 [BASELINE]"),
        (6.667, -5.667, "thresh=0.850 width=0.150 [sweep equiv]"),
        (8.000, -7.200, "thresh=0.900 width=0.125 [sweep equiv]"),
        (10.00, -9.500, "thresh=0.950 width=0.100 [sweep equiv]"),
        # Fixed slope w=5, varying threshold (v4-high territory)
        (5.000, -3.900, "thresh=0.780 width=0.200"),
        (5.000, -3.750, "thresh=0.750 width=0.200 [wide, same slope]"),
        (5.000, -4.125, "thresh=0.825 width=0.200"),
        (5.000, -4.250, "thresh=0.850 width=0.200 [v4-high init]"),
        (5.000, -4.375, "thresh=0.875 width=0.200"),
        # Steeper slopes, same threshold=0.800
        (8.000, -6.400, "thresh=0.800 width=0.125 [steeper at same thresh]"),
        (10.00, -8.000, "thresh=0.800 width=0.100 [steep at same thresh]"),
        # Steeper slopes, lower threshold (wider + steeper)
        (8.000, -6.240, "thresh=0.780 width=0.125 [wide+steep]"),
        (8.000, -6.000, "thresh=0.750 width=0.125 [widest steep]"),
        # ep=10 v4 result
        (5.069, -3.930, "thresh=0.775 width=0.197 [v4 ep=10 approx]"),
    ]

    print("=" * 80)
    print("  GATE GRID EVAL: Testing (w, b) combinations")
    print(f"  dQ_ref mean={dQ_ref.mean(0).tolist()}")
    print("  gate formula: alpha = (w * near_pi + b).clamp(0,1)")
    print(f"  {len(configs)} configs × 2000 steps each")
    print("=" * 80)
    print(f"\n  Config characteristics (near_pi for α=0 (thresh), α=0.5 (mid), α=1.0 (full)):")
    for w, b, desc in configs:
        thresh = -b/w
        full = (1-b)/w
        print(f"    w={w:.3f} b={b:.3f}: thresh={thresh:.3f}  full={full:.3f}  [{desc}]")

    print(f"\n  Evaluating (compiling CVXPY first — ~7-25 min)...")
    t0 = time.time()

    results = {}
    for w, b, desc in configs:
        model = LinearRampGate(lin_net, dQ_ref, dR_ref, w=w, b=b)
        f01, arr, post = eval2k(model, mpc, x0_eval, x_goal)
        results[(w, b)] = (f01, arr, post)
        elapsed = time.time() - t0
        mark = " ★" if f01 > 0.872 else ""
        print(f"  w={w:.3f} b={b:.3f}: f01={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}"
              f"  t={elapsed:.0f}s  [{desc}]{mark}", flush=True)

    print(f"\n  SUMMARY (sorted by f01):")
    print(f"  {'w':>7}  {'b':>7}  {'f01':>8}  {'arr':>6}  {'post':>8}  description")
    for (w, b), (f01, arr, post) in sorted(results.items(), key=lambda x: -x[1][0]):
        mark = " ★" if f01 == max(v[0] for v in results.values()) else ""
        print(f"  {w:>7.3f}  {b:>7.3f}  {f01:>8.1%}  {arr:>6}  {f'{post:.1%}' if post else 'N/A':>8}  {mark}")


if __name__ == "__main__":
    main()
