"""exp_thresh_upper_sweep.py — Sweep HoldBoostWrapper threshold above 0.80.

We know thresh=0.60 gives arr=322 (worse). We NEVER tested thresh>0.80.
This script tests thresh in [0.80, 0.825, 0.85, 0.875, 0.90] in ONE process
(CVXPY compiled once). Fast way to check if wider or narrower activation helps.

The HoldBoostWrapper uses: near_pi = (1+cos(q1-π))/2 > thresh → Q boost ON

thresh=0.80 → q1 within 53° (current best: 87.2%)
thresh=0.85 → q1 within ~46° (tighter zone, approaching like v2 scalegate)
thresh=0.90 → q1 within ~37° (very tight zone)
thresh=0.75 → q1 within ~60° (wider zone, risky)
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


class ScaledBoostWrapper(torch.nn.Module):
    """HoldBoostWrapper with configurable threshold, matching exp_dq_scale_sweep exactly."""
    def __init__(self, lin_net, dQ_ref, dR_ref, thresh=0.80):
        super().__init__()
        self.lin_net = lin_net
        self.register_buffer('dQ_ref', dQ_ref.clone())  # (9,4)
        self.register_buffer('dR_ref', dR_ref.clone())  # (10,2)
        self.thresh = thresh
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim
        self.control_dim = lin_net.control_dim

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - math.pi)) / 2.0
        gate = ((near_pi - self.thresh) / (1.0 - self.thresh)).clamp(0, 1)
        gQ = gQ + gate * self.dQ_ref         # (9,4)
        gR = gR + gate * self.dR_ref         # (10,2)
        fe = fe * (1.0 - gate.detach())
        return gQ, gR, fe, qd, rd, gQf


def eval2k(model, mpc, x0, x_goal, steps=2000):
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
    dQ_ref = torch.tensor(tp['best_delta_Q'], dtype=torch.float64)  # (9,4)
    dR_ref = torch.tensor(tp['best_delta_R'], dtype=torch.float64)  # (10,2)

    thresholds = [0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95]

    print("=" * 70)
    print("  THRESH UPPER SWEEP: near_pi threshold on HoldBoostWrapper")
    print(f"  dQ_ref mean={dQ_ref.mean(0).tolist()}")
    print("  Thresholds tested:", thresholds)
    print("  (CVXPY compiled once, ~25 min then fast sweeps)")
    print("=" * 70)
    print(f"\n  Threshold zone sizes (q1d=q2=q2d=0):")
    for t in thresholds:
        angle_deg = math.degrees(math.acos(2*t - 1))
        print(f"    thresh={t:.3f} → q1 within {angle_deg:.1f}° of top")

    # Compile CVXPY on first eval
    print(f"\n  First eval (thresh=0.80, compiling CVXPY ~25 min)...")
    t0 = time.time()

    results = {}
    for i, thresh in enumerate(thresholds):
        model = ScaledBoostWrapper(lin_net, dQ_ref, dR_ref, thresh=thresh)
        f01, arr, post = eval2k(model, mpc, x0_eval, x_goal)
        results[thresh] = (f01, arr, post)
        elapsed = time.time() - t0
        print(f"  thresh={thresh:.3f}  f01={f01:.1%}  arr={arr}  "
              f"post={f'{post:.1%}' if post else 'N/A'}  t={elapsed:.0f}s", flush=True)

    print(f"\n  SUMMARY:")
    print(f"  {'thresh':>8}  {'f01':>8}  {'arr':>6}  {'post':>8}")
    for thresh, (f01, arr, post) in sorted(results.items()):
        mark = " ★" if f01 == max(v[0] for v in results.values()) else ""
        print(f"  {thresh:>8.3f}  {f01:>8.1%}  {arr:>6}  {f'{post:.1%}' if post else 'N/A':>8}{mark}")


if __name__ == "__main__":
    main()
