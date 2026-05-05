"""eval_noise_compare.py — Compare v1 vs v2 noise-robust on obs noise.

Key question: did training with noise injection (v2) improve robustness?

Tests both checkpoints at:
  - Clean (σ=0)
  - Obs σ=0.002, 0.005, 0.010 (the MAB hardware-relevant range)
  - Obs σ=0.002 + EKF4 filter
  - Obs σ=0.005 + EKF4 filter

Takes ~15-20 min with two parallel rollouts per level.
"""

import glob
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
from ekf import EKF4, EKF6
from eval_ekf import rollout_ekf, R_CLEAN, Q_BIAS_MED

DEVICE    = torch.device("cpu")
X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 2000
SEED      = 42

MODEL_KWARGS = dict(
    state_dim=4, control_dim=2, horizon=10, hidden_dim=128,
    gate_range_q=0.99, gate_range_r=0.20, f_extra_bound=1.5, f_kickstart_amp=0.01,
)

Q_STATE_HW = torch.diag(torch.tensor([1e-6, 1e-4, 1e-6, 1e-4], dtype=torch.float64))
R_HW_CLEAN = torch.diag(torch.tensor([4e-6, 1e-3, 4e-6, 1e-3], dtype=torch.float64))


def load_model(ckpt_path):
    data = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = data.get("model_state_dict", data)
    model = network_module.SeparatedLinearizationNetwork(**MODEL_KWARGS).double()
    model.load_state_dict(sd)
    model.eval()
    return model


def compute_metrics(x_hist, x_goal):
    traj = x_hist.cpu().numpy()
    wraps = np.array([
        math.sqrt(
            math.atan2(math.sin(s[0] - math.pi), math.cos(s[0] - math.pi))**2
            + s[1]**2 + s[2]**2 + s[3]**2
        )
        for s in traj
    ])
    arr  = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    return {
        "f01": float((wraps < 0.10).mean()),
        "arr": arr,
        "post": post,
    }


def fmt(m):
    arr  = m['arr'] if m['arr'] is not None else "---"
    post = f"{m['post']:.1%}" if m['post'] is not None else "  ---"
    return f"f01={m['f01']:.1%}  arr={arr:>4}  post={post}"


def run_one(label, model, mpc, x0, x_goal, obs_sigma=0.0, use_ekf4=False):
    ekf = None
    if use_ekf4:
        R = torch.diag(torch.tensor([obs_sigma**2]*4, dtype=torch.float64))
        ekf = EKF4(mpc, Q_STATE_HW, R)
        ekf.reset(x0)
    xh = rollout_ekf(model, mpc, x0, x_goal, NUM_STEPS,
                     obs_sigma=obs_sigma, ekf=ekf, cancel_bias=False)
    m = compute_metrics(xh, x_goal)
    print(f"    {label:<45}  {fmt(m)}", flush=True)
    return m


def main():
    x0     = torch.tensor(X0,     dtype=torch.float64, device=DEVICE)
    x_goal = torch.tensor(X_GOAL, dtype=torch.float64, device=DEVICE)
    mpc    = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=10, device=DEVICE)

    # Find checkpoints
    v1_paths = glob.glob("saved_models/hw_v1*/*.pth")
    if not v1_paths:
        print("No hw_v1 checkpoint found.")
        return
    v1_ckpt = max(v1_paths, key=os.path.getmtime)

    v2_paths = glob.glob("saved_models/hw_v2_nr_diag*/*.pth") + \
               glob.glob("saved_models/hw_v2_nr_*FINAL*/*.pth") + \
               glob.glob("saved_models/hw_v2_nr_2*/*.pth")
    if not v2_paths:
        print("No hw_v2 checkpoint found. Using v1 for both.")
        v2_ckpt = v1_ckpt
    else:
        v2_ckpt = max(v2_paths, key=os.path.getmtime)

    print(f"\n  v1 checkpoint: {v1_ckpt}")
    print(f"  v2 checkpoint: {v2_ckpt}")

    v1 = load_model(v1_ckpt)
    v2 = load_model(v2_ckpt)

    tests = [
        ("clean (σ=0)",             0.000, False),
        ("obs σ=0.001",             0.001, False),
        ("obs σ=0.002 RAW",         0.002, False),
        ("obs σ=0.002 + EKF4",      0.002, True),
        ("obs σ=0.005 RAW",         0.005, False),
        ("obs σ=0.005 + EKF4",      0.005, True),
        ("obs σ=0.010 RAW",         0.010, False),
        ("obs σ=0.010 + EKF4",      0.010, True),
        ("obs σ=0.020 RAW",         0.020, False),
        ("obs σ=0.020 + EKF4",      0.020, True),
    ]

    print(f"\n{'═'*90}")
    print(f"  NOISE ROBUSTNESS COMPARISON: hw_v1_ep50  vs  hw_v2_ep80")
    print(f"{'═'*90}")

    results = {}
    for label, sigma, ekf4 in tests:
        print(f"\n  [{label}]")
        r1 = run_one(f"  hw_v1_ep50", v1, mpc, x0, x_goal, sigma, ekf4)
        r2 = run_one(f"  hw_v2_ep80", v2, mpc, x0, x_goal, sigma, ekf4)
        delta = r2['f01'] - r1['f01']
        sign = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "≈")
        print(f"    → Δf01 = {delta:+.1%}  {sign}")
        results[label] = {"v1": r1, "v2": r2, "delta": delta}

    print(f"\n\n{'═'*90}")
    print(f"  SUMMARY")
    print(f"{'═'*90}")
    print(f"  {'Test':<45}  {'v1 f01':>8}  {'v2 f01':>8}  {'Δ':>8}")
    print(f"  {'-'*75}")
    for label, res in results.items():
        d = res['delta']
        sign = "▲" if d > 0.005 else ("▼" if d < -0.005 else "≈")
        print(f"  {label:<45}  {res['v1']['f01']:>8.1%}  {res['v2']['f01']:>8.1%}  {d:>+8.1%} {sign}")
    print("\nDone.")


if __name__ == "__main__":
    main()
