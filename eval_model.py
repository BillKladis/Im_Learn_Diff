"""eval_model.py — Comprehensive standalone evaluation of any trained checkpoint.

Runs a standard suite of evaluations on any model (any architecture, any u_lim)
and prints a clean summary table.

Usage
-----
# Evaluate latest v1 (clean + noise suite):
python eval_model.py latest_v1

# Evaluate v3 with its native u_lim from checkpoint:
python eval_model.py latest_v3

# Compare two models side-by-side:
python eval_model.py latest_v1 latest_v3

# Evaluate a specific path, longer rollout:
python eval_model.py saved_models/hw_v3_u010_FINAL*/hw_v3*.pth --steps 5000

# Quick eval (clean only, 2k steps):
python eval_model.py latest_v1 --quick
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import hardware_deploy as hd
import mpc_controller as mpc_module
import Simulate as sim_module
from ekf import EKF4, EKF6

DEVICE = torch.device("cpu")
X0     = [0.0, 0.0, 0.0, 0.0]
X_GOAL = [math.pi, 0.0, 0.0, 0.0]
SEED   = 42


def rollout_clean(model, mpc, x0, x_goal, steps):
    """Standard clean rollout (no noise, no EKF)."""
    model.eval()
    x_t, _ = sim_module.rollout(lin_net=model, mpc=mpc, x0=x0,
                                 x_goal=x_goal, num_steps=steps)
    model.train()
    return x_t


def rollout_noisy(model, mpc, x0, x_goal, steps, obs_sigma, ekf=None):
    """Rollout with observation noise and optional EKF."""
    from eval_ekf import rollout_ekf
    xh = rollout_ekf(model, mpc, x0, x_goal, steps,
                     obs_sigma=obs_sigma, ekf=ekf, cancel_bias=False)
    return xh


def compute_metrics(x_hist):
    """Compute f01, arrival step, and post-arrival fraction."""
    traj  = x_hist.cpu().numpy()
    wraps = np.array([
        math.sqrt(
            math.atan2(math.sin(s[0] - math.pi), math.cos(s[0] - math.pi))**2
            + s[1]**2 + s[2]**2 + s[3]**2
        )
        for s in traj
    ])
    arr  = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    f01  = float((wraps < 0.10).mean())
    f03  = float((wraps < 0.30).mean())
    err_final = float(wraps[-1])
    return {"f01": f01, "f03": f03, "arr": arr, "post": post, "err_final": err_final}


def fmt_metric(m):
    arr  = str(m["arr"]) if m["arr"] is not None else "---"
    post = f"{m['post']:.1%}" if m["post"] is not None else "  ---"
    return f"f01={m['f01']:>6.1%}  f03={m['f03']:>6.1%}  arr={arr:>5}  post={post}  err={m['err_final']:.3f}"


def eval_one(name, model, mpc, x0, x_goal, steps, quick=False):
    """Run evaluation suite for one model."""
    print(f"\n  ── {name} ──")

    # Clean
    xh = rollout_clean(model, mpc, x0, x_goal, steps)
    m  = compute_metrics(xh)
    print(f"    {'clean':42}  {fmt_metric(m)}")

    if quick:
        return {"clean": m}

    results = {"clean": m}

    # Noise levels (try to import eval_ekf, skip if not available)
    try:
        from eval_ekf import rollout_ekf
        from ekf import EKF4, EKF6

        noise_levels = [
            ("obs σ=0.001 RAW",   0.001, None),
            ("obs σ=0.002 RAW",   0.002, None),
            ("obs σ=0.005 RAW",   0.005, None),
            ("obs σ=0.010 RAW",   0.010, None),
            ("obs σ=0.002 +EKF4", 0.002, "ekf4"),
            ("obs σ=0.005 +EKF4", 0.005, "ekf4"),
            ("obs σ=0.010 +EKF4", 0.010, "ekf4"),
        ]

        Q_s = torch.diag(torch.tensor([1e-6, 1e-4, 1e-6, 1e-4], dtype=torch.float64))
        R_hw = torch.diag(torch.tensor([4e-6, 1e-3, 4e-6, 1e-3], dtype=torch.float64))

        for label, sigma, ekf_type in noise_levels:
            ekf = None
            if ekf_type == "ekf4":
                R = torch.diag(torch.tensor([sigma**2]*4, dtype=torch.float64))
                ekf = EKF4(mpc, Q_s, R)
                ekf.reset(x0)
            xh = rollout_ekf(model, mpc, x0, x_goal, steps,
                             obs_sigma=sigma, ekf=ekf, cancel_bias=False)
            m  = compute_metrics(xh)
            print(f"    {label:42}  {fmt_metric(m)}")
            results[label] = m

    except ImportError:
        print("    [skip noise tests — eval_ekf not available]")

    return results


def main():
    p = argparse.ArgumentParser(description="Evaluate trained checkpoint(s)")
    p.add_argument("models", nargs="+", help="Model spec(s): latest_v1/v2/v3/v4/v5/v6 or path")
    p.add_argument("--steps", type=int, default=2000, help="Rollout length (default 2000)")
    p.add_argument("--quick", action="store_true", help="Clean-only fast eval")
    p.add_argument("--u_lim", type=float, default=None, help="Override torque limit")
    args = p.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    x0     = torch.tensor(X0,     dtype=torch.float64, device=DEVICE)
    x_goal = torch.tensor(X_GOAL, dtype=torch.float64, device=DEVICE)

    # Resolve all model specs
    entries = []
    for spec in args.models:
        try:
            path = hd.resolve_model_path(spec)
        except FileNotFoundError:
            print(f"  ERROR: Cannot find model: {spec!r}")
            continue
        model, meta, u_lim_ckpt = hd.load_model(path)
        u_lim = args.u_lim if args.u_lim is not None else u_lim_ckpt
        label = f"{spec}  [{meta['arch'][:12]}  u_lim={u_lim}]"
        entries.append((label, model, u_lim, path))

    if not entries:
        print("No valid models found.")
        return

    print(f"\n{'='*90}")
    print(f"  EVALUATION SUITE  ({args.steps} steps @ 20 Hz = {args.steps*0.05:.0f}s)")
    print(f"{'='*90}")

    all_results = {}
    for label, model, u_lim, path in entries:
        print(f"\n  Checkpoint: {path}")
        print(f"  u_lim: {u_lim} Nm")

        mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=10,
                                         device=DEVICE, u_lim=u_lim)

        results = eval_one(label, model, mpc, x0, x_goal, args.steps, quick=args.quick)
        all_results[label] = results

    # Summary table
    if len(entries) > 1 and not args.quick:
        print(f"\n\n{'='*90}")
        print(f"  COMPARISON SUMMARY  (f01)")
        print(f"{'='*90}")
        labels_short = [e[0].split("[")[0].strip() for e in entries]
        cond_labels  = list(list(all_results.values())[0].keys())

        hdr = f"  {'Condition':<42}" + "".join(f"  {l:>14}" for l in labels_short)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for cond in cond_labels:
            row = f"  {cond:<42}"
            for label in all_results:
                m = all_results[label].get(cond)
                row += f"  {m['f01']:>13.1%}" if m else f"  {'---':>13}"
            print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
