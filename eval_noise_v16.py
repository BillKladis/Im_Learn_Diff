"""eval_noise_v16.py — Quick noise robustness comparison: v14m vs v16 best vs v16 final.

Tests:
  - Clean rollout
  - σ_ctrl = 0.05  (training noise level)
  - σ_ctrl = 0.10  (the level that devastated v14m, 50.7%)
  - σ_ctrl = 0.20  (severe)
  - σ_obs  = 0.10  (should be unaffected by both models)
  - Combined σ_obs=0.05 + σ_ctrl=0.10
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module

# reuse rollout_custom and compute_metrics from eval_comprehensive
from eval_comprehensive import rollout_custom, compute_metrics, fmt

DEVICE = torch.device("cpu")
X0_FIXED   = [0.0, 0.0, 0.0, 0.0]
X_GOAL_VAL = [math.pi, 0.0, 0.0, 0.0]
N, DT = 10, 0.05

MODEL_KWARGS = dict(
    state_dim=4, control_dim=2, horizon=10, hidden_dim=128,
    gate_range_q=0.99, gate_range_r=0.20, f_extra_bound=2.5, f_kickstart_amp=1.0,
)

CHECKPOINTS = {
    "v14m    ": "saved_models/stageF_mixed_v14m_20260503_102608_ep50/stageF_mixed_v14m_20260503_102608_ep50.pth",
    "v16-best": "saved_models/stageF_noiserobust_v16_BEST_20260504_092338_ep50/stageF_noiserobust_v16_BEST_20260504_092338_ep50.pth",
    "v16-ep100": "saved_models/stageF_noiserobust_v16_20260504_094802_ep100/stageF_noiserobust_v16_20260504_094802_ep100.pth",
}

NOISE_TESTS = [
    ("clean",              dict()),
    ("σ_ctrl=0.05",        dict(ctrl_sigma=0.05)),
    ("σ_ctrl=0.10",        dict(ctrl_sigma=0.10)),
    ("σ_ctrl=0.20",        dict(ctrl_sigma=0.20)),
    ("σ_obs=0.10",         dict(obs_sigma=0.10)),
    ("σ_obs=0.05+σ_c=0.10", dict(obs_sigma=0.05, ctrl_sigma=0.10)),
]


def load_model(ckpt_path):
    data = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = data.get("model_state_dict", data)
    model = network_module.SeparatedLinearizationNetwork(**MODEL_KWARGS).double()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    x0     = torch.tensor(X0_FIXED,   dtype=torch.float64, device=DEVICE)
    x_goal = torch.tensor(X_GOAL_VAL, dtype=torch.float64, device=DEVICE)

    results = {}

    for model_name, ckpt_path in CHECKPOINTS.items():
        print(f"\nLoading {model_name.strip()} ...")
        model = load_model(ckpt_path)
        mpc   = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=N, device=DEVICE)
        results[model_name] = {}

        for test_name, kwargs in NOISE_TESTS:
            x_hist, _ = rollout_custom(model, mpc, x0, x_goal, num_steps=2000, **kwargs)
            m = compute_metrics(x_hist, x_goal)
            results[model_name][test_name] = m
            print(f"  {model_name}  {test_name:<24}  {fmt(m)}", flush=True)

    # Summary table
    print("\n\n" + "=" * 100)
    print("  NOISE ROBUSTNESS SUMMARY  (f01 / post)")
    print("=" * 100)
    print(f"  {'Test':<26}", end="")
    for name in CHECKPOINTS:
        print(f"  {name:<12}", end="")
    print()
    print("  " + "-" * 96)

    for test_name, _ in NOISE_TESTS:
        print(f"  {test_name:<26}", end="")
        for name in CHECKPOINTS:
            m = results[name][test_name]
            arr = m['arr'] if m['arr'] is not None else 9999
            print(f"  {m['f01']:.1%} arr={arr:<4}", end="")
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()
