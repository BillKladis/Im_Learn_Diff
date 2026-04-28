"""exp_expansion2.py — Expansion with x0=zero ALTERNATION to prevent forgetting.

Each iteration trains on TWO trajectories:
  - x0=zero (canonical, anchors the swing-up)
  - x0=perturbed (random sample for generalisation)

The shared optimizer accumulates gradients from both.  This prevents the
catastrophic forgetting we saw in exp_expansion.py where the model degraded
on x0=zero as it specialized to recent perturbations.
"""

import math
import os
import sys
import time
import copy
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

PRETRAINED = "saved_models/stageD_nodemo_20260428_123448/stageD_nodemo_20260428_123448.pth"

X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
NUM_OUTER = 30   # outer iterations
LR        = 5e-5
HORIZON   = 10
SAVE_DIR  = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

# Smaller perturbations for stable convergence
X0_PERT = [0.2, 0.5, 0.15, 0.3]
TRAIN_NOISE_SIGMA = [0.03, 0.10, 0.03, 0.10]


def make_synthetic_demo_from(x0_q1, num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = x0_q1 + (math.pi - x0_q1) * t
    return demo


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    x0_zero = torch.zeros(4, device=device, dtype=torch.float64)

    print("=" * 76)
    print("  EXPANSION TRAINING v2: x0=zero ALTERNATION")
    print(f"  X0 perturbation: {X0_PERT}")
    print(f"  TRAIN_NOISE_SIGMA = {TRAIN_NOISE_SIGMA}")
    print(f"  {NUM_OUTER} outer iterations (each: 1 ep x0=zero + 1 ep random x0)")
    print(f"  LR = {LR}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0_zero, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # Pre eval at canonical x0
    x_pre, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=NUM_STEPS)
    pre_dist = float(np.linalg.norm(x_pre.cpu().numpy()[-1] - np.array(X_GOAL)))
    print(f"\n  Pre-fine-tune clean goal_dist: {pre_dist:.4f}")

    rng = np.random.default_rng(42)
    pert = np.array(X0_PERT)

    best_state_dict = copy.deepcopy(lin_net.state_dict())
    best_combined = float('inf')

    demo_zero = make_synthetic_demo_from(0.0, NUM_STEPS, device)

    t0 = time.time()
    for it in range(NUM_OUTER):
        # 1 epoch on x0=zero
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_zero, x_goal=x_goal, demo=demo_zero, num_steps=NUM_STEPS,
            num_epochs=1, lr=LR,
            debug_monitor=None, recorder=None, grad_debug=False,
            track_mode="energy", w_terminal_anchor=0.0,
            w_q_profile=100.0,
            q_profile_pump=[0.01, 0.01, 1.0, 1.0],
            q_profile_stable=[1.0, 1.0, 1.0, 1.0],
            q_profile_state_phase=True,
            w_end_q_high=80.0, end_phase_steps=20,
            train_noise_sigma=TRAIN_NOISE_SIGMA,
        )

        # 1 epoch on random perturbed x0
        x0_np = rng.uniform(-pert, pert, size=4).astype(np.float64)
        x0_t = torch.tensor(x0_np, device=device, dtype=torch.float64)
        demo_pert = make_synthetic_demo_from(x0_np[0], NUM_STEPS, device)
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_t, x_goal=x_goal, demo=demo_pert, num_steps=NUM_STEPS,
            num_epochs=1, lr=LR,
            debug_monitor=None, recorder=None, grad_debug=False,
            track_mode="energy", w_terminal_anchor=0.0,
            w_q_profile=100.0,
            q_profile_pump=[0.01, 0.01, 1.0, 1.0],
            q_profile_stable=[1.0, 1.0, 1.0, 1.0],
            q_profile_state_phase=True,
            w_end_q_high=80.0, end_phase_steps=20,
            train_noise_sigma=TRAIN_NOISE_SIGMA,
        )

        # Eval at both
        x_e0, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=NUM_STEPS)
        d0 = float(np.linalg.norm(x_e0.cpu().numpy()[-1] - np.array(X_GOAL)))
        x_ep, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0_t, x_goal=x_goal, num_steps=NUM_STEPS)
        dp = float(np.linalg.norm(x_ep.cpu().numpy()[-1] - np.array(X_GOAL)))

        # Best = sum of d0 + dp (joint quality)
        combined = d0 + dp
        if d0 < 1.0 and combined < best_combined:
            best_combined = combined
            best_state_dict = copy.deepcopy(lin_net.state_dict())

        if (it+1) % 2 == 0 or it == 0:
            print(f"  [{it+1:>3}/{NUM_OUTER}] x0_pert=({x0_np[0]:+.2f},{x0_np[1]:+.2f},"
                  f"{x0_np[2]:+.2f},{x0_np[3]:+.2f}) "
                  f"d0={d0:.4f}  dp={dp:.4f}  best_combined={best_combined:.4f}",
                  flush=True)

    elapsed = time.time() - t0
    lin_net.load_state_dict(best_state_dict)

    # Final test on multiple x0
    print(f"\n  Final test:")
    test_x0s = [
        ("zero",         [0.0, 0.0, 0.0, 0.0]),
        ("q1=+0.2",      [0.2, 0.0, 0.0, 0.0]),
        ("q1=-0.2",      [-0.2, 0.0, 0.0, 0.0]),
        ("q1d=+0.5",     [0.0, 0.5, 0.0, 0.0]),
        ("q1d=-0.5",     [0.0, -0.5, 0.0, 0.0]),
        ("combined+",    [0.15, 0.4, 0.1, 0.2]),
        ("combined-",    [-0.15, -0.4, -0.1, -0.2]),
    ]
    success = 0
    for name, x0_test in test_x0s:
        x0_t = torch.tensor(x0_test, device=device, dtype=torch.float64)
        x_test, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0_t, x_goal=x_goal, num_steps=NUM_STEPS)
        d = float(np.linalg.norm(x_test.cpu().numpy()[-1] - np.array(X_GOAL)))
        ok = "✓" if d < 1.0 else "✗"
        if d < 1.0:
            success += 1
        print(f"    {name:>12s}  x0={x0_test}: goal_dist={d:.4f} {ok}")

    print(f"\n  {success}/{len(test_x0s)} initial conditions succeed")
    print(f"  Total time: {elapsed:.0f}s")

    session_name = f"stageD_expand2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[], training_params={
            "experiment": "expansion_alternating",
            "x0_pert": X0_PERT,
            "train_noise_sigma": TRAIN_NOISE_SIGMA,
            "num_outer": NUM_OUTER,
            "success_count": success,
        },
        session_name=session_name,
    )
    print(f"  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
