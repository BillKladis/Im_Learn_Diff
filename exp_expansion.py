"""exp_expansion.py — Expansion training: fine-tune with varied initial states.

Phase 1 (already done): no-demo training from x0=[0,0,0,0] gives swing-up
Phase 2 (this):         fine-tune from RANDOMIZED initial states each epoch

Each epoch samples x0 from a perturbation distribution:
  q1  ∈ [-0.4, 0.4]   (small initial angle perturbation)
  q1d ∈ [-1.0, 1.0]   (small initial angular velocity)
  q2  ∈ [-0.3, 0.3]
  q2d ∈ [-0.5, 0.5]

ALSO injects noise on observations.  The combined training teaches the
model to swing up from any nearby starting state under noisy observations.

Custom training loop (extends the existing one with x0 randomization).
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
EPOCHS_PER_X0 = 5
NUM_X0_SAMPLES = 12      # 12 different random x0s × 5 epochs = 60 total
LR        = 1e-4         # slightly higher to make progress visible
HORIZON   = 10
SAVE_DIR  = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

# x0 perturbation ranges (uniform over [-range, +range])
X0_PERT = [0.4, 1.0, 0.3, 0.5]   # q1, q1d, q2, q2d

# Light noise during expansion training
TRAIN_NOISE_SIGMA = [0.03, 0.10, 0.03, 0.10]


def make_synthetic_demo_from(x0_q1, num_steps, device):
    """Build a synthetic energy-ramp demo whose initial q1 matches x0."""
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = x0_q1 + (math.pi - x0_q1) * t
    return demo


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    print("=" * 76)
    print("  EXPANSION TRAINING: varied x0 + noise injection")
    print(f"  Pretrained: {os.path.basename(PRETRAINED)}")
    print(f"  X0 perturbation ranges: {X0_PERT}")
    print(f"  TRAIN_NOISE_SIGMA = {TRAIN_NOISE_SIGMA}")
    print(f"  {NUM_X0_SAMPLES} x0 samples × {EPOCHS_PER_X0} epochs = {NUM_X0_SAMPLES*EPOCHS_PER_X0} total  LR = {LR}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(
        x0=torch.zeros(4, device=device, dtype=torch.float64),
        x_goal=x_goal, N=HORIZON, device=device,
    )
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # Verify clean swing-up before fine-tuning
    x0_zero = torch.zeros(4, device=device, dtype=torch.float64)
    x_pre, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=NUM_STEPS)
    pre_dist = float(np.linalg.norm(x_pre.cpu().numpy()[-1] - np.array(X_GOAL)))
    print(f"\n  Pre-fine-tune clean goal_dist (x0=zero): {pre_dist:.4f}")

    # Custom training loop with x0 randomization
    rng = np.random.default_rng(42)
    pert = np.array(X0_PERT)

    best_state_dict = copy.deepcopy(lin_net.state_dict())
    best_dist = float('inf')

    epoch_dists = []
    t0 = time.time()
    for it in range(NUM_X0_SAMPLES):
        ep_t0 = time.time()
        x0_np = rng.uniform(-pert, pert, size=4).astype(np.float64)
        x0_t = torch.tensor(x0_np, device=device, dtype=torch.float64)
        demo = make_synthetic_demo_from(x0_np[0], NUM_STEPS, device)

        # EPOCHS_PER_X0 epochs of training on this x0 (lets optimizer momentum build).
        loss_history, _ = train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_t, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=EPOCHS_PER_X0, lr=LR,
            debug_monitor=None, recorder=None,
            grad_debug=False, track_mode="energy", w_terminal_anchor=0.0,
            w_q_profile=100.0,
            q_profile_pump=[0.01, 0.01, 1.0, 1.0],
            q_profile_stable=[1.0, 1.0, 1.0, 1.0],
            q_profile_state_phase=True,
            w_end_q_high=80.0,
            end_phase_steps=20,
            train_noise_sigma=TRAIN_NOISE_SIGMA,
        )
        ep_t = time.time() - ep_t0

        # Eval: clean rollout at x0=zero (canonical) AND at the current x0_t
        x_eval0, _ = train_module.rollout(lin_net=lin_net, mpc=mpc,
                                           x0=x0_zero, x_goal=x_goal, num_steps=NUM_STEPS)
        eval_dist0 = float(np.linalg.norm(x_eval0.cpu().numpy()[-1] - np.array(X_GOAL)))
        x_eval_pert, _ = train_module.rollout(lin_net=lin_net, mpc=mpc,
                                               x0=x0_t, x_goal=x_goal, num_steps=NUM_STEPS)
        eval_dist_pert = float(np.linalg.norm(x_eval_pert.cpu().numpy()[-1] - np.array(X_GOAL)))
        epoch_dists.append((eval_dist0, eval_dist_pert))

        if eval_dist0 < best_dist:
            best_dist = eval_dist0
            best_state_dict = copy.deepcopy(lin_net.state_dict())

        print(f"  [{it+1:>3}/{NUM_X0_SAMPLES}] x0=({x0_np[0]:+.2f},{x0_np[1]:+.2f},"
              f"{x0_np[2]:+.2f},{x0_np[3]:+.2f}) "
              f"loss={loss_history[-1]:.3f} "
              f"d(x0=zero)={eval_dist0:.4f} d(x0_pert)={eval_dist_pert:.4f} "
              f"best={best_dist:.4f} t={ep_t:.1f}s",
              flush=True)

    elapsed = time.time() - t0
    lin_net.load_state_dict(best_state_dict)

    # Evaluate at multiple test x0s
    print(f"\n  Testing trained model at multiple x0:")
    test_x0s = [
        [0.0, 0.0, 0.0, 0.0],     # canonical
        [0.2, 0.0, 0.0, 0.0],     # small angle perturbation
        [-0.2, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],     # initial velocity
        [0.0, -0.5, 0.0, 0.0],
        [0.3, 0.5, 0.1, 0.0],     # combined
    ]
    for x0_test in test_x0s:
        x0_t = torch.tensor(x0_test, device=device, dtype=torch.float64)
        x_test, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0_t,
                                          x_goal=x_goal, num_steps=NUM_STEPS)
        d = float(np.linalg.norm(x_test.cpu().numpy()[-1] - np.array(X_GOAL)))
        result = "✓" if d < 1.0 else "✗"
        print(f"    x0={x0_test}: goal_dist = {d:.4f} {result}")

    print(f"\n  Best eval_dist seen during training: {best_dist:.4f}")
    print(f"  Total time: {elapsed:.0f}s")

    session_name = f"stageD_expand_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=epoch_dists,
        training_params={
            "experiment": "expansion_training",
            "pretrained": PRETRAINED,
            "x0_perturbation": X0_PERT,
            "train_noise_sigma": TRAIN_NOISE_SIGMA,
        },
        session_name=session_name,
    )
    print(f"  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
