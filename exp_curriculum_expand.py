"""exp_curriculum_expand.py — Curriculum perturbation: grow x0 spread gradually.

Observation: simple random-x0 alternation (exp_expansion2.py) gives 3/7
success but struggles with medium perturbations.  The issue may be that
jumping straight to ±0.2 perturbations is too hard a first step.

Strategy:
  - Start with ε_0 = 0.02 (tiny, almost canonical)
  - Grow by factor γ each iteration: ε_k = min(ε_0 * γ^k, ε_max)
  - Alternation: 1 epoch x0=zero + 1 epoch perturbed x0
  - After growth phase, add mirror augmentation once ε > 0.1

This is a "don't frighten the optimizer" approach: every new gradient step
is slightly harder than the last, giving the optimizer time to adapt.
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
NUM_OUTER = 50   # many short steps
LR        = 3e-5
HORIZON   = 10
SAVE_DIR  = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

EPS_START  = 0.02
EPS_GROWTH = 1.10   # 10% growth per iteration
EPS_MAX    = [0.35, 0.9, 0.25, 0.5]   # max per component [q1, q1d, q2, q2d]
MIRROR_AFTER_EPS = 0.10   # add mirror augmentation once ε_q1 exceeds this

TRAIN_NOISE_SIGMA = [0.03, 0.10, 0.03, 0.10]


def make_demo(x0_q1, target_q1, num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = x0_q1 + (target_q1 - x0_q1) * t
    return demo


def train_one(lin_net, mpc, x0, x_goal, demo, device, noise_sigma):
    train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=1, lr=LR,
        debug_monitor=None, recorder=None, grad_debug=False,
        track_mode="energy", w_terminal_anchor=0.0,
        w_q_profile=100.0,
        q_profile_pump=[0.01, 0.01, 1.0, 1.0],
        q_profile_stable=[1.0, 1.0, 1.0, 1.0],
        q_profile_state_phase=True,
        w_end_q_high=80.0, end_phase_steps=20,
        train_noise_sigma=noise_sigma,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    x0_zero = torch.zeros(4, device=device, dtype=torch.float64)

    print("=" * 76)
    print("  CURRICULUM EXPANSION: grow perturbation magnitude gradually")
    print(f"  ε_start={EPS_START}  growth×{EPS_GROWTH}  ε_max={EPS_MAX}")
    print(f"  Mirror augmentation kicks in at ε_q1 > {MIRROR_AFTER_EPS}")
    print(f"  NUM_OUTER = {NUM_OUTER}  LR = {LR}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0_zero, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # Pre eval
    x_pre, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    pre_dist = float(np.linalg.norm(x_pre.cpu().numpy()[-1] - np.array(X_GOAL)))
    print(f"\n  Pre clean goal_dist: {pre_dist:.4f}")

    rng = np.random.default_rng(seed=13)
    eps_max = np.array(EPS_MAX)

    demo_zero = make_demo(0.0, math.pi, NUM_STEPS, device)

    best_state_dict = copy.deepcopy(lin_net.state_dict())
    best_score = float('inf')

    eps_q1 = EPS_START

    t0 = time.time()
    for it in range(NUM_OUTER):
        # Current perturbation budget
        eps = min(eps_q1, EPS_MAX[0])
        pert_now = np.array([
            eps,
            min(eps_q1 * (EPS_MAX[1] / EPS_MAX[0]), EPS_MAX[1]),
            min(eps_q1 * (EPS_MAX[2] / EPS_MAX[0]), EPS_MAX[2]),
            min(eps_q1 * (EPS_MAX[3] / EPS_MAX[0]), EPS_MAX[3]),
        ])

        # Anchor: x0=zero
        train_one(lin_net, mpc, x0_zero, x_goal, demo_zero, device, TRAIN_NOISE_SIGMA)

        # Perturbed x0
        x0_np = rng.uniform(-pert_now, pert_now, size=4).astype(np.float64)
        x0_t  = torch.tensor(x0_np, device=device, dtype=torch.float64)
        demo_fwd = make_demo(x0_np[0], math.pi, NUM_STEPS, device)
        train_one(lin_net, mpc, x0_t, x_goal, demo_fwd, device, TRAIN_NOISE_SIGMA)

        # Mirror augmentation once ε_q1 is large enough
        if eps_q1 > MIRROR_AFTER_EPS:
            x0_m  = torch.tensor(-x0_np, device=device, dtype=torch.float64)
            demo_m = make_demo(-x0_np[0], -math.pi, NUM_STEPS, device)
            train_one(lin_net, mpc, x0_m, x_goal, demo_m, device, TRAIN_NOISE_SIGMA)

        # Grow epsilon
        eps_q1 = min(eps_q1 * EPS_GROWTH, EPS_MAX[0])

        # Eval
        test_pts = [
            [0.0,  0.0, 0.0, 0.0],
            [0.2,  0.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0, 0.0],
            [0.0,  0.5, 0.0, 0.0],
            [0.0, -0.5, 0.0, 0.0],
        ]
        dists = []
        for pt in test_pts:
            x0_t2 = torch.tensor(pt, device=device, dtype=torch.float64)
            xt, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0_t2,
                                         x_goal=x_goal, num_steps=NUM_STEPS)
            dists.append(float(np.linalg.norm(xt.cpu().numpy()[-1] - np.array(X_GOAL))))
        d0 = dists[0]
        n_ok = sum(1 for d in dists if d < 1.0)
        score = sum(dists)
        mirror_used = "M" if eps_q1 > MIRROR_AFTER_EPS else " "

        if d0 < 0.5 and score < best_score:
            best_score = score
            best_state_dict = copy.deepcopy(lin_net.state_dict())

        if (it + 1) % 5 == 0 or it == 0:
            print(f"  [{it+1:>3}/{NUM_OUTER}]{mirror_used} ε_q1={eps:.3f} "
                  f"d0={d0:.4f} d+={dists[1]:.4f} d-={dists[2]:.4f} "
                  f"dv+={dists[3]:.4f} dv-={dists[4]:.4f} "
                  f"ok={n_ok}/5  best={best_score:.3f}", flush=True)

    elapsed = time.time() - t0
    lin_net.load_state_dict(best_state_dict)

    # Final full test
    test_x0s = [
        ("zero",     [0.0,   0.0, 0.0,  0.0]),
        ("q1=+0.2",  [0.2,   0.0, 0.0,  0.0]),
        ("q1=-0.2",  [-0.2,  0.0, 0.0,  0.0]),
        ("q1d=+0.5", [0.0,   0.5, 0.0,  0.0]),
        ("q1d=-0.5", [0.0,  -0.5, 0.0,  0.0]),
        ("comb+",    [0.15,  0.4, 0.1,  0.2]),
        ("comb-",    [-0.15,-0.4,-0.1, -0.2]),
        ("q1=+0.3",  [0.3,   0.0, 0.0,  0.0]),
        ("q1=-0.3",  [-0.3,  0.0, 0.0,  0.0]),
    ]
    success = 0
    print(f"\n  Final test:")
    for name, x0_test in test_x0s:
        x0_t = torch.tensor(x0_test, device=device, dtype=torch.float64)
        x_test, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_t, x_goal=x_goal, num_steps=NUM_STEPS,
        )
        d = float(np.linalg.norm(x_test.cpu().numpy()[-1] - np.array(X_GOAL)))
        ok = "✓" if d < 1.0 else "✗"
        if d < 1.0:
            success += 1
        print(f"    {name:>12s}  x0={x0_test}  goal_dist={d:.4f} {ok}")

    print(f"\n  {success}/{len(test_x0s)} succeed  |  total time: {elapsed:.0f}s")

    session_name = f"stageD_currexp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[], training_params={
            "experiment": "curriculum_expansion",
            "eps_start": EPS_START,
            "eps_growth": EPS_GROWTH,
            "eps_max": EPS_MAX,
            "num_outer": NUM_OUTER,
            "success_count": success,
        },
        session_name=session_name,
    )
    print(f"  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
