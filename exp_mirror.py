"""exp_mirror.py — Mirror augmentation for symmetric swing-up.

Root cause of expansion failure:
  The network was only trained from x0=0 (hanging straight down), so it
  learned a fixed CCW swing-up direction.  Random-x0 fine-tuning can't fix
  this without either forgetting the clean case or specialising to one side.

Fix:
  For EVERY sampled x0=(q1, q1d, q2, q2d) we ALSO train on the mirror
  x0_m=(-q1, -q1d, -q2, -q2d) with the matching mirror demo (ramp from
  -q1 toward -π rather than +π).

  Since the MPC wraps angles, q1=-π has zero error to goal [π,0,0,0] — so
  the MPC/loss sees q1=-π as equally valid as q1=+π.

  This teaches the network BOTH swing-up directions simultaneously without
  any architectural change.
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
NUM_OUTER = 40
LR        = 3e-5   # small LR — preserve the clean swing-up
HORIZON   = 10
SAVE_DIR  = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

# Perturbation ranges (uniform over [-range, +range])
X0_PERT = [0.3, 0.8, 0.2, 0.4]
TRAIN_NOISE_SIGMA = [0.03, 0.10, 0.03, 0.10]


def make_demo(x0_q1, target_q1, num_steps, device):
    """Cosine-eased ramp from x0_q1 to target_q1.

    For normal case:   target_q1 = +π
    For mirror case:   target_q1 = -π  (MPC treats -π ≡ +π via angle wrap)
    """
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = x0_q1 + (target_q1 - x0_q1) * t
    return demo


def train_one_epoch(lin_net, mpc, x0, x_goal, demo, device, noise_sigma):
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


def eval_dist(lin_net, mpc, x0_list, x_goal, device):
    dists = []
    for x0_v in x0_list:
        x0_t = torch.tensor(x0_v, device=device, dtype=torch.float64)
        x_traj, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_t, x_goal=x_goal,
            num_steps=NUM_STEPS,
        )
        d = float(np.linalg.norm(x_traj.cpu().numpy()[-1] - np.array(X_GOAL)))
        dists.append(d)
    return dists


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    x0_zero = torch.zeros(4, device=device, dtype=torch.float64)

    print("=" * 76)
    print("  MIRROR AUGMENTATION: train on x0 + mirror(-x0) simultaneously")
    print(f"  X0_PERT = {X0_PERT}")
    print(f"  TRAIN_NOISE_SIGMA = {TRAIN_NOISE_SIGMA}")
    print(f"  NUM_OUTER = {NUM_OUTER}  LR = {LR}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0_zero, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # Pre eval
    pre_dists = eval_dist(lin_net, mpc, [
        [0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.0],
        [-0.2, 0.0, 0.0, 0.0],
    ], x_goal, device)
    print(f"\n  Pre-train: x0=zero:{pre_dists[0]:.4f}  q1=+0.2:{pre_dists[1]:.4f}"
          f"  q1=-0.2:{pre_dists[2]:.4f}")

    rng = np.random.default_rng(seed=7)
    pert = np.array(X0_PERT)

    demo_zero = make_demo(0.0, math.pi, NUM_STEPS, device)

    best_state_dict = copy.deepcopy(lin_net.state_dict())
    best_score = float('inf')

    t0 = time.time()
    for it in range(NUM_OUTER):
        # Anchor: x0=zero, swing CCW to +π
        train_one_epoch(lin_net, mpc, x0_zero, x_goal, demo_zero, device, TRAIN_NOISE_SIGMA)

        # Random x0
        x0_np = rng.uniform(-pert, pert, size=4).astype(np.float64)
        x0_t  = torch.tensor(x0_np, device=device, dtype=torch.float64)
        demo_fwd = make_demo(x0_np[0], math.pi, NUM_STEPS, device)
        train_one_epoch(lin_net, mpc, x0_t, x_goal, demo_fwd, device, TRAIN_NOISE_SIGMA)

        # Mirror: negate x0, demo goes -x0_q1 → -π  (MPC wraps -π ≡ +π)
        x0_mirror_np = -x0_np
        x0_mirror_t  = torch.tensor(x0_mirror_np, device=device, dtype=torch.float64)
        demo_mirror  = make_demo(x0_mirror_np[0], -math.pi, NUM_STEPS, device)
        train_one_epoch(lin_net, mpc, x0_mirror_t, x_goal, demo_mirror, device, TRAIN_NOISE_SIGMA)

        # Eval
        test_pts = [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, -0.5, 0.0, 0.0],
        ]
        dists = eval_dist(lin_net, mpc, test_pts, x_goal, device)
        d0, dp2, dm2, dv, dmv = dists
        n_ok = sum(1 for d in dists if d < 1.0)
        score = sum(dists)

        if d0 < 0.5 and score < best_score:
            best_score = score
            best_state_dict = copy.deepcopy(lin_net.state_dict())

        if (it + 1) % 4 == 0 or it == 0:
            print(f"  [{it+1:>3}/{NUM_OUTER}] "
                  f"x0_np=({x0_np[0]:+.2f},{x0_np[1]:+.2f}) "
                  f"d0={d0:.4f} d(+0.2)={dp2:.4f} d(-0.2)={dm2:.4f} "
                  f"d(v+)={dv:.4f} d(v-)={dmv:.4f} "
                  f"ok={n_ok}/5  best={best_score:.3f}", flush=True)

    elapsed = time.time() - t0
    lin_net.load_state_dict(best_state_dict)

    # Final full test
    test_x0s = [
        ("zero",      [0.0, 0.0, 0.0, 0.0]),
        ("q1=+0.2",   [0.2, 0.0, 0.0, 0.0]),
        ("q1=-0.2",   [-0.2, 0.0, 0.0, 0.0]),
        ("q1d=+0.5",  [0.0, 0.5, 0.0, 0.0]),
        ("q1d=-0.5",  [0.0, -0.5, 0.0, 0.0]),
        ("comb+",     [0.15, 0.4, 0.1, 0.2]),
        ("comb-",     [-0.15, -0.4, -0.1, -0.2]),
        ("q1=+0.3",   [0.3, 0.0, 0.0, 0.0]),
        ("q1=-0.3",   [-0.3, 0.0, 0.0, 0.0]),
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

    session_name = f"stageD_mirror_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[], training_params={
            "experiment": "mirror_augmentation",
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
