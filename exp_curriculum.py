"""
exp_curriculum.py — Curriculum learning to bridge q1_cost gap.

Strategy:
  1. Load the pre-trained model (q1_cost=0, proven swing-up)
  2. Fine-tune with q1_cost=3.0  (first step beyond threshold)
  3. Fine-tune with q1_cost=6.0
  4. Fine-tune with q1_cost=12.0  (full default)

At each stage, we run up to FINETUNE_EPOCHS.  If the model swings up
(goal_dist < SWING_UP_THRESH), continue to next stage.  If not, stop.

The hypothesis: the NN already knows the f_extra pumping pattern from
stage 0; fine-tuning lets the Q-gate learn to suppress the incremental
q1 cost added at each stage.  The gradient landscape is much better at
fine-tune-time because f_extra already generates non-saturated states.
"""

import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

DEMO_CSV  = "run_20260428_001459_rollout_final.csv"
X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM   = 4
CONTROL_DIM = 2
SAVE_DIR    = "saved_models"

# Pre-trained model with q1_cost=0 (baseline, 6/6 seeds successful)
PRETRAINED_PATH = "saved_models/stageD_imit_20260427_231837/stageD_imit_20260427_231837.pth"

# Curriculum stages: (q1_cost, q1d_cost, finetune_epochs, lr)
CURRICULUM = [
    (3.0,  1.5,  30, 5e-4),   # first step beyond the 2→3 threshold
    (6.0,  3.0,  30, 2e-4),   # mid-range
    (12.0, 5.0,  30, 1e-4),   # full default q_base_diag
]

TRACK_MODE       = "energy"
GATE_RANGE_Q     = 0.95
GATE_RANGE_R     = 0.20
F_EXTRA_BOUND    = 3.0
W_TERMINAL       = 0.0
SWING_UP_THRESH  = 1.0


class QuietMonitor:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
    def log_epoch(self, epoch, num_epochs, loss, info):
        if epoch == 0 or (epoch+1) % 5 == 0 or epoch == num_epochs-1:
            gd = info.get('pure_end_error', float('nan'))
            qdev = info.get('mean_Q_gate_dev', float('nan'))
            fnorm = info.get('mean_f_extra_norm', float('nan'))
            print(f"  [{epoch+1:>3}/{num_epochs}] loss={loss:.4f} goal_dist={gd:.4f} "
                  f"QDev={qdev:.4f} fNorm={fnorm:.3f}", flush=True)


def evaluate(lin_net, mpc, x0, x_goal, device):
    x_final, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    return np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL))


def main():
    if not os.path.exists(PRETRAINED_PATH):
        raise FileNotFoundError(f"Pre-trained model not found: {PRETRAINED_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)

    print("=" * 76)
    print("  CURRICULUM LEARNING: q1_cost=0 → 3 → 6 → 12")
    print(f"  Pre-trained: {PRETRAINED_PATH}")
    print("=" * 76)

    # Load pre-trained model
    lin_net = network_module.LinearizationNetwork.load(PRETRAINED_PATH, device=str(device))
    lin_net = lin_net.double()

    # Evaluate at q1_cost=0 baseline
    mpc_base = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc_base.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc_base.q_base_diag = torch.tensor([0.0, 0.0, 50.0, 40.0], device=device, dtype=torch.float64)
    dist_pretrained = evaluate(lin_net, mpc_base, x0, x_goal, device)
    print(f"\n  Pretrained baseline (q1_cost=0): goal_dist={dist_pretrained:.4f}")

    curriculum_results = [(0.0, dist_pretrained, "SUCCESS" if dist_pretrained < SWING_UP_THRESH else "?")]

    for stage_idx, (q1c, q1dc, n_epochs, lr) in enumerate(CURRICULUM):
        print(f"\n{'─' * 76}")
        print(f"  STAGE {stage_idx+1}: q1_cost={q1c}  q1d_cost={q1dc}  epochs={n_epochs}  lr={lr}")

        mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
        mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
        mpc.q_base_diag = torch.tensor(
            [q1c, q1dc, 50.0, 40.0], device=device, dtype=torch.float64,
        )

        # Evaluate BEFORE fine-tuning (using previous stage's model)
        dist_before = evaluate(lin_net, mpc, x0, x_goal, device)
        print(f"  Before fine-tune: goal_dist={dist_before:.4f}")

        recorder = network_module.NetworkOutputRecorder()
        monitor  = QuietMonitor(num_epochs=n_epochs)

        t0 = time.time()
        loss_history, recorder = train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=n_epochs, lr=lr,
            debug_monitor=monitor, recorder=recorder,
            grad_debug=False, track_mode=TRACK_MODE, w_terminal_anchor=W_TERMINAL,
        )
        elapsed = time.time() - t0

        dist_after = evaluate(lin_net, mpc, x0, x_goal, device)
        result = "SUCCESS" if dist_after < SWING_UP_THRESH else "FAIL"
        print(f"  After  fine-tune: goal_dist={dist_after:.4f}  epochs={len(loss_history)}  "
              f"time={elapsed:.0f}s  {result}")

        curriculum_results.append((q1c, dist_after, result))

        # Save this stage's model
        session_name = f"stageD_curr_q1{q1c:.0f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
            model=lin_net, loss_history=loss_history,
            training_params={
                "curriculum_stage": stage_idx + 1,
                "q1_cost": q1c, "q1d_cost": q1dc,
                "fine_tune_lr": lr, "track_mode": TRACK_MODE,
            },
            session_name=session_name, recorder=recorder,
        )

        if result == "FAIL":
            print(f"  Stopped curriculum at q1_cost={q1c} (goal_dist={dist_after:.4f} > {SWING_UP_THRESH})")
            break

    print(f"\n{'=' * 76}")
    print("  CURRICULUM SUMMARY")
    print(f"  {'q1_cost':>10}  {'goal_dist':>10}  {'result':>8}")
    print(f"  {'-' * 35}")
    for q1c, d, r in curriculum_results:
        print(f"  {q1c:>10.1f}  {d:>10.4f}  {r:>8}")

    final = curriculum_results[-1]
    max_q1c = final[0]
    if final[2] == "SUCCESS":
        print(f"\n  CURRICULUM REACHED q1_cost={max_q1c} — NN handles full default Q costs!")
    else:
        prev = curriculum_results[-2] if len(curriculum_results) > 1 else None
        if prev:
            print(f"\n  Curriculum broke at q1_cost={max_q1c}")
            print(f"  Last success: q1_cost={prev[0]}")


if __name__ == "__main__":
    main()
