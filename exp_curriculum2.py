"""
exp_curriculum2.py — Extended curriculum: fine-grained steps to reach
q1_cost=12 (full default).

Starting from the q1_cost=6 checkpoint (proven swing-up at 0.2333),
try:
  A) q1_cost=9 intermediate step, then 12  (finer granularity)
  B) q1_cost=12 with more epochs + lower LR from q1_cost=6  (brute force)

Both start from the same q1_cost=6 checkpoint.
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

# Starting checkpoint (q1_cost=6, goal_dist=0.2333)
CHECKPOINT_Q6 = "saved_models/stageD_curr_q16_20260428_012340/stageD_curr_q16_20260428_012340.pth"

TRACK_MODE      = "energy"
SWING_UP_THRESH = 1.0


def load_checkpoint(path, device):
    lin_net = network_module.LinearizationNetwork.load(path, device=str(device))
    return lin_net.double()


def make_mpc(x0, x_goal, q1c, q1dc, device):
    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor([q1c, q1dc, 50.0, 40.0], device=device, dtype=torch.float64)
    return mpc


def evaluate(lin_net, mpc, x0, x_goal):
    x_final, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    return np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL))


def finetune(lin_net, mpc, x0, x_goal, demo, n_epochs, lr, label, device):
    class QuietMonitor:
        def __init__(self):
            pass
        def log_epoch(self, epoch, num_epochs, loss, info):
            if epoch == 0 or (epoch+1) % 5 == 0 or epoch == num_epochs-1:
                gd = info.get('pure_end_error', float('nan'))
                qdev = info.get('mean_Q_gate_dev', float('nan'))
                fn = info.get('mean_f_extra_norm', float('nan'))
                print(f"    [{epoch+1:>3}/{num_epochs}] loss={loss:.4f} "
                      f"goal_dist={gd:.4f} QDev={qdev:.4f} fNorm={fn:.3f}",
                      flush=True)

    recorder = network_module.NetworkOutputRecorder()
    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=n_epochs, lr=lr,
        debug_monitor=QuietMonitor(), recorder=recorder,
        grad_debug=False, track_mode=TRACK_MODE, w_terminal_anchor=0.0,
    )
    elapsed = time.time() - t0
    dist = evaluate(lin_net, mpc, x0, x_goal)
    result = "SUCCESS" if dist < SWING_UP_THRESH else "FAIL"
    print(f"  {label:40s}  goal_dist={dist:.4f}  "
          f"epochs={len(loss_history):>3}  time={elapsed:.0f}s  {result}")

    session_name = f"stageD_curr2_{label.replace(' ', '_')[:20]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={"label": label, "lr": lr, "n_epochs": n_epochs},
        session_name=session_name, recorder=recorder,
    )
    return dist, result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)

    print("=" * 76)
    print("  CURRICULUM2: bridge q1_cost=6 → 12")
    print(f"  Starting from: {CHECKPOINT_Q6}")
    print("=" * 76)

    results = []

    # ── Arm A: 9 → 12 intermediate step ────────────────────────────────
    print("\n--- ARM A: 6 → 9 → 12 ---")
    lin_A = load_checkpoint(CHECKPOINT_Q6, device)
    mpc9  = make_mpc(x0, x_goal, 9.0, 4.5, device)

    d_before = evaluate(lin_A, mpc9, x0, x_goal)
    print(f"  Before q1=9 fine-tune: goal_dist={d_before:.4f}")

    d9, r9 = finetune(lin_A, mpc9, x0, x_goal, demo,
                      n_epochs=40, lr=1e-4, label="q1=9 from_q6", device=device)
    results.append(("A q1=9 from q6", d9, r9))

    if r9 == "SUCCESS":
        mpc12 = make_mpc(x0, x_goal, 12.0, 5.0, device)
        d_before12 = evaluate(lin_A, mpc12, x0, x_goal)
        print(f"\n  Before q1=12 fine-tune: goal_dist={d_before12:.4f}")
        d12, r12 = finetune(lin_A, mpc12, x0, x_goal, demo,
                            n_epochs=40, lr=5e-5, label="q1=12 from_q9", device=device)
        results.append(("A q1=12 from q9", d12, r12))
    else:
        print("  ARM A stopped at q1=9 (FAIL)")
        results.append(("A q1=12 (not attempted)", float('nan'), "SKIP"))

    # ── Arm B: q1=12 directly from q6, more epochs ──────────────────────
    print("\n--- ARM B: 6 → 12 (direct, more epochs) ---")
    lin_B = load_checkpoint(CHECKPOINT_Q6, device)
    mpc12_B = make_mpc(x0, x_goal, 12.0, 5.0, device)

    d_before_B = evaluate(lin_B, mpc12_B, x0, x_goal)
    print(f"  Before q1=12 fine-tune: goal_dist={d_before_B:.4f}")
    d12_B, r12_B = finetune(lin_B, mpc12_B, x0, x_goal, demo,
                             n_epochs=60, lr=5e-5, label="q1=12 from_q6_60ep", device=device)
    results.append(("B q1=12 from q6 60ep", d12_B, r12_B))

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print("  CURRICULUM2 SUMMARY")
    print(f"  Starting checkpoint: q1_cost=6  goal_dist=0.2333")
    print(f"  {'Label':42s}  {'goal_dist':>10}  {'result':>8}")
    for label, d, r in results:
        dist_str = f"{d:10.4f}" if not math.isnan(d) else f"{'---':>10}"
        print(f"  {label:42s}  {dist_str}  {r:>8}")


if __name__ == "__main__":
    main()
