"""exp_robust_finetune.py — Fine-tune pretrained swing-up model for robustness.

Phase 1 (proven): no-demo training gives clean swing-up at goal_dist=0.061
Phase 2 (this):   fine-tune with noise injection + varied initial states

Strategy:
  - Start from pretrained checkpoint
  - Lower LR (5e-5) so we don't lose the swing-up
  - Inject medium noise σ_q=0.05, σ_qd=0.20 on observations
  - Optional: randomize x0 each epoch (small perturbations)
  - Short training (40 epochs) — fine-tuning, not from scratch
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

PRETRAINED = "saved_models/stageD_nodemo_20260428_123448/stageD_nodemo_20260428_123448.pth"

X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
EPOCHS    = 40
LR        = 5e-5     # lower LR — fine-tune, don't lose swing-up
HORIZON   = 10
SAVE_DIR  = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

# Medium noise — heavy enough to teach robustness, light enough to learn from
TRAIN_NOISE_SIGMA = [0.05, 0.20, 0.05, 0.20]


def make_synthetic_demo(num_steps, device, x0_q1=0.0):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        # Interpolate from initial q1 to goal q1=π
        demo[i, 0] = x0_q1 + (math.pi - x0_q1) * t
    return demo


class PrintMonitor:
    def __init__(self, num_epochs):
        self.num_epochs    = num_epochs
        self._header_shown = False
    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>9}  {'GoalDist':>9}  "
              f"{'fNorm':>7}  {'Time':>6}")
        print("─" * 80)
        self._header_shown = True
    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        if epoch == 0 or (epoch+1) % 2 == 0 or epoch == num_epochs-1:
            print(f"{epoch+1:>4}/{num_epochs:<4}"
                  f"  {loss:>10.3f}"
                  f"  {info.get('loss_track',        float('nan')):>9.3f}"
                  f"  {info.get('pure_end_error',    float('nan')):>9.4f}"
                  f"  {info.get('mean_f_extra_norm', float('nan')):>7.3f}"
                  f"  {info.get('epoch_time',        float('nan')):>5.2f}s",
                  flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo = make_synthetic_demo(NUM_STEPS, device, x0_q1=0.0)

    print("=" * 76)
    print("  Fine-tune pretrained model with NOISE INJECTION")
    print(f"  Pretrained: {os.path.basename(PRETRAINED)}")
    print(f"  TRAIN_NOISE_SIGMA = {TRAIN_NOISE_SIGMA}")
    print(f"  EPOCHS = {EPOCHS}  LR = {LR}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # Verify clean behaviour BEFORE fine-tuning
    x_pre, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS)
    pre_dist = float(np.linalg.norm(x_pre.cpu().numpy()[-1] - np.array(X_GOAL)))
    print(f"\n  Pre-fine-tune clean goal_dist: {pre_dist:.4f}")

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(num_epochs=EPOCHS)

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=False, track_mode="energy", w_terminal_anchor=0.0,
        w_q_profile=100.0,
        q_profile_pump=[0.01, 0.01, 1.0, 1.0],
        q_profile_stable=[1.0, 1.0, 1.0, 1.0],
        q_profile_state_phase=True,
        w_end_q_high=80.0,
        end_phase_steps=20,
        train_noise_sigma=TRAIN_NOISE_SIGMA,
    )
    elapsed = time.time() - t0

    x_final, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS)
    post_dist = float(np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL)))
    print(f"\n  Post-fine-tune clean goal_dist: {post_dist:.4f}")
    print(f"  Training time: {elapsed:.0f}s")

    session_name = f"stageD_robust_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "noise_finetune",
            "pretrained": PRETRAINED,
            "train_noise_sigma": TRAIN_NOISE_SIGMA,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
