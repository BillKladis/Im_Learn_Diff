"""exp_finetune_best.py — Fine-tune the best 0.0612 model to push performance further.

Goal: reduce clean goal_dist from 0.0612 toward 0.03 by:
  1. Running more iterations from the best checkpoint (LR=1e-4)
  2. Stronger end-phase stabilisation (w_end_q_high = 160)
  3. Slightly longer trajectory (NUM_STEPS = 200) to give more time at goal
  4. Keep the same synthetic energy-ramp target and state-phase Q-profile

This is a pure performance fine-tune — no expansion, no noise, just tighter.
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
NUM_STEPS = 200      # longer rollout — more time at goal
DT        = 0.05
EPOCHS    = 60
LR        = 1e-4    # fine-tune LR
HORIZON   = 10
SAVE_DIR  = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

# Stronger end-phase stabilisation + f_extra suppression
W_END_Q_HIGH     = 160.0
END_PHASE_STEPS  = 30    # last 30 steps (1.5 s) get strong stabilisation push
W_F_END_REG      = 50.0  # suppress f_extra in last N steps (stop pumping at goal)
F_END_REG_STEPS  = 40    # last 2s: no pumping allowed


def make_synthetic_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


class PrintMonitor:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self._header_shown = False
        self._best = float('inf')
        self._best_dict = None

    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>9}  {'GoalDist':>9}  "
              f"{'fNorm':>7}  {'Time':>6}  {'Best':>8}")
        print("─" * 80)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        d = info.get('pure_end_error', float('nan'))
        if d < self._best:
            self._best = d
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"{epoch+1:>4}/{num_epochs:<4}"
                  f"  {loss:>10.3f}"
                  f"  {info.get('loss_track', float('nan')):>9.3f}"
                  f"  {d:>9.4f}"
                  f"  {info.get('mean_f_extra_norm', float('nan')):>7.3f}"
                  f"  {info.get('epoch_time', float('nan')):>5.2f}s"
                  f"  {self._best:>8.4f}",
                  flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_synthetic_demo(NUM_STEPS, device)

    print("=" * 76)
    print("  FINE-TUNE BEST: push clean swing-up below 0.06 + fix overlearned pumping")
    print(f"  Checkpoint: {os.path.basename(PRETRAINED)}")
    print(f"  NUM_STEPS={NUM_STEPS}  LR={LR}  w_end_q_high={W_END_Q_HIGH}")
    print(f"  end_phase_steps={END_PHASE_STEPS}  w_f_end_reg={W_F_END_REG}"
          f"  f_end_reg_steps={F_END_REG_STEPS}")
    print(f"  EPOCHS={EPOCHS}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    x_pre, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    pre_dist = float(np.linalg.norm(x_pre.cpu().numpy()[-1] - np.array(X_GOAL)))
    print(f"\n  Pre-fine-tune goal_dist (NUM_STEPS={NUM_STEPS}): {pre_dist:.4f}")

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(num_epochs=EPOCHS)

    best_state_dict = copy.deepcopy(lin_net.state_dict())
    best_dist = pre_dist

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
        w_end_q_high=W_END_Q_HIGH,
        end_phase_steps=END_PHASE_STEPS,
        w_f_end_reg=W_F_END_REG,
        f_end_reg_steps=F_END_REG_STEPS,
    )
    elapsed = time.time() - t0

    x_final, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    post_dist = float(np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL)))

    # Also test at 170 steps for comparison
    x_170, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=170,
    )
    d170 = float(np.linalg.norm(x_170.cpu().numpy()[-1] - np.array(X_GOAL)))

    print(f"\n  Post-fine-tune goal_dist @ {NUM_STEPS} steps: {post_dist:.4f}")
    print(f"  Post-fine-tune goal_dist @ 170 steps:  {d170:.4f}")
    print(f"  Pre-train:   {pre_dist:.4f}")
    print(f"  Training time: {elapsed:.0f}s")

    result = "BETTER" if post_dist < pre_dist else "SAME/WORSE"
    print(f"  → {result}")

    session_name = f"stageD_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "finetune_best",
            "pretrained": PRETRAINED,
            "num_steps": NUM_STEPS,
            "w_end_q_high": W_END_Q_HIGH,
            "end_phase_steps": END_PHASE_STEPS,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
