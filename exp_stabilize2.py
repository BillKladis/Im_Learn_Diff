"""exp_stabilize2.py — Second-pass fine-tune for tighter stability.

Picks up the latest stageD_stabilize_* checkpoint (best=~0.43 from
exp_stabilize.py) and pushes the goal_dist further down with:
  - smaller LR (5e-5) for finer convergence
  - heavier w_stable_phase (50, was 20) for stronger position pull
  - heavier w_f_end_reg (160, was 80) for stronger no-pumping at goal
  - longer NUM_STEPS=240 with 70-step stabilisation window

Goal: wrapped_dist < 0.2 at 220 steps, < 0.5 at 600 steps.
"""

import math
import os
import sys
import time
import glob
import copy
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 240
DT        = 0.05
EPOCHS    = 60
LR        = 5e-5     # smaller for finer convergence
HORIZON   = 10
SAVE_DIR  = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

STABLE_STEPS      = 70
F_END_REG_STEPS   = 70
END_Q_STEPS       = 70
W_STABLE_PHASE    = 50.0
W_F_END_REG       = 160.0
W_END_Q_HIGH      = 200.0


def make_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


def wrapped_goal_dist(x_state, x_goal):
    q1_err = math.atan2(math.sin(x_state[0] - x_goal[0]),
                        math.cos(x_state[0] - x_goal[0]))
    return math.sqrt(q1_err**2 + x_state[1]**2 + x_state[2]**2 + x_state[3]**2)


class PrintMonitor:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self._header_shown = False
        self._best = float('inf')

    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'GoalDist':>9}  "
              f"{'fNorm':>7}  {'Time':>6}  {'Best':>8}")
        print("─" * 70)
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
                  f"  {d:>9.4f}"
                  f"  {info.get('mean_f_extra_norm', float('nan')):>7.3f}"
                  f"  {info.get('epoch_time', float('nan')):>5.2f}s"
                  f"  {self._best:>8.4f}",
                  flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_demo(NUM_STEPS, device)

    # Find latest stabilize checkpoint
    candidates = sorted(glob.glob("saved_models/stageD_stabilize_*"), reverse=True)
    if not candidates:
        print("ERROR: no stageD_stabilize_* found.  Run exp_stabilize.py first.")
        return
    pretrained = glob.glob(f"{candidates[0]}/*.pth")[0]
    print(f"Loading: {pretrained}")

    print("=" * 76)
    print("  EXP STABILIZE2: tighter stabilisation pass")
    print(f"  NUM_STEPS={NUM_STEPS}  LR={LR}  EPOCHS={EPOCHS}")
    print(f"  w_stable_phase={W_STABLE_PHASE}  w_f_end_reg={W_F_END_REG}"
          f"  w_end_q_high={W_END_Q_HIGH}  window={STABLE_STEPS}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(pretrained, device=str(device)).double()

    # Pre-eval
    print(f"\n  Pre-eval:")
    for n in [170, 220, 300, 600]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n,
        )
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = wrapped_goal_dist(last, X_GOAL)
        print(f"    {n:>3} steps: raw={raw:.4f}  wrapped={wrp:.4f}")

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
        w_end_q_high=W_END_Q_HIGH,
        end_phase_steps=END_Q_STEPS,
        w_f_end_reg=W_F_END_REG,
        f_end_reg_steps=F_END_REG_STEPS,
        w_stable_phase=W_STABLE_PHASE,
        stable_phase_steps=STABLE_STEPS,
    )
    elapsed = time.time() - t0

    # Save FIRST
    session_name = f"stageD_stabilize2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "stabilize2",
            "pretrained": pretrained,
            "num_steps": NUM_STEPS,
            "w_stable_phase": W_STABLE_PHASE,
            "w_f_end_reg": W_F_END_REG,
            "w_end_q_high": W_END_Q_HIGH,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"\n  Saved → saved_models/{session_name}/")

    # Post-eval
    print(f"\n  Post-eval:")
    for n in [170, 220, 300, 600]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n,
        )
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = wrapped_goal_dist(last, X_GOAL)
        status = "STABLE" if wrp < 0.3 else ("CLOSE" if wrp < 1.0 else "FAIL")
        print(f"    {n:>3} steps: raw={raw:.4f}  wrapped={wrp:.4f}  {status}")

    print(f"\n  Training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
