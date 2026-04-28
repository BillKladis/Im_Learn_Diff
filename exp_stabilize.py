"""exp_stabilize.py — Teach the best model to HOLD the upright position.

Diagnosis (exp_diag_failure.py) showed:
  - The best model (0.0612) reaches q1≈180° at step 170 (goal reached)
  - But by step 200 the pendulum has FALLEN BACK DOWN (goal_dist=3.47)
  - At step 169, fNorm=6.8 (large f_extra still pumping at the goal)
  - Root cause: network overlearned energy pumping, applies large torques
    even when the pendulum is at the goal, causing it to overshoot and fall

Fix with three loss components active in the last 40 steps:
  1. w_stable_phase: direct position tracking (q1→π, velocities→0)
  2. w_f_end_reg: suppress f_extra to stop pumping at goal
  3. w_end_q_high: push Q-gates up so MPC cares about state error

Training: 250 steps (12.5s) so the network sees the post-170 phase.
The model needs to learn: after reaching the top, let the MPC stabilize.
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
NUM_STEPS = 220     # 170 swing-up + 50 stabilisation window
DT        = 0.05
EPOCHS    = 50
LR        = 2e-4    # higher LR for faster convergence
HORIZON   = 10
SAVE_DIR  = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

# All three stabilisation signals active for last 50 steps
STABLE_STEPS      = 50    # position tracking window
F_END_REG_STEPS   = 50    # same window: no pumping allowed
END_Q_STEPS       = 50    # same window: push Q-gates up

W_STABLE_PHASE    = 20.0  # direct position loss weight
W_F_END_REG       = 80.0  # f_extra suppression weight
W_END_Q_HIGH      = 160.0 # Q-gate increase weight


def make_demo(num_steps, device):
    """Energy ramp demo: cosine ease from q1=0→π (same as best model training)."""
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


def wrapped_goal_dist(x_traj_np, x_goal):
    """Properly wrapped goal distance (treats q1=+π and q1=-π as equivalent)."""
    last = x_traj_np[-1]
    q1_err = math.atan2(math.sin(last[0] - x_goal[0]), math.cos(last[0] - x_goal[0]))
    return math.sqrt(q1_err**2 + last[1]**2 + last[2]**2 + last[3]**2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_demo(NUM_STEPS, device)

    print("=" * 76)
    print("  EXP STABILIZE: reach upright AND HOLD IT")
    print(f"  NUM_STEPS={NUM_STEPS} (170 swing-up + 50 stabilisation)")
    print(f"  LR={LR}  EPOCHS={EPOCHS}")
    print(f"  w_stable_phase={W_STABLE_PHASE} (last {STABLE_STEPS} steps)")
    print(f"  w_f_end_reg={W_F_END_REG} (last {F_END_REG_STEPS} steps)")
    print(f"  w_end_q_high={W_END_Q_HIGH} (last {END_Q_STEPS} steps)")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # Pre-eval at multiple step counts
    print(f"\n  Pre-eval:")
    for n in [170, 200, 250]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n,
        )
        raw_d = float(np.linalg.norm(x_t.cpu().numpy()[-1] - np.array(X_GOAL)))
        wrp_d = wrapped_goal_dist(x_t.cpu().numpy(), X_GOAL)
        print(f"    {n:>3} steps: raw_dist={raw_d:.4f}  wrapped_dist={wrp_d:.4f}")

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(num_epochs=EPOCHS)
    best_state_dict = copy.deepcopy(lin_net.state_dict())
    best_metric = float('inf')

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

    # Post-eval at multiple step counts
    print(f"\n  Post-eval:")
    for n in [170, 200, 250, 400]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n,
        )
        raw_d = float(np.linalg.norm(x_t.cpu().numpy()[-1] - np.array(X_GOAL)))
        wrp_d = wrapped_goal_dist(x_t.cpu().numpy(), X_GOAL)
        stable = "STABLE" if wrp_d < 0.5 else "UNSTABLE"
        print(f"    {n:>3} steps: raw_dist={raw_d:.4f}  wrapped_dist={wrp_d:.4f}  {stable}")

    print(f"\n  Training time: {elapsed:.0f}s")

    session_name = f"stageD_stabilize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "stabilize",
            "pretrained": PRETRAINED,
            "num_steps": NUM_STEPS,
            "w_stable_phase": W_STABLE_PHASE,
            "w_f_end_reg": W_F_END_REG,
            "w_end_q_high": W_END_Q_HIGH,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
