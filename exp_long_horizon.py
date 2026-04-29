"""exp_long_horizon.py — Train for SUSTAINED HOLD with longer horizon.

The state-conditional w_f_stable training (exp_stab_state.py) achieved
raw=0.0755 at the 220-step training horizon — proof of principle —
but the model degrades AFTER step 220 (raw=6.59 at step 400).

Diagnosis: the network was never trained on what happens during a
sustained hold.  At step ~250+ the dynamics drift the pendulum partly
out of stable_zone (cos(q1-π) drops), the gate weakens, f_extra fires
again, and the pendulum is kicked out of the goal.

Fix: extend the training horizon so the network sees the entire hold
window and is rewarded for keeping the pendulum inside stable_zone.
NUM_STEPS = 400 (170 swing-up + 230 stabilise) gives 11.5 s of hold
to learn from.

Critical: we fine-tune from the stabstate checkpoint, not from scratch
or from 0.0612.  The state-conditional gate has already been learned;
we just need to extend its temporal coverage.
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

PRETRAINED = "saved_models/stageD_stabstate_20260428_224856/stageD_stabstate_20260428_224856.pth"

X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 400
DT        = 0.05
EPOCHS    = 40
LR        = 3e-5
HORIZON   = 10
SAVE_DIR  = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
W_F_STABLE = 50.0


def make_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    pump_steps = 170
    for i in range(num_steps):
        if i < pump_steps:
            alpha = i / max(pump_steps - 1, 1)
            t = 0.5 * (1.0 - math.cos(math.pi * alpha))
            demo[i, 0] = math.pi * t
        else:
            demo[i, 0] = math.pi
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
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>8}  {'GoalDist':>9}  "
              f"{'fNorm':>7}  {'Time':>6}  {'Best':>8}")
        print("─" * 70)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        d = info.get('pure_end_error', float('nan'))
        if d < self._best:
            self._best = d
        if epoch == 0 or (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            print(f"{epoch+1:>4}/{num_epochs:<4}"
                  f"  {loss:>10.3f}"
                  f"  {info.get('loss_track', float('nan')):>8.4f}"
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

    print("=" * 76)
    print("  EXP LONG-HORIZON: extended training window for sustained hold")
    print(f"  Pretrained: {os.path.basename(PRETRAINED)}")
    print(f"  NUM_STEPS={NUM_STEPS}  ({NUM_STEPS*DT}s)  LR={LR}  EPOCHS={EPOCHS}")
    print(f"  w_f_stable = {W_F_STABLE}  (state-conditional)")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # Pre-eval
    print(f"\n  Pre-eval:")
    for n in [170, 220, 300, 400, 600, 1000]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n,
        )
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = wrapped_goal_dist(last, X_GOAL)
        print(f"    {n:>4} steps: raw={raw:.4f}  wrapped={wrp:.4f}")

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
        w_f_stable=W_F_STABLE,
    )
    elapsed = time.time() - t0

    session_name = f"stageD_long_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "long_horizon",
            "pretrained": PRETRAINED,
            "num_steps": NUM_STEPS,
            "w_f_stable": W_F_STABLE,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"\n  Saved → saved_models/{session_name}/")

    # Loss monotonicity
    print(f"\n  Loss monotonicity:")
    print(f"    epoch 1: {loss_history[0]:.3f}")
    print(f"    epoch {len(loss_history)//2}: {loss_history[len(loss_history)//2]:.3f}")
    print(f"    epoch {len(loss_history)}: {loss_history[-1]:.3f}")
    decreased = sum(1 for i in range(1, len(loss_history))
                    if loss_history[i] < loss_history[i-1])
    print(f"    decreasing transitions: {decreased}/{len(loss_history)-1}")

    # Post-eval — wide horizon range
    print(f"\n  Post-eval (extensive):")
    for n in [170, 220, 300, 400, 600, 1000, 1500]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n,
        )
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = wrapped_goal_dist(last, X_GOAL)
        status = "STABLE" if wrp < 0.3 else ("CLOSE" if wrp < 1.0 else "FAIL")
        print(f"    {n:>4} steps ({n*DT:>5.1f}s): raw={raw:.4f}  wrapped={wrp:.4f}  {status}")

    print(f"\n  Training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
