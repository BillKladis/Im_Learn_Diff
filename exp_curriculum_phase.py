"""
exp_curriculum_phase.py — Pre-trained ZERO_Q1_COSTS model + fine-tune
with state-phase profile target at default q_base_diag.

Combines two proven techniques:
  1. The ZERO_Q1_COSTS curriculum gives a starting model that already
     knows the f_extra pumping pattern (proven 6/6 swing-up).
  2. The state-phase profile target (proven to enable state-dependent
     q1 gates: 0.01 at bottom, 0.30 near upright).

Hypothesis: pre-trained f_extra is already optimal; fine-tuning only
needs to teach the q_head to produce the state-dependent gate profile
that matches the default q_base_diag environment.
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

# Pre-trained ZERO_Q1_COSTS model (best baseline)
PRETRAINED_PATH = "saved_models/stageD_imit_20260427_231837/stageD_imit_20260427_231837.pth"

# Fine-tune: default q_base_diag + state-phase profile
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
W_Q_PROFILE = 100.0
Q_PROFILE_PUMP   = [0.01, 1.0, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0, 1.0, 1.0]

EPOCHS    = 60
LR        = 5e-4
TRACK_MODE = "energy"
W_TERMINAL_ANCHOR = 0.0


class PrintMonitor:
    def __init__(self, num_epochs):
        self.num_epochs    = num_epochs
        self._header_shown = False
    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>9}  {'GoalDist':>9}  "
              f"{'QDev':>7}  {'fNorm':>7}  {'fτ1[0]':>8}  {'Time':>6}")
        print("─" * 100)
        self._header_shown = True
    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        if epoch == 0 or (epoch+1) % 2 == 0 or epoch == num_epochs-1:
            print(f"{epoch+1:>4}/{num_epochs:<4}"
                  f"  {loss:>10.3f}"
                  f"  {info.get('loss_track',        float('nan')):>9.3f}"
                  f"  {info.get('pure_end_error',    float('nan')):>9.4f}"
                  f"  {info.get('mean_Q_gate_dev',   float('nan')):>7.4f}"
                  f"  {info.get('mean_f_extra_norm', float('nan')):>7.3f}"
                  f"  {info.get('mean_f_tau1_first', float('nan')):>8.3f}"
                  f"  {info.get('epoch_time',        float('nan')):>5.2f}s",
                  flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)

    print("=" * 76)
    print("  EXP: Curriculum + state-phase profile (final integration)")
    print(f"  Pre-trained: {os.path.basename(PRETRAINED_PATH)}")
    print(f"  Fine-tune q_base_diag = {Q_BASE_DIAG}  (DEFAULT)")
    print(f"  W_Q_PROFILE = {W_Q_PROFILE}  state_phase=True  TRACK = {TRACK_MODE}")
    print(f"  Epochs = {EPOCHS}  LR = {LR}")
    print("=" * 76)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED_PATH, device=str(device)).double()
    print(f"\n  Loaded pre-trained model")

    # Evaluate at q1_cost=0 baseline first
    mpc_base = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc_base.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc_base.q_base_diag = torch.tensor([0.0, 0.0, 50.0, 40.0], device=device, dtype=torch.float64)
    x_base, _ = train_module.rollout(lin_net=lin_net, mpc=mpc_base, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS)
    dist_base = float(np.linalg.norm(x_base.cpu().numpy()[-1] - np.array(X_GOAL)))
    print(f"  Baseline at q1_cost=0: goal_dist={dist_base:.4f}")

    # Fine-tune environment: default q_base_diag
    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    # Pre-fine-tune evaluation
    x_pre, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS)
    dist_pre = float(np.linalg.norm(x_pre.cpu().numpy()[-1] - np.array(X_GOAL)))
    print(f"  Pre-fine-tune (default q_base): goal_dist={dist_pre:.4f}")

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(num_epochs=EPOCHS)

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=False, track_mode=TRACK_MODE,
        w_terminal_anchor=W_TERMINAL_ANCHOR,
        w_q_profile=W_Q_PROFILE,
        q_profile_pump=Q_PROFILE_PUMP,
        q_profile_stable=Q_PROFILE_STABLE,
        q_profile_state_phase=True,
    )
    elapsed = time.time() - t0

    x_final, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS)
    dist_final = float(np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL)))
    result = "SUCCESS" if dist_final < 1.0 else "FAIL"
    print(f"\n  Post-fine-tune: goal_dist={dist_final:.4f}  epochs={len(loss_history)}  "
          f"time={elapsed:.0f}s  {result}")

    # Q-gate profile diagnostic
    print("\n  Q-gate profile from final epoch:")
    final_steps = recorder.epochs[-1]["steps"]
    print(f"  step={'q1':>8} {'q1d':>8} {'q2':>8} {'q2d':>8} {'fNorm':>8}")
    for s in [0, 30, 60, 90, 118, 140, 160, 169]:
        if s >= len(final_steps):
            continue
        gates = final_steps[s]["gates_Q"]
        gates_t = torch.tensor(gates)
        avg = gates_t.mean(dim=0).tolist()
        f_extra = torch.tensor(final_steps[s]["f_extra"])
        f_n = float(torch.sqrt((f_extra**2).sum()))
        print(f"  {s:>4}: {avg[0]:>8.4f} {avg[1]:>8.4f} {avg[2]:>8.4f} {avg[3]:>8.4f} {f_n:>8.3f}")

    session_name = f"stageD_currphase_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "curriculum_pretrain_plus_state_phase_profile",
            "pretrained": os.path.basename(PRETRAINED_PATH),
            "q_base_diag": Q_BASE_DIAG,
            "w_q_profile": W_Q_PROFILE,
            "track_mode": TRACK_MODE,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"\n  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
