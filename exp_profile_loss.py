"""
exp_profile_loss.py — Strong Q-gate profile targeting + default q_base_diag.

The user's idea, implemented properly:
  Pump phase (0-70%):  target gates_Q = [0.01, 1.0, 1.0, 1.0]
                       (q1 cost suppressed, others nominal)
  Stable phase (70%+): target gates_Q = [1.0, 1.0, 1.0, 1.0]
                       (full Q costs for stabilisation)

This forces effective q_base_diag during pump = [0.12, 5, 50, 40], which
is well below the 2.0 threshold proven to allow energy-tracking to work.

The penalty weight is HIGH (40) — comparable to W_TRACK=5 × tracking
loss ≈ 2.5, ensuring the profile target dominates the gradient trap.

Also encourages large |f_extra| during pump phase via w_f_phase_reward.
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
EPOCHS    = 80
LR        = 5e-4
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM   = 4
CONTROL_DIM = 2
SAVE_DIR    = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
GATE_RANGE_Q   = 0.99
GATE_RANGE_R   = 0.20
F_EXTRA_BOUND  = 3.0
F_KICKSTART    = 0.0          # REMOVED — let energy tracking shape f_extra
Q_GATE_KICKSTART_BIAS = -4.0  # KEPT — q1 gate near minimum at start

# The KEY parameters
W_Q_PROFILE      = 200.0      # very strong profile penalty (was 40)
Q_PROFILE_PUMP   = [0.01, 1.0, 1.0, 1.0]    # [q1, q1d, q2, q2d]
Q_PROFILE_STABLE = [1.0,  1.0, 1.0, 1.0]
PHASE_SPLIT_FRAC = 0.7

# REMOVED: f-reward was saturating f_extra in wrong direction (no alternation).
# Let pure energy tracking gradient shape f_extra naturally.
W_F_PHASE_REWARD = 0.0

TRACK_MODE = "energy"


def apply_q1_kickstart(lin_net, state_dim, horizon, bias_val):
    q_final = [m for m in lin_net.q_head.modules()
               if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            q_final.bias[k * state_dim + 0] = bias_val


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
        if epoch == 0 or (epoch+1) % 5 == 0 or epoch == num_epochs-1:
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
    print("  EXP: Strong profile target + default q_base_diag")
    print(f"  q_base_diag : {Q_BASE_DIAG}")
    print(f"  pump target : {Q_PROFILE_PUMP}  (effective q_base = "
          f"[{12*Q_PROFILE_PUMP[0]:.2f}, {5*Q_PROFILE_PUMP[1]:.1f}, "
          f"{50*Q_PROFILE_PUMP[2]:.1f}, {40*Q_PROFILE_PUMP[3]:.1f}])")
    print(f"  stable tgt  : {Q_PROFILE_STABLE}  (full default Q costs)")
    print(f"  phase split : pump=0-{int(PHASE_SPLIT_FRAC*NUM_STEPS)}  "
          f"stable={int(PHASE_SPLIT_FRAC*NUM_STEPS)}-{NUM_STEPS}")
    print(f"  W_Q_PROFILE = {W_Q_PROFILE}  W_F_PHASE_REWARD = {W_F_PHASE_REWARD}")
    print(f"  Epochs={EPOCHS}  LR={LR}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=F_KICKSTART,
    ).to(device).double()

    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, Q_GATE_KICKSTART_BIAS)

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(num_epochs=EPOCHS)

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=False, track_mode=TRACK_MODE, w_terminal_anchor=0.0,
        w_q_profile=W_Q_PROFILE,
        q_profile_pump=Q_PROFILE_PUMP,
        q_profile_stable=Q_PROFILE_STABLE,
        w_f_phase_reward=W_F_PHASE_REWARD,
        phase_split_frac=PHASE_SPLIT_FRAC,
    )
    elapsed = time.time() - t0

    x_final, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    dist_final = np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL))
    result = "SUCCESS" if dist_final < 1.0 else "FAIL"

    print(f"\n  goal_dist={dist_final:.4f}  epochs={len(loss_history)}  "
          f"time={elapsed:.0f}s  {result}")

    # Diagnostic: capture Q-gate profile from the recorder (last epoch)
    print("\n  Q-gate profile from final epoch (averaged over horizon):")
    final_steps = recorder.epochs[-1]["steps"]
    print(f"  step={'q1':>8} {'q1d':>8} {'q2':>8} {'q2d':>8} {'fNorm':>8}")
    for s in [0, int(NUM_STEPS*0.25), int(NUM_STEPS*0.5),
              int(NUM_STEPS*PHASE_SPLIT_FRAC), int(NUM_STEPS*0.85), NUM_STEPS-1]:
        if s >= len(final_steps):
            continue
        gates = final_steps[s]["gates_Q"]   # (horizon-1, state_dim) list
        gates_t = torch.tensor(gates)
        avg = gates_t.mean(dim=0).tolist()
        f_extra = torch.tensor(final_steps[s]["f_extra"])
        f_n = float(torch.sqrt((f_extra**2).sum()))
        print(f"  {s:>4}: {avg[0]:>8.4f} {avg[1]:>8.4f} {avg[2]:>8.4f} {avg[3]:>8.4f} {f_n:>8.3f}")

    session_name = f"stageD_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "q_profile_target_pump",
            "q_base_diag": Q_BASE_DIAG,
            "w_q_profile": W_Q_PROFILE,
            "q_profile_pump": Q_PROFILE_PUMP,
            "q_profile_stable": Q_PROFILE_STABLE,
            "phase_split_frac": PHASE_SPLIT_FRAC,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"\n  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
