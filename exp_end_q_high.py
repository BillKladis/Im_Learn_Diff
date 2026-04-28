"""
exp_end_q_high.py — Build on the SUCCESS config (state-phase profile,
q1+q1d kickstart, energy track) and ADD end-phase Q-gate increase.

In the last 20 steps, push q1 and q1d gates UP toward 1.0 — this
activates the QP's q1 cost for stabilisation near the goal.

The user's request: "tighter goal_dist by modulating the weights q at
the top — encourage q increase at the end."
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
EPOCHS    = 100
LR        = 1e-3
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM   = 4
CONTROL_DIM = 2
SAVE_DIR    = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
GATE_RANGE_Q   = 0.99
GATE_RANGE_R   = 0.20
F_EXTRA_BOUND  = 3.0
F_KICKSTART    = 0.0
Q_GATE_KICKSTART_BIAS = -3.0   # softer kickstart so end-phase pull can lift gates

# State-phase profile (proven to allow swing-up at goal_dist=0.25)
W_Q_PROFILE      = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
Q_PROFILE_STATE_PHASE = True

# end-phase q-gate increase (best was W=50, try slightly stronger with softer kickstart)
W_END_Q_HIGH    = 80.0
END_PHASE_STEPS = 20

TRACK_MODE = "energy"


def apply_q1_kickstart(lin_net, state_dim, horizon, bias_val):
    q_final = [m for m in lin_net.q_head.modules()
               if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            q_final.bias[k * state_dim + 0] = bias_val   # q1 dim
            q_final.bias[k * state_dim + 1] = bias_val   # q1d dim


def clean_demo_tail(demo, num_smooth_steps=20):
    demo_clean = demo.clone()
    T = demo_clean.shape[0]
    goal = torch.tensor(X_GOAL, device=demo.device, dtype=demo.dtype)
    start_idx = T - num_smooth_steps
    start = demo_clean[start_idx].clone()
    for k in range(num_smooth_steps):
        alpha = (k + 1) / num_smooth_steps
        a = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo_clean[start_idx + k] = (1.0 - a) * start + a * goal
    return demo_clean


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
    demo_raw = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)
    demo = clean_demo_tail(demo_raw, num_smooth_steps=20)

    print("=" * 76)
    print("  EXP: end-phase Q-gate increase (tighter goal_dist via stabilisation)")
    print(f"  q_base_diag = {Q_BASE_DIAG}  (DEFAULT)")
    print(f"  W_Q_PROFILE = {W_Q_PROFILE}  state_phase = True")
    print(f"  W_END_Q_HIGH = {W_END_Q_HIGH}  END_PHASE_STEPS = {END_PHASE_STEPS}")
    print(f"  EPOCHS = {EPOCHS}  LR = {LR}")
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
        q_profile_state_phase=Q_PROFILE_STATE_PHASE,
        w_end_q_high=W_END_Q_HIGH,
        end_phase_steps=END_PHASE_STEPS,
    )
    elapsed = time.time() - t0

    x_final, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    dist_final = float(np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL)))
    result = "SUCCESS" if dist_final < 1.0 else "FAIL"
    print(f"\n  goal_dist = {dist_final:.4f}  epochs = {len(loss_history)}  "
          f"time = {elapsed:.0f}s  {result}")

    # Q-gate profile diagnostic with focus on END phase
    print("\n  Q-gate profile from final epoch (averaged over horizon):")
    final_steps = recorder.epochs[-1]["steps"]
    print(f"  step={'q1':>8} {'q1d':>8} {'q2':>8} {'q2d':>8} {'fNorm':>8}")
    for s in [0, 30, 60, 90, 118, 140, 144, 150, 160, 165, 169]:
        if s >= len(final_steps):
            continue
        gates = final_steps[s]["gates_Q"]
        gates_t = torch.tensor(gates)
        avg = gates_t.mean(dim=0).tolist()
        f_extra = torch.tensor(final_steps[s]["f_extra"])
        f_n = float(torch.sqrt((f_extra**2).sum()))
        print(f"  {s:>4}: {avg[0]:>8.4f} {avg[1]:>8.4f} {avg[2]:>8.4f} {avg[3]:>8.4f} {f_n:>8.3f}")

    session_name = f"stageD_endqhigh_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "end_phase_q_gate_increase",
            "q_base_diag": Q_BASE_DIAG,
            "w_q_profile": W_Q_PROFILE,
            "w_end_q_high": W_END_Q_HIGH,
            "end_phase_steps": END_PHASE_STEPS,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"\n  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
