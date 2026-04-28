"""
exp_twophase.py — User's idea: train f-head at q1=0 with freedom, then
add q-head training at default q_base while f continues to adapt.

Phase 1 (30 epochs, q_base = [0,0,50,40]):
  - f-head learns the pumping pattern with NO q1 cost interference
  - q-head's q1 dim gets ZERO gradient (since q_base[0]=0)
  - q-head's q2/q2d dims learn nominal gates (≈1.0 since profile=base)
  - Result: a working swing-up policy with q1 gate ≈ 1.0 (untrained on q1)

Phase 2 (50 epochs, q_base = [12,5,50,40] + profile penalty W=100):
  - q-head profile penalty drives q1 gate toward [0.01,1,1,1] state-phased
  - f-head simultaneously adapts to the changing Q-gate profile
  - Both heads co-evolve; f-head doesn't lose its pumping pattern

If this works, the network has learned to handle the FULL default
q_base_diag via Q-gate suppression of q1 cost during pump phase.
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

PHASE1_EPOCHS = 30
PHASE2_EPOCHS = 60
LR_PHASE1     = 1e-3
LR_PHASE2     = 5e-4

# Phase 1 environment
Q_BASE_PHASE1 = [0.0, 0.0, 50.0, 40.0]
# Phase 2 environment
Q_BASE_PHASE2 = [12.0, 5.0, 50.0, 40.0]

GATE_RANGE_Q   = 0.99
GATE_RANGE_R   = 0.20
F_EXTRA_BOUND  = 3.0
F_KICKSTART    = 0.0   # let f learn naturally in phase 1

# Phase 2: profile penalty
W_Q_PROFILE      = 100.0
Q_PROFILE_PUMP   = [0.01, 1.0, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0, 1.0, 1.0]
TRACK_MODE = "energy"


class QuietMonitor:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
    def log_epoch(self, epoch, num_epochs, loss, info):
        if epoch == 0 or (epoch+1) % 2 == 0 or epoch == num_epochs-1:
            gd = info.get('pure_end_error', float('nan'))
            qdev = info.get('mean_Q_gate_dev', float('nan'))
            fn = info.get('mean_f_extra_norm', float('nan'))
            ft = info.get('mean_f_tau1_first', float('nan'))
            print(f"  [{epoch+1:>3}/{num_epochs}] loss={loss:.4f} "
                  f"goal_dist={gd:.4f} QDev={qdev:.4f} fNorm={fn:.3f} "
                  f"fτ1[0]={ft:+.3f}", flush=True)


def evaluate(lin_net, mpc, x0, x_goal):
    x_final, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    return float(np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)

    print("=" * 76)
    print("  TWO-PHASE EXPERIMENT (user's idea)")
    print(f"  Phase 1: q_base={Q_BASE_PHASE1}  {PHASE1_EPOCHS}ep  LR={LR_PHASE1}")
    print(f"  Phase 2: q_base={Q_BASE_PHASE2}  {PHASE2_EPOCHS}ep  LR={LR_PHASE2}")
    print(f"           + profile penalty W={W_Q_PROFILE} (state-phased)")
    print("=" * 76)

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=F_KICKSTART,
    ).to(device).double()
    # NO q1 kickstart — let phase 2 profile penalty pull q1 gate down.
    # In phase 1, q_base[0]=0 means q1 gate doesn't matter anyway.

    # ── PHASE 1 ─────────────────────────────────────────────────────────
    print("\n--- PHASE 1: f-head trains at zero q1 cost ---\n")
    mpc1 = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc1.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc1.q_base_diag = torch.tensor(Q_BASE_PHASE1, device=device, dtype=torch.float64)

    rec1 = network_module.NetworkOutputRecorder()
    t0 = time.time()
    loss1, rec1 = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc1,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=PHASE1_EPOCHS, lr=LR_PHASE1,
        debug_monitor=QuietMonitor(PHASE1_EPOCHS), recorder=rec1,
        grad_debug=False, track_mode=TRACK_MODE, w_terminal_anchor=0.0,
    )
    p1_time = time.time() - t0
    d_p1 = evaluate(lin_net, mpc1, x0, x_goal)
    print(f"\n  Phase 1 done in {p1_time:.0f}s.  goal_dist (at q1=0) = {d_p1:.4f}")

    # Diagnostic: what gates did q_head learn at q1=0?
    final_p1 = rec1.epochs[-1]["steps"]
    q1_avg = float(torch.tensor([s["gates_Q"] for s in final_p1])[:, :, 0].mean())
    print(f"  Phase 1 final mean q1 gate across trajectory: {q1_avg:.4f}")

    # Verify model also broken at default q_base (motivating phase 2)
    mpc2 = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc2.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc2.q_base_diag = torch.tensor(Q_BASE_PHASE2, device=device, dtype=torch.float64)
    d_p1_default = evaluate(lin_net, mpc2, x0, x_goal)
    print(f"  Phase 1 model evaluated at DEFAULT q_base: goal_dist = {d_p1_default:.4f}")

    # ── PHASE 2 ─────────────────────────────────────────────────────────
    print("\n--- PHASE 2: q-head profile penalty + f-head co-training ---\n")
    rec2 = network_module.NetworkOutputRecorder()
    t1 = time.time()
    loss2, rec2 = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc2,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=PHASE2_EPOCHS, lr=LR_PHASE2,
        debug_monitor=QuietMonitor(PHASE2_EPOCHS), recorder=rec2,
        grad_debug=False, track_mode=TRACK_MODE, w_terminal_anchor=0.0,
        w_q_profile=W_Q_PROFILE,
        q_profile_pump=Q_PROFILE_PUMP,
        q_profile_stable=Q_PROFILE_STABLE,
        q_profile_state_phase=True,
    )
    p2_time = time.time() - t1
    d_p2 = evaluate(lin_net, mpc2, x0, x_goal)
    print(f"\n  Phase 2 done in {p2_time:.0f}s.  goal_dist = {d_p2:.4f}")
    result = "SUCCESS" if d_p2 < 1.0 else "FAIL"
    print(f"  {result}")

    # Final Q-gate profile
    print("\n  Final Q-gate profile (averaged over horizon):")
    final_p2 = rec2.epochs[-1]["steps"]
    print(f"  step={'q1':>8} {'q1d':>8} {'q2':>8} {'q2d':>8} {'fNorm':>8}")
    for s in [0, 30, 60, 90, 118, 140, 160, 169]:
        if s >= len(final_p2):
            continue
        gates = final_p2[s]["gates_Q"]
        gates_t = torch.tensor(gates)
        avg = gates_t.mean(dim=0).tolist()
        f_extra = torch.tensor(final_p2[s]["f_extra"])
        f_n = float(torch.sqrt((f_extra**2).sum()))
        print(f"  {s:>4}: {avg[0]:>8.4f} {avg[1]:>8.4f} {avg[2]:>8.4f} {avg[3]:>8.4f} {f_n:>8.3f}")

    session_name = f"stageD_2phase_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss1 + loss2,
        training_params={
            "experiment": "two_phase_q1zero_then_profile",
            "phase1_q_base": Q_BASE_PHASE1,
            "phase2_q_base": Q_BASE_PHASE2,
            "phase1_epochs": PHASE1_EPOCHS,
            "phase2_epochs": PHASE2_EPOCHS,
            "phase1_goal_dist": d_p1,
            "phase1_at_default_goal_dist": d_p1_default,
            "phase2_goal_dist": d_p2,
            "w_q_profile": W_Q_PROFILE,
        },
        session_name=session_name, recorder=rec2,
    )
    print(f"\n  Saved → saved_models/{session_name}/")
    print(f"\n  SUMMARY:")
    print(f"    Phase 1 at q1=0:        goal_dist = {d_p1:.4f}")
    print(f"    Phase 1 at default:     goal_dist = {d_p1_default:.4f}")
    print(f"    Phase 2 (final):        goal_dist = {d_p2:.4f}  {result}")


if __name__ == "__main__":
    main()
