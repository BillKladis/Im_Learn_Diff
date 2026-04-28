"""exp_sincos.py — Train LinearizationNetworkSC (sin/cos angle encoding).

The standard network sees q1 as a scalar in [-π, π] (normalized by π to [-1,1]).
A network trained only from x0=0 (hanging down) doesn't naturally learn to
handle negative q1 because it's never seen that region.

LinearizationNetworkSC encodes joint angles as (sin, cos) pairs:
  [q1, q1d, q2, q2d]  →  [sin(q1), cos(q1), q1d/8, sin(q2), cos(q2), q2d/8]

Benefits:
  1. q1=+π and q1=-π have IDENTICAL encodings (sin=0, cos=-1) — the network
     sees them as the same state, which is physically correct.
  2. sign(sin(q1)) directly encodes "which side of vertical" without any
     training — this is inductive bias for symmetric swing-up.
  3. No extra parameters in the hidden layers, only the first encoder layer
     changes (20→30 input dims, same 128-dim hidden).

Same training setup as exp_no_demo.py (the best result: goal_dist=0.0612).
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

X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
EPOCHS    = 120
LR        = 1e-3
HORIZON   = 10
STATE_DIM, CONTROL_DIM = 4, 2
HIDDEN_DIM = 128
SAVE_DIR  = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
GATE_RANGE_Q   = 0.99
GATE_RANGE_R   = 0.20
F_EXTRA_BOUND  = 3.0
Q_GATE_KICKSTART_BIAS = -3.0

W_Q_PROFILE      = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
W_END_Q_HIGH     = 80.0
END_PHASE_STEPS  = 20
W_F_END_REG      = 30.0   # suppress f_extra in last steps
F_END_REG_STEPS  = 25


def apply_q1_kickstart(lin_net, state_dim, horizon, bias_val):
    q_final = [m for m in lin_net.q_head.modules()
               if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            q_final.bias[k * state_dim + 0] = bias_val
            q_final.bias[k * state_dim + 1] = bias_val


def make_demo(x0_q1, target_q1, num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = x0_q1 + (target_q1 - x0_q1) * t
    return demo


class PrintMonitor:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self._header_shown = False

    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>9}  {'GoalDist':>9}  "
              f"{'QDev':>7}  {'fNorm':>7}  {'Time':>6}")
        print("─" * 90)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"{epoch+1:>4}/{num_epochs:<4}"
                  f"  {loss:>10.3f}"
                  f"  {info.get('loss_track',        float('nan')):>9.3f}"
                  f"  {info.get('pure_end_error',    float('nan')):>9.4f}"
                  f"  {info.get('mean_Q_gate_dev',   float('nan')):>7.4f}"
                  f"  {info.get('mean_f_extra_norm', float('nan')):>7.3f}"
                  f"  {info.get('epoch_time',        float('nan')):>5.2f}s",
                  flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_demo(0.0, math.pi, NUM_STEPS, device)

    print("=" * 76)
    print("  EXP: SIN/COS ENCODING — structural symmetry inductive bias")
    print(f"  Input: [sin(q1), cos(q1), q1d/8, sin(q2), cos(q2), q2d/8] × 5 = 30 dims")
    print(f"  (vs [q1/π, q1d/8, q2/π, q2d/8] × 5 = 20 dims original)")
    print(f"  hidden_dim={HIDDEN_DIM}  EPOCHS={EPOCHS}  LR={LR}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetworkSC(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=0.0,
    ).to(device).double()

    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, Q_GATE_KICKSTART_BIAS)

    n_params = sum(p.numel() for p in lin_net.parameters())
    print(f"  Parameters: {n_params:,}")

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(num_epochs=EPOCHS)

    t0 = time.time()
    best_dist = float('inf')
    best_state_dict = copy.deepcopy(lin_net.state_dict())

    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=False, track_mode="energy", w_terminal_anchor=0.0,
        w_q_profile=W_Q_PROFILE,
        q_profile_pump=Q_PROFILE_PUMP,
        q_profile_stable=Q_PROFILE_STABLE,
        q_profile_state_phase=True,
        w_end_q_high=W_END_Q_HIGH,
        end_phase_steps=END_PHASE_STEPS,
        w_f_end_reg=W_F_END_REG,
        f_end_reg_steps=F_END_REG_STEPS,
    )
    elapsed = time.time() - t0

    # Evaluate at canonical x0
    x_final, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    dist_final = float(np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL)))

    # Test symmetry immediately: positive and negative q1
    test_x0s = [
        ("zero",     [0.0,   0.0, 0.0, 0.0]),
        ("q1=+0.2",  [0.2,   0.0, 0.0, 0.0]),
        ("q1=-0.2",  [-0.2,  0.0, 0.0, 0.0]),
        ("q1d=+0.5", [0.0,   0.5, 0.0, 0.0]),
        ("q1d=-0.5", [0.0,  -0.5, 0.0, 0.0]),
    ]
    result = "SUCCESS" if dist_final < 1.0 else "FAIL"
    print(f"\n  Clean goal_dist = {dist_final:.4f}  time = {elapsed:.0f}s  {result}")
    print(f"\n  Symmetry test (trained only on x0=zero):")
    success = 0
    for name, x0_test in test_x0s:
        x0_t = torch.tensor(x0_test, device=device, dtype=torch.float64)
        xt, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0_t,
                                     x_goal=x_goal, num_steps=NUM_STEPS)
        d = float(np.linalg.norm(xt.cpu().numpy()[-1] - np.array(X_GOAL)))
        ok = "✓" if d < 1.0 else "✗"
        if d < 1.0:
            success += 1
        print(f"    {name:>12s}  goal_dist={d:.4f} {ok}")
    print(f"\n  {success}/{len(test_x0s)} succeed (trained on x0=zero ONLY)")

    session_name = f"stageD_sincos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "sincos_encoding",
            "hidden_dim": HIDDEN_DIM,
            "input_encoding": "sin_cos",
            "input_dims": 30,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
