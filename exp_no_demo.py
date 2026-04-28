"""exp_no_demo.py — Train without reference trajectory.

Replaces the demo's energy curve with a HARDCODED smooth energy ramp:
  E_target(t) = E_min + (E_max - E_min) * (t / T_total)
where E_min = energy at q1=0 (-14.715 J for double pendulum)
      E_max = energy at upright (+14.715 J)

The network only sees:
  - state (current pendulum)
  - energy of current state vs target ramp
  - existing Q-profile + end-Q-high penalties

If swing-up emerges from this purely physics-driven setup, it shows the
network can learn swing-up WITHOUT ANY DEMO TRAJECTORY — just from a
target energy schedule.
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

X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
EPOCHS    = 100
LR        = 1e-3
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM, CONTROL_DIM = 4, 2
SAVE_DIR    = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
GATE_RANGE_Q   = 0.99
GATE_RANGE_R   = 0.20
F_EXTRA_BOUND  = 3.0

W_Q_PROFILE      = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
W_END_Q_HIGH     = 80.0
END_PHASE_STEPS  = 20
Q_GATE_KICKSTART_BIAS = -3.0


def apply_q1_kickstart(lin_net, state_dim, horizon, bias_val):
    q_final = [m for m in lin_net.q_head.modules()
               if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            q_final.bias[k * state_dim + 0] = bias_val
            q_final.bias[k * state_dim + 1] = bias_val


def make_synthetic_demo(num_steps, device):
    """Build a 'demo' trajectory whose energy ramps linearly from
    E(q1=0) to E(q1=π).  States are placeholder (zeros except q1) — the
    energy-tracking loss only uses the energy of demo[i], so the position
    components aren't read by the loss.  We linearly interpolate q1 from
    0 to π to get a smooth energy ramp."""
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        # Smooth cosine ramp from 0 → π
        alpha = i / max(num_steps - 1, 1)
        # cosine-eased ramp
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
        # Velocities zero — energy is purely potential
    return demo


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

    # SYNTHETIC demo: cosine-eased energy ramp from bottom to upright
    demo = make_synthetic_demo(NUM_STEPS, device)

    print("=" * 76)
    print("  EXP: NO DEMO TRAJECTORY — synthetic energy-ramp target")
    print(f"  q_base_diag = {Q_BASE_DIAG}  (DEFAULT)")
    print(f"  Target energy: cosine-eased ramp from E(0)≈-14.7 to E(π)≈+14.7")
    print(f"  Synthetic demo q1: 0 → π over {NUM_STEPS} steps")
    print(f"  EPOCHS = {EPOCHS}  LR = {LR}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    # Verify energy of synthetic demo
    E_demo = torch.stack([mpc.compute_energy_single(demo[i]) for i in range(NUM_STEPS)])
    print(f"  Synthetic E[0]: {E_demo[0].item():.3f}")
    print(f"  Synthetic E[{NUM_STEPS//2}]: {E_demo[NUM_STEPS//2].item():.3f}")
    print(f"  Synthetic E[-1]: {E_demo[-1].item():.3f}")

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=0.0,
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
        grad_debug=False, track_mode="energy", w_terminal_anchor=0.0,
        w_q_profile=W_Q_PROFILE,
        q_profile_pump=Q_PROFILE_PUMP,
        q_profile_stable=Q_PROFILE_STABLE,
        q_profile_state_phase=True,
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
    print(f"  (NO real demo trajectory was used — only synthetic energy ramp)")

    session_name = f"stageD_nodemo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "no_demo_synthetic_energy_ramp",
            "q_base_diag": Q_BASE_DIAG,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
