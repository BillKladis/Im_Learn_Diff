"""exp_no_demo_qf50.py — exp_no_demo's exact recipe + Qf q1d=50.

ZERO architectural changes. ZERO curriculum. ZERO progressive penalties.
ZERO new heads. Literally exp_no_demo.py with one line different:

    mpc.Qf = diag([20, 50, 40, 30])   # was [20, 20, 40, 30]

This is the user's "just test what qf will do in our best no-traj
training" — the cleanest possible isolated test of whether bumping
q1d in the terminal cost helps when the network learns from scratch
against it.

Hypothesis (from probe_qf.py): the trained-against-q1d=20 stab_state
model loses arrival window when Qf jumps to q1d=50, but a network
trained FROM THE START against q1d=50 should adapt — finding a swing-up
that arrives early AND benefits from the tighter terminal velocity
penalty for sustained hold.
"""

import math, os, sys, time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

# Constants — IDENTICAL to exp_no_demo.py
X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
EPOCHS    = 200    # was 100; v1 early-stopped at 52, want longer patience
LR        = 1e-3
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM, CONTROL_DIM = 4, 2
SAVE_DIR  = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
GATE_RANGE_Q = 0.99
GATE_RANGE_R = 0.20
F_EXTRA_BOUND = 3.0

W_Q_PROFILE      = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
W_END_Q_HIGH     = 80.0
END_PHASE_STEPS  = 20
Q_GATE_KICKSTART_BIAS = -3.0

# THE ONLY CHANGE: bump Qf q1d from 20 to 50
QF_DIAG = [20.0, 50.0, 40.0, 30.0]


def apply_q1_kickstart(lin_net, state_dim, horizon, bias_val):
    q_final = [m for m in lin_net.q_head.modules()
               if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            q_final.bias[k * state_dim + 0] = bias_val
            q_final.bias[k * state_dim + 1] = bias_val


def make_synthetic_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


def wrap_pi(x): return math.atan2(math.sin(x), math.cos(x))


def metrics_for(traj, x_goal):
    wraps = np.array([
        math.sqrt(wrap_pi(s[0]-x_goal[0])**2 + s[1]**2 + s[2]**2 + s[3]**2)
        for s in traj
    ])
    in_zone = wraps < 0.3
    arr = next((i for i, v in enumerate(in_zone) if v), None)
    longest = 0; cur = 0
    for v in in_zone:
        cur = cur + 1 if v else 0
        if cur > longest: longest = cur
    return arr, longest, int(np.sum(in_zone))


class PrintMonitor:
    def __init__(self, num_epochs):
        self.num_epochs    = num_epochs
        self._header_shown = False
        self._best = float('inf')
    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>9}  {'GoalDist':>9}  "
              f"{'fNorm':>7}  {'Time':>6}  {'Best':>8}")
        print("─" * 76)
        self._header_shown = True
    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        d = info.get('pure_end_error', float('nan'))
        if d < self._best: self._best = d
        if epoch == 0 or (epoch+1) % 5 == 0 or epoch == num_epochs-1:
            print(f"{epoch+1:>4}/{num_epochs:<4}"
                  f"  {loss:>10.3f}"
                  f"  {info.get('loss_track', float('nan')):>9.3f}"
                  f"  {d:>9.4f}"
                  f"  {info.get('mean_f_extra_norm', float('nan')):>7.3f}"
                  f"  {info.get('epoch_time', float('nan')):>5.2f}s"
                  f"  {self._best:>8.4f}",
                  flush=True)


def main():
    device = torch.device("cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_synthetic_demo(NUM_STEPS, device)

    print("=" * 80)
    print("  EXP NO-DEMO + Qf q1d=50  (ONLY change vs the 0.0612-baseline recipe)")
    print(f"  Q_BASE_DIAG = {Q_BASE_DIAG}")
    print(f"  Qf          = {QF_DIAG}      (was [20, 20, 40, 30])")
    print(f"  EPOCHS={EPOCHS}  LR={LR}  NUM_STEPS={NUM_STEPS}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    mpc.Qf          = torch.diag(torch.tensor(QF_DIAG, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=0.0,
        # gate_range_qf=0 (default) — NO qf_head used in this experiment
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
        early_stop_patience=40,   # was 15; want longer patience for tighter convergence
    )
    elapsed = time.time() - t0

    session_name = f"stageD_nodemo_qf50_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "no_demo_qf50_q1d_bump",
            "q_base_diag": Q_BASE_DIAG,
            "qf_diag":     QF_DIAG,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"\n  Saved → saved_models/{session_name}/")

    # Loss monotonicity
    if len(loss_history) > 2:
        decreased = sum(1 for i in range(1, len(loss_history))
                        if loss_history[i] < loss_history[i-1])
        print(f"\n  Loss monotonicity:")
        print(f"    epoch 1: {loss_history[0]:.3f}")
        print(f"    epoch {len(loss_history)//2}: {loss_history[len(loss_history)//2]:.3f}")
        print(f"    epoch {len(loss_history)}: {loss_history[-1]:.3f}")
        print(f"    decreasing transitions: {decreased}/{len(loss_history)-1}")

    # Post-eval
    print(f"\n  Post-eval:")
    for n in [170, 220, 300, 400, 600, 1000, 1500]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n,
        )
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = math.sqrt(wrap_pi(last[0]-X_GOAL[0])**2 + last[1]**2 + last[2]**2 + last[3]**2)
        status = "STABLE" if wrp < 0.3 else ("CLOSE" if wrp < 1.0 else "FAIL")
        print(f"    {n:>4} steps ({n*DT:>5.1f}s): raw={raw:.4f}  wrap={wrp:.4f}  {status}")

    # Sustained hold metrics on a 1000-step rollout
    x_t, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=1000,
    )
    arr, lng, tot = metrics_for(x_t.cpu().numpy(), x_goal.cpu().numpy())
    print(f"\n  Sustained hold (1000 steps, wrap < 0.3):")
    print(f"    arrival: {'step '+str(arr)+' ('+f'{arr*DT:.2f}s'+')' if arr is not None else 'NEVER'}")
    print(f"    longest contiguous: {lng} steps ({lng*DT:.2f}s)")
    print(f"    total in zone: {tot} steps ({tot*DT:.2f}s)")

    # Comparison reference
    print(f"\n  REFERENCE: original 0.0612 (Qf=[20,20,40,30]):")
    print(f"    arrival=167 (8.35s) longest=74 (3.7s) total=124 (6.2s)")

    print(f"\n  Training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
