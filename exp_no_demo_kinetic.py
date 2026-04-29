"""exp_no_demo_kinetic.py — original recipe + KINETIC-PEAK demo.

User insight: the current demo has q1 ramp 0 → π monotonically with
q1d=0, so total energy ramps monotonically up to 2mgL only at the
very last step. The pendulum never gets a 'lower energy as we
approach the top' signal — and physically that's wrong, because
near upright the kinetic energy MUST go to zero.

ONLY change vs exp_no_demo.py: the demo now has a non-zero q1d
that bell-curves through the trajectory:
   demo[i, 0] = π * cosine_ease(i/T)            ← unchanged: q1 ramp
   demo[i, 1] = q1d_peak * sin(π * i/T)         ← NEW: kinetic peak

So demo's energy E_demo[i] = (1/2)q1d² + mgL(1-cos(q1)) goes:
  i=0       : (potential 0)        + (kinetic 0)              = 0
  i=T/2     : (potential ≈ mgL)    + (kinetic = ½ q1d_peak²)  = HIGH
  i=T       : (potential ≈ 2mgL)   + (kinetic 0)              = 2mgL

Crucially, E_demo[T/2] > E_demo[T]. The energy CURVE goes up, peaks,
and comes DOWN to the upright equilibrium. Network must inject AND
dissipate energy in the right places.

Single-variable change vs the 0.0612-baseline recipe. Qf default
(no q1d=50 bump). Same epochs/LR/everything else.
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

X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
EPOCHS    = 100
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

# THE ONLY CHANGE — demo with kinetic bell curve
Q1D_PEAK = 5.0  # peak q1d in demo (rad/s), tuned for E_mid ≈ 2.5 mgL


def apply_q1_kickstart(lin_net, state_dim, horizon, bias_val):
    q_final = [m for m in lin_net.q_head.modules()
               if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            q_final.bias[k * state_dim + 0] = bias_val
            q_final.bias[k * state_dim + 1] = bias_val


def make_kinetic_peak_demo(num_steps, device):
    """q1: cosine-eased ramp 0 → π. q1d: bell curve 0 → peak → 0.
    Total energy peaks mid-swing and returns to 2mgL at the end —
    explicitly tells the network 'inject kinetic to climb, then
    dissipate it as you reach the top'."""
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
        demo[i, 1] = Q1D_PEAK * math.sin(math.pi * alpha)
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
    demo   = make_kinetic_peak_demo(NUM_STEPS, device)

    print("=" * 80)
    print("  EXP NO-DEMO + KINETIC-PEAK demo  (single change vs 0.0612 baseline)")
    print(f"  q1d peak in demo: {Q1D_PEAK} rad/s")
    print(f"  Q_BASE_DIAG = {Q_BASE_DIAG}")
    print(f"  Qf          = [20, 20, 40, 30]   (default — Qf NOT bumped)")
    print(f"  EPOCHS={EPOCHS}  LR={LR}  NUM_STEPS={NUM_STEPS}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    # Verify: print demo energies at key timesteps
    print("\n  Demo energy trace (first 5, mid, last 5):")
    e_demo = [mpc.compute_energy_single(demo[i]).item() for i in range(NUM_STEPS)]
    for idx in [0, 1, 2, 3, 4, NUM_STEPS//2-2, NUM_STEPS//2, NUM_STEPS//2+2,
                NUM_STEPS-5, NUM_STEPS-3, NUM_STEPS-1]:
        print(f"    step {idx:>3}: q1={demo[idx,0].item():.3f}  "
              f"q1d={demo[idx,1].item():.3f}  E={e_demo[idx]:.3f}")

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

    session_name = f"stageD_kinetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "no_demo_kinetic_peak",
            "q_base_diag": Q_BASE_DIAG,
            "q1d_peak": Q1D_PEAK,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"\n  Saved → saved_models/{session_name}/")

    if len(loss_history) > 2:
        decreased = sum(1 for i in range(1, len(loss_history))
                        if loss_history[i] < loss_history[i-1])
        print(f"\n  Loss monotonicity:")
        print(f"    epoch 1: {loss_history[0]:.3f}")
        print(f"    epoch {len(loss_history)//2}: {loss_history[len(loss_history)//2]:.3f}")
        print(f"    epoch {len(loss_history)}: {loss_history[-1]:.3f}")
        print(f"    decreasing transitions: {decreased}/{len(loss_history)-1}")

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

    x_t, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=1000,
    )
    arr, lng, tot = metrics_for(x_t.cpu().numpy(), x_goal.cpu().numpy())
    print(f"\n  Sustained hold (1000 steps, wrap < 0.3):")
    print(f"    arrival: {'step '+str(arr)+' ('+f'{arr*DT:.2f}s'+')' if arr is not None else 'NEVER'}")
    print(f"    longest contiguous: {lng} steps ({lng*DT:.2f}s)")
    print(f"    total in zone: {tot} steps ({tot*DT:.2f}s)")

    print(f"\n  REFERENCE: 0.0612 baseline (Qf=[20,20,40,30], no kinetic peak):")
    print(f"    arrival=167 (8.35s) longest=74 (3.7s) total=124 (6.2s)")

    print(f"\n  Training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
