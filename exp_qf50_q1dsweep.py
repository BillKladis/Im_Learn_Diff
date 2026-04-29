"""exp_qf50_q1dsweep.py — find the optimal Qf q1d weight.

probe_qf.py (no retraining) showed q1d=50 was the lowest value where
sustained hold became long. Above 50, it stays long; below collapses.
But that was probing the EXISTING (q1d=20) model — the policy didn't
adapt. Now train a fresh model against q1d=40 and q1d=60 to see if:
  - q1d=40 is enough (quicker arrival, slightly less hold tightness)
  - q1d=60 is even better (later arrival but tighter brake)

Each seed runs from scratch with the same recipe as exp_no_demo_qf50,
just with different Qf q1d.
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
W_Q_PROFILE = 100.0
Q_PROFILE_PUMP = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0, 1.0, 1.0, 1.0]
W_END_Q_HIGH = 80.0
END_PHASE_STEPS = 20
Q_GATE_KICKSTART_BIAS = -3.0

# Sweep over Qf q1d. (q1d=50 is the existing winner.)
Q1D_VALUES = [40, 60]   # 50 already tested

QUICK_TEST_X0S = [
    ("canonical",  [0.0,   0.0, 0.0, 0.0]),
    ("q1=+0.2",    [0.2,   0.0, 0.0, 0.0]),
    ("q1=-0.2",    [-0.2,  0.0, 0.0, 0.0]),
    ("q1d=+0.5",   [0.0,   0.5, 0.0, 0.0]),
    ("q1d=-0.5",   [0.0,  -0.5, 0.0, 0.0]),
    ("combined+",  [0.15,  0.4, 0.1, 0.2]),
]


def apply_q1_kickstart(net, state_dim, horizon, bias):
    final = [m for m in net.q_head.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            final.bias[k * state_dim + 0] = bias
            final.bias[k * state_dim + 1] = bias


def make_synthetic_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


def wrap_pi(x): return math.atan2(math.sin(x), math.cos(x))


def metrics(traj, x_goal):
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


def quick_gen(lin_net, mpc, x_goal):
    success = 0; rows = []
    for label, x0_list in QUICK_TEST_X0S:
        x0_t = torch.tensor(x0_list, dtype=torch.float64)
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_t, x_goal=x_goal, num_steps=1000,
        )
        arr, lng, tot = metrics(x_t.cpu().numpy(), x_goal.cpu().numpy())
        ok = "OK" if tot >= 50 else ("WEAK" if tot > 0 else "FAIL")
        if tot >= 50: success += 1
        rows.append(f"    {label:<11}  arr={'-' if arr is None else str(arr):>4}  long={lng:>3}  tot={tot:>3}  {ok}")
    return success, "\n".join(rows)


class SilentMonitor:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self._best = float('inf')
    def log_epoch(self, epoch, num_epochs, loss, info):
        d = info.get('pure_end_error', float('nan'))
        if d < self._best: self._best = d
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"      ep {epoch+1:>3}/{num_epochs}: loss={loss:>7.3f}  goal_d={d:.3f}  best={self._best:.3f}",
                  flush=True)


def train_one_q1d(q1d_value, seed=0):
    qf_diag = [20.0, float(q1d_value), 40.0, 30.0]
    print(f"\n{'='*78}\n  Q1D={q1d_value}: training from scratch  seed={seed}\n{'='*78}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    x0     = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_synthetic_demo(NUM_STEPS, device)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    mpc.Qf          = torch.diag(torch.tensor(qf_diag, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=0.0,
    ).to(device).double()
    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, Q_GATE_KICKSTART_BIAS)

    monitor = SilentMonitor(EPOCHS)
    recorder = network_module.NetworkOutputRecorder()

    t0 = time.time()
    train_module.train_linearization_network(
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
    print(f"    Trained in {elapsed:.0f}s")

    success, summary = quick_gen(lin_net, mpc, x_goal)
    print(f"    Quick generality (6 ICs):")
    print(summary)
    print(f"    >>> q1d={q1d_value}  success={success}/{len(QUICK_TEST_X0S)}")

    name = f"stageD_qf{q1d_value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "qf_q1d_sweep", "qf_diag": qf_diag,
                         "success": success, "seed": seed},
        session_name=name,
    )
    return q1d_value, success, name


def main():
    print("=" * 78)
    print("  EXP QF50 Q1D-SWEEP — find optimal terminal cost q1d weight")
    print(f"  Q1D values: {Q1D_VALUES} (q1d=50 is existing winner: 12/12)")
    print("=" * 78)

    results = []
    t_total = time.time()
    for q1d in Q1D_VALUES:
        try:
            r = train_one_q1d(q1d)
            results.append(r)
        except Exception as e:
            print(f"    q1d={q1d} ERROR: {e}")
            results.append((q1d, -1, None))

    print("\n" + "=" * 78)
    print("  SUMMARY (q1d=50 baseline: 6/6 expected)")
    print("=" * 78)
    for q1d, succ, name in results:
        print(f"  q1d={q1d}  success={succ}/{len(QUICK_TEST_X0S)}  {name or '-'}")
    print(f"\n  Total time: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
