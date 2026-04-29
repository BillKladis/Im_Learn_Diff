"""exp_inline_seedsweep_sc.py — multi-seed of the inline phase transition.

User question: 'how do professionals handle [seed fragility]?'.
Answer: brute-force multi-seed.

Run the inline phase transition recipe (Phase 1 = 0.0612 swing-up,
Phase 2 = + hold reward + higher Q) with 5 different RNG seeds.
After each, run a quick generality test (6 ICs).

Smaller per-seed config than the parent (60+50 epochs vs 80+70) so
all 5 seeds finish in a reasonable wall-clock window. The point is
to find AT LEAST ONE seed that produces good generality, not to
exhaustively train each.
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
DT        = 0.05
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM, CONTROL_DIM = 4, 2
SAVE_DIR  = "saved_models"
LR        = 1e-3

# Smaller per-seed budget (5 seeds × ~12 min each ≈ 1 hour)
PHASE1_EPOCHS = 60
PHASE2_EPOCHS = 50

# Phase 1
P1_NUM_STEPS = 170
P1_QF_DIAG   = [20.0, 20.0, 40.0, 30.0]
Q_BASE_DIAG  = [12.0, 5.0, 50.0, 40.0]
P1_W_Q_PROFILE      = 100.0
P1_Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
P1_Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
P1_W_END_Q_HIGH     = 80.0
P1_END_PHASE_STEPS  = 20
Q_GATE_KICKSTART_BIAS = -3.0

# Phase 2
P2_NUM_STEPS = 350
P2_QF_DIAG   = [20.0, 50.0, 40.0, 30.0]
P2_Q_PROFILE_STABLE = [1.5, 1.5, 1.5, 1.5]
P2_W_END_Q_HIGH     = 150.0
P2_END_PHASE_STEPS  = 100
P2_W_HOLD_REWARD    = 3.0
P2_HOLD_SIGMA       = 0.6
P2_HOLD_START       = 170

SEEDS = [10, 20, 30, 40, 50]   # arbitrary distinct seeds

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


def make_swingup_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


def make_swingup_hold_demo(num_steps, swingup_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        if i < swingup_steps:
            alpha = i / max(swingup_steps - 1, 1)
            t = 0.5 * (1.0 - math.cos(math.pi * alpha))
            demo[i, 0] = math.pi * t
        else:
            demo[i, 0] = math.pi
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
        rows.append(f"      {label:<11}  arr={'-' if arr is None else str(arr):>4}  long={lng:>3}  tot={tot:>3}  {ok}")
    return success, "\n".join(rows)


class Silent:
    def __init__(self): self._best = float('inf')
    def log_epoch(self, epoch, num_epochs, loss, info):
        d = info.get('pure_end_error', float('nan'))
        if d < self._best: self._best = d


def train_one_seed(seed):
    print(f"\n{'='*78}\n  SEED {seed}\n{'='*78}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    x0     = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    demo_p1 = make_swingup_demo(P1_NUM_STEPS, device)
    demo_p2 = make_swingup_hold_demo(P2_NUM_STEPS, P1_NUM_STEPS, device)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    mpc.Qf          = torch.diag(torch.tensor(P1_QF_DIAG, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetworkSC(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=0.99, gate_range_r=0.20,
        f_extra_bound=3.0, f_kickstart_amp=0.0,
    ).to(device).double()
    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, Q_GATE_KICKSTART_BIAS)

    monitor = Silent()
    recorder = network_module.NetworkOutputRecorder()
    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=LR, weight_decay=1e-4)

    t0 = time.time()
    # Phase 1
    for epoch in range(PHASE1_EPOCHS):
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0, x_goal=x_goal, demo=demo_p1, num_steps=P1_NUM_STEPS,
            num_epochs=1, lr=LR,
            debug_monitor=monitor, recorder=recorder, grad_debug=False,
            track_mode="energy", w_terminal_anchor=0.0,
            w_q_profile=P1_W_Q_PROFILE,
            q_profile_pump=P1_Q_PROFILE_PUMP,
            q_profile_stable=P1_Q_PROFILE_STABLE,
            q_profile_state_phase=True,
            w_end_q_high=P1_W_END_Q_HIGH, end_phase_steps=P1_END_PHASE_STEPS,
            external_optimizer=optimizer, restore_best=False,
        )
    t_p1 = time.time() - t0
    print(f"  Phase 1: best GoalDist@170 = {monitor._best:.4f}  in {t_p1:.0f}s", flush=True)

    # Phase 2
    mpc.Qf = torch.diag(torch.tensor(P2_QF_DIAG, device=device, dtype=torch.float64))
    for epoch in range(PHASE2_EPOCHS):
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0, x_goal=x_goal, demo=demo_p2, num_steps=P2_NUM_STEPS,
            num_epochs=1, lr=LR,
            debug_monitor=monitor, recorder=recorder, grad_debug=False,
            track_mode="energy", w_terminal_anchor=0.0,
            w_q_profile=P1_W_Q_PROFILE,
            q_profile_pump=P1_Q_PROFILE_PUMP,
            q_profile_stable=P2_Q_PROFILE_STABLE,
            q_profile_state_phase=True,
            w_end_q_high=P2_W_END_Q_HIGH, end_phase_steps=P2_END_PHASE_STEPS,
            w_hold_reward=P2_W_HOLD_REWARD,
            hold_sigma=P2_HOLD_SIGMA,
            hold_start_step=P2_HOLD_START,
            external_optimizer=optimizer, restore_best=False,
        )
    t_total = time.time() - t0
    print(f"  Phase 2 done. Total: {t_total:.0f}s", flush=True)

    # Quick generality
    success, summary = quick_gen(lin_net, mpc, x_goal)
    print(f"    Quick generality (6 ICs):")
    print(summary, flush=True)
    print(f"    >>> seed={seed}  success={success}/{len(QUICK_TEST_X0S)}", flush=True)

    name = f"stageD_seedsweepSC{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "inline_seedsweep_sc", "seed": seed,
                         "success": success},
        session_name=name,
    )
    return seed, success, name


def main():
    print("=" * 78)
    print(f"  EXP INLINE SEEDSWEEP (sin/cos encoding) — {len(SEEDS)} seeds × ({PHASE1_EPOCHS}+{PHASE2_EPOCHS}) epochs")
    print(f"  Seeds: {SEEDS}")
    print("=" * 78)

    results = []
    t_total = time.time()
    for s in SEEDS:
        try:
            r = train_one_seed(s)
            results.append(r)
        except Exception as e:
            print(f"    seed={s} ERROR: {e}", flush=True)
            results.append((s, -1, None))

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for seed, succ, name in results:
        print(f"  seed={seed:>2}  success={succ}/{len(QUICK_TEST_X0S)}  {name or '-'}")
    print(f"\n  Total time: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
