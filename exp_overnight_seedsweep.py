"""exp_overnight_seedsweep.py — sequential brute-force seed sweep.

Run the qf50 recipe 20 times with different seeds. Save each model.
After each, run quick gen test (6 ICs). Track which seeds succeed.

If we find another working seed → reproducibility solved (we have 2+).
If 0/20 succeed → recipe is fundamentally seed-fragile, need bigger
fix.

Per-seed budget: 80 epochs × 170 steps. Each ~10 min on contended
CPU (~5 min if alone). 20 seeds × 10 min = 3.3 hours.
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
NUM_STEPS = 170
EPOCHS    = 80
LR        = 1e-3
SAVE_DIR  = "saved_models"

QF_DIAG = [20.0, 50.0, 40.0, 30.0]
SEEDS = list(range(100, 120))   # 100, 101, ..., 119 (avoid overlap with prior runs)


def apply_q1_kickstart(net, state_dim, horizon, bias):
    final = [m for m in net.q_head.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            final.bias[k * state_dim + 0] = bias
            final.bias[k * state_dim + 1] = bias


def make_demo(num_steps, device):
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
    tight = float((wraps < 0.1).mean())
    return arr, longest, int(np.sum(in_zone)), tight


def quick_gen(net, mpc, x_goal):
    test_x0s = [
        ("canonical",  [0.0, 0.0, 0.0, 0.0]),
        ("q1=+0.5",    [0.5, 0.0, 0.0, 0.0]),
        ("q1=-0.5",    [-0.5, 0.0, 0.0, 0.0]),
        ("q1d=+1.0",   [0.0, 1.0, 0.0, 0.0]),
        ("q1d=-1.0",   [0.0, -1.0, 0.0, 0.0]),
        ("combined",   [0.3, 0.5, 0.1, 0.3]),
    ]
    succ = 0; rows = []
    tights = []
    for label, x0_list in test_x0s:
        x0_t = torch.tensor(x0_list, dtype=torch.float64)
        x_t, _ = train_module.rollout(lin_net=net, mpc=mpc, x0=x0_t, x_goal=x_goal, num_steps=1000)
        arr, lng, tot, tight = metrics(x_t.cpu().numpy(), x_goal.cpu().numpy())
        ok = "OK" if tot >= 50 else ("WEAK" if tot > 0 else "FAIL")
        if tot >= 50: succ += 1
        rows.append(f"      {label:<11}  arr={'-' if arr is None else str(arr):>4}  long={lng:>3}  tot={tot:>3}  tight={tight:.0%}  {ok}")
        tights.append(tight)
    return succ, np.mean(tights), "\n".join(rows)


class M:
    def __init__(self): self._best = float('inf')
    def log_epoch(self, epoch, num_epochs, loss, info):
        d = info.get('pure_end_error', float('nan'))
        if d < self._best: self._best = d


def train_one(seed):
    print(f"\n{'='*78}\n  SEED {seed}\n{'='*78}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    x0 = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo = make_demo(NUM_STEPS, device)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor([12., 5., 50., 40.], device=device, dtype=torch.float64)
    mpc.Qf = torch.diag(torch.tensor(QF_DIAG, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork(
        state_dim=4, control_dim=2, horizon=HORIZON, hidden_dim=128,
        gate_range_q=0.99, gate_range_r=0.20, f_extra_bound=3.0, f_kickstart_amp=0.0,
    ).to(device).double()
    apply_q1_kickstart(lin_net, 4, HORIZON, -3.0)

    mon = M()
    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=LR, weight_decay=1e-4)
    recorder = network_module.NetworkOutputRecorder()

    t0 = time.time()
    for epoch in range(EPOCHS):
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=1, lr=LR,
            debug_monitor=mon, recorder=recorder, grad_debug=False,
            track_mode="energy",
            w_q_profile=100.0, q_profile_pump=[0.01, 0.01, 1, 1],
            q_profile_stable=[1, 1, 1, 1], q_profile_state_phase=True,
            w_end_q_high=80.0, end_phase_steps=20,
            external_optimizer=optimizer, restore_best=False,
        )
    print(f"    Trained in {time.time()-t0:.0f}s. Best GoalDist@170: {mon._best:.4f}", flush=True)

    succ, avg_tight, summary = quick_gen(lin_net, mpc, x_goal)
    print(f"    Quick gen (6 ICs):")
    print(summary, flush=True)
    print(f"    >>> seed={seed}  success={succ}/6  avg_tight(<0.1)={avg_tight:.1%}", flush=True)

    name = f"stageD_overnight_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "overnight_seed_sweep", "seed": seed,
                         "success": succ, "avg_tight": avg_tight,
                         "best_p1": mon._best},
        session_name=name,
    )
    return seed, succ, avg_tight, name


def main():
    print("=" * 78)
    print(f"  OVERNIGHT SEED SWEEP — {len(SEEDS)} seeds × {EPOCHS} epochs")
    print(f"  Seeds: {SEEDS}")
    print(f"  Recipe: qf50 (Qf q1d=50)")
    print("=" * 78)

    results = []
    t_total = time.time()
    for s in SEEDS:
        try:
            r = train_one(s)
            results.append(r)
        except Exception as e:
            print(f"    seed={s} ERROR: {e}", flush=True)
            results.append((s, -1, 0.0, None))

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for seed, succ, tight, name in results:
        print(f"  seed={seed:>3}  success={succ}/6  tight={tight:.1%}  {name or '-'}")
    print(f"\n  Total time: {time.time() - t_total:.0f}s")

    # Highlight winners
    winners = [r for r in results if r[1] >= 5]
    print(f"\n  WINNERS (success >= 5/6): {len(winners)}")
    for w in winners:
        print(f"    seed={w[0]}  success={w[1]}  tight={w[2]:.1%}")


if __name__ == "__main__":
    main()
