"""trace_rollout.py — Print every step of a rollout so we can SEE what's
happening. Answers: does the pendulum REACH the goal early and then
drift, or does it never reach it?

Reports for each step: q1 (wrapped to ±π), q1d, q2, q2d, raw distance,
wrapped distance, fastest-arrival timestamp, and bins of "time spent
near goal" (wrap < 0.3, < 1.0, etc).
"""

import math, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")
import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 1000
DT        = 0.05
HORIZON   = 10
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

# Compare two models side-by-side
MODELS = [
    ("baseline",
     "saved_models/stageD_nodemo_20260428_123448/stageD_nodemo_20260428_123448.pth"),
    ("stab_state",
     "saved_models/stageD_stabstate_20260428_224856/stageD_stabstate_20260428_224856.pth"),
]


def wrap_pi(x):
    return math.atan2(math.sin(x), math.cos(x))


def wrapped_goal_dist(x_state, x_goal):
    q1_err = wrap_pi(x_state[0] - x_goal[0])
    return math.sqrt(q1_err**2 + x_state[1]**2 + x_state[2]**2 + x_state[3]**2)


def trace_one(name, model_path, mpc, x0, x_goal):
    print("\n" + "=" * 80)
    print(f"  Trace: {name}")
    print(f"  {model_path}")
    print("=" * 80)

    lin_net = network_module.LinearizationNetwork.load(model_path, device="cpu").double()
    x_t, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    x = x_t.cpu().numpy()  # shape (NUM_STEPS+1, 4)
    g = np.array(X_GOAL)

    # Per-step metrics
    raw  = np.linalg.norm(x - g, axis=1)
    q1_w = np.array([wrap_pi(s[0] - g[0]) for s in x])
    wrap = np.sqrt(q1_w**2 + x[:,1]**2 + x[:,2]**2 + x[:,3]**2)

    # First arrival times (when wrapped distance first drops below threshold)
    print(f"\n  First arrival times:")
    for thr in [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]:
        below = np.where(wrap < thr)[0]
        if len(below):
            t0 = below[0]
            print(f"    wrap < {thr:.2f}: step {t0:>4d}  ({t0*DT:>5.2f}s)")
        else:
            print(f"    wrap < {thr:.2f}: NEVER")

    # Sustained hold: longest contiguous run with wrap < 0.3
    print(f"\n  Sustained hold (wrap < 0.30):")
    in_zone = wrap < 0.3
    runs = []
    start = None
    for i, v in enumerate(in_zone):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i-1))
            start = None
    if start is not None:
        runs.append((start, len(in_zone)-1))
    if runs:
        longest = max(runs, key=lambda r: r[1]-r[0])
        print(f"    longest run: steps [{longest[0]:>4d}, {longest[1]:>4d}]  "
              f"({(longest[1]-longest[0]+1)*DT:.2f}s, "
              f"{longest[1]-longest[0]+1} steps)")
        print(f"    total time in zone: {sum(r[1]-r[0]+1 for r in runs)*DT:.2f}s "
              f"({sum(r[1]-r[0]+1 for r in runs)} steps)")
        print(f"    number of zone re-entries: {len(runs)}")
    else:
        print(f"    NEVER entered the zone")

    # Time-binned snapshots
    print(f"\n  Snapshots:")
    print(f"    {'step':>5}  {'time':>5}  {'q1':>7}  {'q1_w':>7}  "
          f"{'q1d':>7}  {'q2':>7}  {'q2d':>7}  {'wrap':>6}")
    for step in [0, 100, 170, 200, 220, 250, 300, 350, 400, 500, 600, 800, 1000]:
        if step < len(x):
            s = x[step]
            print(f"    {step:>5d}  {step*DT:>4.1f}s  "
                  f"{s[0]:>7.3f}  {wrap_pi(s[0]):>7.3f}  "
                  f"{s[1]:>7.3f}  {s[2]:>7.3f}  {s[3]:>7.3f}  "
                  f"{wrap[step]:>6.3f}")

    # When is the pendulum closest to goal?
    arg_min = int(np.argmin(wrap))
    print(f"\n  Closest approach: step {arg_min}  ({arg_min*DT:.2f}s)  wrap={wrap[arg_min]:.4f}")

    return wrap


def main():
    device = torch.device("cpu")
    x0     = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    for name, path in MODELS:
        if os.path.exists(path):
            trace_one(name, path, mpc, x0, x_goal)
        else:
            print(f"  SKIP: {name} ({path} not found)")


if __name__ == "__main__":
    main()
