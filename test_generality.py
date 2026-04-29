"""test_generality.py — compare baseline 0.0612 vs qf50 v2 on perturbed x0.

Quick rollout-only test. No training. Reports for each x0:
  - first arrival time (wrap < 0.3)
  - longest contiguous hold (wrap < 0.3)
  - total time in zone

Tests both models with the same initial conditions and the Qf each
was trained against (baseline = default Qf, qf50 = q1d=50).
"""

import math, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

BASELINE = "saved_models/stageD_nodemo_20260428_123448/stageD_nodemo_20260428_123448.pth"
QF50V2   = "saved_models/stageD_nodemo_qf50_20260429_111711/stageD_nodemo_qf50_20260429_111711.pth"

X_GOAL = [math.pi, 0.0, 0.0, 0.0]
DT = 0.05
HORIZON = 10
NUM_STEPS = 1000

# Perturbed initial conditions
TEST_X0S = [
    ("canonical",     [0.0,   0.0, 0.0, 0.0]),
    ("q1=+0.2",       [0.2,   0.0, 0.0, 0.0]),
    ("q1=-0.2",       [-0.2,  0.0, 0.0, 0.0]),
    ("q1=+0.5",       [0.5,   0.0, 0.0, 0.0]),
    ("q1=-0.5",       [-0.5,  0.0, 0.0, 0.0]),
    ("q1d=+0.5",      [0.0,   0.5, 0.0, 0.0]),
    ("q1d=-0.5",      [0.0,  -0.5, 0.0, 0.0]),
    ("q1d=+1.0",      [0.0,   1.0, 0.0, 0.0]),
    ("q2=+0.1",       [0.0,   0.0, 0.1, 0.0]),
    ("q2d=+0.5",      [0.0,   0.0, 0.0, 0.5]),
    ("combined+",     [0.15,  0.4, 0.1, 0.2]),
    ("combined-",     [-0.15, -0.4, -0.1, -0.2]),
]


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


def test_one(name, model_path, qf_diag, num_steps=NUM_STEPS):
    device = torch.device("cpu")
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    x0_zero = torch.zeros(4, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0_zero, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor([12.0, 5.0, 50.0, 40.0], dtype=torch.float64)
    mpc.Qf = torch.diag(torch.tensor(qf_diag, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork.load(model_path, device="cpu").double()

    print(f"\n--- {name}  (Qf={qf_diag}) ---")
    print(f"  {'x0':<14}  {'Arr':>5}  {'Long':>6}  {'Total':>6}  {'Verdict':>10}")

    success_count = 0
    for label, x0 in TEST_X0S:
        x0_t = torch.tensor(x0, dtype=torch.float64)
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_t, x_goal=x_goal, num_steps=num_steps,
        )
        arr, lng, tot = metrics(x_t.cpu().numpy(), x_goal.cpu().numpy())

        if arr is None:
            verdict = "FAIL"
        elif lng >= 50:
            verdict = "GREAT"
            success_count += 1
        elif tot >= 50:
            verdict = "OK"
            success_count += 1
        else:
            verdict = "WEAK"

        arr_str = "—" if arr is None else f"{arr}"
        print(f"  {label:<14}  {arr_str:>5}  {lng:>6}  {tot:>6}  {verdict:>10}")

    print(f"  >>> SUCCESS: {success_count}/{len(TEST_X0S)}")
    return success_count


def main():
    print("=" * 72)
    print("  GENERALITY TEST")
    print(f"  {NUM_STEPS} steps ({NUM_STEPS*DT:.0f}s) per rollout")
    print(f"  GREAT: longest>=50 (2.5s sustained)")
    print(f"  OK:    total>=50 (any combination of holds totaling 2.5s)")
    print("=" * 72)

    s_base = test_one("BASELINE 0.0612", BASELINE, [20.0, 20.0, 40.0, 30.0])
    s_qf50 = test_one("QF50 v2",         QF50V2,   [20.0, 50.0, 40.0, 30.0])

    print("\n" + "=" * 72)
    print(f"  BASELINE: {s_base}/{len(TEST_X0S)}")
    print(f"  QF50  v2: {s_qf50}/{len(TEST_X0S)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
