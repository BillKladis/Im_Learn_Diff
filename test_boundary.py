"""test_boundary.py — find qf50 v2's perturbation boundary.

We've established 12/12 on q1∈[-0.5,0.5], q1d∈[-1,1]. Now push:
  q1: ±0.7, ±0.9, ±1.1, ±1.3, ±1.5
  q1d: ±1.5, ±2.0, ±2.5, ±3.0
  q2/q2d: wider too
  combinations
"""

import math, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

PRETRAINED = "saved_models/stageD_nodemo_qf50_20260429_111711/stageD_nodemo_qf50_20260429_111711.pth"
QF_DIAG = [20.0, 50.0, 40.0, 30.0]


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


def main():
    device = torch.device("cpu")
    x_goal = torch.tensor([math.pi, 0., 0., 0.], dtype=torch.float64)
    mpc = mpc_module.MPC_controller(x0=torch.zeros(4, dtype=torch.float64),
                                    x_goal=x_goal, N=10, device=device)
    mpc.dt = torch.tensor(0.05, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor([12., 5., 50., 40.], dtype=torch.float64)
    mpc.Qf = torch.diag(torch.tensor(QF_DIAG, dtype=torch.float64))
    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device="cpu").double()

    print("=" * 80)
    print(f"  qf50 v2 BOUNDARY TEST")
    print(f"  Threshold: total time in wrap<0.3 zone >= 50 (= OK)")
    print("=" * 80)

    # Wider q1 sweep
    print(f"\n  {'x0':<22}  {'arr':>5}  {'long':>5}  {'total':>5}  {'verdict':>8}")
    print("  " + "─" * 60)

    test_x0s = []
    # q1 sweep (already know ±0.5 works)
    for v in [-1.5, -1.3, -1.1, -0.9, -0.7, -0.5, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
        test_x0s.append((f"q1={v:+.1f}", [v, 0, 0, 0]))
    # q1d sweep (already know ±1.0 works)
    for v in [-3.0, -2.5, -2.0, -1.5, 1.5, 2.0, 2.5, 3.0]:
        test_x0s.append((f"q1d={v:+.1f}", [0, v, 0, 0]))
    # q2 wider
    for v in [-0.3, -0.2, 0.2, 0.3]:
        test_x0s.append((f"q2={v:+.1f}", [0, 0, v, 0]))
    # q2d wider
    for v in [-1.0, -0.7, 0.7, 1.0, 1.5]:
        test_x0s.append((f"q2d={v:+.1f}", [0, 0, 0, v]))
    # Tougher combinations
    for q1, q1d in [(0.5, 1.0), (-0.5, -1.0), (0.7, 0.5), (-0.7, -0.5), (0.3, 1.5), (-0.3, -1.5)]:
        test_x0s.append((f"q1={q1:+.1f},q1d={q1d:+.1f}", [q1, q1d, 0, 0]))

    succ = 0
    for label, x0_list in test_x0s:
        x0_t = torch.tensor(x0_list, dtype=torch.float64)
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_t, x_goal=x_goal, num_steps=1000,
        )
        arr, lng, tot = metrics(x_t.cpu().numpy(), x_goal.cpu().numpy())
        ok = "OK" if tot >= 50 else ("WEAK" if tot > 0 else "FAIL")
        if tot >= 50: succ += 1
        arr_str = "—" if arr is None else f"{arr}"
        print(f"  {label:<22}  {arr_str:>5}  {lng:>5}  {tot:>5}     {ok}", flush=True)

    print(f"\n  Boundary success: {succ} / {len(test_x0s)}")


if __name__ == "__main__":
    main()
