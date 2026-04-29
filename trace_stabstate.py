"""trace_qf50.py — does stab_state actually STAY at the top?

The 35/35 boundary metric uses 'total ≥ 50 in zone' which is just
total time near goal. Doesn't distinguish between:
  - Pendulum oscillating in/out of loose zone (touching it 50 times briefly)
  - Pendulum actually settling and staying

Trace the wrap distance over 2000 steps to see what's REALLY happening.
"""

import math, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

PRETRAINED = "saved_models/stageD_stabstate_20260428_224856/stageD_stabstate_20260428_224856.pth"
QF_DIAG = [20.0, 20.0, 40.0, 30.0]


def wrap_pi(x): return math.atan2(math.sin(x), math.cos(x))


def main():
    device = torch.device("cpu")
    x_goal = torch.tensor([math.pi, 0., 0., 0.], dtype=torch.float64)
    x0 = torch.zeros(4, dtype=torch.float64)
    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=10, device=device)
    mpc.dt = torch.tensor(0.05, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor([12., 5., 50., 40.], dtype=torch.float64)
    mpc.Qf = torch.diag(torch.tensor(QF_DIAG, dtype=torch.float64))
    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device="cpu").double()

    print(f"Tracing stab_state from x0=zero for 2000 steps")
    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=2000)
    traj = x_t.cpu().numpy()

    wraps = np.array([
        math.sqrt(wrap_pi(s[0]-x_goal[0].item())**2 + s[1]**2 + s[2]**2 + s[3]**2)
        for s in traj
    ])

    # Histogram of wrap distance over time after arrival
    arr_idx = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    print(f"\nFirst arrival (wrap<0.3): step {arr_idx}  ({arr_idx*0.05:.2f}s)")

    if arr_idx is not None:
        post = wraps[arr_idx:]
        print(f"\nWrap-distance stats over post-arrival ({len(post)} steps = {len(post)*0.05:.1f}s):")
        print(f"  mean: {post.mean():.4f}")
        print(f"  median: {np.median(post):.4f}")
        print(f"  min: {post.min():.4f}")
        print(f"  max: {post.max():.4f}")
        print(f"  std: {post.std():.4f}")

        # Time spent in different zones
        for thresh in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
            n = int(np.sum(post < thresh))
            print(f"  fraction wrap<{thresh:.2f}: {n/len(post):.1%}  ({n} steps)")

        # Snapshot trajectory
        print(f"\nSnapshot of wrap distance every 50 steps post-arrival:")
        for i in range(0, len(post), 50):
            t = (arr_idx + i) * 0.05
            print(f"  step {arr_idx+i:>4} ({t:>5.1f}s): wrap={post[i]:.4f}  q1={traj[arr_idx+i, 0]:.3f}  q1d={traj[arr_idx+i, 1]:+.3f}")

    print(f"\nOverall wrap-distance stats (entire 2000 steps):")
    print(f"  fraction wrap<0.1: {(wraps < 0.1).mean():.1%}")
    print(f"  fraction wrap<0.3: {(wraps < 0.3).mean():.1%}")
    print(f"  fraction wrap<1.0: {(wraps < 1.0).mean():.1%}")


if __name__ == "__main__":
    main()
