"""probe_qf.py — Does tweaking Qf improve stab_state's sustained hold?

The network outputs gates_Q (running cost scale) but NOT Qf (terminal
cost). This asymmetry might cause issues: near upright, gates_Q goes
to 1 (high running Q), but Qf stays at default. Test if bumping Qf
on q1 / q1d helps the network's existing policy hold longer.
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


def wrap_pi(x):
    return math.atan2(math.sin(x), math.cos(x))


def metrics(traj, x_goal, dt):
    """For a (N, 4) numpy trajectory, compute:
       - first arrival step (wrap < 0.3)
       - longest contiguous hold (wrap < 0.3)
       - longest hold STARTING in [160, 230]
       - total time in zone
    """
    wraps = np.array([
        math.sqrt(wrap_pi(s[0] - x_goal[0])**2 + s[1]**2 + s[2]**2 + s[3]**2)
        for s in traj
    ])
    in_zone = wraps < 0.3
    arrive = next((i for i, v in enumerate(in_zone) if v), None)
    runs = []
    start = None
    for i, v in enumerate(in_zone):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i-1)); start = None
    if start is not None:
        runs.append((start, len(in_zone)-1))
    longest = max((r[1]-r[0]+1 for r in runs), default=0)
    longest_in_window = max(
        (r[1]-r[0]+1 for r in runs if 160 <= r[0] <= 230),
        default=0,
    )
    return {
        "arrive": arrive,
        "longest": longest,
        "longest_in_window": longest_in_window,
        "total": int(np.sum(in_zone)),
        "wrap_min": float(wraps.min()),
        "wrap_min_at": int(wraps.argmin()),
    }


def test_qf(qf_diag, x0, x_goal, num_steps=600):
    device = torch.device("cpu")
    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=10, device=device)
    mpc.dt          = torch.tensor(0.05, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor([12.0, 5.0, 50.0, 40.0], dtype=torch.float64)
    mpc.Qf = torch.diag(torch.tensor(qf_diag, dtype=torch.float64))
    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device="cpu").double()
    x_t, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=num_steps,
    )
    return metrics(x_t.cpu().numpy(), x_goal.cpu().numpy(), 0.05)


def main():
    x0     = torch.zeros(4, dtype=torch.float64)
    x_goal = torch.tensor([math.pi, 0., 0., 0.], dtype=torch.float64)

    qf_grid = [
        ("BASELINE q1d=20", [20.0, 20.0, 40.0, 30.0]),
        ("q1d=25",          [20.0, 25.0, 40.0, 30.0]),
        ("q1d=30",          [20.0, 30.0, 40.0, 30.0]),
        ("q1d=35",          [20.0, 35.0, 40.0, 30.0]),
        ("q1d=40",          [20.0, 40.0, 40.0, 30.0]),
        ("q1d=45",          [20.0, 45.0, 40.0, 30.0]),
        ("q1d=50",          [20.0, 50.0, 40.0, 30.0]),
        ("q1d=60",          [20.0, 60.0, 40.0, 30.0]),
        ("q1d=70",          [20.0, 70.0, 40.0, 30.0]),
        ("q1d=80",          [20.0, 80.0, 40.0, 30.0]),
    ]

    print("=" * 90)
    print(f"  {'Qf config':<18}  {'arrive':>6}  {'longest':>7}  {'in_win':>6}  {'total':>5}  {'min_wrap':>8}")
    print("=" * 90)
    for name, qf in qf_grid:
        m = test_qf(qf, x0, x_goal, num_steps=600)
        arr = "—" if m['arrive'] is None else f"{m['arrive']}"
        print(f"  {name:<18}  {arr:>6}  {m['longest']:>7}  "
              f"{m['longest_in_window']:>6}  {m['total']:>5}  {m['wrap_min']:>8.4f}",
              flush=True)


if __name__ == "__main__":
    main()
