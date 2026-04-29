"""exp_near_goal.py — Train the LOCAL stabilisation policy directly.

The trace_rollout.py output proved the diagnosis: the network hits the
goal at step 167 (8.35s) but holds for only 74 contiguous steps before
drifting out.  The network has been trained with x0=(0,0,0,0) — it has
NEVER seen the input pattern "5-frame history of states near upright".
So when the pendulum lingers near π, the network's outputs are an
out-of-distribution extrapolation.

Fix: alternate two rollout types each outer iteration:
  1. Canonical swing-up: x0=(0,0,0,0), 220 steps, x_goal=π — preserves
     the breakthrough swing-up policy (anchor against forgetting).
  2. Near-upright hold: x0=(π+δq1, δq1d, δq2, δq2d), 100 steps,
     x_goal=π, FLAT demo at π — teaches the network to handle inputs
     where 5-frame history is "all near upright with various
     velocities".

The "near-upright" demo is constant at π (no swing-up to track).  The
network's existing losses (energy tracking, q_profile_state_phase,
w_f_stable) all naturally drive the rollout TOWARD the goal.
"""

import math
import os
import sys
import time
import copy
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

PRETRAINED = "saved_models/stageD_stabstate_20260428_224856/stageD_stabstate_20260428_224856.pth"

X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS_SU = 220   # swing-up window (anchor)
NUM_STEPS_NG = 150   # near-goal hold window (7.5 s)
DT        = 0.05
NUM_OUTER = 30
LR        = 3e-5
HORIZON   = 10
SAVE_DIR  = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
W_F_STABLE = 50.0

# Perturbation around upright (q1=π, q1d=0, q2=0, q2d=0)
NEAR_PERT = [0.30, 0.50, 0.20, 0.30]


def make_swingup_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


def make_flat_upright_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    demo[:, 0] = math.pi
    return demo


def wrap_pi(x):
    return math.atan2(math.sin(x), math.cos(x))


def wrapped_goal_dist(x_state, x_goal):
    q1_err = wrap_pi(x_state[0] - x_goal[0])
    return math.sqrt(q1_err**2 + x_state[1]**2 + x_state[2]**2 + x_state[3]**2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_goal  = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    x0_zero = torch.zeros(4, device=device, dtype=torch.float64)

    print("=" * 76)
    print("  EXP NEAR-GOAL: alternate swing-up and near-upright training")
    print(f"  Pretrained: {os.path.basename(PRETRAINED)}")
    print(f"  {NUM_OUTER} outer iters: 1 ep swing-up + 1 ep near-upright per iter")
    print(f"  swing-up: {NUM_STEPS_SU} steps  |  near-goal: {NUM_STEPS_NG} steps")
    print(f"  near-goal perturbation around π: {NEAR_PERT}")
    print(f"  LR={LR}  w_f_stable={W_F_STABLE}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0_zero, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # Pre-eval (extensive)
    print(f"\n  Pre-eval:")
    for n in [170, 220, 400, 600, 1000]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=n,
        )
        last = x_t.cpu().numpy()[-1]
        wrp = wrapped_goal_dist(last, X_GOAL)
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        print(f"    {n:>4} steps: raw={raw:.4f}  wrapped={wrp:.4f}")

    rng = np.random.default_rng(42)
    pert = np.array(NEAR_PERT)

    best_state_dict = copy.deepcopy(lin_net.state_dict())
    best_swingup    = float('inf')

    demo_su   = make_swingup_demo(NUM_STEPS_SU, device)
    demo_flat = make_flat_upright_demo(NUM_STEPS_NG, device)

    # Common loss config (matches stab_state)
    base_kwargs = dict(
        lr=LR,
        debug_monitor=None, recorder=None, grad_debug=False,
        track_mode="energy", w_terminal_anchor=0.0,
        w_q_profile=100.0,
        q_profile_pump=[0.01, 0.01, 1.0, 1.0],
        q_profile_stable=[1.0, 1.0, 1.0, 1.0],
        q_profile_state_phase=True,
        w_end_q_high=80.0,
        end_phase_steps=20,
        w_f_stable=W_F_STABLE,
    )

    print(f"\n  {'Iter':>5}  {'q1_p':>6}  {'q1d_p':>6}  {'su_d':>7}  "
          f"{'ng_wrap':>8}  {'su_long':>9}  {'best':>7}")
    print("  " + "-" * 70)

    t0 = time.time()
    for it in range(NUM_OUTER):
        # 1 epoch on swing-up (anchor)
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_zero, x_goal=x_goal, demo=demo_su, num_steps=NUM_STEPS_SU,
            num_epochs=1, **base_kwargs,
        )

        # 1 epoch on near-upright hold (NEW exposure for the network)
        x0_np = np.array([math.pi, 0.0, 0.0, 0.0]) + rng.uniform(-pert, pert, size=4)
        x0_np = x0_np.astype(np.float64)
        x0_t = torch.tensor(x0_np, device=device, dtype=torch.float64)
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_t, x_goal=x_goal, demo=demo_flat, num_steps=NUM_STEPS_NG,
            num_epochs=1, **base_kwargs,
        )

        # Eval: swing-up + near-goal hold
        x_su, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=600,
        )
        su_arr = x_su.cpu().numpy()
        su_d   = float(np.linalg.norm(su_arr[-1] - np.array(X_GOAL)))
        # Sustained hold: longest contiguous wrap < 0.30 in 600-step rollout
        wraps = np.array([wrapped_goal_dist(s, X_GOAL) for s in su_arr])
        in_zone = wraps < 0.3
        runs = []
        start = None
        for i, v in enumerate(in_zone):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i-1)); start = None
        if start is not None:
            runs.append((start, len(in_zone)-1))
        su_long = max((r[1]-r[0]+1 for r in runs), default=0)

        # Near-goal eval
        x_ng, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_t, x_goal=x_goal, num_steps=NUM_STEPS_NG,
        )
        ng_wrap = wrapped_goal_dist(x_ng.cpu().numpy()[-1], X_GOAL)

        # Best metric: prioritise sustained-hold while keeping swing-up
        # Need wrapped(swing-up @ 600) reasonable — derive from wraps.
        su_wrap_600 = float(wraps[-1])
        if su_wrap_600 < 1.0 and su_long > best_swingup:
            best_swingup = su_long
            best_state_dict = copy.deepcopy(lin_net.state_dict())

        if (it + 1) % 1 == 0 or it == 0:
            print(f"  {it+1:>5}  {x0_np[0]-math.pi:>+6.2f}  {x0_np[1]:>+6.2f}  "
                  f"{su_d:>7.4f}  {ng_wrap:>8.4f}  {su_long:>9d}  {best_swingup:>7d}",
                  flush=True)

    elapsed = time.time() - t0
    lin_net.load_state_dict(best_state_dict)

    # SAVE FIRST
    session_name = f"stageD_neargoal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={
            "experiment": "near_goal_alternation",
            "pretrained": PRETRAINED,
            "num_outer": NUM_OUTER,
            "near_pert": NEAR_PERT,
            "best_su_long_steps": best_swingup,
        },
        session_name=session_name,
    )
    print(f"\n  Saved → saved_models/{session_name}/")

    # Post-eval (extensive)
    print(f"\n  Post-eval (canonical swing-up):")
    for n in [170, 220, 300, 400, 600, 1000, 1500]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=n,
        )
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = wrapped_goal_dist(last, X_GOAL)
        status = "STABLE" if wrp < 0.3 else ("CLOSE" if wrp < 1.0 else "FAIL")
        print(f"    {n:>4} steps ({n*DT:>5.1f}s): raw={raw:.4f}  wrapped={wrp:.4f}  {status}")

    # Sustained hold trace on the loaded best
    print(f"\n  Sustained hold (600-step rollout, wrap < 0.3):")
    x_t, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=600,
    )
    arr = x_t.cpu().numpy()
    wraps = np.array([wrapped_goal_dist(s, X_GOAL) for s in arr])
    in_zone = wraps < 0.3
    longest = 0; current = 0
    for v in in_zone:
        current = current + 1 if v else 0
        if current > longest: longest = current
    print(f"    longest contiguous: {longest} steps ({longest*DT:.2f}s)")
    print(f"    total in zone: {int(np.sum(in_zone))} steps ({int(np.sum(in_zone))*DT:.2f}s)")

    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
