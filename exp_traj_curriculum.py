"""exp_traj_curriculum.py — Curriculum along a SUCCESSFUL trajectory.

Idea (from user, after diagnosing that random near-upright x0s give the
network 5-frame histories that aren't on the physically-reached
manifold): pick initial conditions from a known-good rollout instead.

  1. Run a 600-step rollout with the stab_state model from x0=zero.
  2. Save every state along that trajectory.
  3. Stage 1: pick x0 from states 30-40 steps before goal-arrival —
     train short rollouts (60 steps) from there. Network learns to hold
     starting from "nearly there with the right energy".
  4. Stage 2: extend the window to ~80 steps before arrival, training
     longer (120-step) rollouts.
  5. Stage 3: full swing-up from x0=zero (anchor).

States from a real swing-up are physically reachable, have the correct
energy, and the 5-frame history matches the network's training
distribution — much more in-distribution than (π+δ, ε, δ, ε) noise.
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
DT        = 0.05
HORIZON   = 10
SAVE_DIR  = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
W_F_STABLE = 50.0
LR        = 3e-5

# Curriculum stages
STAGE1_OUTER = 12   # pick x0 within 30-40 steps of arrival, 80-step rollout
STAGE2_OUTER = 12   # within 80 steps of arrival, 140-step rollout
STAGE3_OUTER = 6    # full swing-up (anchor)


def make_flat_upright_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    demo[:, 0] = math.pi
    return demo


def make_swingup_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
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
    print("  EXP TRAJ-CURRICULUM: pick x0 from a successful trajectory")
    print(f"  Pretrained: {os.path.basename(PRETRAINED)}")
    print(f"  Stages: S1={STAGE1_OUTER} (near-arrival)  "
          f"S2={STAGE2_OUTER} (mid-swingup)  S3={STAGE3_OUTER} (full)")
    print(f"  LR={LR}  w_f_stable={W_F_STABLE}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0_zero, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # 1. Generate the SUCCESSFUL trajectory (from current best checkpoint)
    print(f"\n  Generating reference trajectory (600 steps from x0=zero)...")
    x_ref, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=600,
    )
    traj = x_ref.cpu().numpy()  # (601, 4)
    wraps = np.array([wrapped_goal_dist(s, X_GOAL) for s in traj])
    arrival = int(np.where(wraps < 0.3)[0][0]) if (wraps < 0.3).any() else 167
    print(f"  Goal-arrival step (wrap<0.3): {arrival}")
    print(f"  Trajectory wrap stats: min={wraps.min():.3f}  "
          f"@step {wraps.argmin()}  ({wraps.argmin()*DT:.2f}s)")

    # Pre-eval (extensive)
    print(f"\n  Pre-eval (canonical):")
    for n in [170, 220, 400, 600, 1000]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=n,
        )
        last = x_t.cpu().numpy()[-1]
        wrp = wrapped_goal_dist(last, X_GOAL)
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        print(f"    {n:>4} steps: raw={raw:.4f}  wrapped={wrp:.4f}")

    base_kwargs = dict(
        lr=LR, debug_monitor=None, recorder=None, grad_debug=False,
        track_mode="energy", w_terminal_anchor=0.0,
        w_q_profile=100.0,
        q_profile_pump=[0.01, 0.01, 1.0, 1.0],
        q_profile_stable=[1.0, 1.0, 1.0, 1.0],
        q_profile_state_phase=True,
        w_end_q_high=80.0, end_phase_steps=20,
        w_f_stable=W_F_STABLE,
    )

    rng = np.random.default_rng(123)
    best_state_dict = copy.deepcopy(lin_net.state_dict())
    best_long = 0  # longest contiguous hold

    def eval_and_maybe_save(label, it):
        nonlocal best_long, best_state_dict
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=600,
        )
        arr = x_t.cpu().numpy()
        wr  = np.array([wrapped_goal_dist(s, X_GOAL) for s in arr])
        in_zone = wr < 0.3
        long = 0; cur = 0
        for v in in_zone:
            cur = cur + 1 if v else 0
            if cur > long: long = cur
        end_wrap = float(wr[-1])
        # Save when the swing-up still arrives AND hold improves
        if (wr[170:230] < 0.3).any() and long > best_long:
            best_long = long
            best_state_dict = copy.deepcopy(lin_net.state_dict())
        print(f"  [{label} {it+1}] su_arrived={'Y' if (wr[170:230]<0.3).any() else 'N'}  "
              f"long={long}({long*DT:.1f}s)  wrap@600={end_wrap:.3f}  best_long={best_long}",
              flush=True)

    t0 = time.time()

    # ---------------- STAGE 1: x0 from {arrival-40, ..., arrival-15} ----
    print(f"\n  STAGE 1: x0 within 30-40 steps of arrival, 80-step holds")
    s1_lo = max(arrival - 40, 0); s1_hi = max(arrival - 15, 1)
    demo_flat_80 = make_flat_upright_demo(80, device)
    demo_su      = make_swingup_demo(220, device)
    for it in range(STAGE1_OUTER):
        # Anchor: 1 ep canonical swing-up
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_zero, x_goal=x_goal, demo=demo_su, num_steps=220,
            num_epochs=1, **base_kwargs,
        )
        # Curriculum: pick a state from the trajectory near arrival
        idx = int(rng.integers(s1_lo, s1_hi))
        x0_pick = torch.tensor(traj[idx], device=device, dtype=torch.float64)
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_pick, x_goal=x_goal, demo=demo_flat_80, num_steps=80,
            num_epochs=1, **base_kwargs,
        )
        if (it+1) % 2 == 0 or it == 0:
            eval_and_maybe_save("S1", it)

    # ---------------- STAGE 2: x0 from {arrival-80, ..., arrival-30} ---
    print(f"\n  STAGE 2: x0 within 30-80 steps of arrival, 140-step holds")
    s2_lo = max(arrival - 80, 0); s2_hi = max(arrival - 30, 1)
    demo_flat_140 = make_flat_upright_demo(140, device)
    for it in range(STAGE2_OUTER):
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_zero, x_goal=x_goal, demo=demo_su, num_steps=220,
            num_epochs=1, **base_kwargs,
        )
        idx = int(rng.integers(s2_lo, s2_hi))
        x0_pick = torch.tensor(traj[idx], device=device, dtype=torch.float64)
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_pick, x_goal=x_goal, demo=demo_flat_140, num_steps=140,
            num_epochs=1, **base_kwargs,
        )
        if (it+1) % 2 == 0 or it == 0:
            eval_and_maybe_save("S2", it)

    # ---------------- STAGE 3: full canonical swing-up only -----------
    print(f"\n  STAGE 3: full canonical swing-up consolidation")
    for it in range(STAGE3_OUTER):
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_zero, x_goal=x_goal, demo=demo_su, num_steps=220,
            num_epochs=1, **base_kwargs,
        )
        eval_and_maybe_save("S3", it)

    elapsed = time.time() - t0
    lin_net.load_state_dict(best_state_dict)

    # SAVE
    session_name = f"stageD_trajcurr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={
            "experiment": "trajectory_curriculum",
            "pretrained": PRETRAINED,
            "stages": [STAGE1_OUTER, STAGE2_OUTER, STAGE3_OUTER],
            "best_long_steps": best_long,
        },
        session_name=session_name,
    )
    print(f"\n  Saved → saved_models/{session_name}/")

    # Final post-eval (extensive)
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

    # Sustained hold
    x_t, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=1000,
    )
    arr = x_t.cpu().numpy()
    wr  = np.array([wrapped_goal_dist(s, X_GOAL) for s in arr])
    in_zone = wr < 0.3
    long = 0; cur = 0
    for v in in_zone:
        cur = cur + 1 if v else 0
        if cur > long: long = cur
    print(f"\n  Sustained hold (1000-step rollout, wrap < 0.3):")
    print(f"    longest contiguous: {long} steps ({long*DT:.2f}s)")
    print(f"    total in zone: {int(np.sum(in_zone))} steps ({int(np.sum(in_zone))*DT:.2f}s)")

    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
