"""exp_traj_curriculum.py — Curriculum along a SUCCESSFUL trajectory,
with proper 5-frame history seeding.

Refinement on the user's idea: when picking x0 = state[t] from the
reference trajectory, also seed the network's 5-frame history with
states {t-4, t-3, t-2, t-1, t} from the same trajectory.  This way
the network's input is exactly what it would have seen at step t in
a real rollout — fully in-distribution.

Pipeline:
  1. Run reference 600-step rollout from the pretrained model.
  2. Export trajectory to CSV (saved next to the output checkpoint).
  3. Stage 1: pick t ∈ [arrival-40, arrival-15], 80-step holds with
     init_history = traj[t-4..t].
  4. Stage 2: t ∈ [arrival-80, arrival-30], 140-step holds.
  5. Stage 3: t ∈ [arrival-150, arrival-60], 200-step holds (further
     from goal, longer hold).
  6. Anchor: each iter, also do 1 ep canonical x0=zero swing-up.
"""

import math
import os
import sys
import time
import copy
import csv
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
LR        = 1e-4   # bumped from 3e-5 — at 1 epoch/iter, 3e-5 was too small to move the network

STAGE1_OUTER = 15
STAGE2_OUTER = 15
STAGE3_OUTER = 10


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


class LossCapture:
    """Captures loss + final goal_dist from train_linearization_network's
    debug_monitor callback so we can verify the gradient is doing work."""
    def __init__(self):
        self.last_loss = float('nan')
        self.last_goal_dist = float('nan')
        self.last_fnorm = float('nan')
    def log_epoch(self, epoch, num_epochs, loss, info):
        self.last_loss      = float(loss)
        self.last_goal_dist = float(info.get('pure_end_error', float('nan')))
        self.last_fnorm     = float(info.get('mean_f_extra_norm', float('nan')))


def make_init_from_traj(traj, t, device):
    """Pick traj[t] as x0 and traj[t-4..t] as the 5-frame init history.

    Returns (x0, init_history) where init_history is shape (5, 4).
    """
    if t < 4:
        raise ValueError(f"Need t >= 4 to build a 5-frame history (got t={t})")
    history_np = traj[t-4:t+1]                    # shape (5, 4)
    init_history = torch.tensor(history_np, device=device, dtype=torch.float64)
    x0 = init_history[-1].clone()
    return x0, init_history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_goal  = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    x0_zero = torch.zeros(4, device=device, dtype=torch.float64)

    print("=" * 76)
    print("  EXP TRAJ-CURRICULUM v2: x0 + 5-frame history from reference trajectory")
    print(f"  Pretrained: {os.path.basename(PRETRAINED)}")
    print(f"  Stages: S1={STAGE1_OUTER}  S2={STAGE2_OUTER}  S3={STAGE3_OUTER}")
    print(f"  LR={LR}  w_f_stable={W_F_STABLE}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0_zero, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # 1. Generate reference trajectory
    print(f"\n  Generating reference trajectory (600 steps, x0=zero)...")
    x_ref, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=600,
    )
    traj = x_ref.cpu().numpy()
    wraps = np.array([wrapped_goal_dist(s, X_GOAL) for s in traj])
    arrival = int(np.where(wraps < 0.3)[0][0]) if (wraps < 0.3).any() else 167
    print(f"  Goal-arrival step (wrap<0.3): {arrival}  ({arrival*DT:.2f}s)")
    print(f"  Min wrap: {wraps.min():.3f} @ step {wraps.argmin()}")

    # 2. Export trajectory to CSV
    session_name = f"stageD_trajcurr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(SAVE_DIR, session_name)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "reference_trajectory.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "time", "q1", "q1d", "q2", "q2d", "wrap_dist"])
        for i, s in enumerate(traj):
            w.writerow([i, f"{i*DT:.3f}", f"{s[0]:.6f}", f"{s[1]:.6f}",
                        f"{s[2]:.6f}", f"{s[3]:.6f}", f"{wraps[i]:.6f}"])
    print(f"  Trajectory exported to {csv_path}")

    # Pre-eval (extensive)
    print(f"\n  Pre-eval (canonical x0=zero):")
    for n in [170, 220, 400, 600, 1000]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=n,
        )
        last = x_t.cpu().numpy()[-1]
        wrp = wrapped_goal_dist(last, X_GOAL)
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        print(f"    {n:>4} steps: raw={raw:.4f}  wrapped={wrp:.4f}")

    cap_anc = LossCapture()
    cap_cur = LossCapture()
    base_kwargs_anc = dict(
        lr=LR, debug_monitor=cap_anc, recorder=None, grad_debug=False,
        track_mode="energy", w_terminal_anchor=0.0,
        w_q_profile=100.0,
        q_profile_pump=[0.01, 0.01, 1.0, 1.0],
        q_profile_stable=[1.0, 1.0, 1.0, 1.0],
        q_profile_state_phase=True,
        w_end_q_high=80.0, end_phase_steps=20,
        w_f_stable=W_F_STABLE,
    )
    base_kwargs_cur = {**base_kwargs_anc, "debug_monitor": cap_cur}

    rng = np.random.default_rng(123)
    best_state_dict = copy.deepcopy(lin_net.state_dict())
    best_long = 0

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
        su_arrived = bool((wr[160:230] < 0.3).any())
        if su_arrived and long > best_long:
            best_long = long
            best_state_dict = copy.deepcopy(lin_net.state_dict())
        print(f"  [{label} {it+1}] su_ok={'Y' if su_arrived else 'N'}  "
              f"long={long}({long*DT:.1f}s)  wrap@600={end_wrap:.3f}  best={best_long}  "
              f"L_anc={cap_anc.last_loss:.3f}  L_cur={cap_cur.last_loss:.3f}  "
              f"f={cap_cur.last_fnorm:.3f}",
              flush=True)

    demo_su = make_swingup_demo(220, device)
    t0 = time.time()

    # ------------------- STAGE 1 -------------------
    s1_lo = max(arrival - 20, 4); s1_hi = max(arrival - 5, 5)
    print(f"\n  STAGE 1: x0 from steps [{s1_lo}, {s1_hi}]  hold=80 steps")
    demo_flat_80 = make_flat_upright_demo(80, device)
    for it in range(STAGE1_OUTER):
        # Anchor swing-up
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_zero, x_goal=x_goal, demo=demo_su, num_steps=220,
            num_epochs=1, **base_kwargs_anc,
        )
        # Curriculum: trajectory state with seeded history
        t = int(rng.integers(s1_lo, s1_hi))
        x0_pick, hist = make_init_from_traj(traj, t, device)
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_pick, x_goal=x_goal, demo=demo_flat_80, num_steps=80,
            num_epochs=1, init_history=hist, **base_kwargs_cur,
        )
        if (it+1) % 2 == 0 or it == 0:
            eval_and_maybe_save("S1", it)

    # ------------------- STAGE 2 -------------------
    s2_lo = max(arrival - 50, 4); s2_hi = max(arrival - 20, 5)
    print(f"\n  STAGE 2: x0 from steps [{s2_lo}, {s2_hi}]  hold=140 steps")
    demo_flat_140 = make_flat_upright_demo(140, device)
    for it in range(STAGE2_OUTER):
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_zero, x_goal=x_goal, demo=demo_su, num_steps=220,
            num_epochs=1, **base_kwargs_anc,
        )
        t = int(rng.integers(s2_lo, s2_hi))
        x0_pick, hist = make_init_from_traj(traj, t, device)
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_pick, x_goal=x_goal, demo=demo_flat_140, num_steps=140,
            num_epochs=1, init_history=hist, **base_kwargs_cur,
        )
        if (it+1) % 2 == 0 or it == 0:
            eval_and_maybe_save("S2", it)

    # ------------------- STAGE 3 -------------------
    s3_lo = max(arrival - 150, 4); s3_hi = max(arrival - 60, 5)
    print(f"\n  STAGE 3: x0 from steps [{s3_lo}, {s3_hi}]  hold=200 steps")
    demo_flat_200 = make_flat_upright_demo(200, device)
    for it in range(STAGE3_OUTER):
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_zero, x_goal=x_goal, demo=demo_su, num_steps=220,
            num_epochs=1, **base_kwargs_anc,
        )
        t = int(rng.integers(s3_lo, s3_hi))
        x0_pick, hist = make_init_from_traj(traj, t, device)
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_pick, x_goal=x_goal, demo=demo_flat_200, num_steps=200,
            num_epochs=1, init_history=hist, **base_kwargs_cur,
        )
        eval_and_maybe_save("S3", it)

    elapsed = time.time() - t0
    lin_net.load_state_dict(best_state_dict)

    # SAVE
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={
            "experiment": "trajectory_curriculum_v2_history_seeded",
            "pretrained": PRETRAINED,
            "stages": [STAGE1_OUTER, STAGE2_OUTER, STAGE3_OUTER],
            "best_long_steps": best_long,
            "arrival_step": arrival,
        },
        session_name=session_name,
    )
    print(f"\n  Saved → {out_dir}/")

    # Post-eval
    print(f"\n  Post-eval (canonical x0=zero):")
    for n in [170, 220, 300, 400, 600, 1000, 1500]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=n,
        )
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = wrapped_goal_dist(last, X_GOAL)
        status = "STABLE" if wrp < 0.3 else ("CLOSE" if wrp < 1.0 else "FAIL")
        print(f"    {n:>4} steps ({n*DT:>5.1f}s): raw={raw:.4f}  wrapped={wrp:.4f}  {status}")

    # Sustained-hold
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
