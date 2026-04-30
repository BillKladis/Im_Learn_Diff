"""exp_optinit_holdboost.py — HoldBoost gradient training from optimal q1/q1d init.

MOTIVATION:
  exp_q1restore_test.py (direct eval) expected to show that:
    delta_Q[:,0] = ~0.987 (restore gates_Q[q1]  from 0.013 → 1.000)
    delta_Q[:,1] = ~0.987 (restore gates_Q[q1d] from 0.013 → 1.000)
  significantly exceeds the 26.2% ZeroFNet baseline.

  This script:
    1. Initializes delta_Q from (dq0, dq1) given on command line
    2. Evaluates immediately (should reproduce q1restore result)
    3. Fine-tunes with gradient training (near-top x0, all-hold phase)
    4. Reports whether fine-tuning improves further

USAGE:
  python exp_optinit_holdboost.py 0.987 0.987     # best q1restore values
  python exp_optinit_holdboost.py 0.75 0.75       # conservative init
  python exp_optinit_holdboost.py 1.5 0.987       # higher q1 init
"""

import math, os, sys, random, time, signal
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

POSONLY_FINAL = "saved_models/stageD_posonly_ft_20260430_083618/stageD_posonly_ft_20260430_083618.pth"
X0          = [0.0, 0.0, 0.0, 0.0]
X_GOAL      = [math.pi, 0.0, 0.0, 0.0]
DT          = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]; THRESH = 0.8

NUM_STEPS   = 200    # all in hold phase
EPOCHS      = 200
LR          = 5e-3   # moderate LR: init already good, fine-tune carefully
SAVE_EVERY  = 20
SAVE_DIR    = "saved_models"

# Near-top perturbation ranges
X0_PERT_Q1, X0_PERT_V1, X0_PERT_Q2, X0_PERT_V2 = 0.25, 0.5, 0.20, 0.5


class HoldBoostWrapper(nn.Module):
    def __init__(self, lin_net, thresh, dq0=0.0, dq1=0.0, dq2=0.0, dq3=0.0,
                 x_goal_q1=math.pi):
        super().__init__()
        self.lin_net       = lin_net; self.thresh = thresh; self.x_goal_q1 = x_goal_q1
        self.f_extra_bound = lin_net.f_extra_bound; self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim; self.control_dim = lin_net.control_dim
        q_shape = (lin_net.horizon - 1, lin_net.state_dim)
        r_shape = (lin_net.horizon,     lin_net.control_dim)
        dQ_init = torch.zeros(q_shape, dtype=torch.float64)
        dQ_init[:, 0] = dq0; dQ_init[:, 1] = dq1
        dQ_init[:, 2] = dq2; dQ_init[:, 3] = dq3
        self.delta_Q = nn.Parameter(dQ_init)
        self.delta_R = nn.Parameter(torch.zeros(r_shape, dtype=torch.float64))

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf = self.lin_net(
            x_sequence, q_base_diag, r_base_diag
        )
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        f_extra = f_extra * (1.0 - gate.detach())
        gates_Q = gates_Q + gate * self.delta_Q
        gates_R = gates_R + gate * self.delta_R
        return gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf


def sample_near_top_x0(device):
    return torch.tensor([
        math.pi + (random.random() * 2 - 1) * X0_PERT_Q1,
        (random.random() * 2 - 1) * X0_PERT_V1,
        (random.random() * 2 - 1) * X0_PERT_Q2,
        (random.random() * 2 - 1) * X0_PERT_V2,
    ], device=device, dtype=torch.float64)


def eval2k(model, mpc, x0, x_goal, steps=2000):
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([math.sqrt(
        math.atan2(math.sin(s[0]-math.pi), math.cos(s[0]-math.pi))**2
        + s[1]**2 + s[2]**2 + s[3]**2) for s in traj])
    arr = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    return float((wraps < 0.10).mean()), float((wraps < 0.30).mean()), arr, post


def main():
    dq0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.987
    dq1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.987
    dq2 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    dq3 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    print("=" * 80)
    print(f"  EXP OPTINIT-HOLDBOOST: init delta_Q = [{dq0:.3f}, {dq1:.3f}, {dq2:.3f}, {dq3:.3f}]")
    print(f"  Rationale: gates_Q[q1/q1d]≈0.013 at top → dq0={dq0:.3f} restores to {0.013+dq0:.3f}")
    print(f"  LR={LR}  NUM_STEPS={NUM_STEPS}  EPOCHS={EPOCHS}  thresh={THRESH}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.requires_grad_(False)

    boost = HoldBoostWrapper(lin_net, thresh=THRESH, dq0=dq0, dq1=dq1, dq2=dq2, dq3=dq3)

    # Eval with fixed init (should match exp_q1restore_test result)
    print(f"\n  Initial eval (delta_Q=[{dq0:.3f},{dq1:.3f},{dq2:.3f},{dq3:.3f}] fixed):")
    f01, f03, arr, post = eval2k(boost, mpc, x0, x_goal, steps=2000)
    print(f"    frac<0.10={f01:.1%}  frac<0.30={f03:.1%}  arr={arr}  "
          f"post={f'{post:.1%}' if post else 'N/A'}  [baseline=26.2%]", flush=True)

    best_f01 = f01
    best_dQ = boost.delta_Q.data.clone()
    best_dR = boost.delta_R.data.clone()

    # Fine-tune from this init
    session_name = (f"stageD_optinit_holdboost_dq{dq0:.2f}x{dq1:.2f}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    optimizer = torch.optim.AdamW([boost.delta_Q, boost.delta_R], lr=LR, weight_decay=1e-4)
    all_losses = []
    t0 = time.time()

    interrupted = [False]
    signal.signal(signal.SIGINT, lambda s, f: interrupted.__setitem__(0, True))

    print(f"\n  Fine-tuning from init (LR={LR}, near-top x0, all-hold-phase)...")
    chunk_start = 0
    while chunk_start < EPOCHS and not interrupted[0]:
        n_ep = min(SAVE_EVERY, EPOCHS - chunk_start)
        x0_train = sample_near_top_x0(device)
        demo = torch.zeros((NUM_STEPS, 4), dtype=torch.float64, device=device)
        demo[:, 0] = math.pi

        loss_chunk, _ = train_module.train_linearization_network(
            lin_net=boost, mpc=mpc,
            x0=x0_train, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=n_ep, lr=LR,
            debug_monitor=None, recorder=network_module.NetworkOutputRecorder(),
            track_mode="phase_aware", phase_split_frac=0.0,
            w_terminal_anchor=0.0, w_q_profile=0.0, w_f_pos_only=0.0,
            w_stable_phase=0.0, f_gate_thresh=0.0,
            w_hold_reward=0.0, hold_sigma=0.5, hold_start_step=0,
            early_stop_patience=n_ep + 5,
            external_optimizer=optimizer, restore_best=False,
        )
        all_losses.extend(loss_chunk)
        chunk_start += n_ep

        f01, f03, arr, post = eval2k(boost, mpc, x0, x_goal, steps=2000)
        dQ_vals = boost.delta_Q.data.mean(0).tolist()
        mark = ""
        if f01 > best_f01:
            mark = " ★"
            best_f01 = f01
            best_dQ = boost.delta_Q.data.clone()
            best_dR = boost.delta_R.data.clone()

        print(f"  [ep={chunk_start:3d}]  {f01:.1%}  frac<0.30={f03:.1%}  arr={arr}  "
              f"post={f'{post:.1%}' if post else 'N/A'}  "
              f"dQ_mean=[{dQ_vals[0]:.3f},{dQ_vals[1]:.3f},{dQ_vals[2]:.3f},{dQ_vals[3]:.3f}]"
              f"  t={time.time()-t0:.0f}s{mark}", flush=True)

        if chunk_start % (2 * SAVE_EVERY) == 0:
            ckpt = f"{session_name}_ep{chunk_start:03d}"
            network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
                model=lin_net, loss_history=all_losses,
                training_params={
                    "experiment": "optinit_holdboost",
                    "init_dq0": dq0, "init_dq1": dq1,
                    "best_frac01": best_f01,
                    "best_delta_Q": best_dQ.tolist(),
                    "best_delta_R": best_dR.tolist(),
                    "checkpoint_epoch": chunk_start,
                },
                session_name=ckpt,
            )

        if best_f01 > 0.5:
            print("  EXCELLENT HOLD — stopping.")
            break

    boost.delta_Q.data.copy_(best_dQ)
    boost.delta_R.data.copy_(best_dR)

    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=all_losses,
        training_params={
            "experiment": "optinit_holdboost_FINAL",
            "init_dq0": dq0, "init_dq1": dq1,
            "best_frac01_2000step": best_f01,
            "best_delta_Q": best_dQ.tolist(), "best_delta_R": best_dR.tolist(),
        },
        session_name=session_name,
    )

    print(f"\n  Init result: frac<0.10={f01:.1%}  →  Best fine-tuned: {best_f01:.1%}")
    print(f"  ZeroFNet baseline: 26.2%  |  Init dq0={dq0:.3f},dq1={dq1:.3f}")
    print(f"  Best dQ mean per dim: {best_dQ.mean(0).tolist()}")
    if best_f01 > 0.262:
        print(f"  ★★★ IMPROVEMENT: {best_f01:.1%} > 26.2% ★★★")
    else:
        print(f"  No improvement — q1 restoration doesn't help with this Q_BASE formula")


if __name__ == "__main__":
    main()
