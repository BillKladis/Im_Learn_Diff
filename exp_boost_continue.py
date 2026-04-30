"""exp_boost_continue.py — Continue gradient training from best saved delta_Q.

MOTIVATION:
  exp_optinit_holdboost.py reached 82.9% frac<0.10 after 20 epochs and stopped
  (EXCELLENT_HOLD threshold 0.50). This script continues from that delta_Q to
  push further: higher excellent-hold bar (0.90), more epochs, diverse x0 seeds.

USAGE:
  python exp_boost_continue.py                      # loads best checkpoint auto
  python exp_boost_continue.py --from_ckpt PATH     # explicit checkpoint
  python exp_boost_continue.py --epochs 400         # override epoch count
"""

import argparse, copy, math, os, random, signal, sys, time
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
BEST_CKPT     = "saved_models/stageD_optinit_holdboost_dq0.99x0.99_20260430_165519/stageD_optinit_holdboost_dq0.99x0.99_20260430_165519.pth"
X0            = [0.0, 0.0, 0.0, 0.0]
X_GOAL        = [math.pi, 0.0, 0.0, 0.0]
DT            = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]; THRESH = 0.8
NUM_STEPS     = 200; LR = 2e-3; SAVE_EVERY = 20; EXCELLENT_HOLD = 0.90
X0_PERT_Q1, X0_PERT_V1, X0_PERT_Q2, X0_PERT_V2 = 0.25, 0.5, 0.20, 0.5


class HoldBoostWrapper(nn.Module):
    def __init__(self, lin_net, thresh, dQ_init, dR_init=None, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net = lin_net; self.thresh = thresh; self.x_goal_q1 = x_goal_q1
        self.f_extra_bound = lin_net.f_extra_bound; self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim; self.control_dim = lin_net.control_dim
        self.delta_Q = nn.Parameter(dQ_init.clone().double())
        r_shape = (lin_net.horizon, lin_net.control_dim)
        self.delta_R = nn.Parameter(
            dR_init.clone().double() if dR_init is not None
            else torch.zeros(r_shape, dtype=torch.float64)
        )

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        fe = fe * (1.0 - gate.detach())
        gQ = gQ + gate * self.delta_Q
        gR = gR + gate * self.delta_R
        return gQ, gR, fe, qd, rd, gQf


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


def load_best_dQ(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    meta = ckpt.get('metadata', {})
    tp = meta.get('training_params', {})
    dQ = tp.get('best_delta_Q')
    dR = tp.get('best_delta_R')
    if dQ is None:
        raise ValueError(f"No best_delta_Q in {ckpt_path}")
    return torch.tensor(dQ, dtype=torch.float64), torch.tensor(dR, dtype=torch.float64) if dR else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_ckpt", default=BEST_CKPT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    dQ_init, dR_init = load_best_dQ(args.from_ckpt)

    print("=" * 80)
    print(f"  EXP BOOST-CONTINUE: gradient training from best delta_Q checkpoint")
    print(f"  Init delta_Q mean: {dQ_init.mean(0).tolist()}")
    print(f"  LR={args.lr}  NUM_STEPS={NUM_STEPS}  EPOCHS={args.epochs}  thresh={THRESH}")
    print(f"  EXCELLENT_HOLD threshold: {EXCELLENT_HOLD:.0%}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.requires_grad_(False)

    boost = HoldBoostWrapper(lin_net, thresh=THRESH, dQ_init=dQ_init, dR_init=dR_init)

    print(f"\n  Initial eval (loaded from checkpoint):")
    f01, f03, arr, post = eval2k(boost, mpc, x0, x_goal)
    print(f"    frac<0.10={f01:.1%}  frac<0.30={f03:.1%}  arr={arr}  "
          f"post={f'{post:.1%}' if post else 'N/A'}  [prev best=82.9%]", flush=True)

    best_f01 = f01
    best_dQ = boost.delta_Q.data.clone()
    best_dR = boost.delta_R.data.clone()

    session_name = f"stageD_boost_continue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    optimizer = torch.optim.AdamW([boost.delta_Q, boost.delta_R], lr=args.lr, weight_decay=1e-5)
    all_losses = []
    t0 = time.time()

    interrupted = [False]
    signal.signal(signal.SIGINT, lambda s, f: interrupted.__setitem__(0, True))

    print(f"\n  Training (LR={args.lr}, diverse near-top x0, all-hold-phase)...")
    chunk_start = 0
    while chunk_start < args.epochs and not interrupted[0]:
        n_ep = min(SAVE_EVERY, args.epochs - chunk_start)
        x0_train = sample_near_top_x0(device)
        demo = torch.zeros((NUM_STEPS, 4), dtype=torch.float64, device=device)
        demo[:, 0] = math.pi

        loss_chunk, _ = train_module.train_linearization_network(
            lin_net=boost, mpc=mpc,
            x0=x0_train, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=n_ep, lr=args.lr,
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

        f01, f03, arr, post = eval2k(boost, mpc, x0, x_goal)
        dQ_vals = boost.delta_Q.data.mean(0).tolist()
        mark = ""
        if f01 > best_f01:
            mark = " ★"
            best_f01 = f01
            best_dQ = boost.delta_Q.data.clone()
            best_dR = boost.delta_R.data.clone()

        print(f"  [ep={chunk_start:3d}]  {f01:.1%}  frac<0.30={f03:.1%}  arr={arr}  "
              f"post={f'{post:.1%}' if post else 'N/A'}  "
              f"dQ=[{dQ_vals[0]:.3f},{dQ_vals[1]:.3f},{dQ_vals[2]:.3f},{dQ_vals[3]:.3f}]"
              f"  t={time.time()-t0:.0f}s{mark}", flush=True)

        if chunk_start % (2 * SAVE_EVERY) == 0:
            ckpt = f"{session_name}_ep{chunk_start:03d}"
            network_module.ModelManager(base_dir="saved_models").save_training_session(
                model=lin_net, loss_history=all_losses,
                training_params={
                    "experiment": "boost_continue",
                    "best_frac01": best_f01,
                    "best_delta_Q": best_dQ.tolist(),
                    "best_delta_R": best_dR.tolist(),
                    "checkpoint_epoch": chunk_start,
                },
                session_name=ckpt,
            )

        if best_f01 >= EXCELLENT_HOLD:
            print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
            break

    boost.delta_Q.data.copy_(best_dQ)
    boost.delta_R.data.copy_(best_dR)

    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net, loss_history=all_losses,
        training_params={
            "experiment": "boost_continue_FINAL",
            "best_frac01_2000step": best_f01,
            "best_delta_Q": best_dQ.tolist(),
            "best_delta_R": best_dR.tolist(),
        },
        session_name=session_name,
    )

    print(f"\n  Init: {f01:.1%}  →  Best: {best_f01:.1%}")
    print(f"  Best dQ mean: {best_dQ.mean(0).tolist()}")
    print(f"  Best dR mean: {best_dR.mean(0).tolist()}")
    print(f"  ZeroFNet baseline: 26.2%  |  prev best: 82.9%")
    if best_f01 > 0.829:
        print(f"  ★★★ NEW RECORD: {best_f01:.1%} ★★★")


if __name__ == "__main__":
    main()
