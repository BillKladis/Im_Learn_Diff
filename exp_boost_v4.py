"""exp_boost_v4.py — Boost training with delta_Qf (terminal cost learning).

NEW AXIS vs boost_v2/v3:
  Adds a learnable delta_Qf correction to the terminal cost matrix near the top.
  Since diag_corrections_Qf is MULTIPLICATIVE (Qf.diag() × gates_Qf), we learn:
    gates_Qf = 1.0 + gate * delta_Qf   (shape: state_dim=4)
  This lets the MPC adjust its terminal planning near [π,0,0,0].

  Combined with existing delta_Q (stage cost) and delta_R corrections, this gives
  3 learnable tensors: delta_Q (9×4), delta_R (10×2), delta_Qf (4,).

USAGE:
  python exp_boost_v4.py                    # starts from best checkpoint
  python exp_boost_v4.py --epochs 300
  python exp_boost_v4.py --no_delta_qf      # disable Qf learning (ablation)
"""

import argparse, math, os, random, signal, sys, time
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
QF_BASE       = [20.0, 20.0, 40.0, 30.0]
NUM_STEPS     = 200; SAVE_EVERY = 10; EXCELLENT_HOLD = 0.92
X0_PERT_Q1, X0_PERT_V1, X0_PERT_Q2, X0_PERT_V2 = 0.30, 0.6, 0.25, 0.6


class HoldBoostWrapperV4(nn.Module):
    """HoldBoost with learnable delta_Q, delta_R, and delta_Qf (terminal cost)."""

    def __init__(self, lin_net, thresh, dQ_init, dR_init=None, learn_qf=True,
                 x_goal_q1=math.pi):
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
        # delta_Qf: additive correction to Qf multiplier near top
        # gates_Qf = 1.0 + gate * delta_Qf → passed as multiplicative correction to MPC
        self.learn_qf = learn_qf
        if learn_qf:
            self.delta_Qf = nn.Parameter(torch.zeros(lin_net.state_dim, dtype=torch.float64))

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, _ = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        fe = fe * (1.0 - gate.detach())
        gQ = gQ + gate * self.delta_Q
        gR = gR + gate * self.delta_R
        # Compute multiplicative Qf correction near top
        if self.learn_qf:
            gQf = (1.0 + gate * self.delta_Qf).clamp(0.05, 10.0)
        else:
            gQf = None
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


def load_best(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    tp = ckpt['metadata'].get('training_params', {})
    dQ = tp.get('best_delta_Q')
    dR = tp.get('best_delta_R')
    return (torch.tensor(dQ, dtype=torch.float64) if dQ else None,
            torch.tensor(dR, dtype=torch.float64) if dR else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_ckpt", default=BEST_CKPT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_delta_qf", action="store_true",
                        help="Disable Qf learning (ablation)")
    args = parser.parse_args()
    learn_qf = not args.no_delta_qf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    dQ_init, dR_init = load_best(args.from_ckpt)

    print("=" * 80)
    print(f"  EXP BOOST-V4: delta_Q + delta_R + delta_Qf (terminal cost learning)")
    print(f"  learn_qf={learn_qf}  LR={args.lr}  NUM_STEPS={NUM_STEPS}  EPOCHS={args.epochs}")
    print(f"  Init dQ mean: {dQ_init.mean(0).tolist()}")
    print(f"  EXCELLENT_HOLD: {EXCELLENT_HOLD:.0%}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.requires_grad_(False)

    boost = HoldBoostWrapperV4(lin_net, thresh=THRESH, dQ_init=dQ_init, dR_init=dR_init,
                               learn_qf=learn_qf)

    params = [boost.delta_Q, boost.delta_R]
    if learn_qf:
        params.append(boost.delta_Qf)
        print(f"  Trainable params: delta_Q({list(boost.delta_Q.shape)}) + "
              f"delta_R({list(boost.delta_R.shape)}) + delta_Qf({list(boost.delta_Qf.shape)})")

    print(f"\n  Initial eval:")
    f01, f03, arr, post = eval2k(boost, mpc, x0, x_goal)
    print(f"    frac<0.10={f01:.1%}  frac<0.30={f03:.1%}  arr={arr}  "
          f"post={f'{post:.1%}' if post else 'N/A'}  [prev=82.9%]", flush=True)

    best_f01 = f01
    best_dQ = boost.delta_Q.data.clone()
    best_dR = boost.delta_R.data.clone()
    best_dQf = boost.delta_Qf.data.clone() if learn_qf else None

    session_name = f"stageD_boost_v4{'_noDqf' if not learn_qf else ''}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    t0 = time.time()

    interrupted = [False]
    signal.signal(signal.SIGINT, lambda s, f: interrupted.__setitem__(0, True))

    print(f"\n  Training (fresh x0 each epoch, cosine LR, learn_qf={learn_qf})...")
    epoch = 0
    losses = []
    while epoch < args.epochs and not interrupted[0]:
        x0_train = sample_near_top_x0(device)
        demo = torch.zeros((NUM_STEPS, 4), dtype=torch.float64, device=device)
        demo[:, 0] = math.pi

        loss_chunk, _ = train_module.train_linearization_network(
            lin_net=boost, mpc=mpc,
            x0=x0_train, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=1, lr=args.lr,
            debug_monitor=None, recorder=network_module.NetworkOutputRecorder(),
            track_mode="phase_aware", phase_split_frac=0.0,
            w_terminal_anchor=0.0, w_q_profile=0.0, w_f_pos_only=0.0,
            w_stable_phase=0.0, f_gate_thresh=0.0,
            w_hold_reward=0.0, hold_sigma=0.5, hold_start_step=0,
            early_stop_patience=5,
            external_optimizer=optimizer, restore_best=False,
        )
        losses.extend(loss_chunk)
        scheduler.step()
        epoch += 1

        if epoch % SAVE_EVERY == 0:
            f01, f03, arr, post = eval2k(boost, mpc, x0, x_goal)
            dQ_vals = boost.delta_Q.data.mean(0).tolist()
            dQf_vals = boost.delta_Qf.data.tolist() if learn_qf else [0]*4
            lr_now = optimizer.param_groups[0]['lr']
            mark = ""
            if f01 > best_f01:
                mark = " ★"
                best_f01 = f01
                best_dQ = boost.delta_Q.data.clone()
                best_dR = boost.delta_R.data.clone()
                if learn_qf:
                    best_dQf = boost.delta_Qf.data.clone()

            print(f"  [ep={epoch:3d}]  {f01:.1%}  f<0.30={f03:.1%}  arr={arr}  "
                  f"post={f'{post:.1%}' if post else 'N/A'}  "
                  f"dQ=[{dQ_vals[0]:.3f},{dQ_vals[1]:.3f},{dQ_vals[2]:.3f},{dQ_vals[3]:.3f}]  "
                  f"dQf=[{dQf_vals[0]:.3f},{dQf_vals[1]:.3f},{dQf_vals[2]:.3f},{dQf_vals[3]:.3f}]"
                  f"  lr={lr_now:.1e}  t={time.time()-t0:.0f}s{mark}", flush=True)

            if epoch % (2 * SAVE_EVERY) == 0:
                tp = {
                    "experiment": "boost_v4",
                    "best_frac01": best_f01,
                    "best_delta_Q": best_dQ.tolist(),
                    "best_delta_R": best_dR.tolist(),
                    "epoch": epoch,
                }
                if learn_qf and best_dQf is not None:
                    tp["best_delta_Qf"] = best_dQf.tolist()
                network_module.ModelManager(base_dir="saved_models").save_training_session(
                    model=lin_net, loss_history=losses,
                    training_params=tp,
                    session_name=f"{session_name}_ep{epoch:03d}",
                )

            if best_f01 >= EXCELLENT_HOLD:
                print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
                break

    boost.delta_Q.data.copy_(best_dQ)
    boost.delta_R.data.copy_(best_dR)
    if learn_qf and best_dQf is not None:
        boost.delta_Qf.data.copy_(best_dQf)

    tp_final = {
        "experiment": "boost_v4_FINAL",
        "best_frac01_2000step": best_f01,
        "best_delta_Q": best_dQ.tolist(),
        "best_delta_R": best_dR.tolist(),
    }
    if learn_qf and best_dQf is not None:
        tp_final["best_delta_Qf"] = best_dQf.tolist()

    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net, loss_history=losses,
        training_params=tp_final,
        session_name=session_name,
    )

    print(f"\n  Init: {f01:.1%}  →  Best: {best_f01:.1%}")
    print(f"  Best dQ mean: {best_dQ.mean(0).tolist()}")
    if learn_qf and best_dQf is not None:
        dQf_eff = [QF_BASE[i] * (1.0 + best_dQf[i].item()) for i in range(4)]
        print(f"  Best dQf: {best_dQf.tolist()}")
        print(f"  Effective Qf near top: {dQf_eff}")
    if best_f01 > 0.829:
        print(f"  ★★★ NEW RECORD: {best_f01:.1%} > 82.9% ★★★")


if __name__ == "__main__":
    main()
