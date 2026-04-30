"""exp_dual_thresh.py — Dual-threshold boost: separate gates for delta_Q and f_extra.

KEY INSIGHT (from scale sweep + optinit_2.0 analysis):
  - dQ=[1.09,1.02] (trained): 82.9%, arr=236, post_arr=93.9%
  - dQ=[2.0,2.0] (larger): 82.6%, arr=327, post_arr=98.7%

  Larger delta_Q → better hold quality (98.7% vs 93.9%) but also ACTIVATES GATE
  earlier, which zeros f_extra over a wider region, hurting swing-up (arr 236→327).

  SOLUTION: Use TWO separate thresholds:
  - thresh_dQ (LOWER, e.g. 0.5): activates delta_Q correction earlier during approach
  - thresh_f (HIGHER, e.g. 0.85): only zeros f_extra very close to top

  This lets MPC benefit from delta_Q throughout the approach phase while preserving
  f_extra control during swing-up (f_extra=0 only right at the top).

USAGE:
  python exp_dual_thresh.py                           # standard run
  python exp_dual_thresh.py --dq_init 2.0 --thresh_dq 0.5 --thresh_f 0.85
  python exp_dual_thresh.py --sweep                   # sweep thresh_dq values
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
DT            = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
NUM_STEPS     = 200; SAVE_EVERY = 20; EXCELLENT_HOLD = 0.92
X0_PERT_Q1, X0_PERT_V1, X0_PERT_Q2, X0_PERT_V2 = 0.30, 0.6, 0.25, 0.6


class DualThreshBoostWrapper(nn.Module):
    """HoldBoost with SEPARATE gate thresholds for delta_Q and f_extra zeroing.

    thresh_dQ (lower): when to activate the Q boost → activates earlier
    thresh_f  (higher): when to zero f_extra → only right at the top
    """

    def __init__(self, lin_net, thresh_dQ, thresh_f, dQ_init, dR_init=None,
                 x_goal_q1=math.pi):
        super().__init__()
        self.lin_net = lin_net
        self.thresh_dQ = thresh_dQ
        self.thresh_f = thresh_f
        self.x_goal_q1 = x_goal_q1
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

        # Separate gates for delta_Q and f_extra
        gate_dQ = ((near_pi - self.thresh_dQ) / max(1e-8, 1.0 - self.thresh_dQ)).clamp(0.0, 1.0)
        gate_f  = ((near_pi - self.thresh_f)  / max(1e-8, 1.0 - self.thresh_f)).clamp(0.0, 1.0)

        fe = fe * (1.0 - gate_f.detach())   # only zero f_extra close to top
        gQ = gQ + gate_dQ * self.delta_Q    # activate Q boost earlier
        gR = gR + gate_dQ * self.delta_R
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


def forward_only_sweep(lin_net, mpc, x0, x_goal, dQ, dR, dq_thresholds, thresh_f, device):
    """Forward-only sweep over thresh_dQ values (no training)."""
    print(f"\n  FORWARD-ONLY SWEEP  thresh_f={thresh_f:.2f}")
    print(f"  {'thresh_dQ':>10}  {'f<0.10':>8}  {'f<0.30':>8}  {'arr':>5}  {'post':>7}  {'note'}")
    print(f"  {'-'*65}")
    for t in dq_thresholds:
        model = DualThreshBoostWrapper(lin_net, thresh_dQ=t, thresh_f=thresh_f,
                                       dQ_init=dQ, dR_init=dR)
        f01, f03, arr, post = eval2k(model, mpc, x0, x_goal)
        note = " ← original" if abs(t - thresh_f) < 0.01 else ""
        print(f"  {t:>10.3f}  {f01:>7.1%}  {f03:>7.1%}  {str(arr):>5}  "
              f"{f'{post:.1%}' if post else 'N/A':>7}{note}", flush=True)


def train_mode(lin_net, mpc, x0, x_goal, dQ, dR, thresh_dQ, thresh_f, epochs, lr, device):
    """Full gradient training with dual-threshold wrapper."""
    boost = DualThreshBoostWrapper(lin_net, thresh_dQ=thresh_dQ, thresh_f=thresh_f,
                                   dQ_init=dQ, dR_init=dR)

    print(f"\n  Initial eval:")
    f01, f03, arr, post = eval2k(boost, mpc, x0, x_goal)
    print(f"    frac<0.10={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}  "
          f"[82.9% baseline]", flush=True)

    best_f01 = f01
    best_dQ = boost.delta_Q.data.clone()
    best_dR = boost.delta_R.data.clone()

    optimizer = torch.optim.AdamW([boost.delta_Q, boost.delta_R], lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    interrupted = [False]
    signal.signal(signal.SIGINT, lambda s, f: interrupted.__setitem__(0, True))
    t0 = time.time()

    print(f"  Training (thresh_dQ={thresh_dQ:.2f}, thresh_f={thresh_f:.2f})...")
    epoch = 0
    losses = []
    while epoch < epochs and not interrupted[0]:
        x0_train = sample_near_top_x0(device)
        demo = torch.zeros((NUM_STEPS, 4), dtype=torch.float64, device=device)
        demo[:, 0] = math.pi

        loss_chunk, _ = train_module.train_linearization_network(
            lin_net=boost, mpc=mpc,
            x0=x0_train, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=1, lr=lr,
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
            mark = ""
            if f01 > best_f01:
                mark = " ★"
                best_f01 = f01
                best_dQ = boost.delta_Q.data.clone()
                best_dR = boost.delta_R.data.clone()
            dQ_vals = boost.delta_Q.data.mean(0).tolist()
            print(f"  [ep={epoch:3d}]  {f01:.1%}  arr={arr}  "
                  f"post={f'{post:.1%}' if post else 'N/A'}  "
                  f"dQ=[{dQ_vals[0]:.3f},{dQ_vals[1]:.3f},{dQ_vals[2]:.3f},{dQ_vals[3]:.3f}]"
                  f"  t={time.time()-t0:.0f}s{mark}", flush=True)
            if best_f01 >= EXCELLENT_HOLD:
                print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
                break

    print(f"\n  Best: {best_f01:.1%} (init was {f01:.1%})")
    if best_f01 > 0.829:
        print(f"  ★★★ NEW RECORD: {best_f01:.1%} ★★★")

    session_name = f"stageD_dual_thresh_dQ{thresh_dQ:.2f}_f{thresh_f:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net, loss_history=losses,
        training_params={
            "experiment": "dual_thresh",
            "best_frac01_2000step": best_f01,
            "best_delta_Q": best_dQ.tolist(),
            "best_delta_R": best_dR.tolist(),
            "thresh_dQ": thresh_dQ,
            "thresh_f": thresh_f,
        },
        session_name=session_name,
    )
    return best_f01, best_dQ, best_dR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_ckpt", default=BEST_CKPT)
    parser.add_argument("--thresh_dq", type=float, default=0.5,
                        help="Gate threshold for delta_Q activation (lower = activates earlier)")
    parser.add_argument("--thresh_f", type=float, default=0.85,
                        help="Gate threshold for f_extra zeroing (higher = only at very top)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--sweep", action="store_true",
                        help="Forward-only sweep over thresh_dQ values (no training)")
    parser.add_argument("--dq_init", type=float, default=None,
                        help="If set, reinitialize delta_Q to this uniform value")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    dQ, dR = load_best(args.from_ckpt)
    if args.dq_init is not None:
        dQ = torch.full_like(dQ, 0.0)
        dQ[:, 0] = args.dq_init
        dQ[:, 1] = args.dq_init
        print(f"  Reinitializing delta_Q to [{args.dq_init},{args.dq_init},0,0]")

    print("=" * 80)
    print(f"  EXP DUAL-THRESH: separate gates for delta_Q and f_extra")
    print(f"  thresh_dQ={args.thresh_dq:.2f}  thresh_f={args.thresh_f:.2f}")
    print(f"  dQ mean: {dQ.mean(0).tolist()}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.eval()

    if args.sweep:
        sweep_thresholds = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90]
        forward_only_sweep(lin_net, mpc, x0, x_goal, dQ, dR, sweep_thresholds,
                           args.thresh_f, device)
    else:
        lin_net.requires_grad_(False)
        train_mode(lin_net, mpc, x0, x_goal, dQ, dR,
                   args.thresh_dq, args.thresh_f, args.epochs, args.lr, device)


if __name__ == "__main__":
    main()
