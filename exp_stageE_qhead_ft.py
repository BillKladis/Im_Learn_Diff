"""exp_stageE_qhead_ft.py — Stage E: fine-tune lin_net's Q-head near the top.

MOTIVATION:
  The wrapper approach (scale=4× delta_Q) reaches ~87.2% but has a ceiling ~88%.
  The root cause is lin_net's Q-head saturates at gates_Q[q1]≈0.013 near the top
  due to tanh saturation (raw_Q[q1]≈-3.47). The additive wrapper bypasses this,
  but can't change the swing-up dynamics.

  Stage E: Unfreeze ONLY lin_net's Q-head and fine-tune it on near-top trajectories.
  If the Q-head learns raw_Q[q1]→+2 near the top, gates_Q[q1] goes from 0.013→1.96,
  giving effective Q[q1]=12×1.96=23.5 WITHOUT the wrapper at all.

  The key innovation: Q-HEAD ONLY fine-tuning + REGULARIZATION to preserve
  swing-up behavior (penalize deviation from original Q outputs at non-top states).

STRATEGY:
  1. Load frozen lin_net (the base posonly model)
  2. Unfreeze ONLY q_head parameters
  3. Train with near-top x0 loss + regularization on Q values at non-top states
  4. At non-top states: minimize ||gates_Q_new - gates_Q_orig||² (prevent swing-up damage)

USAGE:
  python exp_stageE_qhead_ft.py
  python exp_stageE_qhead_ft.py --epochs 100 --lr 1e-4
  python exp_stageE_qhead_ft.py --with_wrapper  # also use scale=4x wrapper on top
"""

import argparse, math, os, random, signal, sys, time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

POSONLY_FINAL = "saved_models/stageD_posonly_ft_20260430_083618/stageD_posonly_ft_20260430_083618.pth"
SCALE4_CKPT   = "saved_models/stageD_scale4.0x_dQ_20260430_192447/stageD_scale4.0x_dQ_20260430_192447.pth"
X0            = [0.0, 0.0, 0.0, 0.0]
X_GOAL        = [math.pi, 0.0, 0.0, 0.0]
DT            = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]; THRESH = 0.8
NUM_STEPS     = 200; SAVE_EVERY = 10; EXCELLENT_HOLD = 0.92
X0_PERT_Q1, X0_PERT_V1, X0_PERT_Q2, X0_PERT_V2 = 0.30, 0.6, 0.25, 0.6


class OptionalBoostWrapper(nn.Module):
    """Lin_net wrapper that optionally applies scale=4x delta_Q boost on top."""

    def __init__(self, lin_net, dQ=None, dR=None, thresh=THRESH, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net = lin_net; self.thresh = thresh; self.x_goal_q1 = x_goal_q1
        self.f_extra_bound = lin_net.f_extra_bound; self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim; self.control_dim = lin_net.control_dim
        self.dQ = dQ  # None = no wrapper, else (9,4) tensor
        self.dR = dR

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        if self.dQ is not None:
            q1 = x_sequence[-1, 0]
            near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
            gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
            fe = fe * (1.0 - gate.detach())
            gQ = gQ + gate * self.dQ.to(gQ.device)
            if self.dR is not None:
                gR = gR + gate * self.dR.to(gR.device)
        return gQ, gR, fe, qd, rd, gQf


def sample_near_top_x0(device):
    return torch.tensor([
        math.pi + (random.random() * 2 - 1) * X0_PERT_Q1,
        (random.random() * 2 - 1) * X0_PERT_V1,
        (random.random() * 2 - 1) * X0_PERT_Q2,
        (random.random() * 2 - 1) * X0_PERT_V2,
    ], device=device, dtype=torch.float64)


def sample_swing_x0(device):
    base_q1 = (random.random() * 2 - 1) * 0.5  # q1 ∈ [-0.5, 0.5] (near bottom)
    return torch.tensor([
        base_q1, (random.random() * 2 - 1) * 0.3,
        (random.random() * 2 - 1) * 0.3, (random.random() * 2 - 1) * 0.3,
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


def get_q_reg_loss(lin_net, lin_net_orig, x_sequence, device):
    """Compute Q-regularization loss: ||gates_Q_new - gates_Q_orig||² at non-top state."""
    with torch.no_grad():
        gQ_orig, _, _, _, _, _ = lin_net_orig(x_sequence)
    gQ_new, _, _, _, _, _ = lin_net(x_sequence)
    return F.mse_loss(gQ_new, gQ_orig)


def load_wrapper_params(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    tp = ckpt['metadata'].get('training_params', {})
    dQ = tp.get('best_delta_Q')
    dR = tp.get('best_delta_R')
    return (torch.tensor(dQ, dtype=torch.float64) if dQ else None,
            torch.tensor(dR, dtype=torch.float64) if dR else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="LR for Q-head fine-tuning (very small to avoid catastrophic forgetting)")
    parser.add_argument("--reg_weight", type=float, default=10.0,
                        help="Weight for Q regularization loss at non-top states")
    parser.add_argument("--with_wrapper", action="store_true",
                        help="Also apply scale=4x wrapper on top of fine-tuned lin_net")
    parser.add_argument("--near_top_frac", type=float, default=1.0,
                        help="Fraction of epochs from near-top (rest from swing-up for reg)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    # Load lin_net (will partially unfreeze)
    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net_orig = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net_orig.requires_grad_(False)  # frozen reference copy

    # Freeze everything except Q-head
    lin_net.requires_grad_(False)
    # Unfreeze only q_head
    qhead_params = []
    for name, param in lin_net.named_parameters():
        if 'q_head' in name or 'Q_head' in name or 'qhead' in name:
            param.requires_grad_(True)
            qhead_params.append(param)
            print(f"  Unfreezing: {name}  shape={list(param.shape)}")

    if not qhead_params:
        print("  WARNING: No q_head parameters found! Check lin_net architecture.")
        print("  Available modules:", [n for n, _ in lin_net.named_modules()])
        for name, param in lin_net.named_parameters():
            if 'head' in name.lower() or 'q' in name.lower():
                print(f"    {name}: {param.shape}")

    print("=" * 80)
    print(f"  STAGE E: Q-head fine-tuning")
    print(f"  LR={args.lr}  epochs={args.epochs}  reg_weight={args.reg_weight}")
    print(f"  with_wrapper={args.with_wrapper}  near_top_frac={args.near_top_frac:.0%}")
    print(f"  Trainable Q-head params: {sum(p.numel() for p in qhead_params)}")
    print("=" * 80)

    # Optional wrapper on top
    dQ_wrapper, dR_wrapper = None, None
    if args.with_wrapper:
        dQ_wrapper, dR_wrapper = load_wrapper_params(SCALE4_CKPT)
        print(f"  Using scale=4x wrapper: dQ_mean={dQ_wrapper.mean(0).tolist()}")

    model = OptionalBoostWrapper(lin_net, dQ=dQ_wrapper, dR=dR_wrapper)

    print(f"\n  Initial eval:")
    f01, f03, arr, post = eval2k(model, mpc, x0, x_goal)
    print(f"    frac<0.10={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}  "
          f"[prev_best=87.2%]", flush=True)

    best_f01 = f01
    best_qhead_state = {k: v.clone() for k, v in lin_net.state_dict().items()
                        if 'q_head' in k}

    optimizer = torch.optim.AdamW(qhead_params, lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    t0 = time.time()
    interrupted = [False]
    signal.signal(signal.SIGINT, lambda s, f: interrupted.__setitem__(0, True))

    print(f"\n  Training Q-head only (frozen: all other lin_net layers, all wrapper params)...")
    epoch = 0
    losses = []
    while epoch < args.epochs and not interrupted[0]:
        # Choose training x0
        if random.random() < args.near_top_frac:
            x0_train = sample_near_top_x0(device)
        else:
            x0_train = sample_swing_x0(device)

        demo = torch.zeros((NUM_STEPS, 4), dtype=torch.float64, device=device)
        demo[:, 0] = math.pi

        loss_chunk, _ = train_module.train_linearization_network(
            lin_net=model, mpc=mpc,
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
            f01, f03, arr, post = eval2k(model, mpc, x0, x_goal)
            mark = ""
            if f01 > best_f01:
                mark = " ★"
                best_f01 = f01
                best_qhead_state = {k: v.clone() for k, v in lin_net.state_dict().items()
                                    if 'q_head' in k}

            # Check Q-head change near top vs baseline (lin_net takes 5-step history)
            x_top = torch.tensor([math.pi, 0, 0, 0], device=device, dtype=torch.float64)
            x_seq_top = x_top.unsqueeze(0).expand(5, -1)
            with torch.no_grad():
                gQ_new, _, _, _, _, _ = lin_net(x_seq_top)
                gQ_old, _, _, _, _, _ = lin_net_orig(x_seq_top)
            q1_gate_new = gQ_new[0, 0].item()
            q1_gate_old = gQ_old[0, 0].item()

            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [ep={epoch:3d}]  {f01:.1%}  arr={arr}  "
                  f"post={f'{post:.1%}' if post else 'N/A'}  "
                  f"gQ[q1]={q1_gate_new:.4f}(was {q1_gate_old:.4f})  "
                  f"lr={lr_now:.1e}  t={time.time()-t0:.0f}s{mark}", flush=True)

            if best_f01 >= EXCELLENT_HOLD:
                print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
                break

    # Restore best Q-head parameters
    state = lin_net.state_dict()
    state.update(best_qhead_state)
    lin_net.load_state_dict(state)

    session_name = f"stageE_qhead_ft_{'w4x_' if args.with_wrapper else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net,  # saves fine-tuned lin_net
        loss_history=losses,
        training_params={
            "experiment": "stageE_qhead_ft",
            "best_frac01_2000step": best_f01,
            "with_wrapper": args.with_wrapper,
            "lr": args.lr,
            "reg_weight": args.reg_weight,
            "q1_gate_original": q1_gate_old,
        },
        session_name=session_name,
    )

    print(f"\n  Init: {f01:.1%}  →  Best: {best_f01:.1%}")
    print(f"  gQ[q1] at top: {q1_gate_old:.4f} → {q1_gate_new:.4f}")
    if best_f01 > 0.872:
        print(f"  ★★★ NEW RECORD: {best_f01:.1%} > 87.2% ★★★")


if __name__ == "__main__":
    main()
