"""exp_stageE_alternating.py — Joint bottom/top training with alternating losses.

USER MOTIVATION:
  Train lin_net from BOTH bottom states (swing-up) and top states (hold),
  alternating so neither dominates. Use different losses for each:
  - BOTTOM epoch: standard energy/tracking loss → preserve swing-up
  - TOP epoch: tracking loss + Q-max bonus (push gates_Q[q1,q1d] → max)

  Key hypothesis: the trunk already encodes state regime (bottom vs top)
  via the input state sequence. The q_head can learn different Q profiles
  for each regime through the alternating gradients.

  Two axes to explore:
  1. Alternating loss training (this script)
  2. Optional: state-conditioned q_head that explicitly conditions on near_pi

WHAT THE Q-MAX BONUS DOES:
  The tanh gate in lin_net bounds gates_Q ∈ [0.01, 1.99].
  Near the top, raw_Q[q1] ≈ -3.47 → gates_Q[q1] ≈ 0.013 (saturated LOW).
  The Q-max bonus pushes q_head to output raw_Q[q1] → large positive near top.
  Target: gates_Q[q1] ≈ 1.95 → effective Q[q1] = 12 × 1.95 = 23.4 (from lin_net alone).
  With scale=4x wrapper on top: 12 × (1.95 + 4.354) = 75.6 — even better.

  Gates_Q bound [0.01, 1.99]. Tanh saturation means best achievable is ~1.99.
  Without wrapper: Q[q1] = 12 × 1.99 ≈ 23.9. Target for Stage E baseline.

TRAINING:
  - Every epoch: flip between top (near [π,0,0,0]) and bottom ([0,0,0,0] area)
  - TOP: tracking loss + w_q_bonus × MSE(gates_Q[:,:2], 1.95) (Q-max supervision)
  - BOTTOM: tracking loss only (standard, preserves swing-up)
  - Only q_head is unfrozen by default (trunk + other heads frozen)

USAGE:
  python exp_stageE_alternating.py                    # standard run
  python exp_stageE_alternating.py --unfreeze_trunk   # also unfreeze trunk (risky)
  python exp_stageE_alternating.py --top_frac 0.7     # 70% top epochs
  python exp_stageE_alternating.py --w_q_bonus 5.0    # stronger Q supervision
  python exp_stageE_alternating.py --with_wrapper     # use scale=4x wrapper for eval
"""

import argparse, math, os, random, signal, sys, time
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
STATE_HIST    = 5    # lin_net input: last 5 states (hardcoded in lin_net.py:80)
NUM_STEPS     = 200; SAVE_EVERY = 10; EXCELLENT_HOLD = 0.92

# Near-top: q1 close to π
TOP_PERT_Q1, TOP_PERT_V1, TOP_PERT_Q2, TOP_PERT_V2 = 0.30, 0.6, 0.25, 0.6
# Near-bottom: diverse starting conditions for swing-up
BOTTOM_X0_LIST = [
    [0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0], [-0.2, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0], [0.0, -0.5, 0.0, 0.0],
]
Q_MAX_TARGET = 1.95  # target gates_Q for q1 and q1d near top (close to tanh ceiling of 1.99)


class BoostEvalWrapper(nn.Module):
    """Wrapper for evaluation only: applies scale=4x delta_Q on top of lin_net."""
    def __init__(self, lin_net, dQ, dR, thresh=THRESH, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net = lin_net; self.thresh = thresh; self.x_goal_q1 = x_goal_q1
        self.f_extra_bound = lin_net.f_extra_bound; self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim; self.control_dim = lin_net.control_dim
        self.dQ = dQ; self.dR = dR

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        fe = fe * (1.0 - gate.detach())
        gQ = gQ + gate * self.dQ.to(gQ.device)
        gR = gR + gate * self.dR.to(gR.device)
        return gQ, gR, fe, qd, rd, gQf


def sample_top_x0(device):
    return torch.tensor([
        math.pi + (random.random() * 2 - 1) * TOP_PERT_Q1,
        (random.random() * 2 - 1) * TOP_PERT_V1,
        (random.random() * 2 - 1) * TOP_PERT_Q2,
        (random.random() * 2 - 1) * TOP_PERT_V2,
    ], device=device, dtype=torch.float64)


def sample_bottom_x0(device):
    base = random.choice(BOTTOM_X0_LIST)
    return torch.tensor([
        base[0] + (random.random() * 2 - 1) * 0.1,
        base[1] + (random.random() * 2 - 1) * 0.2,
        base[2] + (random.random() * 2 - 1) * 0.1,
        base[3] + (random.random() * 2 - 1) * 0.2,
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


def q_max_aux_step(lin_net, optimizer, device, w_q_bonus):
    """Auxiliary gradient step: push gates_Q[q1, q1d] → Q_MAX_TARGET at top states.

    This runs independently of the MPC/rollout — just a direct regression
    on the q_head's output for top-state inputs.
    """
    if w_q_bonus <= 0:
        return 0.0

    # Sample a batch of near-top states
    batch_size = 8
    q1_samples = torch.tensor([
        math.pi + (random.random() * 2 - 1) * TOP_PERT_Q1
        for _ in range(batch_size)
    ], device=device, dtype=torch.float64)

    losses = []
    for q1 in q1_samples:
        # Create a state sequence at this near-top state
        x_top = torch.stack([
            q1,
            torch.zeros(1, device=device, dtype=torch.float64).squeeze(),
            torch.zeros(1, device=device, dtype=torch.float64).squeeze(),
            torch.zeros(1, device=device, dtype=torch.float64).squeeze(),
        ])
        x_seq = x_top.unsqueeze(0).expand(STATE_HIST, -1)

        # Forward through lin_net (only q_head contributes to grad since trunk frozen)
        gates_Q, _, _, _, _, _ = lin_net(x_seq)

        # Push gates_Q[q1] and gates_Q[q1d] toward Q_MAX_TARGET
        target = torch.full_like(gates_Q[:, :2], Q_MAX_TARGET)
        loss = F.mse_loss(gates_Q[:, :2], target)
        losses.append(loss)

    total_loss = w_q_bonus * torch.stack(losses).mean()
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item()


def load_wrapper_params(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    tp = ckpt['metadata'].get('training_params', {})
    dQ = tp.get('best_delta_Q')
    dR = tp.get('best_delta_R')
    return (torch.tensor(dQ, dtype=torch.float64) if dQ else None,
            torch.tensor(dR, dtype=torch.float64) if dR else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--top_frac", type=float, default=0.7,
                        help="Fraction of epochs using near-top x0 + Q-max bonus")
    parser.add_argument("--w_q_bonus", type=float, default=2.0,
                        help="Weight for Q-max auxiliary loss at top states")
    parser.add_argument("--unfreeze_trunk", action="store_true",
                        help="Also unfreeze trunk (more capacity, higher risk)")
    parser.add_argument("--with_wrapper", action="store_true",
                        help="Apply scale=4x wrapper during evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0_eval = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal  = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0_eval, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()

    # Selectively unfreeze
    lin_net.requires_grad_(False)
    trainable_params = []
    for name, param in lin_net.named_parameters():
        if 'q_head' in name:
            param.requires_grad_(True)
            trainable_params.append(param)
        elif args.unfreeze_trunk and 'trunk' in name:
            param.requires_grad_(True)
            trainable_params.append(param)

    total_trainable = sum(p.numel() for p in trainable_params)
    print("=" * 80)
    print(f"  STAGE E: Alternating bottom/top training with Q-max bonus")
    print(f"  Trainable params: {total_trainable}  (unfreeze_trunk={args.unfreeze_trunk})")
    print(f"  LR={args.lr}  top_frac={args.top_frac:.0%}  w_q_bonus={args.w_q_bonus}")
    print(f"  Q-max target: gates_Q[q1,q1d] → {Q_MAX_TARGET} (vs 0.013 currently)")
    print(f"  with_wrapper={args.with_wrapper}  prev_best=87.2%")
    print("=" * 80)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    # Check initial gQ[q1] at top
    x_top_test = torch.tensor([math.pi, 0, 0, 0], device=device, dtype=torch.float64)
    x_seq_test = x_top_test.unsqueeze(0).expand(STATE_HIST, -1)
    with torch.no_grad():
        gQ_init, _, _, _, _, _ = lin_net(x_seq_test)
    print(f"  Initial gates_Q at top: {gQ_init[0].tolist()}")
    print(f"  Initial gates_Q[q1] = {gQ_init[0, 0].item():.4f}  (target={Q_MAX_TARGET})")

    # Optional scale=4x wrapper for eval
    dQ_w, dR_w = (None, None)
    if args.with_wrapper:
        dQ_w, dR_w = load_wrapper_params(SCALE4_CKPT)
        eval_model = BoostEvalWrapper(lin_net, dQ=dQ_w, dR=dR_w)
    else:
        eval_model = lin_net

    print(f"\n  Initial eval:")
    f01, f03, arr, post = eval2k(eval_model, mpc, x0_eval, x_goal)
    print(f"    frac<0.10={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}", flush=True)

    best_f01 = f01
    best_state = {k: v.clone() for k, v in lin_net.state_dict().items()
                  if 'q_head' in k or (args.unfreeze_trunk and 'trunk' in k)}

    interrupted = [False]
    signal.signal(signal.SIGINT, lambda s, f: interrupted.__setitem__(0, True))
    t0 = time.time()

    print(f"\n  Training (alternating top={args.top_frac:.0%} / bottom={(1-args.top_frac):.0%})...")
    epoch = 0
    top_count, bot_count = 0, 0

    while epoch < args.epochs and not interrupted[0]:
        is_top_epoch = (random.random() < args.top_frac)

        if is_top_epoch:
            x0_train = sample_top_x0(device)
            demo = torch.zeros((NUM_STEPS, 4), dtype=torch.float64, device=device)
            demo[:, 0] = math.pi
            top_count += 1
        else:
            x0_train = sample_bottom_x0(device)
            demo = torch.zeros((NUM_STEPS, 4), dtype=torch.float64, device=device)
            demo[:, 0] = math.pi
            bot_count += 1

        # Standard tracking training step
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0_train, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=1, lr=args.lr,
            debug_monitor=None, recorder=network_module.NetworkOutputRecorder(),
            track_mode="phase_aware", phase_split_frac=0.0 if is_top_epoch else 0.5,
            w_terminal_anchor=0.0, w_q_profile=0.0, w_f_pos_only=0.0,
            w_stable_phase=0.0, f_gate_thresh=0.0,
            w_hold_reward=0.0, hold_sigma=0.5, hold_start_step=0,
            early_stop_patience=5,
            external_optimizer=optimizer, restore_best=False,
        )

        # Q-max auxiliary step (only for top epochs)
        if is_top_epoch and args.w_q_bonus > 0:
            q_max_aux_step(lin_net, optimizer, device, args.w_q_bonus)

        scheduler.step()
        epoch += 1

        if epoch % SAVE_EVERY == 0:
            # Eval (with or without wrapper)
            if args.with_wrapper:
                eval_model = BoostEvalWrapper(lin_net, dQ=dQ_w, dR=dR_w)
            f01, f03, arr, post = eval2k(eval_model, mpc, x0_eval, x_goal)

            # Check gates_Q at top
            with torch.no_grad():
                gQ_now, _, _, _, _, _ = lin_net(x_seq_test)
            gQ_q1 = gQ_now[0, 0].item()
            gQ_q1d = gQ_now[0, 1].item()

            mark = ""
            if f01 > best_f01:
                mark = " ★"
                best_f01 = f01
                best_state = {k: v.clone() for k, v in lin_net.state_dict().items()
                              if 'q_head' in k or (args.unfreeze_trunk and 'trunk' in k)}

            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [ep={epoch:3d}]  {f01:.1%}  arr={arr}  "
                  f"post={f'{post:.1%}' if post else 'N/A'}  "
                  f"gQ[q1]={gQ_q1:.3f}(↑target {Q_MAX_TARGET})  "
                  f"gQ[q1d]={gQ_q1d:.3f}  top/bot={top_count}/{bot_count}"
                  f"  lr={lr_now:.1e}  t={time.time()-t0:.0f}s{mark}", flush=True)

            if best_f01 >= EXCELLENT_HOLD:
                print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
                break

    # Restore best
    state = lin_net.state_dict()
    state.update(best_state)
    lin_net.load_state_dict(state)

    session_name = (f"stageE_alt_{'w4x_' if args.with_wrapper else ''}"
                    f"top{args.top_frac:.0f}_qb{args.w_q_bonus:.0f}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net,
        loss_history=[],
        training_params={
            "experiment": "stageE_alternating",
            "best_frac01_2000step": best_f01,
            "with_wrapper": args.with_wrapper,
            "top_frac": args.top_frac,
            "w_q_bonus": args.w_q_bonus,
            "lr": args.lr,
            "epochs": epoch,
        },
        session_name=session_name,
    )

    print(f"\n  Init: {f01:.1%}  →  Best: {best_f01:.1%}")
    with torch.no_grad():
        gQ_final, _, _, _, _, _ = lin_net(x_seq_test)
    print(f"  gates_Q at top: {gQ_init[0, :2].tolist()} → {gQ_final[0, :2].tolist()}")
    if best_f01 > 0.872:
        print(f"  ★★★ NEW RECORD: {best_f01:.1%} > 87.2% ★★★")


if __name__ == "__main__":
    main()
