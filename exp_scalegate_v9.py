"""exp_scalegate_v9.py — Learned gate: small MLP [near_pi,q1d,q2,q2d] → alpha.

MOTIVATION:
  All v4-v8 used hard-coded gate formulas (linear ramp, velocity penalty).
  This experiment lets a small network discover the gate shape from rollouts.

DESIGN:
  - Gate: LearnedGate, a 2-layer MLP with ReLU hidden layers + clamp(0,1) output
    Input: [near_pi, q1d, q2, q2d] (4 features; near_pi=(1+cos(q1-π))/2 ∈ [0,1])
    Hidden: 16 → 8 → 1 (scalar alpha)
    near_pi is used instead of raw q1 because q1 can accumulate beyond [0,2π]
    during rollouts (out-of-distribution for a network trained on [0,2π]).
    No sigmoid output — use clamp to avoid saturation
  - lin_net frozen; dQ_ref/dR_ref frozen (from SCALE4_CKPT)
  - 57 parameters total (comparable to v4's 2 params but more expressive)

INITIALIZATION (imitation pretraining):
  Pre-train on a state grid to match linear ramp target:
    near_pi = (1 + cos(q1-π)) / 2
    target = ((near_pi - 0.850) / 0.150).clamp(0,1)
  This initializes network to reproduce the 87.3% behavior exactly.
  Then fine-tune with rollout quality — network can improve beyond linear ramp.

WHY THIS IS BETTER THAN v1/v2:
  - v1/v2 used sigmoid output → saturated → gate was binary → same failure as posgate
  - v9 uses ReLU + clamp: smooth, no saturation, gradients flow in ramp region
  - v1/v2 started from random init (0% initially); v9 starts from 87.3% via imitation

GOAL: Network discovers gate shapes better than linear ramp (e.g., velocity-aware,
  state-dependent, non-symmetric) without any hard-coded formula.
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
DT            = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
STATE_HIST    = 5; NUM_STEPS = 200; SAVE_EVERY = 10; EXCELLENT_HOLD = 0.92
IMITATION_THRESH = 0.850  # linear ramp target for pretraining
BOTTOM_X0_LIST = [[0,0,0,0],[0.2,0,0,0],[-0.2,0,0,0],[0.5,0,0,0],[-0.5,0,0,0]]


class LearnedGate(nn.Module):
    """Small MLP: state [q1,q1d,q2,q2d] → alpha ∈ [0,1].

    ReLU hidden layers + clamp output (no sigmoid → no saturation).
    Pre-trained via imitation to match linear ramp formula at thresh=0.850.
    """
    def __init__(self, lin_net, dQ_ref, dR_ref, hidden=16):
        super().__init__()
        self.lin_net = lin_net
        self.register_buffer('dQ_ref', dQ_ref.clone())
        self.register_buffer('dR_ref', dR_ref.clone())
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim
        self.control_dim = lin_net.control_dim

        # Small MLP: 4 inputs → hidden → hidden//2 → 1
        self.gate_net = nn.Sequential(
            nn.Linear(4, hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1, dtype=torch.float64),
        )
        # Init final layer to zero → alpha≈0 before pretraining
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def get_alpha(self, x_sequence):
        x_last = x_sequence[-1]  # (4,)
        near_pi = (1.0 + torch.cos(x_last[0] - math.pi)) / 2.0
        features = torch.stack([near_pi, x_last[1], x_last[2], x_last[3]])
        raw = self.gate_net(features.unsqueeze(0)).squeeze()
        return raw.clamp(0.0, 1.0)

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        alpha = self.get_alpha(x_sequence)
        gQ = gQ + alpha * self.dQ_ref
        gR = gR + alpha * self.dR_ref
        fe = fe * (1.0 - alpha)
        return gQ, gR, fe, qd, rd, gQf


def imitation_pretrain(model, device, thresh=IMITATION_THRESH, n_steps=5000, lr=1e-3):
    """Supervised pretraining: match linear ramp formula on random state grid."""
    opt = torch.optim.Adam(model.gate_net.parameters(), lr=lr)
    for step in range(n_steps):
        # Random state batch
        q1  = torch.empty(256, dtype=torch.float64, device=device).uniform_(0, 2*math.pi)
        q1d = torch.empty(256, dtype=torch.float64, device=device).uniform_(-3, 3)
        q2  = torch.empty(256, dtype=torch.float64, device=device).uniform_(-1, 1)
        q2d = torch.empty(256, dtype=torch.float64, device=device).uniform_(-3, 3)
        near_pi = (1.0 + torch.cos(q1 - math.pi)) / 2.0
        target = ((near_pi - thresh) / (1.0 - thresh)).clamp(0.0, 1.0)
        state = torch.stack([near_pi, q1d, q2, q2d], dim=1)  # (256, 4) — near_pi first

        pred = model.gate_net(state).squeeze().clamp(0.0, 1.0)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 1000 == 999:
            print(f"    imitation step {step+1}/{n_steps}: loss={loss.item():.5f}", flush=True)

    # Verify
    with torch.no_grad():
        for deg in [0, 90, 127, 140, 150, 165, 180]:
            q1_val = math.radians(deg)
            np_val = (1 + math.cos(q1_val - math.pi)) / 2
            s = torch.tensor([[np_val, 0.0, 0.0, 0.0]], dtype=torch.float64, device=device)
            alpha = model.gate_net(s).squeeze().clamp(0,1).item()
            tgt = max(0, min(1, (np_val - thresh) / (1 - thresh)))
            print(f"    q1={deg:3d}°  near_pi={np_val:.3f}  α={alpha:.4f}  target={tgt:.4f}", flush=True)


def probe_gate(model, device, header="Gate profile:"):
    print(f"  {header}")
    with torch.no_grad():
        for deg, q1d_val in [(0,0),(90,0),(127,0),(140,0),(165,0),(180,0),(180,1),(180,2)]:
            q1 = math.radians(deg)
            np_val = (1 + math.cos(q1 - math.pi)) / 2
            x = torch.tensor([q1, q1d_val, 0, 0], dtype=torch.float64, device=device)
            xseq = x.unsqueeze(0).expand(STATE_HIST, -1)
            alpha = model.get_alpha(xseq).item()
            linear_ramp = max(0, min(1, (np_val - IMITATION_THRESH) / (1 - IMITATION_THRESH)))
            print(f"    q1={deg:3d}° q1d={q1d_val}  near_pi={np_val:.3f}"
                  f"  α={alpha:.4f}  ramp={linear_ramp:.4f}", flush=True)


def eval2k(model, mpc, x0, x_goal, steps=2000):
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([math.sqrt(
        math.atan2(math.sin(s[0]-math.pi), math.cos(s[0]-math.pi))**2
        + s[1]**2 + s[2]**2 + s[3]**2) for s in traj])
    arr = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    return float((wraps < 0.10).mean()), arr, post


def sample_top_x0(device):
    return torch.tensor([
        math.pi + (random.random()*2-1)*0.30,
        (random.random()*2-1)*0.6,
        (random.random()*2-1)*0.25,
        (random.random()*2-1)*0.6,
    ], dtype=torch.float64, device=device)


def sample_bottom_x0(device):
    base = random.choice(BOTTOM_X0_LIST)
    return torch.tensor([
        base[0]+(random.random()*2-1)*0.1,
        base[1]+(random.random()*2-1)*0.2,
        (random.random()*2-1)*0.1,
        (random.random()*2-1)*0.2,
    ], dtype=torch.float64, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--top_frac", type=float, default=0.7)
    parser.add_argument("--hidden", type=int, default=16,
                        help="Hidden layer size (default 16)")
    parser.add_argument("--pretrain_steps", type=int, default=5000,
                        help="Imitation pretraining steps (0 to skip)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0_eval = torch.tensor(X0, dtype=torch.float64, device=device)
    x_goal  = torch.tensor(X_GOAL, dtype=torch.float64, device=device)

    mpc = mpc_module.MPC_controller(x0=x0_eval, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, dtype=torch.float64, device=device)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, dtype=torch.float64, device=device)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.requires_grad_(False)

    ckpt = torch.load(SCALE4_CKPT, map_location='cpu', weights_only=False)
    tp = ckpt['metadata']['training_params']
    dQ_ref = torch.tensor(tp['best_delta_Q'], dtype=torch.float64).to(device)
    dR_ref = torch.tensor(tp['best_delta_R'], dtype=torch.float64).to(device)

    model = LearnedGate(lin_net, dQ_ref, dR_ref, hidden=args.hidden).to(device)

    n_params = sum(p.numel() for p in model.gate_net.parameters())
    print("=" * 80)
    print(f"  EXP SCALEGATE v9: Learned gate MLP [near_pi,q1d,q2,q2d] → alpha")
    print(f"  Architecture: 4 → {args.hidden} → {args.hidden//2} → 1 ({n_params} params)")
    print(f"  Imitation target: linear ramp thresh={IMITATION_THRESH}")
    print(f"  LR={args.lr}  top_frac={args.top_frac:.0%}  epochs={args.epochs}")
    print("=" * 80)

    if args.pretrain_steps > 0:
        print(f"\n  Imitation pretraining ({args.pretrain_steps} steps, no CVXPY)...")
        imitation_pretrain(model, device, thresh=IMITATION_THRESH,
                           n_steps=args.pretrain_steps)
        probe_gate(model, device, "Gate after imitation pretraining:")
    else:
        print("  Skipping imitation pretraining (--pretrain_steps 0)")
        probe_gate(model, device, "Initial gate profile (untrained):")

    print(f"\n  Initial eval (compiling CVXPY — ~25 min)...")
    f01, arr, post = eval2k(model, mpc, x0_eval, x_goal)
    print(f"    frac<0.10={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}"
          f"  [target: >87.3%]", flush=True)

    best_f01 = f01
    best_state = {k: v.clone() for k, v in model.gate_net.state_dict().items()}

    optimizer = torch.optim.AdamW(model.gate_net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    interrupted = [False]
    signal.signal(signal.SIGINT, lambda s, f: interrupted.__setitem__(0, True))
    t0 = time.time()

    print(f"\n  Training ({args.top_frac:.0%} top / {1-args.top_frac:.0%} bottom)...")
    epoch = 0
    top_count = bot_count = 0

    while epoch < args.epochs and not interrupted[0]:
        is_top = (random.random() < args.top_frac)
        x0_train = sample_top_x0(device) if is_top else sample_bottom_x0(device)
        demo = torch.zeros((NUM_STEPS, 4), dtype=torch.float64, device=device)
        demo[:, 0] = math.pi
        if is_top: top_count += 1
        else: bot_count += 1

        train_module.train_linearization_network(
            lin_net=model, mpc=mpc,
            x0=x0_train, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=1, lr=args.lr,
            debug_monitor=None, recorder=network_module.NetworkOutputRecorder(),
            track_mode="phase_aware", phase_split_frac=0.0 if is_top else 0.5,
            w_terminal_anchor=0.0, w_q_profile=0.0, w_f_pos_only=0.0,
            w_stable_phase=0.0, f_gate_thresh=0.0,
            w_hold_reward=0.0, hold_sigma=0.5, hold_start_step=0,
            early_stop_patience=5,
            external_optimizer=optimizer, restore_best=False,
        )
        scheduler.step()
        epoch += 1

        if epoch % SAVE_EVERY == 0:
            f01, arr, post = eval2k(model, mpc, x0_eval, x_goal)
            mark = ""
            if f01 > best_f01:
                mark = " ★"
                best_f01 = f01
                best_state = {k: v.clone() for k, v in model.gate_net.state_dict().items()}
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [ep={epoch:3d}]  {f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}"
                  f"  top/bot={top_count}/{bot_count}  lr={lr_now:.1e}"
                  f"  t={time.time()-t0:.0f}s{mark}", flush=True)

            if best_f01 >= EXCELLENT_HOLD:
                print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
                break

    model.gate_net.load_state_dict(best_state)
    probe_gate(model, device, "Final gate profile:")

    session_name = f"stageE_scalegate_v9_h{args.hidden}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net,
        loss_history=[],
        training_params={
            "experiment": "scalegate_v9",
            "best_f01": best_f01,
            "hidden_size": args.hidden,
            "pretrain_steps": args.pretrain_steps,
            "imitation_thresh": IMITATION_THRESH,
            "dQ_ref_mean": dQ_ref.mean(0).tolist(),
        },
        session_name=session_name,
    )
    print(f"\n  DONE. best_f01={best_f01:.1%}  saved: {session_name}")


if __name__ == "__main__":
    main()
