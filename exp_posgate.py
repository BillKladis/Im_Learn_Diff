"""exp_posgate.py — Learned position gate replacing the hardcoded HoldBoostWrapper.

USER MOTIVATION (session 6):
  "Maybe the network can learn to scale a single scalar. That thing that created the
  switch instead of hard coding it in. Maybe an informative loss can do that.
  Different representation of states or extra inputs could be worth experimenting."

DESIGN (simple):
  1. lin_net is FULLY FROZEN (preserves swing-up knowledge)
  2. Tiny Q-gate MLP: [cos(q1-π), sin(q1-π), q1d, q2, q2d] → Q_adj[q1, q1d, q2, q2d]
     - cos(q1-π) = +1 at top, -1 at bottom: trivially distinguishable
     - Only 5→32→4 = 322 parameters
  3. Phase 1 (no CVXPY): Supervised init from hardcoded wrapper behavior
     - Top: Q_adj → dQ_wrapper = [4.354, 4.090, -0.419, 0.307]
     - Bottom: Q_adj → 0
  4. Phase 2 (needs CVXPY): End-to-end fine-tune with MPC tracking loss
     - Alternating top and bottom states, standard tracking

KEY ADVANTAGE: cos(q1-π) gives the gate a trivially distinguishable feature, unlike the
trunk features that overlap significantly between top and bottom (cos_sim up to 0.586).
The gate can learn any smooth function of position+velocity, not just a fixed threshold.
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

TOP_PERT_Q1, TOP_PERT_V1, TOP_PERT_Q2, TOP_PERT_V2 = 0.30, 0.6, 0.25, 0.6
BOTTOM_X0_LIST = [
    [0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0], [-0.2, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0], [0.0, -0.5, 0.0, 0.0],
]


class PositionGateWrapper(nn.Module):
    """Tiny learned Q-gate replacing the hardcoded HoldBoostWrapper.

    Gate input: [cos(q1-pi), sin(q1-pi), q1d, q2, q2d]  (5 features, informative)
    Gate output: Q_adj for [q1, q1d, q2, q2d]            (4 values)

    lin_net is frozen. Only gate_net parameters are trained.
    """
    def __init__(self, lin_net, hidden=32, also_gate_f=True):
        super().__init__()
        self.lin_net = lin_net
        self.also_gate_f = also_gate_f
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim
        self.control_dim = lin_net.control_dim

        # Q adjustment gate: 5 position features → 4 Q adjustments
        self.q_gate = nn.Sequential(
            nn.Linear(5, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, 4, dtype=torch.float64),
        )
        # f_extra gate: scalar ∈ [0,1] for zeroing out f_extra near top
        self.f_gate = nn.Sequential(
            nn.Linear(5, 16, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(16, 1, dtype=torch.float64),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize gate to approximately match the hardcoded 4× wrapper behavior.

        Target: q_adj ≈ dQ_4x × sigmoid(k*(cos(q1-π)-0.6)) where dQ_4x=[4.354,4.090,-0.419,0.307]
        We achieve this by setting the last layer bias to near-zero and weight small.
        This gives a neutral starting point (zero adjustment everywhere).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def get_position_features(self, x_sequence):
        """Extract 5-dim position features from current state."""
        x_curr = x_sequence[-1]  # last state in sequence
        q1  = x_curr[0]
        q1d = x_curr[1]
        q2  = x_curr[2]
        q2d = x_curr[3]
        return torch.stack([
            torch.cos(q1 - math.pi),  # +1 at top (q1=π), -1 at bottom (q1=0)
            torch.sin(q1 - math.pi),  # 0 at top/bottom, ±1 at 90°
            q1d, q2, q2d
        ])

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)

        pos_feat = self.get_position_features(x_sequence)  # (5,)
        q_adj = self.q_gate(pos_feat)          # (4,)  Q adjustment
        gQ = (gQ + q_adj.unsqueeze(0)).clamp(min=0.01)  # ensure non-negative Q weights

        if self.also_gate_f:
            f_suppress = self.f_gate(pos_feat)  # (1,)  ∈ [0,1]
            fe = fe * (1.0 - f_suppress)

        return gQ, gR, fe, qd, rd, gQf


def load_wrapper_params(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    tp = ckpt['metadata'].get('training_params', {})
    dQ = tp.get('best_delta_Q')
    dR = tp.get('best_delta_R')
    return (torch.tensor(dQ, dtype=torch.float64) if dQ else None,
            torch.tensor(dR, dtype=torch.float64) if dR else None)


def supervised_pretrain(model, device, dQ_target, n_steps=500, lr=1e-2):
    """Phase 1: no CVXPY. Train gate to match hardcoded wrapper.

    Top states (cos>0.5): Q_adj → dQ_target (wrapper values)
    Bottom states (cos<0): Q_adj → 0 (no correction)
    """
    dQ_t = dQ_target.to(device)
    gate_params = list(model.q_gate.parameters()) + list(model.f_gate.parameters())
    opt = torch.optim.Adam(gate_params, lr=lr)

    print(f"  Supervised pre-train: {n_steps} steps (no CVXPY)...")
    for step in range(1, n_steps + 1):
        opt.zero_grad()
        losses = []

        # TOP batch: cos(q1-π) ≈ +1 → Q_adj should → dQ_target
        for _ in range(8):
            q1 = math.pi + (random.random() * 2 - 1) * 0.30
            q1d = (random.random() * 2 - 1) * 0.6
            q2  = (random.random() * 2 - 1) * 0.25
            q2d = (random.random() * 2 - 1) * 0.6
            x_curr = torch.tensor([q1, q1d, q2, q2d], device=device, dtype=torch.float64)
            x_seq  = x_curr.unsqueeze(0).expand(STATE_HIST, -1)
            pos = model.get_position_features(x_seq)
            q_adj = model.q_gate(pos)
            f_gate_val = model.f_gate(pos)
            losses.append(F.mse_loss(q_adj, dQ_t))
            losses.append(F.mse_loss(f_gate_val, torch.ones(1, device=device, dtype=torch.float64)))

        # BOTTOM batch: cos(q1-π) ≈ -1 → Q_adj should → 0, f_gate → 0
        for _ in range(8):
            base = random.choice(BOTTOM_X0_LIST)
            q1  = base[0] + (random.random() * 2 - 1) * 0.1
            q1d = base[1] + (random.random() * 2 - 1) * 0.2
            q2  = (random.random() * 2 - 1) * 0.1
            q2d = (random.random() * 2 - 1) * 0.2
            x_curr = torch.tensor([q1, q1d, q2, q2d], device=device, dtype=torch.float64)
            x_seq  = x_curr.unsqueeze(0).expand(STATE_HIST, -1)
            pos = model.get_position_features(x_seq)
            q_adj = model.q_gate(pos)
            f_gate_val = model.f_gate(pos)
            losses.append(F.mse_loss(q_adj, torch.zeros_like(dQ_t)))
            losses.append(F.mse_loss(f_gate_val, torch.zeros(1, device=device, dtype=torch.float64)))

        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            # Quick check: q_adj at top vs bottom
            with torch.no_grad():
                x_top = torch.tensor([math.pi, 0, 0, 0], device=device, dtype=torch.float64)
                pos_top = model.get_position_features(x_top.unsqueeze(0).expand(STATE_HIST, -1))
                adj_top = model.q_gate(pos_top)
                x_bot = torch.tensor([0.0, 0, 0, 0], device=device, dtype=torch.float64)
                pos_bot = model.get_position_features(x_bot.unsqueeze(0).expand(STATE_HIST, -1))
                adj_bot = model.q_gate(pos_bot)
            print(f"    step {step}: loss={loss.item():.4f} "
                  f"| top Q_adj={adj_top.tolist()[:2]} "
                  f"| bot Q_adj={adj_bot.tolist()[:2]}", flush=True)

    print(f"  Pre-train complete.", flush=True)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--top_frac", type=float, default=0.7)
    parser.add_argument("--pretrain_steps", type=int, default=500,
                        help="Supervised pre-train steps (no CVXPY)")
    parser.add_argument("--hidden", type=int, default=32,
                        help="Hidden size of the gate MLP")
    parser.add_argument("--no_pretrain", action="store_true",
                        help="Skip supervised pre-train (random init → end-to-end only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0_eval = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal  = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0_eval, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.requires_grad_(False)

    # Load reference dQ from scale=4× checkpoint (the hardcoded wrapper target)
    dQ_ref, _ = load_wrapper_params(SCALE4_CKPT)
    # dQ is stored per horizon step (shape [N, 4]); take mean across horizon
    if dQ_ref.dim() > 1:
        dQ_ref = dQ_ref.mean(dim=0)
    dQ_ref = dQ_ref.to(device)

    model = PositionGateWrapper(lin_net, hidden=args.hidden).to(device)
    gate_params = list(model.q_gate.parameters()) + list(model.f_gate.parameters())
    n_params = sum(p.numel() for p in gate_params)

    print("=" * 80)
    print("  EXP POSGATE: Learned position-aware Q gate (replaces hardcoded wrapper)")
    print(f"  lin_net: FROZEN  gate_net: {n_params} params  hidden={args.hidden}")
    print(f"  Gate input: [cos(q1-π), sin(q1-π), q1d, q2, q2d] → Q_adj[4]")
    print(f"  cos(q1-π)=+1 at top, -1 at bottom: trivially distinguishable")
    print(f"  LR={args.lr}  top_frac={args.top_frac:.0%}  epochs={args.epochs}")
    print(f"  Reference dQ (from scale=4×): {dQ_ref.tolist()}")
    print("=" * 80)

    # Phase 1: Supervised pre-train (fast, no CVXPY)
    if not args.no_pretrain and args.pretrain_steps > 0:
        supervised_pretrain(model, device, dQ_ref, n_steps=args.pretrain_steps)

    # Check gate output before CVXPY eval
    with torch.no_grad():
        x_top = torch.tensor([math.pi, 0, 0, 0], device=device, dtype=torch.float64)
        pos_top = model.get_position_features(x_top.unsqueeze(0).expand(STATE_HIST, -1))
        adj_top = model.q_gate(pos_top)
        fg_top  = model.f_gate(pos_top)
        x_bot = torch.tensor([0, 0, 0, 0], device=device, dtype=torch.float64)
        pos_bot = model.get_position_features(x_bot.unsqueeze(0).expand(STATE_HIST, -1))
        adj_bot = model.q_gate(pos_bot)
        fg_bot  = model.f_gate(pos_bot)
    print(f"\n  Gate check before eval:")
    print(f"    top: Q_adj={[f'{v:.3f}' for v in adj_top.tolist()]}  f_suppress={fg_top.item():.3f}")
    print(f"    bot: Q_adj={[f'{v:.3f}' for v in adj_bot.tolist()]}  f_suppress={fg_bot.item():.3f}")

    # Initial eval (triggers CVXPY compilation)
    print(f"\n  Initial eval (compiling CVXPY — ~25 min)...")
    f01, f03, arr, post = eval2k(model, mpc, x0_eval, x_goal)
    print(f"    frac<0.10={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}  [baseline=87.2%]",
          flush=True)

    best_f01 = f01
    best_state = {k: v.clone() for k, v in model.state_dict().items()
                  if 'gate' in k}
    optimizer = torch.optim.AdamW(gate_params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

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
        if is_top:
            top_count += 1
        else:
            bot_count += 1

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
            f01, f03, arr, post = eval2k(model, mpc, x0_eval, x_goal)
            with torch.no_grad():
                adj_top = model.q_gate(model.get_position_features(
                    torch.tensor([math.pi, 0, 0, 0], device=device, dtype=torch.float64)
                    .unsqueeze(0).expand(STATE_HIST, -1)))
                adj_bot = model.q_gate(model.get_position_features(
                    torch.tensor([0, 0, 0, 0], device=device, dtype=torch.float64)
                    .unsqueeze(0).expand(STATE_HIST, -1)))
            mark = ""
            if f01 > best_f01:
                mark = " ★"
                best_f01 = f01
                best_state = {k: v.clone() for k, v in model.state_dict().items() if 'gate' in k}
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [ep={epoch:3d}]  {f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}"
                  f"  top_adj[q1]={adj_top[0].item():.3f}  bot_adj[q1]={adj_bot[0].item():.3f}"
                  f"  top/bot={top_count}/{bot_count}  lr={lr_now:.1e}  t={time.time()-t0:.0f}s{mark}",
                  flush=True)

            if best_f01 >= EXCELLENT_HOLD:
                print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
                break

    # Restore best
    state = model.state_dict()
    state.update(best_state)
    model.load_state_dict(state)

    session_name = (f"stageE_posgate_h{args.hidden}_top{args.top_frac:.0f}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net,  # save the base (frozen) lin_net
        loss_history=[],
        training_params={
            "experiment": "posgate",
            "best_f01": best_f01,
            "gate_state_dict": {k: v.cpu().tolist() for k, v in best_state.items()},
            "hidden": args.hidden,
            "dQ_ref": dQ_ref.tolist(),
        },
        session_name=session_name,
    )

    print(f"\n  DONE. best_f01={best_f01:.1%}  saved: {session_name}")


if __name__ == "__main__":
    main()
