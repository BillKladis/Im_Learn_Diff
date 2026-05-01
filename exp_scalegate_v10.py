"""exp_scalegate_v10.py — Learned gate + learned boost direction.

LAYER PEELED vs v9:
  v9: hand-crafted dQ_ref direction + learned gate (WHEN to boost)
  v10: learned gate (WHEN) + learned direction (WHAT to boost) — no external dQ_ref

DESIGN:
  CombinedGate: [q1,q1d,q2,q2d] → (alpha scalar) AND (delta_q 4-dim vector)
    alpha = clamp(linear_head(hidden), 0, 1)
    delta_q = direction_head(hidden)   [unconstrained, learned direction]
  Apply: gQ += alpha * delta_q.unsqueeze(0).expand(H-1, -1)
         gR += alpha * delta_r.unsqueeze(0).expand(H, -1)   [delta_r also learned]
         fe *= (1 - alpha)

INITIALIZATION (imitation from dQ_ref):
  alpha_head: match linear ramp thresh=0.850 (same as v9)
  direction_head: match dQ_ref.mean(0) and dR_ref.mean(0)
  (We use the mean across horizon steps since we broadcast over steps)

This removes the hard dependency on SCALE4_CKPT dQ_ref. The network discovers
both activation timing AND boost direction from rollout quality.

WHAT WE EXPECT:
  - Initial eval ≈ 87.3% (imitation from dQ_ref mean)
  - Fine-tuning: may discover that the per-step variation in dQ_ref matters,
    or that different dimensions need different scaling, or that q2/q2d info
    helps determine the right boost
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
X0 = [0.0,0.0,0.0,0.0]; X_GOAL = [math.pi,0.0,0.0,0.0]
DT = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0,5.0,50.0,40.0]
STATE_HIST = 5; NUM_STEPS = 200; SAVE_EVERY = 10; EXCELLENT_HOLD = 0.92
IMITATION_THRESH = 0.850
BOTTOM_X0_LIST = [[0,0,0,0],[0.2,0,0,0],[-0.2,0,0,0],[0.5,0,0,0],[-0.5,0,0,0]]


class CombinedGate(nn.Module):
    """Learned gate + learned boost direction.

    Shared trunk: [q1,q1d,q2,q2d] → hidden features
    Two heads:
      alpha_head: hidden → scalar ∈ [0,1]  (WHEN to boost)
      dQ_head:    hidden → 4-dim vector     (WHAT Q correction to apply)
      dR_head:    hidden → 2-dim vector     (WHAT R correction to apply)

    Apply: gQ += alpha * dQ_head(state).unsqueeze(0)  (broadcast over H-1 steps)
           gR += alpha * dR_head(state).unsqueeze(0)  (broadcast over H steps)
           fe *= (1 - alpha)
    """
    def __init__(self, lin_net, hidden=16):
        super().__init__()
        self.lin_net = lin_net
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim
        self.control_dim = lin_net.control_dim

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(4, hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
            nn.ReLU(),
        )
        # Alpha head (gating)
        self.alpha_head = nn.Linear(hidden, 1, dtype=torch.float64)
        # Direction heads
        self.dQ_head = nn.Linear(hidden, 4, dtype=torch.float64)
        self.dR_head = nn.Linear(hidden, 2, dtype=torch.float64)

        # Init alpha head to zero (alpha≈0 before pretraining)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        # Init direction heads to zero (will be set by imitation)
        nn.init.zeros_(self.dQ_head.weight)
        nn.init.zeros_(self.dR_head.weight)

    def get_alpha(self, x_sequence):
        x_last = x_sequence[-1]
        h = self.trunk(x_last.unsqueeze(0))
        return self.alpha_head(h).squeeze().clamp(0.0, 1.0)

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        x_last = x_sequence[-1]
        h = self.trunk(x_last.unsqueeze(0))
        alpha = self.alpha_head(h).squeeze().clamp(0.0, 1.0)
        dq = self.dQ_head(h).squeeze()   # (4,)
        dr = self.dR_head(h).squeeze()   # (2,)
        # Broadcast over horizon steps
        gQ = gQ + alpha * dq.unsqueeze(0)    # (H-1, 4)
        gR = gR + alpha * dr.unsqueeze(0)    # (H, 2)
        fe = fe * (1.0 - alpha)
        return gQ, gR, fe, qd, rd, gQf


def imitation_pretrain(model, dQ_target, dR_target, device,
                       thresh=IMITATION_THRESH, n_steps=5000, lr=1e-3):
    """Imitation: alpha matches linear ramp; dQ/dR heads match dQ_ref mean."""
    params = list(model.trunk.parameters()) + list(model.alpha_head.parameters()) + \
             list(model.dQ_head.parameters()) + list(model.dR_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    dq_mean = dQ_target.mean(0).to(device)  # (4,)
    dr_mean = dR_target.mean(0).to(device)  # (2,)

    for step in range(n_steps):
        q1  = torch.empty(256, dtype=torch.float64, device=device).uniform_(0, 2*math.pi)
        q1d = torch.empty(256, dtype=torch.float64, device=device).uniform_(-3, 3)
        q2  = torch.empty(256, dtype=torch.float64, device=device).uniform_(-1, 1)
        q2d = torch.empty(256, dtype=torch.float64, device=device).uniform_(-3, 3)
        state = torch.stack([q1, q1d, q2, q2d], dim=1)  # (256, 4)

        near_pi = (1.0 + torch.cos(q1 - math.pi)) / 2.0
        alpha_target = ((near_pi - thresh) / (1.0 - thresh)).clamp(0.0, 1.0)

        h = model.trunk(state)
        alpha_pred = model.alpha_head(h).squeeze().clamp(0.0, 1.0)
        dq_pred = model.dQ_head(h)   # (256, 4)
        dr_pred = model.dR_head(h)   # (256, 2)

        # Alpha: match linear ramp
        loss_alpha = F.mse_loss(alpha_pred, alpha_target)
        # dQ/dR: match reference mean (broadcast target to all batch elements)
        loss_dq = F.mse_loss(dq_pred, dq_mean.unsqueeze(0).expand(256, -1))
        loss_dr = F.mse_loss(dr_pred, dr_mean.unsqueeze(0).expand(256, -1))

        loss = loss_alpha + loss_dq + loss_dr
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 1000 == 999:
            print(f"    step {step+1}/{n_steps}: alpha_loss={loss_alpha.item():.5f}"
                  f"  dq_loss={loss_dq.item():.5f}  dr_loss={loss_dr.item():.5f}", flush=True)

    # Verify
    with torch.no_grad():
        print(f"  dQ_target mean: {dq_mean.tolist()}")
        for deg in [0, 90, 127, 150, 180]:
            q1_val = math.radians(deg)
            np_val = (1 + math.cos(q1_val - math.pi)) / 2
            s = torch.tensor([[q1_val, 0, 0, 0]], dtype=torch.float64, device=device)
            h = model.trunk(s)
            alpha = model.alpha_head(h).squeeze().clamp(0,1).item()
            dq = model.dQ_head(h).squeeze().tolist()
            tgt = max(0, min(1, (np_val - thresh) / (1 - thresh)))
            print(f"    q1={deg:3d}°  α={alpha:.4f}(tgt={tgt:.4f})  dq={[f'{v:.2f}' for v in dq]}", flush=True)


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
    return torch.tensor([math.pi+(random.random()*2-1)*0.30, (random.random()*2-1)*0.6,
                          (random.random()*2-1)*0.25, (random.random()*2-1)*0.6],
                        dtype=torch.float64, device=device)

def sample_bottom_x0(device):
    base = random.choice(BOTTOM_X0_LIST)
    return torch.tensor([base[0]+(random.random()*2-1)*0.1, base[1]+(random.random()*2-1)*0.2,
                          (random.random()*2-1)*0.1, (random.random()*2-1)*0.2],
                        dtype=torch.float64, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--top_frac", type=float, default=0.7)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--pretrain_steps", type=int, default=5000)
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
    dQ_ref = torch.tensor(tp['best_delta_Q'], dtype=torch.float64)  # (9,4) — imitation target only
    dR_ref = torch.tensor(tp['best_delta_R'], dtype=torch.float64)  # (10,2)

    model = CombinedGate(lin_net, hidden=args.hidden).to(device)

    n_params = sum(p.numel() for p in [*model.trunk.parameters(),
                                        *model.alpha_head.parameters(),
                                        *model.dQ_head.parameters(),
                                        *model.dR_head.parameters()])
    print("=" * 80)
    print(f"  EXP SCALEGATE v10: Learned gate + learned boost direction")
    print(f"  Trunk: 4→{args.hidden}→{args.hidden}  |  alpha_head: {args.hidden}→1"
          f"  |  dQ_head: {args.hidden}→4  |  dR_head: {args.hidden}→2")
    print(f"  Total params: {n_params}  |  NO hard-coded dQ_ref at inference")
    print(f"  Imitation init: alpha→linear ramp(0.850), dQ→dQ_ref mean, dR→dR_ref mean")
    print(f"  LR={args.lr}  top_frac={args.top_frac:.0%}  epochs={args.epochs}")
    print("=" * 80)

    if args.pretrain_steps > 0:
        print(f"\n  Imitation pretraining ({args.pretrain_steps} steps)...")
        imitation_pretrain(model, dQ_ref, dR_ref, device,
                           thresh=IMITATION_THRESH, n_steps=args.pretrain_steps)

    print(f"\n  Initial eval (compiling CVXPY ~25 min)...")
    f01, arr, post = eval2k(model, mpc, x0_eval, x_goal)
    print(f"    frac<0.10={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}"
          f"  [target: >87.3%]", flush=True)

    best_f01 = f01
    best_state = {k: v.clone() for k, v in model.state_dict().items()
                  if not k.startswith('lin_net')}

    all_params = list(model.trunk.parameters()) + list(model.alpha_head.parameters()) + \
                 list(model.dQ_head.parameters()) + list(model.dR_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    interrupted = [False]
    signal.signal(signal.SIGINT, lambda s, f: interrupted.__setitem__(0, True))
    t0 = time.time()

    print(f"\n  Training ({args.top_frac:.0%} top / {1-args.top_frac:.0%} bottom)...")
    epoch = 0; top_count = bot_count = 0

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
                best_state = {k: v.clone() for k, v in model.state_dict().items()
                              if not k.startswith('lin_net')}
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [ep={epoch:3d}]  {f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}"
                  f"  top/bot={top_count}/{bot_count}  lr={lr_now:.1e}"
                  f"  t={time.time()-t0:.0f}s{mark}", flush=True)
            if best_f01 >= EXCELLENT_HOLD:
                print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
                break

    for k, v in best_state.items():
        model.state_dict()[k].copy_(v)

    session_name = f"stageE_scalegate_v10_h{args.hidden}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "scalegate_v10", "best_f01": best_f01,
                         "hidden_size": args.hidden, "pretrain_steps": args.pretrain_steps},
        session_name=session_name,
    )
    print(f"\n  DONE. best_f01={best_f01:.1%}  saved: {session_name}")


if __name__ == "__main__":
    main()
