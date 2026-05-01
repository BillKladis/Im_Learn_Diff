"""exp_scalegate_v10b.py — Learned gate + learned per-dimension direction scaling.

MOTIVATION:
  v10 (original) tried to learn a completely free (4,) direction vector broadcast over
  all H-1 horizon steps. This LOSES the per-step structure of dQ_ref, which matters:
  initial eval was 82.5% instead of the expected 87.3%.

  v10b fixes this: use dQ_ref as a frozen buffer (preserving per-step structure) but
  add a state-dependent per-DIMENSION scale vector lambda ∈ R^4. This allows the
  network to learn e.g. "when approaching from the right, reduce the q2 correction"
  while preserving the time-varying structure of dQ_ref.

DESIGN:
  GateWithDimScale: [near_pi,q1d,q2,q2d] → (alpha scalar, lambda 4-dim)
    near_pi = (1+cos(q1-π))/2 ∈ [0,1]
    alpha = clamp(alpha_head(trunk), 0, 1)       WHEN to boost
    lambda = lambda_head(trunk)                   HOW MUCH each dimension

  Apply: gQ += alpha * (lambda ⊙ dQ_ref)    — preserves per-step structure, scales dims
         gR += alpha * (mu ⊙ dR_ref)         — mu is 2-dim for R dimensions
         fe *= (1 - alpha)

  ⊙ = element-wise broadcast: lambda:(4,) × dQ_ref:(H-1,4) → (H-1,4)

INITIALIZATION:
  alpha_head: match linear ramp thresh=0.850  [same as v9]
  lambda_head: output ones  [identical to v9 at init → expects 87.3% initial eval]
  mu_head: output ones

WHY THIS IS BETTER THAN v10:
  v10: broadcast dQ_mean over all steps → loses per-step variation → 82.5%
  v10b: scale dQ_ref per-dimension → preserves per-step structure → expect 87.3%

  Additionally, lambda can vary by state (e.g. larger q2 correction when q2 is large)
  without destroying the per-step horizon structure.

GOAL: Discover dimension-wise reweighting of dQ_ref that improves upon 87.3%.
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


class GateWithDimScale(nn.Module):
    """Learned gate + state-dependent per-dimension scale on dQ_ref/dR_ref.

    Shared trunk: [near_pi,q1d,q2,q2d] → hidden features
    Heads:
      alpha_head: hidden → scalar alpha ∈ [0,1]  (WHEN to boost)
      lambda_head: hidden → 4-dim lambda          (HOW MUCH per Q dimension)
      mu_head:     hidden → 2-dim mu              (HOW MUCH per R dimension)

    Apply: gQ += alpha * (lambda ⊙ dQ_ref)   — lambda broadcast over H-1 steps
           gR += alpha * (mu ⊙ dR_ref)        — mu broadcast over H steps
           fe *= (1 - alpha)

    At init: lambda=ones, mu=ones → identical to v9/linear ramp → 87.3%.
    Training discovers per-dimension reweighting from rollout quality.
    """
    def __init__(self, lin_net, dQ_ref, dR_ref, hidden=16):
        super().__init__()
        self.lin_net = lin_net
        self.register_buffer('dQ_ref', dQ_ref.clone())  # (H-1, 4)
        self.register_buffer('dR_ref', dR_ref.clone())  # (H, 2)
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim
        self.control_dim = lin_net.control_dim

        self.trunk = nn.Sequential(
            nn.Linear(4, hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
            nn.ReLU(),
        )
        self.alpha_head  = nn.Linear(hidden, 1, dtype=torch.float64)
        self.lambda_head = nn.Linear(hidden, 4, dtype=torch.float64)  # Q dim scales
        self.mu_head     = nn.Linear(hidden, 2, dtype=torch.float64)  # R dim scales

        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        # lambda/mu init to output ones: bias=1, weight=0
        nn.init.zeros_(self.lambda_head.weight)
        nn.init.ones_(self.lambda_head.bias)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.ones_(self.mu_head.bias)

    def _features(self, x_last):
        near_pi = (1.0 + torch.cos(x_last[0] - math.pi)) / 2.0
        return torch.stack([near_pi, x_last[1], x_last[2], x_last[3]])

    def _heads(self, x_last):
        h = self.trunk(self._features(x_last).unsqueeze(0))
        alpha  = self.alpha_head(h).squeeze().clamp(0.0, 1.0)
        lam    = self.lambda_head(h).squeeze()   # (4,) — unconstrained
        mu     = self.mu_head(h).squeeze()       # (2,) — unconstrained
        return alpha, lam, mu

    def get_alpha(self, x_sequence):
        x_last = x_sequence[-1]
        h = self.trunk(self._features(x_last).unsqueeze(0))
        return self.alpha_head(h).squeeze().clamp(0.0, 1.0)

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        alpha, lam, mu = self._heads(x_sequence[-1])
        # lam:(4,) × dQ_ref:(H-1,4) — broadcasts correctly
        gQ = gQ + alpha * (lam * self.dQ_ref)
        gR = gR + alpha * (mu  * self.dR_ref)
        fe = fe * (1.0 - alpha)
        return gQ, gR, fe, qd, rd, gQf


def imitation_pretrain(model, device, thresh=IMITATION_THRESH, n_steps=5000, lr=1e-3):
    """Pretrain: alpha → linear ramp, lambda → ones, mu → ones."""
    params = (list(model.trunk.parameters()) + list(model.alpha_head.parameters()) +
              list(model.lambda_head.parameters()) + list(model.mu_head.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    ones4 = torch.ones(4, dtype=torch.float64, device=device)
    ones2 = torch.ones(2, dtype=torch.float64, device=device)

    for step in range(n_steps):
        q1  = torch.empty(256, dtype=torch.float64, device=device).uniform_(0, 2*math.pi)
        q1d = torch.empty(256, dtype=torch.float64, device=device).uniform_(-3, 3)
        q2  = torch.empty(256, dtype=torch.float64, device=device).uniform_(-1, 1)
        q2d = torch.empty(256, dtype=torch.float64, device=device).uniform_(-3, 3)
        near_pi = (1.0 + torch.cos(q1 - math.pi)) / 2.0
        alpha_target = ((near_pi - thresh) / (1.0 - thresh)).clamp(0.0, 1.0)
        state = torch.stack([near_pi, q1d, q2, q2d], dim=1)  # (256, 4)

        h = model.trunk(state)
        alpha_pred  = model.alpha_head(h).squeeze().clamp(0.0, 1.0)
        lambda_pred = model.lambda_head(h)   # (256, 4)
        mu_pred     = model.mu_head(h)       # (256, 2)

        loss = (F.mse_loss(alpha_pred, alpha_target) +
                F.mse_loss(lambda_pred, ones4.unsqueeze(0).expand(256, -1)) +
                F.mse_loss(mu_pred,     ones2.unsqueeze(0).expand(256, -1)))
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 1000 == 999:
            print(f"    step {step+1}/{n_steps}: loss={loss.item():.5f}", flush=True)

    with torch.no_grad():
        print(f"  Verification (q1d=0, q2=0, q2d=0):")
        for deg in [0, 90, 127, 150, 180]:
            q1_val = math.radians(deg)
            np_val = (1 + math.cos(q1_val - math.pi)) / 2
            s = torch.tensor([[np_val, 0.0, 0.0, 0.0]], dtype=torch.float64, device=device)
            h = model.trunk(s)
            alpha = model.alpha_head(h).squeeze().clamp(0,1).item()
            lam   = model.lambda_head(h).squeeze().tolist()
            tgt   = max(0, min(1, (np_val - thresh) / (1 - thresh)))
            print(f"    q1={deg:3d}°  α={alpha:.4f}(tgt={tgt:.4f})  λ={[f'{v:.3f}' for v in lam]}", flush=True)


def probe_gate(model, device, header="Gate profile:"):
    print(f"  {header}")
    with torch.no_grad():
        for deg, q1d_val in [(0,0),(90,0),(127,0),(140,0),(165,0),(180,0),(180,1),(180,2)]:
            q1 = math.radians(deg)
            np_val = (1 + math.cos(q1 - math.pi)) / 2
            x = torch.tensor([q1, q1d_val, 0, 0], dtype=torch.float64, device=device)
            xseq = x.unsqueeze(0).expand(STATE_HIST, -1)
            alpha, lam, mu = model._heads(x)
            ramp = max(0, min(1, (np_val - IMITATION_THRESH) / (1 - IMITATION_THRESH)))
            print(f"    q1={deg:3d}° q1d={q1d_val}  α={alpha:.4f}  ramp={ramp:.4f}"
                  f"  λ={[f'{v:.3f}' for v in lam.tolist()]}", flush=True)


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
    dQ_ref = torch.tensor(tp['best_delta_Q'], dtype=torch.float64).to(device)
    dR_ref = torch.tensor(tp['best_delta_R'], dtype=torch.float64).to(device)

    model = GateWithDimScale(lin_net, dQ_ref, dR_ref, hidden=args.hidden).to(device)

    n_params = sum(p.numel() for p in [*model.trunk.parameters(),
                                        *model.alpha_head.parameters(),
                                        *model.lambda_head.parameters(),
                                        *model.mu_head.parameters()])
    print("=" * 80)
    print(f"  EXP SCALEGATE v10b: Learned gate + per-dimension dQ_ref scaling")
    print(f"  Trunk: 4→{args.hidden}→{args.hidden}  |  alpha_head: 1  |  lambda_head: 4  |  mu_head: 2")
    print(f"  Total trainable params: {n_params}")
    print(f"  Apply: gQ += alpha * (lambda ⊙ dQ_ref)  [preserves per-step structure]")
    print(f"  Init: alpha_head zeros (α=0), lambda_head ones (λ=1), pretrain_steps={args.pretrain_steps}")
    print(f"  LR={args.lr}  top_frac={args.top_frac:.0%}  epochs={args.epochs}")
    print("=" * 80)

    if args.pretrain_steps > 0:
        print(f"\n  Imitation pretraining ({args.pretrain_steps} steps)...")
        imitation_pretrain(model, device, thresh=IMITATION_THRESH, n_steps=args.pretrain_steps)
        probe_gate(model, device, "Gate after imitation:")
    else:
        probe_gate(model, device, "Initial gate (no pretraining):")

    print(f"\n  Initial eval (compiling CVXPY ~25 min)...")
    f01, arr, post = eval2k(model, mpc, x0_eval, x_goal)
    print(f"    frac<0.10={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}"
          f"  [baseline: 87.3%]", flush=True)

    best_f01 = f01
    non_linnet_params = {k: v for k, v in model.state_dict().items() if not k.startswith('lin_net')}
    best_state = {k: v.clone() for k, v in non_linnet_params.items()}

    all_params = (list(model.trunk.parameters()) + list(model.alpha_head.parameters()) +
                  list(model.lambda_head.parameters()) + list(model.mu_head.parameters()))
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
        dest = model.state_dict().get(k)
        if dest is not None:
            dest.copy_(v)

    probe_gate(model, device, "Final gate profile:")

    session_name = f"stageE_scalegate_v10b_h{args.hidden}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "scalegate_v10b", "best_f01": best_f01,
                         "hidden_size": args.hidden, "pretrain_steps": args.pretrain_steps},
        session_name=session_name,
    )
    print(f"\n  DONE. best_f01={best_f01:.1%}  saved: {session_name}")


if __name__ == "__main__":
    main()
