"""exp_scalegate_v10c.py — Learned gate with structural velocity suppressor.

LESSONS FROM v10b FAILURES:
  v10b (LR=0.01 and LR=0.001, top_frac=0.5-0.7): collapsed to 0.0% at ep=10 both times.
  Root cause: MLP with no velocity inductive bias. Top-start gradient pushed alpha up
  for all near-π states (including high-velocity approach), disrupting swing-up.
  The MLP cannot learn the velocity-position interaction without structural help.

DESIGN:
  Same as v10b but adds a structural velocity suppressor:

    k_eff = softplus(gate_k_raw)  [global scalar, ≥0, init ≈ 0.313]
    vel_factor = clamp(1 - k_eff × q1d², 0, 1)
    alpha_effective = alpha_raw × vel_factor

  Where alpha_raw comes from the MLP trunk (same as v10b).

  At init (k_eff≈0.313):
    q1d=0:   vel_factor=1.00  (no suppression — holding)
    q1d=1:   vel_factor=0.69  (31% suppression)
    q1d=1.8: vel_factor≈0     (gate fully off at ≥1.78 rad/s)
    q1d=2:   vel_factor=0     (clamped — swing-up approach suppressed)

  This structurally prevents gate from disrupting swing-up (where q1d≈2-4 rad/s).
  The specific value of k_eff is LEARNED from rollout quality, not hand-tuned.

  Additionally: NUM_STEPS=300 (> arr=242) so bottom-start rollouts see BOTH
  the swing-up phase AND the hold phase in a single training rollout. This gives
  the network proper velocity-specific learning signal.

PARAMETERS:
  Trunk: [near_pi, q1d, q2, q2d] → 16 → 16 → hidden
  alpha_head: hidden → scalar raw gate (zeros init → alpha_raw=0 at start)
  gate_k_raw: global scalar, init=-1.0 → k_eff=softplus(-1.0)≈0.313
  lambda_head: hidden → 4 Q scales (ones bias init → lambda=ones at start)
  mu_head: hidden → 2 R scales (ones bias init → mu=ones at start)

INIT BEHAVIOR (before training):
  alpha_effective=0 everywhere → same as no gate → ~5% f01 (no fe suppression)
  Training must discover BOTH when to activate (alpha_raw) AND fix velocity (gate_k_raw).
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
STATE_HIST = 5; NUM_STEPS = 300; SAVE_EVERY = 10; EXCELLENT_HOLD = 0.92
BOTTOM_X0_LIST = [[0,0,0,0],[0.2,0,0,0],[-0.2,0,0,0],[0.5,0,0,0],[-0.5,0,0,0]]
GATE_K_INIT = -1.0  # → k_eff = softplus(-1.0) ≈ 0.313, suppresses gate at q1d≥1.78


class GateVelMLP(nn.Module):
    """MLP gate with structural velocity suppressor.

    alpha_effective = alpha_raw(trunk) × clamp(1 - k_eff × q1d², 0, 1)
    gQ += alpha_effective × (lambda ⊙ dQ_ref)
    gR += alpha_effective × (mu ⊙ dR_ref)
    fe *= (1 - alpha_effective)
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

        self.trunk = nn.Sequential(
            nn.Linear(4, hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
            nn.ReLU(),
        )
        self.alpha_head  = nn.Linear(hidden, 1, dtype=torch.float64)
        self.lambda_head = nn.Linear(hidden, 4, dtype=torch.float64)
        self.mu_head     = nn.Linear(hidden, 2, dtype=torch.float64)

        # gate_k_raw: global velocity suppressor (not state-dependent)
        self.gate_k_raw = nn.Parameter(
            torch.tensor(GATE_K_INIT, dtype=torch.float64))

        # alpha starts at 0 (no gate initially)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        # lambda/mu start at ones (neutral direction scaling)
        nn.init.zeros_(self.lambda_head.weight)
        nn.init.ones_(self.lambda_head.bias)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.ones_(self.mu_head.bias)

    @property
    def gate_k_eff(self):
        return F.softplus(self.gate_k_raw)

    def _features(self, x_last):
        near_pi = (1.0 + torch.cos(x_last[0] - math.pi)) / 2.0
        return torch.stack([near_pi, x_last[1], x_last[2], x_last[3]])

    def _gate(self, x_last):
        """Returns (alpha_effective, alpha_raw, vel_factor, lam, mu)."""
        features = self._features(x_last)
        h = self.trunk(features.unsqueeze(0))
        alpha_raw  = self.alpha_head(h).squeeze().clamp(0.0, 1.0)
        lam = self.lambda_head(h).squeeze()
        mu  = self.mu_head(h).squeeze()
        q1d = x_last[1]
        vel_factor = (1.0 - self.gate_k_eff * q1d * q1d).clamp(0.0, 1.0)
        alpha_eff = alpha_raw * vel_factor
        return alpha_eff, alpha_raw, vel_factor, lam, mu

    def get_alpha(self, x_sequence):
        alpha_eff, _, _, _, _ = self._gate(x_sequence[-1])
        return alpha_eff

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        alpha, _, _, lam, mu = self._gate(x_sequence[-1])
        gQ = gQ + alpha * (lam * self.dQ_ref)
        gR = gR + alpha * (mu  * self.dR_ref)
        fe = fe * (1.0 - alpha)
        return gQ, gR, fe, qd, rd, gQf


def probe_gate(model, device, header="Gate profile:"):
    print(f"  {header}")
    with torch.no_grad():
        k_eff = model.gate_k_eff.item()
        vel_thresh = (1.0 / k_eff) ** 0.5 if k_eff > 0 else float('inf')
        print(f"    k_eff={k_eff:.4f}  vel_gate_off_at_q1d={vel_thresh:.2f}")
        for deg, q1d_val in [(0,0),(90,0),(127,0),(140,0),(165,0),(180,0),(180,1),(180,2)]:
            q1 = math.radians(deg)
            np_val = (1 + math.cos(q1 - math.pi)) / 2
            x = torch.tensor([q1, q1d_val, 0, 0], dtype=torch.float64, device=device)
            alpha_eff, alpha_raw, vel_f, lam, _ = model._gate(x)
            print(f"    q1={deg:3d}° q1d={q1d_val}"
                  f"  α_raw={alpha_raw:.4f}  vel={vel_f:.4f}  α_eff={alpha_eff:.4f}"
                  f"  λ={[f'{v:.2f}' for v in lam.tolist()]}", flush=True)


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
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--top_frac", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=16)
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

    model = GateVelMLP(lin_net, dQ_ref, dR_ref, hidden=args.hidden).to(device)

    n_params = sum(p.numel() for p in [*model.trunk.parameters(),
                                        *model.alpha_head.parameters(),
                                        model.gate_k_raw,
                                        *model.lambda_head.parameters(),
                                        *model.mu_head.parameters()])
    print("=" * 80)
    print(f"  EXP SCALEGATE v10c: MLP gate + structural velocity suppressor")
    print(f"  alpha_eff = alpha_raw(MLP) × clamp(1 - k_eff×q1d², 0, 1)")
    print(f"  k_eff = softplus({GATE_K_INIT}) ≈ {F.softplus(torch.tensor(GATE_K_INIT)).item():.3f}")
    print(f"  Trunk: 4→{args.hidden}→{args.hidden}  |  {n_params} total params")
    print(f"  NO pretraining. alpha=0 start, k_eff={F.softplus(torch.tensor(GATE_K_INIT)).item():.3f} (suppresses q1d≥{(1/F.softplus(torch.tensor(GATE_K_INIT)).item())**0.5:.2f})")
    print(f"  NUM_STEPS={NUM_STEPS} (> arr=242 so bottom-start sees hold phase)")
    print(f"  LR={args.lr}  top_frac={args.top_frac:.0%}  epochs={args.epochs}")
    print("=" * 80)

    probe_gate(model, device, "Initial gate profile:")

    print(f"\n  Initial eval (compiling CVXPY ~25 min)...")
    f01, arr, post = eval2k(model, mpc, x0_eval, x_goal)
    print(f"    frac<0.10={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}", flush=True)

    best_f01 = f01
    best_state = {k: v.clone() for k, v in model.state_dict().items()
                  if not k.startswith('lin_net')}

    all_params = (list(model.trunk.parameters()) + list(model.alpha_head.parameters()) +
                  [model.gate_k_raw] +
                  list(model.lambda_head.parameters()) + list(model.mu_head.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    interrupted = [False]
    signal.signal(signal.SIGINT, lambda s, f: interrupted.__setitem__(0, True))
    t0 = time.time()

    print(f"\n  Training ({args.top_frac:.0%} top / {1-args.top_frac:.0%} bottom, "
          f"NUM_STEPS={NUM_STEPS})...")
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
            k_eff = model.gate_k_eff.item()
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [ep={epoch:3d}]  {f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}"
                  f"  k={k_eff:.4f}  top/bot={top_count}/{bot_count}  lr={lr_now:.1e}"
                  f"  t={time.time()-t0:.0f}s{mark}", flush=True)
            if best_f01 >= EXCELLENT_HOLD:
                print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
                break

    probe_gate(model, device, "Final gate profile:")

    session_name = f"stageE_scalegate_v10c_h{args.hidden}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "scalegate_v10c", "best_f01": best_f01,
                         "hidden_size": args.hidden, "gate_k_init": GATE_K_INIT},
        session_name=session_name,
    )
    print(f"\n  DONE. best_f01={best_f01:.1%}  saved: {session_name}")


if __name__ == "__main__":
    main()
