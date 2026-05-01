"""exp_scalegate_v11b.py — v10h + larger hold perturbations + w_stable_phase.

Same architecture as v10h (structural pos_gate(0.85), frozen lambda/mu, QPC tracking).

TWO CHANGES from v10h:
1. Top-start perturbations: q1d up to ±1.5 rad/s (was ±0.6)
   - Targets the exact hold-failure scenarios (near π, moderate velocity)
   - Gives gradient signal for velocity-aware gating near top
2. w_stable_phase=0.5 for bottom-start trajectories (extra direct position loss in last 50 steps)
   - Bottom-start: first 150 steps are swing-up, last 150 steps are hold
   - w_stable_phase adds a direct position tracking loss in the last 50 steps
   - Stronger hold-quality gradient for bottom-start trajectories

HYPOTHESIS: The hold failures (0.8% of post-arrival steps, ~14 steps) happen when
  the pendulum receives q2-q2d perturbations and the gate doesn't provide enough
  restoring force. Larger training perturbations should give the gate more
  direct gradient for these recovery scenarios.

EXPECTED: Same 87.2% initial as v10h (same structural_thresh=0.85), but potentially
  better training convergence or higher ceiling from harder training distribution.
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
GATE_K_INIT       = -1.0
NEAR_PI_SKIP_W    = 6.0
NEAR_PI_SKIP_B    = -5.0
STRUCTURAL_THRESH = 0.85   # same as v10h (optimal)
STRUCTURAL_K      = 50.0
W_STABLE_PHASE    = 0.5    # extra position loss in last 50 steps of bottom-start
STABLE_PHASE_STEPS = 50


class GateVelMLP(nn.Module):
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
        self.alpha_head = nn.Linear(hidden + 1, 1, dtype=torch.float64)
        self.gate_k_raw = nn.Parameter(
            torch.tensor(GATE_K_INIT, dtype=torch.float64))

        nn.init.zeros_(self.alpha_head.weight)
        self.alpha_head.weight.data[0, hidden] = NEAR_PI_SKIP_W
        nn.init.constant_(self.alpha_head.bias, NEAR_PI_SKIP_B)

    @property
    def gate_k_eff(self):
        return F.softplus(self.gate_k_raw)

    def _features(self, x_last):
        near_pi = (1.0 + torch.cos(x_last[0] - math.pi)) / 2.0
        return torch.stack([near_pi, x_last[1], x_last[2], x_last[3]])

    def _gate(self, x_last):
        features = self._features(x_last)
        near_pi  = features[0]
        h = self.trunk(features.unsqueeze(0))
        near_pi_skip = near_pi.unsqueeze(0).unsqueeze(0)
        alpha_input  = torch.cat([h, near_pi_skip], dim=1)
        alpha_raw = torch.sigmoid(self.alpha_head(alpha_input).squeeze())
        q1d = x_last[1]
        vel_gate = (1.0 - self.gate_k_eff * q1d * q1d).clamp(0.0, 1.0)
        pos_gate = torch.sigmoid(
            torch.tensor(STRUCTURAL_K, dtype=torch.float64) * (near_pi - STRUCTURAL_THRESH))
        alpha_eff = alpha_raw * vel_gate * pos_gate
        return alpha_eff, alpha_raw, vel_gate, pos_gate

    def get_alpha(self, x_sequence):
        alpha_eff, _, _, _ = self._gate(x_sequence[-1])
        return alpha_eff

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        alpha, _, _, _ = self._gate(x_sequence[-1])
        gQ = gQ + alpha * self.dQ_ref
        gR = gR + alpha * self.dR_ref
        fe = fe * (1.0 - alpha)
        return gQ, gR, fe, qd, rd, gQf


def probe_gate(model, device, header="Gate profile:"):
    print(f"  {header}")
    with torch.no_grad():
        k_eff = model.gate_k_eff.item()
        vel_thresh = (1.0 / k_eff) ** 0.5 if k_eff > 0 else float('inf')
        print(f"    k_eff={k_eff:.4f}  vel_gate_off_at_q1d={vel_thresh:.2f}"
              f"  struct_thresh={STRUCTURAL_THRESH}")
        for deg, q1d_val in [(0,0),(90,0),(127,0),(140,0),(165,0),(180,0),(180,1),(180,2)]:
            q1 = math.radians(deg)
            x = torch.tensor([q1, q1d_val, 0, 0], dtype=torch.float64, device=device)
            alpha_eff, alpha_raw, vel_f, pos_f = model._gate(x)
            print(f"    q1={deg:3d}° q1d={q1d_val}"
                  f"  α_raw={alpha_raw:.4f}  vel={vel_f:.4f}  pos={pos_f:.4f}"
                  f"  α_eff={alpha_eff:.4f}", flush=True)


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
    """Larger perturbation: q1d up to ±1.5 rad/s."""
    return torch.tensor([math.pi + (random.random()*2-1)*0.40,
                         (random.random()*2-1)*1.5,
                         (random.random()*2-1)*0.30,
                         (random.random()*2-1)*0.8],
                        dtype=torch.float64, device=device)

def sample_bottom_x0(device):
    base = random.choice(BOTTOM_X0_LIST)
    return torch.tensor([base[0]+(random.random()*2-1)*0.1, base[1]+(random.random()*2-1)*0.2,
                          (random.random()*2-1)*0.1, (random.random()*2-1)*0.2],
                        dtype=torch.float64, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--top_frac", type=float, default=0.7)
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

    trainable = (list(model.trunk.parameters()) +
                 list(model.alpha_head.parameters()) +
                 [model.gate_k_raw])
    n_trainable = sum(p.numel() for p in trainable)

    k_eff_init = F.softplus(torch.tensor(GATE_K_INIT)).item()
    vel_thresh_init = (1.0 / k_eff_init) ** 0.5
    print("=" * 80)
    print(f"  EXP SCALEGATE v11b: v10h + larger perturbations + w_stable_phase")
    print(f"  Same structural_thresh={STRUCTURAL_THRESH} as v10h (optimal)")
    print(f"  Top perturbation: q1d up to ±1.5 rad/s (was ±0.6) — targets hold failures")
    print(f"  Bottom-start: w_stable_phase={W_STABLE_PHASE} (extra pos loss, last {STABLE_PHASE_STEPS} steps)")
    print(f"  Init skip: W={NEAR_PI_SKIP_W}, B={NEAR_PI_SKIP_B} → alpha_raw: 0.007..0.731")
    print(f"  vel_gate: k_eff≈{k_eff_init:.3f} → off at q1d≥{vel_thresh_init:.2f}")
    print(f"  Trunk: 4→{args.hidden}→{args.hidden}  |  {n_trainable} trainable params  |  lambda/mu FROZEN")
    print(f"  NO pretraining. LR={args.lr}  top_frac={args.top_frac:.0%}  epochs={args.epochs}")
    print(f"  NUM_STEPS={NUM_STEPS}")
    print("=" * 80)

    probe_gate(model, device, "Initial gate profile:")

    print(f"\n  Initial eval (compiling CVXPY ~25 min)...")
    f01, arr, post = eval2k(model, mpc, x0_eval, x_goal)
    print(f"    frac<0.10={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}", flush=True)

    best_f01 = f01
    best_state = {k: v.clone() for k, v in model.state_dict().items()
                  if not k.startswith('lin_net')}

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
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
            w_stable_phase=0.0 if is_top else W_STABLE_PHASE,
            stable_phase_steps=STABLE_PHASE_STEPS,
            f_gate_thresh=0.0,
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
            with torch.no_grad():
                x_pi  = torch.tensor([math.pi, 0.0, 0.0, 0.0], dtype=torch.float64, device=device)
                x_mid = torch.tensor([math.radians(127), 0.0, 0.0, 0.0], dtype=torch.float64, device=device)
                ae_pi,  ar_pi,  _, _ = model._gate(x_pi)
                ae_mid, ar_mid, _, _ = model._gate(x_mid)
            print(f"  [ep={epoch:3d}]  {f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}"
                  f"  k={k_eff:.4f}  α_raw@π={ar_pi:.4f}  α_eff@π={ae_pi:.4f}"
                  f"  α_eff@127°={ae_mid:.4f}"
                  f"  top/bot={top_count}/{bot_count}  lr={lr_now:.1e}"
                  f"  t={time.time()-t0:.0f}s{mark}", flush=True)
            if best_f01 >= EXCELLENT_HOLD:
                print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
                break

    probe_gate(model, device, "Final gate profile:")

    session_name = f"stageE_scalegate_v11b_h{args.hidden}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "scalegate_v11b", "best_f01": best_f01,
                         "hidden_size": args.hidden, "gate_k_init": GATE_K_INIT,
                         "structural_thresh": STRUCTURAL_THRESH,
                         "near_pi_skip_w": NEAR_PI_SKIP_W, "near_pi_skip_b": NEAR_PI_SKIP_B,
                         "w_stable_phase": W_STABLE_PHASE},
        session_name=session_name,
    )
    print(f"\n  DONE. best_f01={best_f01:.1%}  saved: {session_name}")


if __name__ == "__main__":
    main()
