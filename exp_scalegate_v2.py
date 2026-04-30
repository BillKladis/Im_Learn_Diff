"""exp_scalegate_v2.py — Scalar gate with error_norm as direct input.

MOTIVATION: v1 uses [cos, sin, q1d, q2, q2d] and has to DISCOVER the error_norm
threshold. v2 passes error_norm explicitly — the pre-train trivially learns the
exact 0.80 threshold (α = heaviside(0.8 - error_norm)), and end-to-end training
can then discover if a different threshold (0.85? 0.90?) beats 87.2%.

INPUT: [error_norm, cos(q1-π)] — norm captures distance from goal, cos tells
which direction we're approaching from.
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

BOTTOM_X0_LIST = [
    [0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0], [-0.2, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0],
]


def compute_error_norm(q1, q1d, q2, q2d):
    dq1 = math.atan2(math.sin(q1 - math.pi), math.cos(q1 - math.pi))
    return math.sqrt(dq1**2 + q1d**2 + q2**2 + q2d**2)


def compute_error_norm_tensor(x):
    """Compute error norm from state tensor [q1, q1d, q2, q2d]."""
    dq1 = torch.atan2(torch.sin(x[0] - math.pi), torch.cos(x[0] - math.pi))
    return torch.sqrt(dq1**2 + x[1]**2 + x[2]**2 + x[3]**2)


class ErrorNormGateWrapper(nn.Module):
    """Scalar gate using error_norm as the primary input.

    gate input: [error_norm, cos(q1-π)]  (2 features)
    gate output: α ∈ [0,1] via Sigmoid
    Q_adj = α × dQ_ref (fixed direction)
    """
    def __init__(self, lin_net, dQ_ref, hidden=8):
        super().__init__()
        self.lin_net = lin_net
        self.register_buffer('dQ_ref', dQ_ref.clone())
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim
        self.control_dim = lin_net.control_dim

        self.gate_net = nn.Sequential(
            nn.Linear(2, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=torch.float64),
            nn.Sigmoid(),
        )
        self._init_gate()

    def _init_gate(self):
        """Initialize so α starts near 0 everywhere."""
        for m in self.gate_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.05)
                nn.init.zeros_(m.bias)

    def get_gate_features(self, x_sequence):
        x = x_sequence[-1]
        en = compute_error_norm_tensor(x)
        cos_q1 = torch.cos(x[0] - math.pi)
        return torch.stack([en, cos_q1])

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        alpha = self.gate_net(self.get_gate_features(x_sequence)).squeeze()
        Q_adj = alpha * self.dQ_ref
        gQ = (gQ + Q_adj.unsqueeze(0)).clamp(min=0.01)
        fe = fe * (1.0 - alpha)
        return gQ, gR, fe, qd, rd, gQf


def supervised_pretrain(model, device, boost_thresh, n_steps=1000, lr=1e-2):
    """Train gate to match error_norm < boost_thresh threshold."""
    opt = torch.optim.Adam(model.gate_net.parameters(), lr=lr)
    print(f"  Supervised pre-train: {n_steps} steps, threshold={boost_thresh}...")

    for step in range(1, n_steps + 1):
        opt.zero_grad()
        losses = []

        # Active: sample states with error_norm < boost_thresh
        for _ in range(8):
            for _ in range(50):
                q1 = math.pi + random.gauss(0, 0.25)
                q1d, q2, q2d = random.gauss(0, 0.25), random.gauss(0, 0.20), random.gauss(0, 0.20)
                if compute_error_norm(q1, q1d, q2, q2d) < boost_thresh:
                    break
            x = torch.tensor([q1, q1d, q2, q2d], device=device, dtype=torch.float64)
            xseq = x.unsqueeze(0).expand(STATE_HIST, -1)
            alpha = model.gate_net(model.get_gate_features(xseq)).squeeze()
            losses.append(F.binary_cross_entropy(alpha, torch.tensor(1.0, device=device, dtype=torch.float64)))

        # Inactive: sample states with error_norm >= boost_thresh
        for _ in range(8):
            choice = random.random()
            if choice < 0.4:
                # Bottom
                base = random.choice(BOTTOM_X0_LIST)
                q1 = base[0] + random.gauss(0, 0.3)
                q1d, q2, q2d = base[1] + random.gauss(0, 0.5), random.gauss(0, 0.3), random.gauss(0, 0.5)
            else:
                # Intermediate (50°-127° from top)
                angle = math.radians(50 + random.random() * 77)
                q1 = math.pi + (1 if random.random() > 0.5 else -1) * angle
                q1d, q2, q2d = random.gauss(0, 0.5), random.gauss(0, 0.3), random.gauss(0, 0.5)
            if compute_error_norm(q1, q1d, q2, q2d) < boost_thresh:
                continue  # skip if accidentally active
            x = torch.tensor([q1, q1d, q2, q2d], device=device, dtype=torch.float64)
            xseq = x.unsqueeze(0).expand(STATE_HIST, -1)
            alpha = model.gate_net(model.get_gate_features(xseq)).squeeze()
            losses.append(F.binary_cross_entropy(alpha, torch.tensor(0.0, device=device, dtype=torch.float64)))

        if not losses:
            continue
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            with torch.no_grad():
                xt = torch.tensor([math.pi, 0, 0, 0], device=device, dtype=torch.float64)
                at = model.gate_net(model.get_gate_features(xt.unsqueeze(0).expand(STATE_HIST,-1))).item()
                xb = torch.tensor([0.0, 0, 0, 0], device=device, dtype=torch.float64)
                ab = model.gate_net(model.get_gate_features(xb.unsqueeze(0).expand(STATE_HIST,-1))).item()
                # at exactly the threshold
                xm = torch.tensor([math.pi + 0.93, 0, 0, 0], device=device, dtype=torch.float64)
                am = model.gate_net(model.get_gate_features(xm.unsqueeze(0).expand(STATE_HIST,-1))).item()
            print(f"    step {step:4d}: loss={loss.item():.4f} | "
                  f"α@top={at:.3f}  α@thresh(en={boost_thresh})={am:.3f}  α@bot={ab:.3f}", flush=True)

    print(f"  Pre-train complete.", flush=True)


def probe_gate(model, device):
    print(f"  Gate profile (q1d=q2=q2d=0):")
    for deg in [0, 30, 60, 90, 120, 127, 150, 180]:
        q1 = math.radians(deg)
        x = torch.tensor([q1, 0, 0, 0], device=device, dtype=torch.float64)
        xseq = x.unsqueeze(0).expand(STATE_HIST, -1)
        with torch.no_grad():
            a = model.gate_net(model.get_gate_features(xseq)).item()
            en = compute_error_norm(q1, 0, 0, 0)
        print(f"    q1={deg:3d}°  en={en:.3f}  α={a:.4f}", flush=True)


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
        math.pi + (random.random() * 2 - 1) * 0.30,
        (random.random() * 2 - 1) * 0.6,
        (random.random() * 2 - 1) * 0.25,
        (random.random() * 2 - 1) * 0.6,
    ], device=device, dtype=torch.float64)


def sample_bottom_x0(device):
    base = random.choice(BOTTOM_X0_LIST)
    return torch.tensor([
        base[0] + (random.random() * 2 - 1) * 0.1,
        base[1] + (random.random() * 2 - 1) * 0.2,
        (random.random() * 2 - 1) * 0.1,
        (random.random() * 2 - 1) * 0.2,
    ], device=device, dtype=torch.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--top_frac", type=float, default=0.7)
    parser.add_argument("--pretrain_steps", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--boost_thresh", type=float, default=0.80,
                        help="Pre-train threshold for error_norm (default=0.80 matches HoldBoostWrapper)")
    parser.add_argument("--no_pretrain", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0_eval = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal  = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0_eval, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.requires_grad_(False)

    ckpt = torch.load(SCALE4_CKPT, map_location='cpu', weights_only=False)
    tp = ckpt['metadata']['training_params']
    dQ = torch.tensor(tp['best_delta_Q'], dtype=torch.float64)
    if dQ.dim() > 1:
        dQ = dQ.mean(dim=0)
    dQ_ref = dQ.to(device)

    model = ErrorNormGateWrapper(lin_net, dQ_ref, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.gate_net.parameters())

    print("=" * 80)
    print("  EXP SCALEGATE v2: Error-norm gate (trivially learnable threshold)")
    print(f"  lin_net: FROZEN  gate_net: {n_params} params  hidden={args.hidden}")
    print(f"  Gate: [error_norm, cos(q1-π)] → α ∈ [0,1]")
    print(f"  Q_adj = α × dQ_ref  dQ_ref={dQ_ref.tolist()}")
    print(f"  Pre-train threshold: {args.boost_thresh}")
    print(f"  LR={args.lr}  top_frac={args.top_frac:.0%}  epochs={args.epochs}")
    print("=" * 80)

    if not args.no_pretrain and args.pretrain_steps > 0:
        supervised_pretrain(model, device, args.boost_thresh, n_steps=args.pretrain_steps)

    probe_gate(model, device)

    print(f"\n  Initial eval (compiling CVXPY — ~25 min)...")
    f01, arr, post = eval2k(model, mpc, x0_eval, x_goal)
    print(f"    frac<0.10={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}  [baseline=87.2%]",
          flush=True)

    best_f01 = f01
    best_state = {k: v.clone() for k, v in model.state_dict().items() if 'gate' in k}
    gate_params = list(model.gate_net.parameters())
    optimizer = torch.optim.AdamW(gate_params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

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
            f01, arr, post = eval2k(model, mpc, x0_eval, x_goal)
            with torch.no_grad():
                xt = torch.tensor([math.pi, 0, 0, 0], device=device, dtype=torch.float64)
                at = model.gate_net(model.get_gate_features(xt.unsqueeze(0).expand(STATE_HIST,-1))).item()
                xb = torch.tensor([0.0, 0, 0, 0], device=device, dtype=torch.float64)
                ab = model.gate_net(model.get_gate_features(xb.unsqueeze(0).expand(STATE_HIST,-1))).item()
            mark = ""
            if f01 > best_f01:
                mark = " ★"
                best_f01 = f01
                best_state = {k: v.clone() for k, v in model.state_dict().items() if 'gate' in k}
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [ep={epoch:3d}]  {f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}"
                  f"  α@top={at:.3f}  α@bot={ab:.3f}"
                  f"  top/bot={top_count}/{bot_count}  lr={lr_now:.1e}"
                  f"  t={time.time()-t0:.0f}s{mark}", flush=True)

            if best_f01 >= EXCELLENT_HOLD:
                print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
                break

    state = model.state_dict()
    state.update(best_state)
    model.load_state_dict(state)

    session_name = (f"stageE_scalegate_v2_h{args.hidden}_thr{args.boost_thresh:.2f}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net,
        loss_history=[],
        training_params={
            "experiment": "scalegate_v2",
            "best_f01": best_f01,
            "gate_state_dict": {k: v.cpu().tolist() for k, v in best_state.items()},
            "hidden": args.hidden,
            "dQ_ref": dQ_ref.tolist(),
            "boost_thresh": args.boost_thresh,
        },
        session_name=session_name,
    )
    print(f"\n  DONE. best_f01={best_f01:.1%}  saved: {session_name}")


if __name__ == "__main__":
    main()
