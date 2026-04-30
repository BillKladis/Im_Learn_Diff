"""exp_scalegate_v5.py — Linear ramp gate with learnable dQ scale (3 params).

Extension of v4: adds a scalar `scale` multiplier on dQ/dR.
  gate: α = (w * near_pi + b).clamp(0,1)     [shape: scalar]
  gQ += α * scale * dQ_ref                    [dQ_ref: (9,4)]
  gR += α * scale * dR_ref                    [dR_ref: (10,2)]
  fe *= (1 - α)

Init: w=5.0, b=-4.0, scale=1.0 → EXACTLY 87.2% wrapper.

Why this might help:
  - scale alone: if optimal threshold needs higher/lower boost magnitude
  - e.g., tighter thresh (wider w) + scale=1.5 could give same average boost
    as thresh=0.80 but with sharper gate onset
  - scale can compensate for changes in threshold without relearning dQ values
  - Adds only 1 extra parameter vs v4; still very interpretable

3 parameters: gate_w, gate_b, dq_scale
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
SCALE4_CKPT   = "saved_models/stageD_scale4.0x_dQ_20260430_192447/stageD_scale4.0x_dQ_20260430_192447.pth"
X0            = [0.0, 0.0, 0.0, 0.0]
X_GOAL        = [math.pi, 0.0, 0.0, 0.0]
DT            = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
STATE_HIST    = 5; NUM_STEPS = 200; SAVE_EVERY = 10; EXCELLENT_HOLD = 0.92

BOTTOM_X0_LIST = [[0,0,0,0],[0.2,0,0,0],[-0.2,0,0,0],[0.5,0,0,0],[-0.5,0,0,0]]


class LinearRampGateScaled(nn.Module):
    """3-param gate: α=(w*near_pi+b).clamp(0,1), boost = α * scale * dQ_ref.

    Init: w=5, b=-4, scale=1 → exactly HoldBoostWrapper with thresh=0.80.
    Extra degree of freedom: scale adjusts dQ/dR magnitude independently of
    gate shape. Allows compensating for threshold shift with magnitude change.
    """
    def __init__(self, lin_net, dQ_ref, dR_ref, init_w=5.0, init_b=-4.0, init_scale=1.0):
        super().__init__()
        self.lin_net = lin_net
        self.register_buffer('dQ_ref', dQ_ref.clone())  # (9,4)
        self.register_buffer('dR_ref', dR_ref.clone())  # (10,2)
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim
        self.control_dim = lin_net.control_dim

        self.gate_w  = nn.Parameter(torch.tensor(init_w,     dtype=torch.float64))
        self.gate_b  = nn.Parameter(torch.tensor(init_b,     dtype=torch.float64))
        self.dq_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float64))

    def get_near_pi(self, x_sequence):
        return (1.0 + torch.cos(x_sequence[-1, 0] - math.pi)) / 2.0

    def get_alpha(self, x_sequence):
        near_pi = self.get_near_pi(x_sequence)
        return (self.gate_w * near_pi + self.gate_b).clamp(0.0, 1.0)

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        alpha = self.get_alpha(x_sequence)
        scale = self.dq_scale
        gQ = (gQ + alpha * scale * self.dQ_ref).clamp(min=0.01)
        gR = gR + alpha * scale * self.dR_ref
        fe = fe * (1.0 - alpha)
        return gQ, gR, fe, qd, rd, gQf


def probe_gate(model, device, header="Gate profile (q1d=q2=q2d=0):"):
    print(f"  {header}")
    thresh = (-model.gate_b / model.gate_w).item() if abs(model.gate_w.item()) > 1e-6 else float('nan')
    print(f"  gate_w={model.gate_w.item():.4f}  gate_b={model.gate_b.item():.4f}"
          f"  thresh={thresh:.4f}  scale={model.dq_scale.item():.4f}")
    with torch.no_grad():
        for deg in [0, 60, 90, 120, 127, 130, 140, 150, 165, 180]:
            q1 = math.radians(deg)
            np_val = (1 + math.cos(q1 - math.pi)) / 2
            x = torch.tensor([q1, 0, 0, 0], device=device, dtype=torch.float64)
            xseq = x.unsqueeze(0).expand(STATE_HIST, -1)
            alpha = model.get_alpha(xseq).item()
            wrapper_gate = max(0, min(1, (np_val - 0.80) / 0.20))
            print(f"    q1={deg:3d}°  near_pi={np_val:.3f}"
                  f"  wrapper={wrapper_gate:.3f}  α={alpha:.4f}", flush=True)


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
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--top_frac", type=float, default=0.7)
    parser.add_argument("--init_w", type=float, default=5.0)
    parser.add_argument("--init_b", type=float, default=-4.0)
    parser.add_argument("--init_scale", type=float, default=1.0,
                        help="Initial dQ/dR scale (1.0 = exact wrapper)")
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
    dQ_ref = torch.tensor(tp['best_delta_Q'], dtype=torch.float64).to(device)  # (9,4)
    dR_ref = torch.tensor(tp['best_delta_R'], dtype=torch.float64).to(device)  # (10,2)

    model = LinearRampGateScaled(lin_net, dQ_ref, dR_ref,
                                  init_w=args.init_w, init_b=args.init_b,
                                  init_scale=args.init_scale).to(device)

    init_thresh = -args.init_b / args.init_w if abs(args.init_w) > 1e-6 else 0.80

    print("=" * 80)
    print("  EXP SCALEGATE v5: Linear ramp + learnable scale (3 params)")
    print(f"  lin_net: FROZEN  gate: 3 params (w={args.init_w}, b={args.init_b}, scale={args.init_scale})")
    print(f"  Init threshold: near_pi={init_thresh:.3f}  "
          f"→ q1 within {math.degrees(math.acos(2*init_thresh-1)):.1f}° of top")
    print(f"  boost = α × scale × dQ_ref  dQ_ref mean={dQ_ref.mean(0).tolist()}")
    print(f"  LR={args.lr}  top_frac={args.top_frac:.0%}  epochs={args.epochs}")
    print("=" * 80)

    probe_gate(model, device, "Initial gate profile (= exact wrapper at scale=1):")

    print(f"\n  Initial eval (compiling CVXPY — ~25 min)...")
    f01, arr, post = eval2k(model, mpc, x0_eval, x_goal)
    print(f"    frac<0.10={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}  [baseline=87.2%]",
          flush=True)

    best_f01 = f01
    best_w = model.gate_w.item()
    best_b = model.gate_b.item()
    best_scale = model.dq_scale.item()

    gate_params = [model.gate_w, model.gate_b, model.dq_scale]
    optimizer = torch.optim.AdamW(gate_params, lr=args.lr, weight_decay=0.0)
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
            w_now = model.gate_w.item()
            b_now = model.gate_b.item()
            s_now = model.dq_scale.item()
            thresh_now = -b_now / w_now if abs(w_now) > 1e-6 else float('nan')
            mark = ""
            if f01 > best_f01:
                mark = " ★"
                best_f01 = f01
                best_w, best_b, best_scale = w_now, b_now, s_now
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [ep={epoch:3d}]  {f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}"
                  f"  w={w_now:.3f}  thresh={thresh_now:.4f}  scale={s_now:.3f}"
                  f"  top/bot={top_count}/{bot_count}  lr={lr_now:.1e}"
                  f"  t={time.time()-t0:.0f}s{mark}", flush=True)

            if best_f01 >= EXCELLENT_HOLD:
                print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
                break

    model.gate_w.data.fill_(best_w)
    model.gate_b.data.fill_(best_b)
    model.dq_scale.data.fill_(best_scale)

    probe_gate(model, device, "Final gate profile:")

    session_name = (f"stageE_scalegate_v5_w{args.init_w:.1f}_b{args.init_b:.1f}_s{args.init_scale:.1f}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net,
        loss_history=[],
        training_params={
            "experiment": "scalegate_v5",
            "best_f01": best_f01,
            "gate_w": best_w,
            "gate_b": best_b,
            "dq_scale": best_scale,
            "gate_thresh": -best_b / best_w if abs(best_w) > 1e-6 else 0.80,
            "dQ_ref_mean": dQ_ref.mean(0).tolist(),
            "init_w": args.init_w,
            "init_b": args.init_b,
            "init_scale": args.init_scale,
        },
        session_name=session_name,
    )
    print(f"\n  DONE. best_f01={best_f01:.1%}  saved: {session_name}")


if __name__ == "__main__":
    main()
