"""exp_scalegate_v6.py — Decoupled gates: separate thresholds for fe vs Q/R boost.

MOTIVATION (from thresh sweep):
  thresh=0.75 (wider): arr=239 (faster!) but post=93.4% (worse hold)
  thresh=0.80 (current): arr=242 but post=99.1%
  Wider fe suppression helps arrival; wider Q boost hurts hold quality.

HYPOTHESIS: Decouple the two effects:
  - Suppress fe EARLIER (thresh_fe = 0.75) → faster capture (arr<242)
  - Boost Q/R LATER (thresh_Q = 0.80+) → maintain hold quality (post≥99.1%)

4 learnable parameters:
  alpha_fe = (w_fe * near_pi + b_fe).clamp(0,1)  → fe suppression
  alpha_Q  = (w_Q  * near_pi + b_Q ).clamp(0,1)  → Q/R boost

Init A (two-gate): w_fe=5,b_fe=-4, w_Q=5,b_Q=-4 → same as wrapper (both at thresh=0.80)
Init B (decoupled): w_fe=4,b_fe=-3 (thresh=0.75), w_Q=5,b_Q=-4 (thresh=0.80)
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


class DecoupledRampGate(nn.Module):
    """4-param decoupled gate: separate linear ramps for fe and Q/R boost.

    alpha_fe = (w_fe * near_pi + b_fe).clamp(0,1)  → controls fe suppression
    alpha_Q  = (w_Q  * near_pi + b_Q ).clamp(0,1)  → controls Q/R boost

    fe gate activating EARLIER than Q gate allows faster capture
    while keeping hold quality intact from the tighter Q gate.
    """
    def __init__(self, lin_net, dQ_ref, dR_ref,
                 init_w_fe=5.0, init_b_fe=-4.0,
                 init_w_Q=5.0,  init_b_Q=-4.0):
        super().__init__()
        self.lin_net = lin_net
        self.register_buffer('dQ_ref', dQ_ref.clone())  # (9,4)
        self.register_buffer('dR_ref', dR_ref.clone())  # (10,2)
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim
        self.control_dim = lin_net.control_dim

        self.w_fe = nn.Parameter(torch.tensor(init_w_fe, dtype=torch.float64))
        self.b_fe = nn.Parameter(torch.tensor(init_b_fe, dtype=torch.float64))
        self.w_Q  = nn.Parameter(torch.tensor(init_w_Q,  dtype=torch.float64))
        self.b_Q  = nn.Parameter(torch.tensor(init_b_Q,  dtype=torch.float64))

    def get_near_pi(self, x_sequence):
        return (1.0 + torch.cos(x_sequence[-1, 0] - math.pi)) / 2.0

    def get_alphas(self, x_sequence):
        near_pi = self.get_near_pi(x_sequence)
        alpha_fe = (self.w_fe * near_pi + self.b_fe).clamp(0.0, 1.0)
        alpha_Q  = (self.w_Q  * near_pi + self.b_Q ).clamp(0.0, 1.0)
        return alpha_fe, alpha_Q

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        alpha_fe, alpha_Q = self.get_alphas(x_sequence)
        gQ = (gQ + alpha_Q * self.dQ_ref).clamp(min=0.01)
        gR = gR + alpha_Q * self.dR_ref
        fe = fe * (1.0 - alpha_fe)
        return gQ, gR, fe, qd, rd, gQf


def probe_gate(model, device, header="Gate profile (q1d=q2=q2d=0):"):
    print(f"  {header}")
    thresh_fe = (-model.b_fe / model.w_fe).item() if abs(model.w_fe.item()) > 1e-6 else float('nan')
    thresh_Q  = (-model.b_Q  / model.w_Q ).item() if abs(model.w_Q.item())  > 1e-6 else float('nan')
    print(f"  fe: w={model.w_fe.item():.4f}  b={model.b_fe.item():.4f}  thresh={thresh_fe:.4f}")
    print(f"  Q:  w={model.w_Q.item():.4f}  b={model.b_Q.item():.4f}  thresh={thresh_Q:.4f}")
    with torch.no_grad():
        for deg in [0, 60, 90, 114, 120, 127, 130, 140, 150, 165, 180]:
            q1 = math.radians(deg)
            np_val = (1 + math.cos(q1 - math.pi)) / 2
            x = torch.tensor([q1, 0, 0, 0], device=device, dtype=torch.float64)
            xseq = x.unsqueeze(0).expand(STATE_HIST, -1)
            alpha_fe, alpha_Q = model.get_alphas(xseq)
            wrapper_gate = max(0, min(1, (np_val - 0.80) / 0.20))
            print(f"    q1={deg:3d}°  near_pi={np_val:.3f}"
                  f"  α_fe={alpha_fe.item():.4f}  α_Q={alpha_Q.item():.4f}"
                  f"  wrapper={wrapper_gate:.3f}", flush=True)


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
    # fe gate init (wider by default: thresh=0.75)
    parser.add_argument("--w_fe", type=float, default=4.0,
                        help="fe gate slope (4.0=thresh 0.75 w/ width 0.25)")
    parser.add_argument("--b_fe", type=float, default=-3.0,
                        help="fe gate intercept (-3.0 → thresh=0.75)")
    # Q gate init (same as wrapper by default)
    parser.add_argument("--w_Q", type=float, default=5.0,
                        help="Q gate slope (5.0=thresh 0.80)")
    parser.add_argument("--b_Q", type=float, default=-4.0,
                        help="Q gate intercept (-4.0 → thresh=0.80)")
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
    dQ_ref = torch.tensor(tp['best_delta_Q'], dtype=torch.float64).to(device)
    dR_ref = torch.tensor(tp['best_delta_R'], dtype=torch.float64).to(device)

    model = DecoupledRampGate(lin_net, dQ_ref, dR_ref,
                               init_w_fe=args.w_fe, init_b_fe=args.b_fe,
                               init_w_Q=args.w_Q,   init_b_Q=args.b_Q).to(device)

    thresh_fe = -args.b_fe / args.w_fe if abs(args.w_fe) > 1e-6 else 0.80
    thresh_Q  = -args.b_Q  / args.w_Q  if abs(args.w_Q)  > 1e-6 else 0.80

    print("=" * 80)
    print("  EXP SCALEGATE v6: Decoupled fe/Q gates (4 params)")
    print(f"  lin_net: FROZEN")
    print(f"  fe gate:  w={args.w_fe}, b={args.b_fe}  → thresh={thresh_fe:.3f}"
          f"  ({math.degrees(math.acos(2*thresh_fe-1)):.1f}° from top)")
    print(f"  Q gate:   w={args.w_Q},  b={args.b_Q}  → thresh={thresh_Q:.3f}"
          f"  ({math.degrees(math.acos(2*thresh_Q-1)):.1f}° from top)")
    print(f"  LR={args.lr}  top_frac={args.top_frac:.0%}  epochs={args.epochs}")
    print("=" * 80)

    probe_gate(model, device, "Initial gate profile:")

    print(f"\n  Initial eval (compiling CVXPY — ~25 min)...")
    f01, arr, post = eval2k(model, mpc, x0_eval, x_goal)
    print(f"    frac<0.10={f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}  [target: >87.2%]",
          flush=True)

    best_f01 = f01
    best_params = {k: v.item() for k, v in [
        ('w_fe', model.w_fe), ('b_fe', model.b_fe),
        ('w_Q', model.w_Q), ('b_Q', model.b_Q)
    ]}

    gate_params = [model.w_fe, model.b_fe, model.w_Q, model.b_Q]
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
            th_fe = (-model.b_fe / model.w_fe).item() if abs(model.w_fe.item()) > 1e-6 else float('nan')
            th_Q  = (-model.b_Q  / model.w_Q ).item() if abs(model.w_Q.item())  > 1e-6 else float('nan')
            mark = ""
            if f01 > best_f01:
                mark = " ★"
                best_f01 = f01
                best_params = {k: v.item() for k, v in [
                    ('w_fe', model.w_fe), ('b_fe', model.b_fe),
                    ('w_Q', model.w_Q), ('b_Q', model.b_Q)
                ]}
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [ep={epoch:3d}]  {f01:.1%}  arr={arr}  post={f'{post:.1%}' if post else 'N/A'}"
                  f"  th_fe={th_fe:.3f}  th_Q={th_Q:.3f}"
                  f"  top/bot={top_count}/{bot_count}  lr={lr_now:.1e}"
                  f"  t={time.time()-t0:.0f}s{mark}", flush=True)

            if best_f01 >= EXCELLENT_HOLD:
                print(f"  REACHED {EXCELLENT_HOLD:.0%} — stopping.")
                break

    for k, v in best_params.items():
        getattr(model, k).data.fill_(v)

    probe_gate(model, device, "Final gate profile:")

    session_name = (f"stageE_scalegate_v6_"
                    f"fe{args.w_fe:.1f}_{args.b_fe:.1f}_Q{args.w_Q:.1f}_{args.b_Q:.1f}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net,
        loss_history=[],
        training_params={
            "experiment": "scalegate_v6",
            "best_f01": best_f01,
            **best_params,
            "gate_thresh_fe": -best_params['b_fe'] / best_params['w_fe'] if abs(best_params['w_fe']) > 1e-6 else 0.80,
            "gate_thresh_Q":  -best_params['b_Q']  / best_params['w_Q']  if abs(best_params['w_Q'])  > 1e-6 else 0.80,
            "dQ_ref_mean": dQ_ref.mean(0).tolist(),
        },
        session_name=session_name,
    )
    print(f"\n  DONE. best_f01={best_f01:.1%}  saved: {session_name}")


if __name__ == "__main__":
    main()
