"""exp_holdboost_perdiminit.py — Per-dim Q boost fine-tuning from scalar optimal.

USAGE:
  Run AFTER exp_qboost_targeted.py has found the best scalar q_boost.
  Initialize delta_Q = INIT_QBOOST * ones(9,4) and optionally fine-tune
  per-dimension to find improvements beyond the scalar.

  python exp_holdboost_perdiminit.py          # uses INIT_QBOOST below
  python exp_holdboost_perdiminit.py 0.3      # override from command line

APPROACH:
  1. Start from delta_Q = init_scalar * ones(H-1, state_dim)
  2. Evaluate immediately (reproduces qboost_targeted result for same scalar)
  3. Fine-tune with gradient training to find per-dim improvements
  4. Report: was fine-tuning better than fixed scalar?
"""

import math, os, sys, time, signal
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
X0     = [0.0, 0.0, 0.0, 0.0]
X_GOAL = [math.pi, 0.0, 0.0, 0.0]
DT     = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]; THRESH = 0.8

INIT_QBOOST = 0.0   # set to optimal from qboost_targeted, or override via argv
ARRIVAL_STEP = 300
NUM_STEPS    = 600
EPOCHS       = 100
LR           = 5e-4   # small LR for fine-tuning from near-optimal init
SAVE_EVERY   = 10
SAVE_DIR     = "saved_models"


class HoldBoostWrapper(nn.Module):
    def __init__(self, lin_net, thresh, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net       = lin_net; self.thresh = thresh; self.x_goal_q1 = x_goal_q1
        self.f_extra_bound = lin_net.f_extra_bound; self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim; self.control_dim = lin_net.control_dim
        q_shape = (lin_net.horizon - 1, lin_net.state_dim)
        r_shape = (lin_net.horizon,     lin_net.control_dim)
        self.delta_Q = nn.Parameter(torch.zeros(q_shape, dtype=torch.float64))
        self.delta_R = nn.Parameter(torch.zeros(r_shape, dtype=torch.float64))

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf = self.lin_net(
            x_sequence, q_base_diag, r_base_diag
        )
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        f_extra = f_extra * (1.0 - gate.detach())
        gates_Q = gates_Q + gate * self.delta_Q
        gates_R = gates_R + gate * self.delta_R
        return gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf


def eval2k(model, mpc, x0, x_goal, steps=2000):
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([math.sqrt(math.atan2(math.sin(s[0]-math.pi), math.cos(s[0]-math.pi))**2
                                + s[1]**2 + s[2]**2 + s[3]**2) for s in traj])
    arr = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    return float((wraps < 0.10).mean()), float((wraps < 0.30).mean()), arr, post


def main():
    init_qboost = float(sys.argv[1]) if len(sys.argv) > 1 else INIT_QBOOST

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    print("=" * 80)
    print(f"  EXP HOLDBOOST-PERDIMINIT: init delta_Q = {init_qboost:.3f} * ones")
    print(f"  Then fine-tune per-dim (LR={LR}) for {EPOCHS} epochs")
    print(f"  Expected starting frac<0.10 ≈ {max(0.262, 0.262 + init_qboost * 0.1):.1%} (est.)")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.requires_grad_(False)

    boost = HoldBoostWrapper(lin_net, thresh=THRESH, x_goal_q1=X_GOAL[0])

    # Initialize delta_Q to the scalar optimal value from qboost_targeted
    with torch.no_grad():
        boost.delta_Q.fill_(init_qboost)
        boost.delta_R.fill_(0.0)

    print(f"\n  Initial eval (delta_Q = {init_qboost:.3f}):")
    f01, f03, arr, post = eval2k(boost, mpc, x0, x_goal, steps=2000)
    print(f"    frac<0.10={f01:.1%}  frac<0.30={f03:.1%}  arr={arr}  "
          f"post={f'{post:.1%}' if post else 'N/A'}", flush=True)

    if init_qboost == 0.0:
        print(f"\n  NOTE: init_qboost=0.0 → same as ZeroFNet baseline (26.2%)")
        print(f"  Pass the best q_boost value from qboost_targeted as argv[1]")
        if len(sys.argv) < 2:
            print(f"  Usage: python {sys.argv[0]} <q_boost_value>")
            print(f"  Example: python {sys.argv[0]} 0.30")
            return

    # Fine-tune delta_Q from this init
    session_name = f"stageD_holdboost_perdiminit{init_qboost:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    demo = torch.zeros((NUM_STEPS, 4), dtype=torch.float64, device=device)
    for i in range(NUM_STEPS):
        alpha = min(i / max(ARRIVAL_STEP, 1), 1.0)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t

    optimizer = torch.optim.AdamW([boost.delta_Q, boost.delta_R], lr=LR, weight_decay=0.0)
    best_f01 = f01
    best_dQ = boost.delta_Q.data.clone()
    best_dR = boost.delta_R.data.clone()
    all_losses = []
    t0 = time.time()

    interrupted = [False]
    signal.signal(signal.SIGINT, lambda s, f: interrupted.__setitem__(0, True))

    chunk_start = 0
    while chunk_start < EPOCHS and not interrupted[0]:
        n_ep = min(SAVE_EVERY, EPOCHS - chunk_start)
        loss_chunk, _ = train_module.train_linearization_network(
            lin_net=boost, mpc=mpc,
            x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=n_ep, lr=LR,
            debug_monitor=None, recorder=network_module.NetworkOutputRecorder(),
            track_mode="phase_aware", phase_split_frac=0.5,
            w_terminal_anchor=0.0, w_q_profile=0.0, w_f_pos_only=0.0, w_stable_phase=0.0,
            f_gate_thresh=0.0, w_hold_reward=0.0, hold_sigma=0.5, hold_start_step=200,
            early_stop_patience=n_ep + 5,
            external_optimizer=optimizer, restore_best=False,
        )
        all_losses.extend(loss_chunk)
        chunk_start += n_ep

        f01, f03, arr, post = eval2k(boost, mpc, x0, x_goal, steps=2000)
        dQ_norm = boost.delta_Q.data.abs().mean().item()
        print(f"  [ep={chunk_start}]  {f01:.1%}  frac<0.30={f03:.1%}  arr={arr}  "
              f"post={f'{post:.1%}' if post else 'N/A'}  "
              f"|dQ|={dQ_norm:.4f}  t={time.time()-t0:.0f}s", flush=True)

        if f01 > best_f01:
            best_f01 = f01
            best_dQ = boost.delta_Q.data.clone()
            best_dR = boost.delta_R.data.clone()
            print(f"  ★ New best: {best_f01:.1%}  dQ_mean={best_dQ.mean():.4f}", flush=True)

        if best_f01 > 0.5:
            print("  EXCELLENT HOLD — stopping.")
            break

    boost.delta_Q.data.copy_(best_dQ)
    boost.delta_R.data.copy_(best_dR)
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=all_losses,
        training_params={"experiment": "holdboost_perdiminit_FINAL",
                         "init_qboost": init_qboost, "best_frac01": best_f01,
                         "best_delta_Q": best_dQ.tolist(), "best_delta_R": best_dR.tolist()},
        session_name=session_name,
    )
    print(f"\n  Init: {init_qboost:.3f} → best fine-tuned: {best_f01:.1%}")
    print(f"  ZeroFNet: 26.2%  |  Fixed scalar {init_qboost:.3f}: {f01:.1%}  |  Fine-tuned: {best_f01:.1%}")
    if best_f01 > 0.262:
        print(f"  ★ IMPROVEMENT over 26.2%!")


if __name__ == "__main__":
    main()
