"""exp_holdboost_nearstart.py — HoldBoost with near-top initial states for strong gradients.

MOTIVATION:
  holdboost_ft.py (v2, phase_aware) trains from x0=[0,0,0,0], 600-step rollout.
  Only the last ~150 steps (near-top portion) have gate>0 → delta_Q gets gradient
  from ~25% of steps. Training is also slow due to the long rollout.

THIS APPROACH:
  Start every training rollout from [π±δq, ±δv, ±δq2, ±δv2].
  All 200 steps are in the hold phase (gate≈1.0 throughout).
  delta_Q gets gradient from 100% of steps.
  LR=1e-2 (10× higher than v2): 60× more effective update per unit time.

DECOUPLING PRESERVED:
  At eval time, rollout starts from [0,0,0,0]. Swing-up is unaffected because
  gate≈0 during swing-up regardless of delta_Q values.

TARGET: Exceed 26.2% frac<0.10 (2000-step eval from [0,0,0,0])
"""

import glob, math, os, random, sys, time, signal, copy
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
DT     = 0.05
HORIZON = 10
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
THRESH  = 0.8

# Near-top perturbation ranges
X0_PERT_Q1 = 0.25   # ±0.25 rad from π
X0_PERT_V1 = 0.5    # ±0.5 rad/s
X0_PERT_Q2 = 0.20   # ±0.20 rad
X0_PERT_V2 = 0.5    # ±0.5 rad/s

NUM_STEPS    = 200   # short rollout — all in hold phase
EPOCHS       = 300   # more epochs to compensate for shorter rollouts
LR           = 1e-2  # 10× higher than v2 (all steps contribute gradient)
SAVE_EVERY   = 20
SAVE_DIR     = "saved_models"


class HoldBoostWrapper(nn.Module):
    """Same as in exp_holdboost_ft.py — frozen lin_net + trainable delta_Q, delta_R."""
    def __init__(self, lin_net, thresh, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net       = lin_net
        self.thresh        = thresh
        self.x_goal_q1     = x_goal_q1
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon       = lin_net.horizon
        self.state_dim     = lin_net.state_dim
        self.control_dim   = lin_net.control_dim
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


def sample_near_top_x0(device):
    dq1 = (random.random() * 2 - 1) * X0_PERT_Q1
    dv1 = (random.random() * 2 - 1) * X0_PERT_V1
    dq2 = (random.random() * 2 - 1) * X0_PERT_Q2
    dv2 = (random.random() * 2 - 1) * X0_PERT_V2
    return torch.tensor([math.pi + dq1, dv1, dq2, dv2], device=device, dtype=torch.float64)


def make_hold_demo(num_steps, device):
    """Target: stay at [π, 0, 0, 0] throughout."""
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    demo[:, 0] = math.pi
    return demo


def eval_hold_quality(model, mpc, x0, x_goal, steps=2000):
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([
        math.sqrt(
            math.atan2(math.sin(s[0] - math.pi), math.cos(s[0] - math.pi)) ** 2
            + s[1] ** 2 + s[2] ** 2 + s[3] ** 2
        )
        for s in traj
    ])
    arr_idx = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post_01 = float((wraps[arr_idx:] < 0.10).mean()) if arr_idx is not None else None
    return {
        "arr_idx": arr_idx,
        "frac_01": float((wraps < 0.10).mean()),
        "frac_03": float((wraps < 0.30).mean()),
        "post_arr_01": post_01,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    print("=" * 80)
    print("  EXP HOLDBOOST-NEARSTART: additive Q/R with near-top initial states")
    print(f"  Frozen lin_net + delta_Q(9×4) + delta_R(10×2) = 56 trainable params")
    print(f"  LR={LR}  NUM_STEPS={NUM_STEPS}  EPOCHS={EPOCHS}  thresh={THRESH}")
    print(f"  Training x0: [π±{X0_PERT_Q1}, ±{X0_PERT_V1}, ±{X0_PERT_Q2}, ±{X0_PERT_V2}]")
    print(f"  phase_split=0.0 → all steps use wrapped-q1 hold tracking")
    print(f"  Target: exceed 26.2% frac<0.10 (2000-step from [0,0,0,0])")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.requires_grad_(False)

    boost = HoldBoostWrapper(lin_net, thresh=THRESH, x_goal_q1=X_GOAL[0])
    session_name = f"stageD_holdboost_nearstart_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n  Pre-eval (delta=0, identical to ZeroFNet baseline):")
    r = eval_hold_quality(boost, mpc, x0, x_goal, steps=2000)
    post_str = f"  post<0.10={r['post_arr_01']:.1%}" if r['post_arr_01'] is not None else ""
    print(f"    frac<0.10={r['frac_01']:.1%}  frac<0.30={r['frac_03']:.1%}  "
          f"arr={r['arr_idx']}{post_str}", flush=True)

    optimizer = torch.optim.AdamW([boost.delta_Q, boost.delta_R], lr=LR, weight_decay=0.0)
    best_frac01 = r['frac_01']
    best_delta_Q = boost.delta_Q.data.clone()
    best_delta_R = boost.delta_R.data.clone()
    all_losses = []
    t0 = time.time()

    interrupted = [False]
    signal.signal(signal.SIGINT, lambda s, f: interrupted.__setitem__(0, True))

    chunk_start = 0
    while chunk_start < EPOCHS and not interrupted[0]:
        n_ep = min(SAVE_EVERY, EPOCHS - chunk_start)

        # Random near-top x0 for this chunk
        x0_train = sample_near_top_x0(device)
        demo = make_hold_demo(NUM_STEPS, device)

        loss_chunk, _ = train_module.train_linearization_network(
            lin_net=boost, mpc=mpc,
            x0=x0_train, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=n_ep, lr=LR,
            debug_monitor=None, recorder=network_module.NetworkOutputRecorder(),
            grad_debug=False,
            track_mode="phase_aware",
            phase_split_frac=0.0,      # all 200 steps = hold tracking
            w_terminal_anchor=0.0,
            w_q_profile=0.0, w_f_pos_only=0.0, w_stable_phase=0.0,
            f_gate_thresh=0.0,
            w_hold_reward=0.0,
            hold_sigma=0.5,
            hold_start_step=0,
            early_stop_patience=n_ep + 5,
            external_optimizer=optimizer,
            restore_best=False,
        )
        all_losses.extend(loss_chunk)
        chunk_start += n_ep

        r2000 = eval_hold_quality(boost, mpc, x0, x_goal, steps=2000)
        dQ_norm = boost.delta_Q.data.abs().mean().item()
        dR_norm = boost.delta_R.data.abs().mean().item()
        post_str = f"  post<0.10={r2000['post_arr_01']:.1%}" if r2000['post_arr_01'] is not None else ""
        print(f"  [ep={chunk_start}]  2000: {r2000['frac_01']:.1%}  "
              f"frac<0.30={r2000['frac_03']:.1%}{post_str}  "
              f"|dQ|={dQ_norm:.4f}  |dR|={dR_norm:.4f}  t={time.time()-t0:.0f}s",
              flush=True)

        if r2000['frac_01'] > best_frac01:
            best_frac01 = r2000['frac_01']
            best_delta_Q = boost.delta_Q.data.clone()
            best_delta_R = boost.delta_R.data.clone()
            print(f"  ★ New best: {best_frac01:.1%}  "
                  f"dQ_mean={boost.delta_Q.data.mean():.4f}  dR_mean={boost.delta_R.data.mean():.4f}",
                  flush=True)

        if chunk_start % (2 * SAVE_EVERY) == 0:
            ckpt_name = f"{session_name}_ep{chunk_start:03d}"
            network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
                model=lin_net, loss_history=all_losses,
                training_params={
                    "experiment": "holdboost_nearstart",
                    "thresh": THRESH,
                    "lr": LR,
                    "best_frac01_2000step": best_frac01,
                    "delta_Q": best_delta_Q.tolist(),
                    "delta_R": best_delta_R.tolist(),
                    "checkpoint_epoch": chunk_start,
                },
                session_name=ckpt_name,
            )
            print(f"  Checkpoint → saved_models/{ckpt_name}/", flush=True)

        if best_frac01 > 0.5:
            print(f"  EXCELLENT HOLD ({best_frac01:.1%}) — stopping early.")
            break

    # Restore best and save final
    boost.delta_Q.data.copy_(best_delta_Q)
    boost.delta_R.data.copy_(best_delta_R)

    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=all_losses,
        training_params={
            "experiment": "holdboost_nearstart_FINAL",
            "thresh": THRESH,
            "lr": LR,
            "best_frac01_2000step": best_frac01,
            "best_delta_Q": best_delta_Q.tolist(),
            "best_delta_R": best_delta_R.tolist(),
        },
        session_name=session_name,
    )
    print(f"\n  Final → saved_models/{session_name}/  best_frac01={best_frac01:.1%}")
    print(f"  dQ mean per dim: {best_delta_Q.mean(0).tolist()}")
    print(f"  dR mean per dim: {best_delta_R.mean(0).tolist()}")

    print(f"\n  Post-eval:")
    for n in [600, 1000, 2000]:
        r = eval_hold_quality(boost, mpc, x0, x_goal, steps=n)
        tag = "HOLD" if r['frac_01'] > 0.3 else ("PART" if r['frac_01'] > 0.05 else "FAIL")
        post_str = f"  post<0.10={r['post_arr_01']:.1%}" if r['post_arr_01'] is not None else ""
        print(f"    {n:>4} steps: frac<0.10={r['frac_01']:.1%}  "
              f"frac<0.30={r['frac_03']:.1%}{post_str}  [{tag}]")

    print(f"\n  ZeroFNet baseline: 26.2% | Best: {best_frac01:.1%}")
    if best_frac01 > 0.262:
        print(f"  ★★★ IMPROVEMENT: {best_frac01:.1%} > 26.2% ★★★")


if __name__ == "__main__":
    main()
