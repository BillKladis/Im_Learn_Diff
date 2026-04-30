"""exp_holdboost_ft.py — Minimal additive Q/R corrections active only near top.

ROOT CAUSE OF ALL TRAINING FAILURES:
  Every approach changes Q/R behavior during BOTH swing-up AND hold phases.
  Modified Q/R during swing-up → different trajectory → different arrival velocity
  → different hold quality. The system is entangled: can't improve hold without
  disrupting swing-up.

THIS APPROACH: Additive Q/R delta that activates ONLY near top (gate × delta).
  - delta_Q: (9,4) additive correction to gates_Q, scaled by ZeroFNet gate
  - delta_R: (10,2) additive correction to gates_R, scaled by ZeroFNet gate
  - Total: 56 trainable parameters (vs 73K in Q/R heads)
  - gate ≈ 0 during swing-up → ZERO effect on swing-up trajectory
  - gate ≈ 1 during hold → additive Q/R correction
  - Initialized to zero → starts IDENTICAL to ZeroFNet baseline (26.2%)

DECOUPLING GUARANTEE:
  Swing-up: gate(near_pi) ≈ 0 → delta inactive → EXACT same as ZeroFNet baseline
  Hold: gate ≈ 1 → effective Q/R = Q/R_baseline + delta_Q/R

TRAINING:
  600-step rollout (covers hold phase). Only delta_Q, delta_R get gradient.
  hold_reward fires when actually near top → trains delta toward better hold.
  Energy tracking gradient for delta: ZERO during swing-up (gate ≈ 0).

TARGET: Exceed 26.2% frac<0.10 (2000-step)
"""

import glob
import math
import os
import sys
import time
import signal
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

# ── Config ──────────────────────────────────────────────────────────────────
POSONLY_FINAL = "saved_models/stageD_posonly_ft_20260430_083618/stageD_posonly_ft_20260430_083618.pth"
BASELINE      = "saved_models/stageD_nodemo_20260428_123448/stageD_nodemo_20260428_123448.pth"

X0           = [0.0, 0.0, 0.0, 0.0]
X_GOAL       = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS    = 600
ARRIVAL_STEP = 300
DT           = 0.05
EPOCHS       = 150
LR           = 1e-3   # high LR: only 56 params, well-conditioned
HORIZON      = 10
SAVE_DIR     = "saved_models"
Q_BASE_DIAG  = [12.0, 5.0, 50.0, 40.0]

THRESH        = 0.8    # ZeroFNet gate threshold (matches best eval)
W_HOLD_REWARD = 80.0   # strong hold reward
HOLD_SIGMA    = 0.5
HOLD_START    = 200
SAVE_EVERY    = 15


def _find_pretrained():
    if os.path.isfile(POSONLY_FINAL):
        return POSONLY_FINAL, "posonly_ft_final"
    ckpts = sorted(glob.glob("saved_models/stageD_posonly_ft_20260430_083618_ep*/*.pth"))
    if ckpts:
        return ckpts[-1], "posonly_ep" + os.path.basename(os.path.dirname(ckpts[-1]))[-3:]
    return BASELINE, "0.0612_baseline"


def make_demo_with_hold(num_steps, arrival_step, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        if i <= arrival_step:
            alpha = i / max(arrival_step, 1)
            t = 0.5 * (1.0 - math.cos(math.pi * alpha))
            demo[i, 0] = math.pi * t
        else:
            demo[i, 0] = math.pi
    return demo


class HoldBoostWrapper(nn.Module):
    """Wraps lin_net with:
    1. ZeroFNet soft-ramp gate: f_extra *= (1-gate)
    2. Additive Q/R deltas activated by the same gate: gates_Q += gate*delta_Q
    Only delta_Q and delta_R are trainable. lin_net is fully frozen.
    """
    def __init__(self, lin_net, thresh, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net    = lin_net
        self.thresh     = thresh
        self.x_goal_q1  = x_goal_q1

        # Expose attributes needed by rollout/training utilities
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon       = lin_net.horizon
        self.state_dim     = lin_net.state_dim
        self.control_dim   = lin_net.control_dim

        # Additive Q/R corrections near top, init=0 → identical to ZeroFNet baseline
        q_shape = (lin_net.horizon - 1, lin_net.state_dim)   # (9, 4)
        r_shape = (lin_net.horizon,     lin_net.control_dim)  # (10, 2)
        self.delta_Q = nn.Parameter(torch.zeros(q_shape, dtype=torch.float64))
        self.delta_R = nn.Parameter(torch.zeros(r_shape, dtype=torch.float64))

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf = self.lin_net(
            x_sequence, q_base_diag, r_base_diag
        )
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)

        # ZeroFNet: zero f_extra near top
        f_extra = f_extra * (1.0 - gate.detach())

        # Additive Q/R boost near top (gate is differentiable here)
        gates_Q = gates_Q + gate * self.delta_Q
        gates_R = gates_R + gate * self.delta_R

        return gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf


def eval_hold_quality(model, mpc, x0, x_goal, steps=2000):
    # NOTE: no torch.no_grad() — cvxpylayers QP solver needs autograd active
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
    post_01 = None
    if arr_idx is not None:
        post = wraps[arr_idx:]
        post_01 = float((post < 0.10).mean())
    return {
        "arr_idx": arr_idx,
        "frac_01": float((wraps < 0.10).mean()),
        "frac_03": float((wraps < 0.30).mean()),
        "wrap_final": float(wraps[-1]),
        "post_arr_01": post_01,
        "wraps": wraps,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_demo_with_hold(NUM_STEPS, ARRIVAL_STEP, device)

    pretrained, src_label = _find_pretrained()
    print("=" * 80)
    print("  EXP HOLDBOOST-FT: additive Q/R corrections near top only")
    print(f"  Pretrained: {src_label} (FULLY FROZEN)")
    print(f"  Trainable: delta_Q (9×4=36) + delta_R (10×2=20) = 56 params only")
    print(f"  gate = clamp((near_pi - {THRESH}) / {1-THRESH:.1f}, 0, 1)")
    print(f"  Effect: gates_Q += gate*delta_Q, gates_R += gate*delta_R, f_extra *= (1-gate)")
    print(f"  DECOUPLING: gate≈0 during swing-up → ZERO effect on swing-up trajectory")
    print(f"  NUM_STEPS={NUM_STEPS}  w_hold_reward={W_HOLD_REWARD}  sigma={HOLD_SIGMA}")
    print(f"  Target: exceed 26.2% frac<0.10 (2000-step)")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(pretrained, device=str(device)).double()
    lin_net.requires_grad_(False)  # freeze entire lin_net

    boost = HoldBoostWrapper(lin_net, thresh=THRESH, x_goal_q1=X_GOAL[0])
    session_name = f"stageD_holdboost_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n  Pre-eval (delta=0, identical to ZeroFNet baseline):")
    r = eval_hold_quality(boost, mpc, x0, x_goal, steps=2000)
    post_str = f"  post<0.10={r['post_arr_01']:.1%}" if r['post_arr_01'] is not None else ""
    print(f"    frac<0.10={r['frac_01']:.1%}  frac<0.30={r['frac_03']:.1%}  "
          f"arr={r['arr_idx']}{post_str}")

    trainable = [boost.delta_Q, boost.delta_R]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)

    best_frac01 = 0.0
    best_delta_Q = boost.delta_Q.data.clone()
    best_delta_R = boost.delta_R.data.clone()
    all_losses = []
    t0 = time.time()

    interrupted = [False]
    def on_sig(sig, frame):
        interrupted[0] = True
    signal.signal(signal.SIGINT, on_sig)

    chunk_start = 0
    while chunk_start < EPOCHS and not interrupted[0]:
        n_ep = min(SAVE_EVERY, EPOCHS - chunk_start)

        loss_chunk, recorder = train_module.train_linearization_network(
            lin_net=boost, mpc=mpc,
            x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=n_ep, lr=LR,
            debug_monitor=None, recorder=network_module.NetworkOutputRecorder(),
            grad_debug=False, track_mode="energy", w_terminal_anchor=0.0,
            w_q_profile=0.0, w_f_pos_only=0.0, w_stable_phase=0.0,
            f_gate_thresh=0.0,            # gate applied in forward() already
            w_hold_reward=W_HOLD_REWARD,
            hold_sigma=HOLD_SIGMA,
            hold_start_step=HOLD_START,
            early_stop_patience=n_ep + 5,
            external_optimizer=optimizer,
            restore_best=False,
        )
        all_losses.extend(loss_chunk)
        chunk_start += n_ep

        r600  = eval_hold_quality(boost, mpc, x0, x_goal, steps=600)
        r2000 = eval_hold_quality(boost, mpc, x0, x_goal, steps=2000)
        dQ_norm = boost.delta_Q.data.abs().mean().item()
        dR_norm = boost.delta_R.data.abs().mean().item()
        post_str = f"  post<0.10={r2000['post_arr_01']:.1%}" if r2000['post_arr_01'] is not None else ""

        print(f"  [ep={chunk_start}]  600: {r600['frac_01']:.1%}  arr={r600['arr_idx']}  |  "
              f"2000: {r2000['frac_01']:.1%}  frac<0.30={r2000['frac_03']:.1%}{post_str}  "
              f"|dQ|={dQ_norm:.4f}  |dR|={dR_norm:.4f}  t={time.time()-t0:.0f}s",
              flush=True)

        if r2000['frac_01'] > best_frac01:
            best_frac01 = r2000['frac_01']
            best_delta_Q = boost.delta_Q.data.clone()
            best_delta_R = boost.delta_R.data.clone()
            print(f"  ★ New best: {best_frac01:.1%}  "
                  f"dQ={boost.delta_Q.data.mean():.4f}  dR={boost.delta_R.data.mean():.4f}")

        if chunk_start % (2 * SAVE_EVERY) == 0:
            ckpt_name = f"{session_name}_ep{chunk_start:03d}"
            network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
                model=lin_net, loss_history=all_losses,
                training_params={
                    "experiment": "holdboost_ft",
                    "src": src_label,
                    "thresh": THRESH,
                    "w_hold_reward": W_HOLD_REWARD,
                    "delta_Q": best_delta_Q.tolist(),
                    "delta_R": best_delta_R.tolist(),
                    "checkpoint_epoch": chunk_start,
                    "best_frac01_2000step": best_frac01,
                },
                session_name=ckpt_name,
            )
            print(f"  Checkpoint → saved_models/{ckpt_name}/", flush=True)

        if best_frac01 > 0.5:
            print(f"  EXCELLENT HOLD ({best_frac01:.1%}) — stopping early.")
            break

    # Restore best deltas
    boost.delta_Q.data.copy_(best_delta_Q)
    boost.delta_R.data.copy_(best_delta_R)

    session_params = {
        "experiment": "holdboost_ft_FINAL",
        "src": src_label,
        "thresh": THRESH,
        "w_hold_reward": W_HOLD_REWARD,
        "best_frac01_2000step": best_frac01,
        "best_delta_Q": best_delta_Q.tolist(),
        "best_delta_R": best_delta_R.tolist(),
    }
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=all_losses,
        training_params=session_params,
        session_name=session_name,
    )
    print(f"\n  Final → saved_models/{session_name}/  best_frac01={best_frac01:.1%}")
    print(f"  best_delta_Q (mean per state dim): {best_delta_Q.mean(0).tolist()}")
    print(f"  best_delta_R (mean per control dim): {best_delta_R.mean(0).tolist()}")

    print(f"\n  Post-eval (with best Q/R boost):")
    for n in [600, 1000, 2000]:
        r = eval_hold_quality(boost, mpc, x0, x_goal, steps=n)
        tag = "HOLD" if r['frac_01'] > 0.3 else ("PART" if r['frac_01'] > 0.05 else "FAIL")
        post_str = f"  post<0.10={r['post_arr_01']:.1%}" if r['post_arr_01'] is not None else ""
        print(f"    {n:>4} steps: frac<0.10={r['frac_01']:.1%}  "
              f"frac<0.30={r['frac_03']:.1%}{post_str}  [{tag}]")

    print(f"\n  ZeroFNet baseline (delta=0): 26.2% frac<0.10 (2000-step)")
    if best_frac01 > 0.262:
        print(f"  ★★★ IMPROVEMENT: {best_frac01:.1%} > 26.2% ★★★")
    elif best_frac01 > 0.10:
        print(f"  Partial improvement — consider different delta init or higher LR.")
    else:
        print(f"  No improvement — ZeroFNet baseline is optimal for this approach.")


if __name__ == "__main__":
    main()
