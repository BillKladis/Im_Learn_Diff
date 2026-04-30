"""exp_holdonly_ft.py — Train Q/R heads from near-top initial states.

PROBLEM WITH 600-STEP ROLLOUT:
  Only steps 300-600 are hold phase. Steps 0-300 (swing-up) create:
  - Energy tracking gradient that conflicts with hold quality gradient
  - Wasted gradient computation on non-hold states
  - Slow per-epoch training (2× longer rollout)

THIS APPROACH: Start each epoch from a RANDOM NEAR-TOP STATE.
  x0 = [π + δq, δv, δq2, δv2] with small δ (pendulum already near top)
  - ALL 280 training steps are hold phase → 100% gradient efficiency
  - No swing-up states → no energy tracking conflict
  - Swing-up quality preserved: trunk FROZEN (encoder, trunk, f_head frozen)
  - Q/R heads learn: "from this near-top state, output Q/R that improves hold"

RANDOMIZATION: Different x0 each epoch → Q/R adapts robustly to varied hold disturbances.
  δq  ~ U[-0.3, 0.3] (q1 perturbation ~ 17°)
  δv  ~ U[-0.5, 0.5] (q1d perturbation)
  δq2 ~ U[-0.3, 0.3] (q2 perturbation)
  δv2 ~ U[-0.5, 0.5] (q2d perturbation)
  (Vary per epoch so Q/R adapts across the distribution of near-top states)

SIGNAL: w_stable_phase fires in last 60 of 280 steps (all hold phase when near top).
        w_hold_reward fires whenever near top with low velocity.

TARGET: Exceed 26.2% frac<0.10 (2000-step, ZeroFNetWrapper thresh=0.80)
"""

import glob
import math
import os
import random
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

X0_SWING     = [0.0, 0.0, 0.0, 0.0]    # for evaluation only
X_GOAL       = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS    = 280    # all hold phase (from near-top x0)
DT           = 0.05
EPOCHS       = 100
LR           = 5e-5   # q_head + r_head only
HORIZON      = 10
SAVE_DIR     = "saved_models"
Q_BASE_DIAG  = [12.0, 5.0, 50.0, 40.0]

F_GATE_THRESH    = 0.8    # matches best ZeroFNet eval (26.2%)
W_STABLE_PHASE   = 50.0   # direct position+velocity penalty in last N steps
STABLE_PHASE_STEPS = 60   # last 60 of 280 steps (all hold since x0 near top)
W_HOLD_REWARD    = 30.0   # complementary: fires whenever actually near top
HOLD_SIGMA       = 0.5    # zone width
HOLD_START_STEP  = 0      # all steps (since x0 is already near top)

# Near-top initial state perturbations (randomized per epoch)
X0_PERT_Q1   = 0.3   # max |δq1| in rad (~17°)
X0_PERT_V    = 0.5   # max |δq1d|, |δq2d| in rad/s
X0_PERT_Q2   = 0.3   # max |δq2| in rad

SAVE_EVERY   = 10


def _find_pretrained():
    if os.path.isfile(POSONLY_FINAL):
        return POSONLY_FINAL, "posonly_ft_final"
    ckpts = sorted(glob.glob("saved_models/stageD_posonly_ft_20260430_083618_ep*/*.pth"))
    if ckpts:
        return ckpts[-1], "posonly_ep" + os.path.basename(os.path.dirname(ckpts[-1]))[-3:]
    return BASELINE, "0.0612_baseline"


def sample_near_top_x0(device, rng):
    dq1  = (rng.random() * 2 - 1) * X0_PERT_Q1
    dv1  = (rng.random() * 2 - 1) * X0_PERT_V
    dq2  = (rng.random() * 2 - 1) * X0_PERT_Q2
    dv2  = (rng.random() * 2 - 1) * X0_PERT_V
    return torch.tensor([math.pi + dq1, dv1, dq2, dv2], device=device, dtype=torch.float64)


def make_hold_demo(num_steps, x_goal, device):
    return x_goal.unsqueeze(0).expand(num_steps, -1).contiguous()


class ZeroFNetWrapper(nn.Module):
    def __init__(self, lin_net, thresh, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net       = lin_net
        self.thresh        = thresh
        self.x_goal_q1     = x_goal_q1
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon       = lin_net.horizon
        self.state_dim     = lin_net.state_dim
        self.control_dim   = lin_net.control_dim

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf = self.lin_net(
            x_sequence, q_base_diag, r_base_diag
        )
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        f_extra = f_extra * (1.0 - gate)
        return gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf


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


class PrintMonitor:
    def __init__(self, num_epochs, start_epoch=0):
        self.num_epochs  = num_epochs
        self.start_epoch = start_epoch
        self._header_shown = False
        self._best = float('inf')

    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>8}  {'GoalDist':>9}  "
              f"{'fNorm':>7}  {'Time':>6}  {'BestGD':>8}")
        print("─" * 76)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        abs_epoch = self.start_epoch + epoch + 1
        d = info.get('pure_end_error', float('nan'))
        if d < self._best:
            self._best = d
        if epoch == 0 or abs_epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"{abs_epoch:>4}/{self.num_epochs:<4}"
                  f"  {loss:>10.3f}"
                  f"  {info.get('loss_track', float('nan')):>8.4f}"
                  f"  {d:>9.4f}"
                  f"  {info.get('mean_f_extra_norm', float('nan')):>7.3f}"
                  f"  {info.get('epoch_time', float('nan')):>5.2f}s"
                  f"  {self._best:>8.4f}",
                  flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0_eval = torch.tensor(X0_SWING,  device=device, dtype=torch.float64)
    x_goal  = torch.tensor(X_GOAL,    device=device, dtype=torch.float64)

    pretrained, src_label = _find_pretrained()
    print("=" * 80)
    print("  EXP HOLDONLY-FT: train Q/R-only from near-top initial states")
    print(f"  Pretrained: {src_label}")
    print(f"  FROZEN: trunk, encoder, f_head, qf_head")
    print(f"  TRAINABLE: q_head, r_head only")
    print(f"  x0 ~ [π+δ, δv, δq2, δv2]: near-top, random per epoch")
    print(f"  All {NUM_STEPS} steps are hold phase → 100% gradient efficiency")
    print(f"  f_gate_thresh={F_GATE_THRESH}  w_stable_phase={W_STABLE_PHASE}  w_hold_reward={W_HOLD_REWARD}")
    print(f"  Evaluation: x0=[0,0,0,0] → ZeroFNetWrapper(thresh={F_GATE_THRESH})")
    print(f"  Target: exceed 26.2% frac<0.10 (2000-step)")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0_eval, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(pretrained, device=str(device)).double()

    # ── Freeze trunk, encoder, f_head, qf_head ──────────────────────────────
    frozen_modules = [lin_net.state_encoder, lin_net.trunk, lin_net.f_head]
    if hasattr(lin_net, 'qf_head'):
        frozen_modules.append(lin_net.qf_head)
    for m in frozen_modules:
        for p in m.parameters():
            p.requires_grad_(False)

    trainable_params = [p for p in lin_net.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in lin_net.parameters())
    print(f"\n  Trainable: {n_trainable:,} / {n_total:,} params (q_head + r_head only)")

    zf_net = ZeroFNetWrapper(lin_net, thresh=F_GATE_THRESH, x_goal_q1=X_GOAL[0])
    session_name = f"stageD_holdonly_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n  Pre-eval (ZeroFNet gate thresh=0.80, 2000 steps from [0,0,0,0]):")
    r = eval_hold_quality(zf_net, mpc, x0_eval, x_goal, steps=2000)
    post_str = f"  post<0.10={r['post_arr_01']:.1%}" if r['post_arr_01'] is not None else ""
    print(f"    frac<0.10={r['frac_01']:.1%}  frac<0.30={r['frac_03']:.1%}  "
          f"arr={r['arr_idx']}{post_str}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)
    rng = random.Random(42)
    all_losses = []
    best_frac01 = 0.0
    best_global_state = copy.deepcopy(lin_net.state_dict())
    t0 = time.time()
    epoch_global = 0

    interrupted = [False]
    def on_sig(sig, frame):
        interrupted[0] = True
    signal.signal(signal.SIGINT, on_sig)

    while epoch_global < EPOCHS and not interrupted[0]:
        chunk_size = min(SAVE_EVERY, EPOCHS - epoch_global)
        monitor = PrintMonitor(num_epochs=EPOCHS, start_epoch=epoch_global)
        recorder = network_module.NetworkOutputRecorder()
        chunk_losses = []

        for local_ep in range(chunk_size):
            if interrupted[0]:
                break
            # Random near-top initial state for this epoch
            x0_train = sample_near_top_x0(device, rng)
            demo_train = make_hold_demo(NUM_STEPS, x_goal, device)

            ep_losses, _ = train_module.train_linearization_network(
                lin_net=lin_net, mpc=mpc,
                x0=x0_train, x_goal=x_goal, demo=demo_train, num_steps=NUM_STEPS,
                num_epochs=1, lr=LR,
                debug_monitor=monitor, recorder=recorder,
                grad_debug=False, track_mode="energy", w_terminal_anchor=0.0,
                w_q_profile=0.0,
                w_f_pos_only=0.0,
                w_stable_phase=W_STABLE_PHASE,
                stable_phase_steps=STABLE_PHASE_STEPS,
                f_gate_thresh=F_GATE_THRESH,
                w_hold_reward=W_HOLD_REWARD,
                hold_sigma=HOLD_SIGMA,
                hold_start_step=HOLD_START_STEP,
                early_stop_patience=999,          # no early stopping per epoch
                external_optimizer=optimizer,
                restore_best=False,
            )
            chunk_losses.extend(ep_losses)

        all_losses.extend(chunk_losses)
        epoch_global += chunk_size

        r600  = eval_hold_quality(zf_net, mpc, x0_eval, x_goal, steps=600)
        r2000 = eval_hold_quality(zf_net, mpc, x0_eval, x_goal, steps=2000)
        post_str = f"  post<0.10={r2000['post_arr_01']:.1%}" if r2000['post_arr_01'] is not None else ""
        print(f"\n  [ep={epoch_global}]  600: frac<0.10={r600['frac_01']:.1%}  "
              f"arr={r600['arr_idx']}  |  "
              f"2000: frac<0.10={r2000['frac_01']:.1%}  frac<0.30={r2000['frac_03']:.1%}"
              f"{post_str}  elapsed={time.time()-t0:.0f}s", flush=True)

        if r2000['frac_01'] > best_frac01:
            best_frac01 = r2000['frac_01']
            best_global_state = copy.deepcopy(lin_net.state_dict())
            print(f"  ★ New best: {best_frac01:.1%}")

        ckpt_name = f"{session_name}_ep{epoch_global:03d}"
        network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
            model=lin_net, loss_history=all_losses,
            training_params={
                "experiment": "holdonly_ft",
                "src": src_label,
                "num_steps": NUM_STEPS,
                "f_gate_thresh": F_GATE_THRESH,
                "w_stable_phase": W_STABLE_PHASE,
                "w_hold_reward": W_HOLD_REWARD,
                "x0_type": "near_top_random",
                "frozen": "trunk,encoder,f_head,qf_head",
                "checkpoint_epoch": epoch_global,
                "best_frac01_2000step": best_frac01,
            },
            session_name=ckpt_name, recorder=recorder,
        )
        print(f"  Checkpoint → saved_models/{ckpt_name}/", flush=True)

        if best_frac01 > 0.5:
            print(f"  EXCELLENT HOLD ({best_frac01:.1%}) — stopping early.")
            break

    lin_net.load_state_dict(best_global_state)
    for m in frozen_modules:
        for p in m.parameters():
            p.requires_grad_(True)

    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=all_losses,
        training_params={
            "experiment": "holdonly_ft_FINAL",
            "src": src_label,
            "f_gate_thresh": F_GATE_THRESH,
            "w_stable_phase": W_STABLE_PHASE,
            "w_hold_reward": W_HOLD_REWARD,
            "x0_type": "near_top_random",
            "frozen": "trunk,encoder,f_head,qf_head",
            "best_frac01_2000step": best_frac01,
        },
        session_name=session_name,
    )
    print(f"\n  Final → saved_models/{session_name}/  best_frac01={best_frac01:.1%}")

    print(f"\n  Post-eval (with ZeroFNet gate, best weights):")
    for n in [600, 1000, 2000]:
        r = eval_hold_quality(zf_net, mpc, x0_eval, x_goal, steps=n)
        tag = "HOLD" if r['frac_01'] > 0.3 else ("PART" if r['frac_01'] > 0.05 else "FAIL")
        post_str = f"  post<0.10={r['post_arr_01']:.1%}" if r['post_arr_01'] is not None else ""
        print(f"    {n:>4} steps: frac<0.10={r['frac_01']:.1%}  "
              f"frac<0.30={r['frac_03']:.1%}{post_str}  [{tag}]")

    print(f"\n  Baseline (ZeroFNet thresh=0.80, no retraining): 26.2% frac<0.10 (2000-step)")
    if best_frac01 > 0.262:
        print(f"  ★★★ IMPROVEMENT: {best_frac01:.1%} > 26.2% ★★★")
    elif best_frac01 > 0.10:
        print(f"  Partial vs baseline — consider longer training or larger perturbations.")
    else:
        print(f"  No improvement — consider architecture changes.")


if __name__ == "__main__":
    main()
