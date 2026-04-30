"""exp_holdreward_ft.py — Train with hold-reward: state-conditional reward near top.

KEY INSIGHT (why all previous training attempts failed):
  All previous approaches used step-conditional signals (w_stable_phase fires in
  last N steps). If the pendulum falls before those steps, the signal fires on
  swing-up states → wrong gradient. w_f_tight_top/pos_only fire globally via
  f_head weights → fNorm collapse.

CORRECT APPROACH: w_hold_reward is STATE-CONDITIONAL.
  in_zone_soft(t) = exp(-wrap_sq(t) / sigma²)
  - Near top with low velocity: in_zone_soft ≈ 1.0 → strong reward
  - Swing-through (q1≈π but high q1d): wrap_sq ≈ large → in_zone_soft ≈ 0
  - Swing-up (far from top): in_zone_soft ≈ 0 → no signal
  Loss = W_TRACK*track_loss - w_hold_reward*mean(in_zone_soft[hold_start:])

GRADIENT SPLIT (natural, no conflict):
  Steps 0-300 (swing-up): E ≠ E_demo → energy tracking dominates
  Steps 300-600 (hold):   E ≈ E_demo → energy tracking ≈ 0 → hold_reward dominates
  Q/R heads receive clean hold gradient. f_head blocked by f_gate_thresh=0.8.

TARGET: Exceed 26.2% frac<0.10 (2000-step, ZeroFNet thresh=0.80).
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
NUM_STEPS    = 600    # covers hold phase (arrival ~step 294-310)
ARRIVAL_STEP = 300    # cosine ramp 300 steps, hold 300 more
DT           = 0.05
EPOCHS       = 80     # allow plenty of epochs
LR           = 2e-5   # conservative
HORIZON      = 10
SAVE_DIR     = "saved_models"
Q_BASE_DIAG  = [12.0, 5.0, 50.0, 40.0]

# KEY: state-conditional hold reward
W_HOLD_REWARD    = 50.0   # reward for being near top with low velocity
HOLD_SIGMA       = 0.5    # zone width (swing-through with q1d=4: exp(-16/0.25)≈0)
HOLD_START_STEP  = 200    # skip first 200 pure swing-up steps

# ZeroFNet gate during training: matches best eval threshold
F_GATE_THRESH    = 0.8    # soft ramp 0.80→1.0, zeroes f_extra at top
                          # matches ZeroFNetWrapper(thresh=0.80) = 26.2% at eval

W_STABLE_PHASE   = 0.0   # NO: step-conditional, fires on wrong states if pendulum falls
W_F_POS_ONLY     = 0.0   # NO: collapses fNorm globally
W_Q_PROFILE      = 0.0   # NO: keep Q/R heads free to adapt

SAVE_EVERY       = 10


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
        if epoch == 0 or abs_epoch % 5 == 0 or epoch == num_epochs - 1:
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
    x0     = torch.tensor(X0,     device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_demo_with_hold(NUM_STEPS, ARRIVAL_STEP, device)

    pretrained, src_label = _find_pretrained()
    print("=" * 80)
    print("  EXP HOLDREWARD-FT: state-conditional hold reward")
    print(f"  Pretrained: {src_label}")
    print(f"  NUM_STEPS={NUM_STEPS}  ARRIVAL_STEP={ARRIVAL_STEP}")
    print(f"  f_gate_thresh={F_GATE_THRESH}  (soft ramp, matches ZeroFNet eval best)")
    print(f"  w_hold_reward={W_HOLD_REWARD}  hold_sigma={HOLD_SIGMA}  hold_start={HOLD_START_STEP}")
    print(f"  KEY: hold_reward is STATE-conditional (exp(-wrap²/σ²))")
    print(f"       Energy tracking → 0 during hold → hold_reward dominates at top")
    print(f"       f_extra blocked by f_gate_thresh → Q/R heads learn hold quality")
    print(f"  Evaluation: ZeroFNetWrapper(thresh={F_GATE_THRESH})")
    print(f"  Target: exceed 26.2% frac<0.10 (2000-step)")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(pretrained, device=str(device)).double()
    zf_net  = ZeroFNetWrapper(lin_net, thresh=F_GATE_THRESH, x_goal_q1=X_GOAL[0])
    session_name = f"stageD_holdreward_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n  Pre-eval (ZeroFNet gate thresh=0.80, 2000 steps):")
    r = eval_hold_quality(zf_net, mpc, x0, x_goal, steps=2000)
    post_str = f"  post<0.10={r['post_arr_01']:.1%}" if r['post_arr_01'] is not None else ""
    print(f"    frac<0.10={r['frac_01']:.1%}  frac<0.30={r['frac_03']:.1%}  "
          f"arr={r['arr_idx']}{post_str}")

    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=LR, weight_decay=1e-4)
    all_losses = []
    best_frac01 = 0.0
    best_global_state = copy.deepcopy(lin_net.state_dict())
    t0 = time.time()

    interrupted = [False]
    def on_sig(sig, frame):
        interrupted[0] = True
    signal.signal(signal.SIGINT, on_sig)

    chunk_start = 0
    while chunk_start < EPOCHS and not interrupted[0]:
        n_ep = min(SAVE_EVERY, EPOCHS - chunk_start)
        monitor = PrintMonitor(num_epochs=EPOCHS, start_epoch=chunk_start)
        recorder = network_module.NetworkOutputRecorder()

        loss_chunk, recorder = train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=n_ep, lr=LR,
            debug_monitor=monitor, recorder=recorder,
            grad_debug=False, track_mode="energy", w_terminal_anchor=0.0,
            w_q_profile=W_Q_PROFILE,
            q_profile_state_phase=True,
            w_f_pos_only=W_F_POS_ONLY,
            w_stable_phase=W_STABLE_PHASE,
            f_gate_thresh=F_GATE_THRESH,
            w_hold_reward=W_HOLD_REWARD,
            hold_sigma=HOLD_SIGMA,
            hold_start_step=HOLD_START_STEP,
            early_stop_patience=SAVE_EVERY + 5,
            external_optimizer=optimizer,
            restore_best=False,
        )
        all_losses.extend(loss_chunk)
        chunk_start += n_ep

        # Evaluate at 600 (fast) and 2000 steps (key metric)
        r600  = eval_hold_quality(zf_net, mpc, x0, x_goal, steps=600)
        r2000 = eval_hold_quality(zf_net, mpc, x0, x_goal, steps=2000)
        post_str = f"  post<0.10={r2000['post_arr_01']:.1%}" if r2000['post_arr_01'] is not None else ""
        print(f"\n  [ep={chunk_start}]  600: frac<0.10={r600['frac_01']:.1%}  "
              f"arr={r600['arr_idx']}  |  "
              f"2000: frac<0.10={r2000['frac_01']:.1%}  frac<0.30={r2000['frac_03']:.1%}"
              f"{post_str}  elapsed={time.time()-t0:.0f}s", flush=True)

        if r2000['frac_01'] > best_frac01:
            best_frac01 = r2000['frac_01']
            best_global_state = copy.deepcopy(lin_net.state_dict())
            print(f"  ★ New best: {best_frac01:.1%}")

        ckpt_name = f"{session_name}_ep{chunk_start:03d}"
        network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
            model=lin_net, loss_history=all_losses,
            training_params={
                "experiment": "holdreward_ft",
                "src": src_label,
                "num_steps": NUM_STEPS,
                "f_gate_thresh": F_GATE_THRESH,
                "w_hold_reward": W_HOLD_REWARD,
                "hold_sigma": HOLD_SIGMA,
                "checkpoint_epoch": chunk_start,
                "best_frac01_2000step": best_frac01,
            },
            session_name=ckpt_name, recorder=recorder,
        )
        print(f"  Checkpoint → saved_models/{ckpt_name}/", flush=True)

        if best_frac01 > 0.5:
            print(f"  EXCELLENT HOLD ({best_frac01:.1%}) — stopping early.")
            break

    lin_net.load_state_dict(best_global_state)
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=all_losses,
        training_params={
            "experiment": "holdreward_ft_FINAL",
            "src": src_label,
            "num_steps": NUM_STEPS,
            "f_gate_thresh": F_GATE_THRESH,
            "w_hold_reward": W_HOLD_REWARD,
            "hold_sigma": HOLD_SIGMA,
            "best_frac01_2000step": best_frac01,
        },
        session_name=session_name,
    )
    print(f"\n  Final → saved_models/{session_name}/  best_frac01={best_frac01:.1%}")

    print(f"\n  Post-eval (with ZeroFNet gate, best weights):")
    for n in [600, 1000, 2000]:
        r = eval_hold_quality(zf_net, mpc, x0, x_goal, steps=n)
        tag = "HOLD" if r['frac_01'] > 0.3 else ("PART" if r['frac_01'] > 0.05 else "FAIL")
        post_str = f"  post<0.10={r['post_arr_01']:.1%}" if r['post_arr_01'] is not None else ""
        print(f"    {n:>4} steps: frac<0.10={r['frac_01']:.1%}  "
              f"frac<0.30={r['frac_03']:.1%}{post_str}  [{tag}]")

    print(f"\n  Baseline (no retraining, ZeroFNet thresh=0.80): 26.2% frac<0.10 (2000-step)")
    if best_frac01 > 0.262:
        print(f"  ★★★ IMPROVEMENT: {best_frac01:.1%} > 26.2% ★★★")
    elif best_frac01 > 0.10:
        print(f"  Partial improvement vs baseline.")
    else:
        print(f"  No improvement over ZeroFNet baseline.")


if __name__ == "__main__":
    main()
