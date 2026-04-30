"""exp_zftrain_ft.py — Train WITH ZeroFNet gate applied during training.

KEY INSIGHT:
  All previous fine-tuning attempts failed because of conflicting gradients:
  - Energy tracking wants f_extra high during swing-up
  - Suppression penalty wants f_extra low near top
  - Gradient descent takes easy path: reduce f_extra globally

  DIFFERENT APPROACH: Apply ZeroFNet gate as a HARD ZERO during training.
    f_extra = f_extra * (1 - zerofnet_gate)  (no gradient to f_head near top)

  With gate applied:
  - f_head gets NO gradient from near-top states (output zeroed before loss)
  - f_head only learns from swing-up states → preserved/improved
  - Q/R gates DO get gradient from near-top states → adapt to stabilize WITHOUT f_extra
  - Energy tracking with f_extra=0 near top trains Q/R for stronger damping at top

  After training: evaluate WITH the same gate applied at inference.
  Q/R gates now optimized for the ZeroFNet regime → should exceed 25.9%.

BASELINE:
  ZeroFNet on posonly_ft ep75: 25.9% frac<0.10 (with f_extra=0 when near_pi>0.9)
  Original posonly_ft ep75:     3.8% frac<0.10 (no gate)
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
NUM_STEPS    = 280
ARRIVAL_STEP = 170
DT           = 0.05
EPOCHS       = 90
LR           = 3e-5        # conservative: same as tighttop_ft
HORIZON      = 10
SAVE_DIR     = "saved_models"
Q_BASE_DIAG  = [12.0, 5.0, 50.0, 40.0]

# KEY: ZeroFNet gate applied during training
F_GATE_THRESH    = 0.9     # matches best ZeroFNet result (25.9% frac<0.10)

# No posonly — it globally collapses fNorm through f_head weights
# Use w_stable_phase instead: direct hold quality signal, gradient via Q/R gates
W_F_POS_ONLY     = 0.0   # eliminated: penalizes f_extra globally through f_head weights
W_Q_PROFILE      = 0.0   # eliminated: no profile needed with gate + stable_phase
W_END_Q_HIGH     = 0.0

# Hold stability loss: direct position+velocity tracking in last N steps
W_STABLE_PHASE   = 50.0  # penalize deviation from goal in last stable_phase_steps
STABLE_PHASE_STEPS = 60  # last 60 of 280 training steps

END_PHASE_STEPS  = 20
SAVE_EVERY       = 15


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
    """Wraps lin_net to apply ZeroFNet gate at inference."""
    def __init__(self, lin_net, thresh, x_goal_q1):
        super().__init__()
        self.lin_net    = lin_net
        self.thresh     = thresh
        self.x_goal_q1  = x_goal_q1
        # Expose attributes needed by rollout/training utilities
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


def eval_hold_quality(model, mpc, x0, x_goal, steps=600):
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
    return {
        "arr_idx": arr_idx,
        "frac_01": float((wraps < 0.10).mean()),
        "frac_03": float((wraps < 0.30).mean()),
        "wrap_final": float(wraps[-1]),
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
    print("  EXP ZFTRAIN-FT: train WITH ZeroFNet gate hard-applied")
    print(f"  Pretrained: {src_label}")
    print(f"  f_gate_thresh={F_GATE_THRESH}  (f_extra=0 when near_pi>{F_GATE_THRESH} during training)")
    print(f"  w_stable_phase={W_STABLE_PHASE}  stable_phase_steps={STABLE_PHASE_STEPS}")
    print(f"  Q/R heads adapt from energy tracking + hold stability loss with gated f_extra")
    print(f"  Evaluation also uses gate: ZeroFNetWrapper(thresh={F_GATE_THRESH})")
    print(f"  Target: exceed 25.9% frac<0.10 by optimising Q/R for no-f regime")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(pretrained, device=str(device)).double()
    zf_net  = ZeroFNetWrapper(lin_net, thresh=F_GATE_THRESH, x_goal_q1=X_GOAL[0])
    session_name = f"stageD_zftrain_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n  Pre-eval (with ZeroFNet gate):")
    r = eval_hold_quality(zf_net, mpc, x0, x_goal, steps=600)
    print(f"    600 steps: frac<0.10={r['frac_01']:.1%}  frac<0.30={r['frac_03']:.1%}  "
          f"arr={r['arr_idx']}  wrap_final={r['wrap_final']:.4f}")
    print(f"  (Baseline posonly_ft ep75 + ZeroFNet = 25.9% in 2000-step eval)")

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

        # Train lin_net (not zf_net) with f_gate_thresh applied internally
        loss_chunk, recorder = train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=n_ep, lr=LR,
            debug_monitor=monitor, recorder=recorder,
            grad_debug=False, track_mode="energy", w_terminal_anchor=0.0,
            w_q_profile=W_Q_PROFILE,
            q_profile_state_phase=True,
            w_end_q_high=W_END_Q_HIGH,
            end_phase_steps=END_PHASE_STEPS,
            w_f_pos_only=W_F_POS_ONLY,
            w_stable_phase=W_STABLE_PHASE,
            stable_phase_steps=STABLE_PHASE_STEPS,
            f_gate_thresh=F_GATE_THRESH,    # KEY: ZeroFNet gate during training
            early_stop_patience=SAVE_EVERY + 5,
            external_optimizer=optimizer,
            restore_best=False,
        )
        all_losses.extend(loss_chunk)
        chunk_start += n_ep

        # Evaluate WITH ZeroFNet gate applied
        r = eval_hold_quality(zf_net, mpc, x0, x_goal, steps=600)
        print(f"\n  [ep={chunk_start}] frac<0.10={r['frac_01']:.1%}  "
              f"frac<0.30={r['frac_03']:.1%}  arr={r['arr_idx']}  "
              f"wrap_final={r['wrap_final']:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)

        if r['frac_01'] > best_frac01:
            best_frac01 = r['frac_01']
            best_global_state = copy.deepcopy(lin_net.state_dict())

        ckpt_name = f"{session_name}_ep{chunk_start:03d}"
        network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
            model=lin_net, loss_history=all_losses,
            training_params={
                "experiment": "zftrain_ft",
                "src": src_label,
                "f_gate_thresh": F_GATE_THRESH,
                "w_f_pos_only": W_F_POS_ONLY,
                "checkpoint_epoch": chunk_start,
            },
            session_name=ckpt_name, recorder=recorder,
        )
        print(f"  Checkpoint → saved_models/{ckpt_name}/", flush=True)

        if r['frac_01'] > 0.5:
            print(f"  EXCELLENT HOLD ({r['frac_01']:.1%}) — stopping early.")
            break

    lin_net.load_state_dict(best_global_state)
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=all_losses,
        training_params={
            "experiment": "zftrain_ft_FINAL",
            "src": src_label,
            "best_frac01": best_frac01,
            "f_gate_thresh": F_GATE_THRESH,
        },
        session_name=session_name,
    )
    print(f"\n  Final → saved_models/{session_name}/  best_frac01={best_frac01:.1%}")

    print(f"\n  Post-eval (with ZeroFNet gate):")
    for n in [280, 600, 1000, 2000]:
        r = eval_hold_quality(zf_net, mpc, x0, x_goal, steps=n)
        tag = "HOLD" if r['frac_01'] > 0.3 else ("PART" if r['frac_01'] > 0.05 else "FAIL")
        print(f"    {n:>4} steps: frac<0.10={r['frac_01']:.1%}  "
              f"frac<0.30={r['frac_03']:.1%}  [{tag}]")

    r2k = eval_hold_quality(zf_net, mpc, x0, x_goal, steps=2000)
    arr = r2k['arr_idx']
    if arr is not None:
        post = r2k['wraps'][arr:]
        print(f"\n  2000-step post-arrival (step {arr}):")
        for thresh in [0.05, 0.10, 0.30]:
            n_in = int(np.sum(post < thresh))
            print(f"    wrap<{thresh:.2f}: {n_in/len(post):.1%} ({n_in}/{len(post)} steps)")
    else:
        print("\n  Never arrived in 2000 steps!")


if __name__ == "__main__":
    main()
