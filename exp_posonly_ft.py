"""exp_posonly_ft.py — Fine-tune with POSITION-ONLY f_extra penalty.

Key insight (HANDOFF section 7):
  w_f_stable (velocity-gated) misses swing-through oscillations. At v=2 rad/s
  the penalty is only ~13% of full strength. The pendulum can freely pump
  through the upright with high velocity.

Fix: w_f_pos_only fires regardless of velocity:
  penalty = w * (1 + cos(q1 - π)) / 2 * ||f_extra||²

Demo fix: original stab_state ramped demo q1 0→π over all 220 steps, so the
energy target was BELOW E_max at step 170 (arrival) — the track loss kept
telling the network to pump more. New demo: ramp 0→π over first ARRIVAL_STEP
steps, then hold at π. Track loss target = E_max for hold phase → gradient
pushes to brake (reduce KE) when pendulum has excess energy at top.

Starting from 0.0612 (real f_extra swing-up), NOT from scratch.
f_extra swing-up is preserved; only the hold phase is improved.
"""

import math
import os
import sys
import time
import signal
import copy
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

# ── Config ─────────────────────────────────────────────────────────────────
PRETRAINED = "saved_models/stageD_nodemo_20260428_123448/stageD_nodemo_20260428_123448.pth"

X0           = [0.0, 0.0, 0.0, 0.0]
X_GOAL       = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS    = 280          # 170 swing-up + 110 stabilisation window
ARRIVAL_STEP = 170          # demo ramp → hold transition step
DT           = 0.05
EPOCHS       = 90
LR           = 5e-5         # small to preserve swing-up
HORIZON      = 10
SAVE_DIR     = "saved_models"
Q_BASE_DIAG  = [12.0, 5.0, 50.0, 40.0]

W_F_POS_ONLY     = 50.0
W_Q_PROFILE      = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
W_END_Q_HIGH     = 80.0
END_PHASE_STEPS  = 20
SAVE_EVERY       = 15   # periodic checkpoint interval (epochs)


def make_demo_with_hold(num_steps, arrival_step, device):
    """Cosine-eased ramp 0→π for steps 0..arrival_step, then hold at π.

    Energy track loss effect after arrival_step:
    - pendulum at top, KE>0: E_now > E_max → gradient pushes to brake
    - pendulum at bottom: E_now << E_max → gradient pushes to pump
    Net effect with w_f_pos_only suppressing f_extra at top: pendulum
    decelerates when it reaches the top, QP holds it there.
    """
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        if i <= arrival_step:
            alpha = i / max(arrival_step, 1)
            t = 0.5 * (1.0 - math.cos(math.pi * alpha))
            demo[i, 0] = math.pi * t
        else:
            demo[i, 0] = math.pi
    return demo


def eval_hold_quality(lin_net, mpc, x0, x_goal, steps=600):
    """600-step rollout → wrap<0.10 fraction over post-arrival window."""
    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([
        math.sqrt(
            math.atan2(math.sin(s[0] - math.pi), math.cos(s[0] - math.pi))**2
            + s[1]**2 + s[2]**2 + s[3]**2
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
        if epoch == 0 or (abs_epoch) % 5 == 0 or epoch == num_epochs - 1:
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

    print("=" * 80)
    print("  EXP POSONLY-FT: position-only f_extra penalty + hold-demo")
    print(f"  Pretrained: 0.0612 baseline")
    print(f"  NUM_STEPS={NUM_STEPS}  ARRIVAL_STEP={ARRIVAL_STEP}  LR={LR}  EPOCHS={EPOCHS}")
    print(f"  w_f_pos_only={W_F_POS_ONLY}  (no velocity gating!)")
    print(f"  Demo: ramp 0→π steps 0-{ARRIVAL_STEP}, then hold at π")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()
    session_name = f"stageD_posonly_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ── Pre-eval ─────────────────────────────────────────────────────────
    print("\n  Pre-eval (0.0612 baseline before fine-tuning):")
    r = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=600)
    print(f"    600 steps: frac<0.10={r['frac_01']:.1%}  frac<0.30={r['frac_03']:.1%}  "
          f"arr={r['arr_idx']}  wrap_final={r['wrap_final']:.4f}")

    # ── Chunked training with periodic checkpoints ────────────────────────
    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=LR, weight_decay=1e-4)
    all_losses = []
    best_global_dist = float("inf")
    best_global_state = copy.deepcopy(lin_net.state_dict())
    t0 = time.time()

    # Save on interrupt
    interrupted = [False]
    def on_sig(sig, frame):
        interrupted[0] = True
        print("\n  SIGINT — saving checkpoint before exit...")
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
            q_profile_pump=Q_PROFILE_PUMP,
            q_profile_stable=Q_PROFILE_STABLE,
            q_profile_state_phase=True,
            w_end_q_high=W_END_Q_HIGH,
            end_phase_steps=END_PHASE_STEPS,
            w_f_pos_only=W_F_POS_ONLY,
            early_stop_patience=SAVE_EVERY + 5,  # don't stop within a chunk
            external_optimizer=optimizer,
            restore_best=False,   # don't rollback within chunk; we track globally
        )
        all_losses.extend(loss_chunk)
        chunk_start += n_ep

        # Eval this checkpoint
        r = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=600)
        print(f"\n  [Chunk end ep={chunk_start}] frac<0.10={r['frac_01']:.1%}  "
              f"frac<0.30={r['frac_03']:.1%}  arr={r['arr_idx']}  "
              f"wrap_final={r['wrap_final']:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)

        # Track global best (by frac_01 — real hold quality)
        if r['frac_01'] > 0.0 or r['wrap_final'] < best_global_dist:
            best_global_dist = r['wrap_final']
            best_global_state = copy.deepcopy(lin_net.state_dict())

        # Save checkpoint
        ckpt_name = f"{session_name}_ep{chunk_start:03d}"
        network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
            model=lin_net, loss_history=all_losses,
            training_params={
                "experiment": "posonly_ft",
                "pretrained": PRETRAINED,
                "num_steps": NUM_STEPS,
                "arrival_step": ARRIVAL_STEP,
                "w_f_pos_only": W_F_POS_ONLY,
                "lr": LR,
                "checkpoint_epoch": chunk_start,
            },
            session_name=ckpt_name, recorder=recorder,
        )
        print(f"  Checkpoint → saved_models/{ckpt_name}/", flush=True)

        # Early stop if holding well
        if r['frac_01'] > 0.5:
            print(f"  EXCELLENT HOLD ({r['frac_01']:.1%} wrap<0.10) — stopping early.", flush=True)
            break

    elapsed = time.time() - t0

    # ── Final save with best weights ──────────────────────────────────────
    lin_net.load_state_dict(best_global_state)
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=all_losses,
        training_params={
            "experiment": "posonly_ft_FINAL",
            "pretrained": PRETRAINED,
            "num_steps": NUM_STEPS,
            "arrival_step": ARRIVAL_STEP,
            "w_f_pos_only": W_F_POS_ONLY,
            "lr": LR,
            "total_epochs": chunk_start,
        },
        session_name=session_name,
    )
    print(f"\n  Final model → saved_models/{session_name}/")

    # ── Full post-eval ────────────────────────────────────────────────────
    print(f"\n  Post-eval (best model, {elapsed:.0f}s training):")
    for n in [280, 600, 1000, 2000]:
        r = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=n)
        hold = "HOLD" if r['frac_01'] > 0.3 else ("PART" if r['frac_01'] > 0.05 else "FAIL")
        print(f"    {n:>4} steps ({n*DT:>5.1f}s): frac<0.10={r['frac_01']:.1%}  "
              f"frac<0.30={r['frac_03']:.1%}  wrap_final={r['wrap_final']:.4f}  [{hold}]")

    # Detailed 2000-step trace
    r2k = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=2000)
    arr = r2k['arr_idx']
    if arr is not None:
        post = r2k['wraps'][arr:]
        print(f"\n  2000-step trace (arrival at step {arr}, {arr*DT:.1f}s):")
        for thresh in [0.05, 0.10, 0.30, 1.0]:
            n = int(np.sum(post < thresh))
            print(f"    wrap<{thresh:.2f}: {n/len(post):.1%} ({n}/{len(post)} steps)")
    else:
        print("\n  Never reached wrap<0.3 in 2000 steps!")

    print(f"\n  Total training time: {elapsed:.0f}s  epochs={chunk_start}")


if __name__ == "__main__":
    main()
