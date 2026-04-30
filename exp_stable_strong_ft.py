"""exp_stable_strong_ft.py — Strong velocity-gated f_extra penalty.

INSIGHT: w_f_stable has velocity gating that HELPS preserve the swing-up:
  - During approach (v=3 rad/s): gate = exp(-9/2) ≈ 0.011 → penalty 1% of w
  - During oscillation (v=2 rad/s): gate ≈ 0.13 → penalty 13% of w
  - At hold (v=0.5 rad/s): gate ≈ 0.88 → penalty 88% of w

Original stab_state used w=50, giving:
  - approach: 50×0.011 = 0.55 (trivial)
  - oscillation: 50×0.13 = 6.5 (too weak to stop oscillation!)
  - hold: 50×0.88 = 44 (meaningful but still pumped)

This experiment uses w=500, giving:
  - approach: 500×0.011 = 5.5 (still trivial, swing-up preserved!)
  - oscillation: 500×0.13 = 65 (now meaningful: 10× stronger than stab_state)
  - hold: 500×0.88 = 440 (very strong: correctly suppresses at low vel)

The key advantage over w_f_pos_only=50:
  - w_f_pos_only fires at 50× during approach (45 per element) → weakens swing-up
  - w_f_stable=500 fires at only 5.5× during approach → preserves swing-up

Demo: same hold-demo (ramp 0→π then hold at π) as posonly_ft.

Fine-tuning from 0.0612 baseline with LR=5e-5.
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

PRETRAINED   = "saved_models/stageD_nodemo_20260428_123448/stageD_nodemo_20260428_123448.pth"
X0           = [0.0, 0.0, 0.0, 0.0]
X_GOAL       = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS    = 280
ARRIVAL_STEP = 170
DT           = 0.05
EPOCHS       = 90
LR           = 5e-5
HORIZON      = 10
SAVE_DIR     = "saved_models"
Q_BASE_DIAG  = [12.0, 5.0, 50.0, 40.0]

W_F_STABLE   = 500.0    # 10× stronger than stab_state's 50
W_Q_PROFILE  = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
W_END_Q_HIGH = 80.0
END_PHASE_STEPS = 20
SAVE_EVERY = 15


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


def eval_hold_quality(lin_net, mpc, x0, x_goal, steps=600):
    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([
        math.sqrt(
            math.atan2(math.sin(s[0]-math.pi), math.cos(s[0]-math.pi))**2
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
    }


class PrintMonitor:
    def __init__(self, num_epochs, start_epoch=0):
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self._header_shown = False
        self._best = float('inf')

    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>8}  {'GoalDist':>9}  "
              f"{'fNorm':>7}  {'Time':>6}")
        print("─" * 70)
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
                  f"  {info.get('epoch_time', float('nan')):>5.2f}s",
                  flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_demo_with_hold(NUM_STEPS, ARRIVAL_STEP, device)

    print("=" * 80)
    print("  EXP STABLE-STRONG-FT: velocity-gated f_extra penalty × 10 stronger")
    print(f"  w_f_stable={W_F_STABLE} (vs stab_state's 50 — 10× stronger)")
    print(f"  Velocity gating preserves swing-up approach (unlike w_f_pos_only)")
    print(f"  At v=2 (oscillation): effective w = {W_F_STABLE*0.135:.0f}")
    print(f"  At v=0.5 (hold):      effective w = {W_F_STABLE*0.88:.0f}")
    print(f"  At v=3 (approach):    effective w = {W_F_STABLE*0.011:.1f}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()
    session_name = f"stageD_stablestrong_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n  Pre-eval:")
    r = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=600)
    print(f"    600 steps: frac<0.10={r['frac_01']:.1%}  arr={r['arr_idx']}")

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
            q_profile_pump=Q_PROFILE_PUMP,
            q_profile_stable=Q_PROFILE_STABLE,
            q_profile_state_phase=True,
            w_end_q_high=W_END_Q_HIGH,
            end_phase_steps=END_PHASE_STEPS,
            w_f_stable=W_F_STABLE,     # strong velocity-gated penalty
            w_f_pos_only=0.0,          # don't add position-only (would hurt approach)
            early_stop_patience=SAVE_EVERY + 5,
            external_optimizer=optimizer,
            restore_best=False,
        )
        all_losses.extend(loss_chunk)
        chunk_start += n_ep

        r = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=600)
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
                "experiment": "stablestrong_ft",
                "w_f_stable": W_F_STABLE,
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
        training_params={"experiment": "stablestrong_ft_FINAL", "best_frac01": best_frac01},
        session_name=session_name,
    )
    print(f"\n  Final → saved_models/{session_name}/  best_frac01={best_frac01:.1%}")

    print(f"\n  Post-eval:")
    for n in [280, 600, 1000, 2000]:
        r = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=n)
        tag = "HOLD" if r['frac_01'] > 0.3 else ("PART" if r['frac_01'] > 0.05 else "FAIL")
        print(f"    {n:>4} steps: frac<0.10={r['frac_01']:.1%}  "
              f"frac<0.30={r['frac_03']:.1%}  [{tag}]")


if __name__ == "__main__":
    main()
