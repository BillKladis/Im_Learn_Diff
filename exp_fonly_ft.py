"""exp_fonly_ft.py — f_head ONLY fine-tuning (everything else frozen).

ROOT CAUSE OF PREVIOUS FAILURES:
  Training trunk+encoder (even at LR×0.05) causes trunk features to drift.
  Even though q_head/r_head parameters are frozen, they receive DIFFERENT
  inputs from the modified trunk → gate values change → ZeroFNet analogy breaks.
  Network still takes easy path: global f_extra reduction.

FIX: FREEZE EVERYTHING EXCEPT f_head.
  - state_encoder, trunk, q_head, r_head ALL frozen
  - f_head receives EXACTLY THE SAME trunk features as posonly_ft ep75
  - q_head and r_head outputs IDENTICAL to posonly_ft ep75 (same params + same features)
  - f_head just needs to learn: near_pi > 0.8 → output f_extra≈0

WHY THIS SHOULD WORK:
  The trunk features encode q1 (position). near_pi = (1+cos(q1-π))/2.
  Trunk features strongly encode position, so f_head CAN learn:
    if near_pi > 0.8: output 0
    else: output large (for energy pumping)
  This is exactly what ZeroFNet does, but baked into f_head weights.

ZeroFNet test showed: 25.9% frac<0.10 (2000-step), 31.0% post-arrival.
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
EPOCHS       = 120
LR           = 2e-4       # higher: only f_head updates
HORIZON      = 10
SAVE_DIR     = "saved_models"
Q_BASE_DIAG  = [12.0, 5.0, 50.0, 40.0]

W_F_TIGHT_TOP    = 1000.0  # stronger: need hard suppression near top
TIGHT_TOP_THRESH = 0.8

W_F_PHASE_REWARD = 50.0   # prevent global collapse: reward large f_extra during swing-up

W_F_POS_ONLY  = 0.0   # tight_top handles near-top suppression
W_Q_PROFILE   = 0.0   # frozen
W_END_Q_HIGH  = 0.0   # frozen
END_PHASE_STEPS = 20
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


def eval_hold_quality(lin_net, mpc, x0, x_goal, steps=600):
    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
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
    print("  EXP FONLY-FT: f_head-ONLY training (all other layers frozen)")
    print(f"  Pretrained: {src_label}")
    print(f"  FROZEN: state_encoder, trunk, q_head, r_head")
    print(f"  TRAINED: f_head only (LR={LR})")
    print(f"  w_f_tight_top={W_F_TIGHT_TOP}  thresh={TIGHT_TOP_THRESH}  w_f_phase_reward={W_F_PHASE_REWARD}")
    print(f"  ZeroFNet target: 25.9% frac<0.10 (2000-step)")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(pretrained, device=str(device)).double()

    # Freeze EVERYTHING except f_head
    for param in lin_net.state_encoder.parameters():
        param.requires_grad = False
    for param in lin_net.trunk.parameters():
        param.requires_grad = False
    for param in lin_net.q_head.parameters():
        param.requires_grad = False
    for param in lin_net.r_head.parameters():
        param.requires_grad = False

    frozen_params = sum(p.numel() for p in lin_net.parameters() if not p.requires_grad)
    total_params  = sum(p.numel() for p in lin_net.parameters())
    f_head_params = sum(p.numel() for p in lin_net.f_head.parameters())
    print(f"\n  Frozen: {frozen_params}/{total_params} params")
    print(f"  f_head trainable: {f_head_params} params")

    session_name = f"stageD_fonly_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n  Pre-eval:")
    r = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=600)
    print(f"    600 steps: frac<0.10={r['frac_01']:.1%}  frac<0.30={r['frac_03']:.1%}  "
          f"arr={r['arr_idx']}  wrap_final={r['wrap_final']:.4f}")

    optimizer = torch.optim.AdamW(
        lin_net.f_head.parameters(), lr=LR, weight_decay=1e-4
    )

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
            w_end_q_high=W_END_Q_HIGH,
            end_phase_steps=END_PHASE_STEPS,
            w_f_pos_only=W_F_POS_ONLY,
            w_f_tight_top=W_F_TIGHT_TOP,
            tight_top_thresh=TIGHT_TOP_THRESH,
            w_f_phase_reward=W_F_PHASE_REWARD,
            phase_split_frac=0.6,
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
                "experiment": "fonly_ft",
                "src": src_label,
                "w_f_tight_top": W_F_TIGHT_TOP,
                "tight_top_thresh": TIGHT_TOP_THRESH,
                "w_f_phase_reward": W_F_PHASE_REWARD,
                "frozen": ["state_encoder", "trunk", "q_head", "r_head"],
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
            "experiment": "fonly_ft_FINAL",
            "src": src_label,
            "best_frac01": best_frac01,
        },
        session_name=session_name,
    )
    print(f"\n  Final → saved_models/{session_name}/  best_frac01={best_frac01:.1%}")

    print(f"\n  Post-eval:")
    for n in [280, 600, 1000, 2000]:
        r = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=n)
        tag = "HOLD" if r['frac_01'] > 0.3 else ("PART" if r['frac_01'] > 0.05 else "FAIL")
        print(f"    {n:>4} steps: frac<0.10={r['frac_01']:.1%}  "
              f"frac<0.30={r['frac_03']:.1%}  [{tag}]")

    r2k = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=2000)
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
