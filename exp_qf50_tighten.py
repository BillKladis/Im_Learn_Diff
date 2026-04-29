"""exp_qf50_tighten.py — fine-tune nodemo_qf50 for tighter hold.

The nodemo_qf50 baseline (just trained, raw=0.30 at step 170, wrap<1.0
at 75s) HOVERS near upright but oscillates within wrap [0, 1]. We
have the sustained-hold property; we need tighter pinning.

Fine-tune from the qf50 checkpoint with:
  1. NUM_STEPS=300 (covers 130 steps of post-arrival dynamics)
  2. w_stable_phase=10 in last 100 steps (state pin: wrapped(q1-π)² +
     velocities — the 'higher q' the user wants emerges from the
     network needing to track a tight target)
  3. LR=5e-5 (gentle, don't ruin the swing-up)
  4. Qf q1d=50 KEPT (same as the loaded model was trained against)
  5. All other params identical to nodemo_qf50

Single change vs the new baseline: longer rollout + state pin.
"""

import math, os, sys, time, copy
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

PRETRAINED = "saved_models/stageD_nodemo_qf50_20260429_111711/stageD_nodemo_qf50_20260429_111711.pth"

X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 300       # 170 swing-up + 130 hold (was 170)
DT        = 0.05
EPOCHS    = 60
LR        = 5e-5      # fine-tune
HORIZON   = 10
SAVE_DIR  = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
QF_DIAG     = [20.0, 50.0, 40.0, 30.0]   # SAME as the loaded model

W_Q_PROFILE      = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
W_END_Q_HIGH     = 80.0
END_PHASE_STEPS  = 20

# NEW: state pin in last 100 steps
W_STABLE_PHASE     = 10.0
STABLE_PHASE_STEPS = 100


def make_synthetic_demo(num_steps, device):
    """q1 cosine-ramps 0 → π over the whole 300-step window so demo's
    energy ramp covers both the swing-up and the post-arrival hold
    region (where demo q1 stays near π for the last ~130 steps)."""
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    pump_steps = 170  # swing-up portion
    for i in range(num_steps):
        if i < pump_steps:
            alpha = i / max(pump_steps - 1, 1)
            t = 0.5 * (1.0 - math.cos(math.pi * alpha))
            demo[i, 0] = math.pi * t
        else:
            demo[i, 0] = math.pi  # hold phase: target stays at upright
    return demo


def wrap_pi(x): return math.atan2(math.sin(x), math.cos(x))


def metrics_for(traj, x_goal):
    wraps = np.array([
        math.sqrt(wrap_pi(s[0]-x_goal[0])**2 + s[1]**2 + s[2]**2 + s[3]**2)
        for s in traj
    ])
    in_zone = wraps < 0.3
    arr = next((i for i, v in enumerate(in_zone) if v), None)
    longest = 0; cur = 0
    for v in in_zone:
        cur = cur + 1 if v else 0
        if cur > longest: longest = cur
    return arr, longest, int(np.sum(in_zone))


class PrintMonitor:
    def __init__(self, num_epochs):
        self.num_epochs    = num_epochs
        self._header_shown = False
        self._best = float('inf')
    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>9}  {'GoalDist':>9}  "
              f"{'fNorm':>7}  {'Time':>6}  {'Best':>8}")
        print("─" * 76)
        self._header_shown = True
    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        d = info.get('pure_end_error', float('nan'))
        if d < self._best: self._best = d
        if epoch == 0 or (epoch+1) % 3 == 0 or epoch == num_epochs-1:
            print(f"{epoch+1:>4}/{num_epochs:<4}"
                  f"  {loss:>10.3f}"
                  f"  {info.get('loss_track', float('nan')):>9.3f}"
                  f"  {d:>9.4f}"
                  f"  {info.get('mean_f_extra_norm', float('nan')):>7.3f}"
                  f"  {info.get('epoch_time', float('nan')):>5.2f}s"
                  f"  {self._best:>8.4f}",
                  flush=True)


def main():
    device = torch.device("cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_synthetic_demo(NUM_STEPS, device)

    print("=" * 80)
    print("  EXP QF50-TIGHTEN: fine-tune qf50 baseline for tighter hold")
    print(f"  Pretrained: {os.path.basename(PRETRAINED)}")
    print(f"  Qf          = {QF_DIAG}")
    print(f"  NUM_STEPS = {NUM_STEPS}  (was 170)")
    print(f"  w_stable_phase = {W_STABLE_PHASE}  (last {STABLE_PHASE_STEPS} steps)")
    print(f"  LR={LR} (gentle fine-tune)  EPOCHS={EPOCHS}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    mpc.Qf          = torch.diag(torch.tensor(QF_DIAG, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # Pre-eval — to confirm the loaded baseline performs as expected
    print(f"\n  Pre-eval (loaded baseline):")
    for n in [170, 220, 400, 600, 1000, 1500]:
        x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n)
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = math.sqrt(wrap_pi(last[0]-X_GOAL[0])**2 + last[1]**2 + last[2]**2 + last[3]**2)
        print(f"    {n:>4} steps: raw={raw:.4f}  wrap={wrp:.4f}")

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(num_epochs=EPOCHS)

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=False, track_mode="energy", w_terminal_anchor=0.0,
        w_q_profile=W_Q_PROFILE,
        q_profile_pump=Q_PROFILE_PUMP,
        q_profile_stable=Q_PROFILE_STABLE,
        q_profile_state_phase=True,
        w_end_q_high=W_END_Q_HIGH,
        end_phase_steps=END_PHASE_STEPS,
        w_stable_phase=W_STABLE_PHASE,
        stable_phase_steps=STABLE_PHASE_STEPS,
    )
    elapsed = time.time() - t0

    session_name = f"stageD_qf50tight_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "qf50_tighten_finetune",
            "pretrained": PRETRAINED,
            "qf_diag": QF_DIAG,
            "num_steps": NUM_STEPS,
            "w_stable_phase": W_STABLE_PHASE,
            "stable_phase_steps": STABLE_PHASE_STEPS,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"\n  Saved → saved_models/{session_name}/")

    if len(loss_history) > 2:
        decreased = sum(1 for i in range(1, len(loss_history))
                        if loss_history[i] < loss_history[i-1])
        print(f"\n  Loss monotonicity:")
        print(f"    epoch 1: {loss_history[0]:.3f}")
        print(f"    epoch {len(loss_history)//2}: {loss_history[len(loss_history)//2]:.3f}")
        print(f"    epoch {len(loss_history)}: {loss_history[-1]:.3f}")
        print(f"    decreasing transitions: {decreased}/{len(loss_history)-1}")

    print(f"\n  Post-eval:")
    for n in [170, 220, 300, 400, 600, 1000, 1500, 2000]:
        x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n)
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = math.sqrt(wrap_pi(last[0]-X_GOAL[0])**2 + last[1]**2 + last[2]**2 + last[3]**2)
        status = "STABLE" if wrp < 0.3 else ("CLOSE" if wrp < 1.0 else "FAIL")
        print(f"    {n:>4} steps ({n*DT:>5.1f}s): raw={raw:.4f}  wrap={wrp:.4f}  {status}")

    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=2000)
    arr, lng, tot = metrics_for(x_t.cpu().numpy(), x_goal.cpu().numpy())
    print(f"\n  Sustained hold (2000 steps, wrap < 0.3):")
    print(f"    arrival: {'step '+str(arr)+' ('+f'{arr*DT:.2f}s'+')' if arr is not None else 'NEVER'}")
    print(f"    longest contiguous: {lng} steps ({lng*DT:.2f}s)")
    print(f"    total in zone: {tot} steps ({tot*DT:.2f}s)")

    print(f"\n  REFERENCE 0.0612 (1000 steps): arr=167 longest=74 total=124")
    print(f"  REFERENCE qf50 v2 (1000 steps): arr=219 longest=13 total=98")

    print(f"\n  Training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
