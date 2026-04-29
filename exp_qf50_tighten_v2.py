"""exp_qf50_tighten_v2.py — aggressive fine-tune with proper best-metric.

v1 was a no-op (LR=5e-5 too gentle, best-metric was noisy snapshot).
This version:
  1. LR=2e-4 (4x higher)
  2. q_profile_stable raised from [1,1,1,1] to [1.95, 1.95, 1.95, 1.95]
     (max possible with gate_range_q=0.99). QP gets MAX braking near goal.
  3. Custom best-tracker: longest contiguous wrap < 0.3 over 1000-step
     rollout. This is the metric we actually care about.
  4. NUM_STEPS=300, w_stable_phase=10 (last 100 steps). Same as v1.
  5. EPOCHS=80, no early stop (run full schedule)

Pretrained: nodemo_qf50 (the new baseline that hovers wrap<1 for 75s).
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
NUM_STEPS = 300
DT        = 0.05
EPOCHS    = 80
LR        = 2e-4
HORIZON   = 10
SAVE_DIR  = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
QF_DIAG     = [20.0, 50.0, 40.0, 30.0]

W_Q_PROFILE      = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.95, 1.95, 1.95, 1.95]   # MAX (was [1,1,1,1])
W_END_Q_HIGH     = 80.0
END_PHASE_STEPS  = 20

W_STABLE_PHASE     = 10.0
STABLE_PHASE_STEPS = 100


def make_synthetic_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    pump_steps = 170
    for i in range(num_steps):
        if i < pump_steps:
            alpha = i / max(pump_steps - 1, 1)
            t = 0.5 * (1.0 - math.cos(math.pi * alpha))
            demo[i, 0] = math.pi * t
        else:
            demo[i, 0] = math.pi
    return demo


def wrap_pi(x): return math.atan2(math.sin(x), math.cos(x))


def hold_metric(traj, x_goal, threshold=0.3):
    """Longest contiguous run with wrap < threshold."""
    wraps = np.array([
        math.sqrt(wrap_pi(s[0]-x_goal[0])**2 + s[1]**2 + s[2]**2 + s[3]**2)
        for s in traj
    ])
    in_zone = wraps < threshold
    longest = 0; cur = 0
    for v in in_zone:
        cur = cur + 1 if v else 0
        if cur > longest: longest = cur
    arr = next((i for i, v in enumerate(in_zone) if v), None)
    total = int(np.sum(in_zone))
    return arr, longest, total


class HoldMonitor:
    """Custom monitor that runs a 1000-step rollout periodically and uses
    longest-contiguous-hold as the BEST metric. Saves the state when the
    longest hold improves."""
    def __init__(self, num_epochs, mpc, lin_net, x0, x_goal):
        self.num_epochs = num_epochs
        self.mpc, self.lin_net = mpc, lin_net
        self.x0, self.x_goal = x0, x_goal
        self._best_long  = 0
        self._best_state = copy.deepcopy(lin_net.state_dict())
        self._header_shown = False

    def _header(self):
        print(f"\n{'Epoch':>6}  {'Loss':>9}  {'Track':>7}  {'fNorm':>7}  "
              f"{'Arr':>5}  {'Long':>5}  {'Total':>5}  {'BestL':>6}")
        print("─" * 72)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown: self._header()
        # Eval every 2 epochs to save time
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                x_t, _ = train_module.rollout(
                    lin_net=self.lin_net, mpc=self.mpc,
                    x0=self.x0, x_goal=self.x_goal, num_steps=1000,
                )
            arr, long, total = hold_metric(x_t.cpu().numpy(), self.x_goal.cpu().numpy())
            if long > self._best_long:
                self._best_long  = long
                self._best_state = copy.deepcopy(self.lin_net.state_dict())
            arr_str = "—" if arr is None else f"{arr}"
            print(f"  {epoch+1:>4}  {loss:>9.3f}  "
                  f"{info.get('loss_track', 0):>7.3f}  "
                  f"{info.get('mean_f_extra_norm', 0):>7.3f}  "
                  f"{arr_str:>5}  {long:>5}  {total:>5}  "
                  f"{self._best_long:>6}", flush=True)


def main():
    device = torch.device("cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_synthetic_demo(NUM_STEPS, device)

    print("=" * 78)
    print("  EXP QF50-TIGHTEN v2")
    print(f"  Pretrained: {os.path.basename(PRETRAINED)}")
    print(f"  LR={LR}  EPOCHS={EPOCHS}  NUM_STEPS={NUM_STEPS}")
    print(f"  Q_PROFILE_STABLE = {Q_PROFILE_STABLE}  (MAX)")
    print(f"  w_stable_phase = {W_STABLE_PHASE}  steps={STABLE_PHASE_STEPS}")
    print(f"  Best-metric: longest contiguous wrap<0.3 over 1000-step rollout")
    print("=" * 78)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    mpc.Qf          = torch.diag(torch.tensor(QF_DIAG, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # Pre-eval
    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=1000)
    arr0, long0, tot0 = hold_metric(x_t.cpu().numpy(), x_goal.cpu().numpy())
    print(f"\n  Pre-eval (loaded baseline, 1000-step):")
    print(f"    arrival={arr0}  longest={long0}  total={tot0}")

    monitor = HoldMonitor(EPOCHS, mpc, lin_net, x0, x_goal)
    recorder = network_module.NetworkOutputRecorder()

    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=LR, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(EPOCHS):
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=1, lr=LR,
            debug_monitor=monitor, recorder=recorder, grad_debug=False,
            track_mode="energy", w_terminal_anchor=0.0,
            w_q_profile=W_Q_PROFILE,
            q_profile_pump=Q_PROFILE_PUMP,
            q_profile_stable=Q_PROFILE_STABLE,
            q_profile_state_phase=True,
            w_end_q_high=W_END_Q_HIGH, end_phase_steps=END_PHASE_STEPS,
            w_stable_phase=W_STABLE_PHASE,
            stable_phase_steps=STABLE_PHASE_STEPS,
            external_optimizer=optimizer,
            restore_best=False,
        )
    elapsed = time.time() - t0

    # Restore best
    lin_net.load_state_dict(monitor._best_state)

    session_name = f"stageD_qf50tightv2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={
            "experiment": "qf50_tighten_v2_aggressive",
            "pretrained": PRETRAINED,
            "qf_diag": QF_DIAG,
            "best_long": monitor._best_long,
            "lr": LR,
            "q_profile_stable": Q_PROFILE_STABLE,
        },
        session_name=session_name,
    )
    print(f"\n  Saved → saved_models/{session_name}/  best_long={monitor._best_long}")

    # Post-eval
    print(f"\n  Post-eval:")
    for n in [170, 220, 300, 400, 600, 1000, 1500, 2000]:
        x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n)
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = math.sqrt(wrap_pi(last[0]-X_GOAL[0])**2 + last[1]**2 + last[2]**2 + last[3]**2)
        status = "STABLE" if wrp < 0.3 else ("CLOSE" if wrp < 1.0 else "FAIL")
        print(f"    {n:>4} steps ({n*DT:>5.1f}s): raw={raw:.4f}  wrap={wrp:.4f}  {status}")

    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=2000)
    arr, lng, tot = hold_metric(x_t.cpu().numpy(), x_goal.cpu().numpy())
    print(f"\n  Sustained hold (2000 steps, wrap < 0.3):")
    print(f"    arrival: {'step '+str(arr)+' ('+f'{arr*DT:.2f}s'+')' if arr is not None else 'NEVER'}")
    print(f"    longest contiguous: {lng} steps ({lng*DT:.2f}s)")
    print(f"    total in zone: {tot} steps ({tot*DT:.2f}s)")

    print(f"\n  REFERENCE qf50 v2: arr=219 longest=13 total=98 (1000-step)")
    print(f"  Training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
