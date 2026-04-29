"""exp_qf50_hold_tight.py — fine-tune qf50 v2 to tighten the hold.

qf50 v2 hovers near upright for 75s but oscillates within wrap [0, 1].
The longest contiguous wrap<0.3 hold is only 13-14 steps because the
pendulum keeps drifting just out of zone and coming back. This is
caused by residual f_extra firing when the state moves slightly away
from the stable zone — kicking the pendulum out.

Approach:
  1. Load qf50 v2 (the working baseline)
  2. NUM_STEPS=300 (covers 130 steps of post-arrival dynamics)
  3. **w_f_stable=30** (NEW): state-conditional f_extra penalty.
     stable_zone = (1+cos(q1-π))/2 * exp(-(q1d²+q2d²)/2). Penalises
     ‖f_extra‖² in this zone. Directly suppresses the residual
     pumping that causes the oscillation.
  4. LR=1e-4 (moderate — v1 at 5e-5 was a no-op, v2 at 2e-4 broke
     swing-up at iter 1)
  5. HoldMonitor: best metric is **longest contiguous wrap<0.3** in
     a 1000-step rollout (the actual metric we care about), saved
     when it improves
  6. Persistent optimizer + restore_best=False (the curriculum-style
     bug fixes from earlier)
  7. EPOCHS=60, no patience (run full schedule)
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
LR        = 3e-5    # gentler than v1 (5e-5) since fine-tune from a trained model
HORIZON   = 10
SAVE_DIR  = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
QF_DIAG     = [20.0, 50.0, 40.0, 30.0]   # SAME as qf50 v2

W_Q_PROFILE      = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]   # KEEP default (max-out broke v2)
W_END_Q_HIGH     = 80.0
END_PHASE_STEPS  = 20
W_F_STABLE       = 5.0    # gentle (was 30; the trained qf50 had high f_extra so the gradient was huge and broke swing-up at iter 1)


def make_demo(num_steps, device):
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
    """Best metric = longest contiguous wrap<0.3 over a 1000-step
    rollout from x0=zero. Saves state when best improves."""
    def __init__(self, num_epochs, mpc, lin_net, x0, x_goal):
        self.num_epochs = num_epochs
        self.mpc, self.lin_net = mpc, lin_net
        self.x0, self.x_goal = x0, x_goal
        self._best_long = 0
        self._best_total = 0
        self._best_state = copy.deepcopy(lin_net.state_dict())
        self._header_shown = False

    def _header(self):
        print(f"\n{'Iter':>4}  {'Loss':>9}  {'Track':>7}  {'fNorm':>7}  "
              f"{'Arr':>5}  {'Long':>5}  {'Total':>5}  {'BestL':>6}  {'BestT':>6}")
        print("─" * 78)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown: self._header()
        # Cheap eval every iter (1000-step rollout takes ~30s on contended CPU)
        with torch.no_grad():
            x_t, _ = train_module.rollout(
                lin_net=self.lin_net, mpc=self.mpc,
                x0=self.x0, x_goal=self.x_goal, num_steps=1000,
            )
        arr, long, total = hold_metric(x_t.cpu().numpy(), self.x_goal.cpu().numpy())
        # Save when EITHER longest or total improves (don't lose progress)
        improved = False
        if long > self._best_long:
            self._best_long = long; improved = True
        if total > self._best_total:
            self._best_total = total
            if not improved:  # only save if longest didn't already trigger
                self._best_state = copy.deepcopy(self.lin_net.state_dict())
        if improved:
            self._best_state = copy.deepcopy(self.lin_net.state_dict())
        arr_str = "—" if arr is None else f"{arr}"
        print(f"  {epoch+1:>2}  {loss:>9.3f}  "
              f"{info.get('loss_track', 0):>7.3f}  "
              f"{info.get('mean_f_extra_norm', 0):>7.3f}  "
              f"{arr_str:>5}  {long:>5}  {total:>5}  "
              f"{self._best_long:>6}  {self._best_total:>6}", flush=True)


def main():
    device = torch.device("cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_demo(NUM_STEPS, device)

    print("=" * 80)
    print("  EXP QF50-HOLD-TIGHT: fine-tune qf50 v2 with w_f_stable=30")
    print(f"  Pretrained: {os.path.basename(PRETRAINED)}")
    print(f"  Qf={QF_DIAG}  NUM_STEPS={NUM_STEPS}  LR={LR}  EPOCHS={EPOCHS}")
    print(f"  w_f_stable = {W_F_STABLE}  (state-conditional f_extra suppression)")
    print(f"  Best metric: longest contiguous wrap<0.3 over 1000-step rollout")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    mpc.Qf          = torch.diag(torch.tensor(QF_DIAG, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device="cpu").double()

    # Pre-eval
    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=1000)
    arr0, long0, tot0 = hold_metric(x_t.cpu().numpy(), x_goal.cpu().numpy())
    print(f"\n  Pre-eval (loaded baseline, 1000 steps):")
    print(f"    arrival={arr0}  longest={long0}  total={tot0}")

    monitor = HoldMonitor(EPOCHS, mpc, lin_net, x0, x_goal)
    recorder = network_module.NetworkOutputRecorder()

    # Persistent optimizer
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
            w_f_stable=W_F_STABLE,
            external_optimizer=optimizer,
            restore_best=False,
        )
    elapsed = time.time() - t0

    lin_net.load_state_dict(monitor._best_state)
    session_name = f"stageD_qf50holdtight_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={
            "experiment": "qf50_hold_tight_finetune",
            "pretrained": PRETRAINED,
            "qf_diag": QF_DIAG,
            "lr": LR,
            "w_f_stable": W_F_STABLE,
            "best_long": monitor._best_long,
            "best_total": monitor._best_total,
        },
        session_name=session_name,
    )
    print(f"\n  Saved → saved_models/{session_name}/  best_long={monitor._best_long}  best_total={monitor._best_total}")

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

    print(f"\n  Reference qf50 v2 (1000 steps): arr=219 longest=14 total=96")
    print(f"  Training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
