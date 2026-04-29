"""exp_qf50_progressive.py — Retrain stab_state with Qf q1d=50 and a
progressive terminal/stable-phase penalty.

Findings from probe_qf.py:
  - Qf q1d=20 (default) → long=74 contiguous hold, total=112
  - Qf q1d=50 (no retrain) → long=98 (+32%), total=151 (+35%) — but
    arrives at step 211 (vs 167) because the policy isn't adapted
  - Higher q1d → even longer total but later arrival

The network was trained against q1d=20. To get BOTH early arrival AND
long sustained hold, retrain with Qf q1d=50 fixed in the controller.
The network adapts its swing-up energy/timing to compensate.

Per user's suggestion ("start adding a terminal penalty as epochs
progress"): also progressively scale w_stable_phase from 0 to a high
value across epochs. Early epochs preserve the swing-up; later epochs
tighten the position pin. Combined with w_f_stable (state-conditional
f_extra penalty) this gives the network three layers of pressure to
hold the upright.
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

PRETRAINED = "saved_models/stageD_stabstate_20260428_224856/stageD_stabstate_20260428_224856.pth"

X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 300
DT        = 0.05
EPOCHS    = 60
LR        = 5e-5
HORIZON   = 10
SAVE_DIR  = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
QF_DIAG     = [20.0, 50.0, 40.0, 30.0]   # q1d bumped 20 → 50

W_F_STABLE        = 50.0
W_STABLE_PHASE_MAX = 30.0     # final value of progressive penalty
STABLE_PHASE_STEPS = 100      # last 100 steps pinned to goal


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


def wrap_pi(x):
    return math.atan2(math.sin(x), math.cos(x))


def metrics(traj, x_goal):
    wraps = np.array([
        math.sqrt(wrap_pi(s[0] - x_goal[0])**2 + s[1]**2 + s[2]**2 + s[3]**2)
        for s in traj
    ])
    in_zone = wraps < 0.3
    arrive = next((i for i, v in enumerate(in_zone) if v), None)
    runs = []
    start = None
    for i, v in enumerate(in_zone):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i-1)); start = None
    if start is not None:
        runs.append((start, len(in_zone)-1))
    longest = max((r[1]-r[0]+1 for r in runs), default=0)
    longest_in_window = max(
        (r[1]-r[0]+1 for r in runs if 140 <= r[0] <= 250),
        default=0,
    )
    return arrive, longest, longest_in_window, int(np.sum(in_zone))


class PrintMonitor:
    def __init__(self, num_epochs, mpc, lin_net, x0, x_goal):
        self.num_epochs = num_epochs
        self.mpc = mpc
        self.lin_net = lin_net
        self.x0 = x0
        self.x_goal = x_goal
        self._best = (-1, -1)  # (longest_in_window, longest)
        self._best_state = None
        self._header_shown = False

    def _header(self):
        print(f"\n{'Epoch':>6}  {'w_sp':>6}  {'Loss':>9}  {'Arrive':>6}  "
              f"{'Long':>5}  {'In_W':>5}  {'Tot':>5}  {'BestIW':>6}  {'BestL':>5}")
        print("─" * 75)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        # Run rollout to evaluate
        if (epoch + 1) % 3 == 0 or epoch == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                x_t, _ = train_module.rollout(
                    lin_net=self.lin_net, mpc=self.mpc,
                    x0=self.x0, x_goal=self.x_goal, num_steps=600,
                )
            arr, lng, iw, tot = metrics(x_t.cpu().numpy(), self.x_goal.cpu().numpy())
            arr_str = "—" if arr is None else f"{arr}"
            current = (iw, lng)
            if current > self._best:
                self._best = current
                self._best_state = copy.deepcopy(self.lin_net.state_dict())
            print(f"  {epoch+1:>4}  {info.get('_w_sp', 0.0):>6.2f}  {loss:>9.3f}  "
                  f"{arr_str:>6}  {lng:>5}  {iw:>5}  {tot:>5}  "
                  f"{self._best[0]:>6}  {self._best[1]:>5}", flush=True)


def main():
    device = torch.device("cpu")
    x0     = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_demo(NUM_STEPS, device)

    print("=" * 80)
    print("  EXP QF50-PROGRESSIVE: retrain with Qf q1d=50 and growing position pin")
    print(f"  Pretrained: {os.path.basename(PRETRAINED)}")
    print(f"  Qf = {QF_DIAG}")
    print(f"  NUM_STEPS={NUM_STEPS}  EPOCHS={EPOCHS}  LR={LR}")
    print(f"  w_f_stable={W_F_STABLE} (constant)")
    print(f"  w_stable_phase: 0 → {W_STABLE_PHASE_MAX} progressive over epochs")
    print(f"  stable_phase_steps={STABLE_PHASE_STEPS}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    mpc.Qf          = torch.diag(torch.tensor(QF_DIAG, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device=str(device)).double()

    # Pre-eval
    print("\n  Pre-eval (with new Qf, no retraining yet):")
    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=600)
    arr, lng, iw, tot = metrics(x_t.cpu().numpy(), x_goal.cpu().numpy())
    print(f"    arrive={arr}  longest={lng}  in_window={iw}  total={tot}")

    # Single persistent optimizer
    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=LR, weight_decay=1e-4)

    monitor = PrintMonitor(EPOCHS, mpc, lin_net, x0, x_goal)
    recorder = network_module.NetworkOutputRecorder()

    base_kwargs = dict(
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=1,
        debug_monitor=monitor, recorder=recorder, grad_debug=False,
        track_mode="energy", w_terminal_anchor=0.0,
        w_q_profile=100.0,
        q_profile_pump=[0.01, 0.01, 1.0, 1.0],
        q_profile_stable=[1.0, 1.0, 1.0, 1.0],
        q_profile_state_phase=True,
        w_end_q_high=80.0, end_phase_steps=20,
        w_f_stable=W_F_STABLE,
        stable_phase_steps=STABLE_PHASE_STEPS,
        external_optimizer=optimizer,
        restore_best=False,
        lr=LR,
    )

    t0 = time.time()
    for epoch in range(EPOCHS):
        # Progressive: w_stable_phase ramps linearly from 0 to W_STABLE_PHASE_MAX
        progress = epoch / max(EPOCHS - 1, 1)
        w_sp = W_STABLE_PHASE_MAX * progress
        # Inject into monitor info via a hack — just stash it
        monitor._w_sp_pending = w_sp
        # Override info dict's _w_sp by passing through monitor
        old_log = monitor.log_epoch
        def patched_log(e, ne, l, info):
            info['_w_sp'] = w_sp
            old_log(e, ne, l, info)
        monitor.log_epoch = patched_log

        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            w_stable_phase=w_sp,
            **base_kwargs,
        )
        monitor.log_epoch = old_log

    elapsed = time.time() - t0

    if monitor._best_state is not None:
        lin_net.load_state_dict(monitor._best_state)

    session_name = f"stageD_qf50prog_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={
            "experiment": "qf50_progressive",
            "pretrained": PRETRAINED,
            "qf_diag": QF_DIAG,
            "w_f_stable": W_F_STABLE,
            "w_stable_phase_max": W_STABLE_PHASE_MAX,
            "stable_phase_steps": STABLE_PHASE_STEPS,
            "best_in_window": monitor._best[0],
            "best_longest": monitor._best[1],
        },
        session_name=session_name,
    )
    print(f"\n  Saved → saved_models/{session_name}/")

    # Post-eval
    print(f"\n  Post-eval (canonical x0=zero):")
    for n in [170, 220, 300, 400, 600, 1000, 1500]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n,
        )
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = math.sqrt(wrap_pi(last[0]-X_GOAL[0])**2 + last[1]**2 + last[2]**2 + last[3]**2)
        status = "STABLE" if wrp < 0.3 else ("CLOSE" if wrp < 1.0 else "FAIL")
        print(f"    {n:>4} steps ({n*DT:>5.1f}s): raw={raw:.4f}  wrap={wrp:.4f}  {status}")

    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=1000)
    arr, lng, iw, tot = metrics(x_t.cpu().numpy(), x_goal.cpu().numpy())
    print(f"\n  Final hold metrics (1000 steps):")
    print(f"    arrive={arr}  longest={lng}  in_window={iw}  total={tot}")
    print(f"\n  Training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
