"""exp_inline_phase_transition.py — single training run with mid-run
loss-config switch. NO checkpoint reload (avoids fine-tune fragility).

User direction: 'take what worked. Right? And force a hold reward and
a higher q reward in the end but after a few epochs showing steady
swing up'.

PHASE 1 (epochs 0-79): EXACT 0.0612 recipe.
  - NUM_STEPS=170 (just swing-up)
  - track + q_profile (state-phase) + end_q_high
  - Qf default [20, 20, 40, 30]
  - NO hold_reward, NO higher-Q forcing
  This is the recipe that produced the 0.0612 baseline.

PHASE 2 (epochs 80+): SAME loss continues + hold_reward + higher Q.
  - NUM_STEPS=350 (170 swing-up + 180 hold)
  - SAME track + q_profile + end_q_high
  - Qf bumped to [20, 50, 40, 30] (q1d brake)
  - NEW: w_hold_reward = 3 (hold_start=170, σ=0.6)
  - NEW: q_profile_stable bumped to [1.5, 1.5, 1.5, 1.5] (max-out gates
        at goal — within the [0.01, 1.99] gate range)

Single optimizer, single training loop, gradient flow continuous.
The Phase 1 model in this same training run produces the swing-up,
THEN the Phase 2 loss adjusts that policy WITHIN the same gradient
descent (not via checkpoint reload).
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

X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
DT        = 0.05
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM, CONTROL_DIM = 4, 2
SAVE_DIR  = "saved_models"
LR        = 1e-3

# Phase boundaries
PHASE1_EPOCHS = 80      # full 0.0612 swing-up training
PHASE2_EPOCHS = 70      # then add hold+higher-Q

# Phase 1: EXACT 0.0612 config
P1_NUM_STEPS = 170
P1_QF_DIAG   = [20.0, 20.0, 40.0, 30.0]
Q_BASE_DIAG  = [12.0, 5.0, 50.0, 40.0]
P1_W_Q_PROFILE      = 100.0
P1_Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
P1_Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
P1_W_END_Q_HIGH     = 80.0
P1_END_PHASE_STEPS  = 20
Q_GATE_KICKSTART_BIAS = -3.0

# Phase 2: ADDITIONS on top of Phase 1
P2_NUM_STEPS = 350
P2_QF_DIAG   = [20.0, 50.0, 40.0, 30.0]    # q1d brake
P2_W_Q_PROFILE      = 100.0                 # same weight
P2_Q_PROFILE_STABLE = [1.5, 1.5, 1.5, 1.5]  # bumped — force higher Q at goal
P2_W_END_Q_HIGH     = 150.0                 # higher than Phase 1's 80
P2_END_PHASE_STEPS  = 100                   # last 100 (was 20)
P2_W_HOLD_REWARD    = 3.0
P2_HOLD_SIGMA       = 0.6
P2_HOLD_START       = 170                   # only after swing-up window


def apply_q1_kickstart(net, state_dim, horizon, bias):
    final = [m for m in net.q_head.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            final.bias[k * state_dim + 0] = bias
            final.bias[k * state_dim + 1] = bias


def make_swingup_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


def make_swingup_hold_demo(num_steps, swingup_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        if i < swingup_steps:
            alpha = i / max(swingup_steps - 1, 1)
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
    arr = next((i for i, v in enumerate(in_zone) if v), None)
    longest = 0; cur = 0
    for v in in_zone:
        cur = cur + 1 if v else 0
        if cur > longest: longest = cur
    return arr, longest, int(np.sum(in_zone))


class Monitor:
    """Tracks both phases. In Phase 1, best metric is GoalDist@170 (single-step
    is OK for arrival-only). In Phase 2, best metric switches to (longest,
    total) over a 1000-step rollout."""
    def __init__(self, mpc, lin_net, x0, x_goal, phase1_epochs):
        self.mpc, self.lin_net = mpc, lin_net
        self.x0, self.x_goal = x0, x_goal
        self.phase1_epochs = phase1_epochs
        self._call_count = 0
        # Phase 1 best (single-step)
        self._p1_best = float('inf')
        # Phase 2 best (hold metric)
        self._p2_best_long, self._p2_best_total = 0, 0
        self._p2_best_state = copy.deepcopy(lin_net.state_dict())
        self._header_shown = False

    def _header(self):
        print(f"\n{'It':>3}  {'Ph':>2}  {'Loss':>9}  {'Track':>7}  {'fNorm':>7}  "
              f"{'P1Best':>7}  {'P2Arr':>5}  {'P2Long':>6}  {'P2Tot':>6}")
        print("─" * 80)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown: self._header()
        self._call_count += 1
        in_phase2 = self._call_count > self.phase1_epochs

        if not in_phase2:
            d = info.get('pure_end_error', float('nan'))
            if d < self._p1_best: self._p1_best = d
            if (self._call_count) % 5 == 0 or self._call_count == 1:
                print(f"  {self._call_count:>2}  P1  {loss:>9.3f}  "
                      f"{info.get('loss_track', 0):>7.3f}  "
                      f"{info.get('mean_f_extra_norm', 0):>7.3f}  "
                      f"{self._p1_best:>7.4f}  {'—':>5}  {'—':>6}  {'—':>6}",
                      flush=True)
        else:
            # Phase 2: do hold metric eval every 3 calls
            if (self._call_count - self.phase1_epochs) % 3 == 0 or self._call_count == self.phase1_epochs + 1:
                with torch.no_grad():
                    x_t, _ = train_module.rollout(
                        lin_net=self.lin_net, mpc=self.mpc,
                        x0=self.x0, x_goal=self.x_goal, num_steps=1000,
                    )
                arr, lng, tot = hold_metric(x_t.cpu().numpy(), self.x_goal.cpu().numpy())
                improved = (lng > self._p2_best_long) or (tot > self._p2_best_total)
                if improved:
                    self._p2_best_long  = max(self._p2_best_long,  lng)
                    self._p2_best_total = max(self._p2_best_total, tot)
                    self._p2_best_state = copy.deepcopy(self.lin_net.state_dict())
                arr_str = "—" if arr is None else f"{arr}"
                print(f"  {self._call_count:>2}  P2  {loss:>9.3f}  "
                      f"{info.get('loss_track', 0):>7.3f}  "
                      f"{info.get('mean_f_extra_norm', 0):>7.3f}  "
                      f"{self._p1_best:>7.4f}  {arr_str:>5}  {lng:>6}  {tot:>6}",
                      flush=True)


def main():
    print("=" * 80)
    print("  EXP INLINE PHASE TRANSITION")
    print(f"  Phase 1: epochs 1-{PHASE1_EPOCHS}  (0.0612 recipe, NUM_STEPS={P1_NUM_STEPS})")
    print(f"  Phase 2: epochs {PHASE1_EPOCHS+1}-{PHASE1_EPOCHS+PHASE2_EPOCHS}  (+ hold + higher Q, NUM_STEPS={P2_NUM_STEPS})")
    print(f"  Qf:  P1={P1_QF_DIAG}  P2={P2_QF_DIAG}")
    print(f"  Hold: w={P2_W_HOLD_REWARD}  σ={P2_HOLD_SIGMA}  start={P2_HOLD_START}")
    print(f"  Q-stable target: P1={P1_Q_PROFILE_STABLE}  P2={P2_Q_PROFILE_STABLE}")
    print("=" * 80)

    device = torch.device("cpu")
    x0     = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    demo_p1 = make_swingup_demo(P1_NUM_STEPS, device)
    demo_p2 = make_swingup_hold_demo(P2_NUM_STEPS, P1_NUM_STEPS, device)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    mpc.Qf          = torch.diag(torch.tensor(P1_QF_DIAG, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=0.99, gate_range_r=0.20,
        f_extra_bound=3.0, f_kickstart_amp=0.0,
    ).to(device).double()
    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, Q_GATE_KICKSTART_BIAS)

    monitor = Monitor(mpc, lin_net, x0, x_goal, PHASE1_EPOCHS)
    recorder = network_module.NetworkOutputRecorder()
    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=LR, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(PHASE1_EPOCHS + PHASE2_EPOCHS):
        if epoch < PHASE1_EPOCHS:
            # Phase 1: exactly the 0.0612 recipe
            train_module.train_linearization_network(
                lin_net=lin_net, mpc=mpc,
                x0=x0, x_goal=x_goal, demo=demo_p1, num_steps=P1_NUM_STEPS,
                num_epochs=1, lr=LR,
                debug_monitor=monitor, recorder=recorder, grad_debug=False,
                track_mode="energy", w_terminal_anchor=0.0,
                w_q_profile=P1_W_Q_PROFILE,
                q_profile_pump=P1_Q_PROFILE_PUMP,
                q_profile_stable=P1_Q_PROFILE_STABLE,
                q_profile_state_phase=True,
                w_end_q_high=P1_W_END_Q_HIGH,
                end_phase_steps=P1_END_PHASE_STEPS,
                external_optimizer=optimizer,
                restore_best=False,
            )
        else:
            # Phase 2: same loss components + hold + higher Q stable
            if epoch == PHASE1_EPOCHS:
                # Switch to Phase 2 config: bump Qf
                mpc.Qf = torch.diag(torch.tensor(P2_QF_DIAG, device=device, dtype=torch.float64))
                print(f"\n  --- Phase 2 starts at epoch {epoch+1}: Qf bumped to {P2_QF_DIAG}, hold reward enabled ---")
            train_module.train_linearization_network(
                lin_net=lin_net, mpc=mpc,
                x0=x0, x_goal=x_goal, demo=demo_p2, num_steps=P2_NUM_STEPS,
                num_epochs=1, lr=LR,
                debug_monitor=monitor, recorder=recorder, grad_debug=False,
                track_mode="energy", w_terminal_anchor=0.0,
                w_q_profile=P2_W_Q_PROFILE,
                q_profile_pump=P1_Q_PROFILE_PUMP,
                q_profile_stable=P2_Q_PROFILE_STABLE,
                q_profile_state_phase=True,
                w_end_q_high=P2_W_END_Q_HIGH,
                end_phase_steps=P2_END_PHASE_STEPS,
                w_hold_reward=P2_W_HOLD_REWARD,
                hold_sigma=P2_HOLD_SIGMA,
                hold_start_step=P2_HOLD_START,
                external_optimizer=optimizer,
                restore_best=False,
            )
    elapsed = time.time() - t0

    # Load best Phase 2 state if any improvement; else use last
    if monitor._p2_best_long > 0 or monitor._p2_best_total > 0:
        lin_net.load_state_dict(monitor._p2_best_state)

    name = f"stageD_inlinephase_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "inline_phase_transition",
                         "p1_qf": P1_QF_DIAG, "p2_qf": P2_QF_DIAG,
                         "p1_epochs": PHASE1_EPOCHS, "p2_epochs": PHASE2_EPOCHS,
                         "p1_best": monitor._p1_best,
                         "p2_best_long": monitor._p2_best_long,
                         "p2_best_total": monitor._p2_best_total},
        session_name=name,
    )
    print(f"\n  Saved → saved_models/{name}/")
    print(f"  P1 best (GoalDist@170): {monitor._p1_best:.4f}")
    print(f"  P2 best (long, total): {monitor._p2_best_long}, {monitor._p2_best_total}")

    # Post-eval
    print(f"\n  Post-eval:")
    for n in [170, 220, 300, 400, 600, 1000, 1500]:
        x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n)
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = math.sqrt(wrap_pi(last[0]-X_GOAL[0])**2 + last[1]**2 + last[2]**2 + last[3]**2)
        status = "STABLE" if wrp < 0.3 else ("CLOSE" if wrp < 1.0 else "FAIL")
        print(f"    {n:>4} steps ({n*DT:>5.1f}s): raw={raw:.4f}  wrap={wrp:.4f}  {status}")

    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=1000)
    arr, lng, tot = hold_metric(x_t.cpu().numpy(), x_goal.cpu().numpy())
    print(f"\n  Sustained hold (1000 steps): arr={arr} long={lng} total={tot}")
    print(f"\n  Reference qf50 v2 (1000 steps): arr=219 longest=14 total=96")
    print(f"  Training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
