"""exp_phased_swingup_hold.py — clean phase-separated training.

User direction: 'we need to find a reliable loss function. try what
you recommended. a reward for when it stabilizes and stays there for
a long time. ... keep the loss function components separated by state
or time or epoch progress or something. just dont mix the signals.
... first swing up training or whatever we were doing in our
successful pipelines'.

Two ENTIRELY SEPARATE training phases:

  PHASE 1 — Swing-up (from scratch, the 0.0612 recipe):
    - Demo: cosine-eased q1 ramp 0 → π over 170 steps
    - Loss components: track (energy) + q_profile + end_q_high
    - NUM_STEPS = 170 (just swing-up, no hold horizon)
    - Qf = default [20, 20, 40, 30]
    - Train until best_goal_dist plateaus (early-stop, original metric
      is fine HERE because the goal is ARRIVAL only — single-step
      proximity at step 170 is the actual signal).
    - Save model A.

  PHASE 2 — Hold (load model A, fine-tune):
    - Demo: q1 ramps to π then HOLDS at π (no kinetic peak).
    - Loss components: ONLY w_hold_reward + tiny w_q_profile to
      preserve gates. NO track loss (energy already correct), NO
      end_q_high (already at max).
    - NUM_STEPS = 500 (170 swing-up + 330 of post-arrival hold).
    - hold_start_step = 170 (reward only fires post-arrival).
    - hold_sigma = 0.5 (zone width: ~0.3 wrap = ~14% of full reward,
      ~0.1 wrap = 96% of full reward → strong gradient toward tight
      hold).
    - Best metric: longest contiguous wrap<0.3 over 1000-step rollout
      (real hold quality, not GoalDist@170).
    - LR = 5e-5 (gentle — fine-tune, don't break swing-up).
    - Qf = [20, 50, 40, 30] (q1d=50 helps brake — already verified).
    - Save model B.

The signals are CLEAN:
  - Phase 1 only tells the network 'arrive at goal'
  - Phase 2 only tells the network 'stay at goal' (no swing-up signal
    interference because the swing-up is already learned and the demo
    just holds at π — track loss is near zero throughout phase 2).

If the swing-up breaks during phase 2, w_hold_reward is too high.
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

# --- Common Q config ---
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
GATE_RANGE_Q = 0.99
GATE_RANGE_R = 0.20
F_EXTRA_BOUND = 3.0

# --- Phase 1 (swing-up) ---
P1_NUM_STEPS = 170
P1_EPOCHS    = 100
P1_LR        = 1e-3
P1_QF_DIAG   = [20.0, 20.0, 40.0, 30.0]   # default — for arrival
P1_W_Q_PROFILE = 100.0
P1_Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
P1_Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
P1_W_END_Q_HIGH     = 80.0
P1_END_PHASE_STEPS  = 20
P1_Q_GATE_KICKSTART_BIAS = -3.0

# --- Phase 2 (hold) ---
P2_NUM_STEPS = 500
P2_EPOCHS    = 80
P2_LR        = 5e-5     # gentle fine-tune
P2_QF_DIAG   = [20.0, 50.0, 40.0, 30.0]   # bumped q1d for braking
P2_W_HOLD_REWARD = 5.0    # was 100 (too aggressive — saturated f_extra in 1 step)
P2_HOLD_SIGMA    = 1.0    # was 0.5 (smoother gradient, less surgical)
P2_HOLD_START    = 170    # reward starts after arrival window
P2_W_Q_PROFILE   = 30.0   # higher — keep gates anchored at the trained values
                          #  (was 5.0; need stronger anchor against hold gradient)
P2_HOLD_EVAL_EVERY = 3
P2_HOLD_EVAL_STEPS = 1000
P2_HOLD_PATIENCE   = 12   # 36 epochs without hold improvement

# Use whatever default RNG state Python has at script run time (which is
# what produced qf50 v2). Don't explicit-seed unless we know which seed
# produces a working swing-up.


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
    """Phase 2 demo: ramp q1 from 0 → π over swingup_steps, then hold at π."""
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


# ──────────────────────────────────────────────────────────────────────
class Phase1Monitor:
    """Standard early-stop on GoalDist@170 (single-step is OK for arrival-only)."""
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self._best = float('inf')
    def log_epoch(self, epoch, num_epochs, loss, info):
        d = info.get('pure_end_error', float('nan'))
        if d < self._best: self._best = d
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"      P1 ep {epoch+1:>3}/{num_epochs}: loss={loss:>7.3f}  goal_d={d:.3f}  best={self._best:.3f}",
                  flush=True)


class Phase2Monitor:
    """Hold-quality early-stop. Best metric = (longest_in_zone, total_in_zone)."""
    def __init__(self, mpc, lin_net, x0, x_goal, eval_every, eval_steps, patience):
        self.mpc, self.lin_net = mpc, lin_net
        self.x0, self.x_goal = x0, x_goal
        self.eval_every = eval_every
        self.eval_steps = eval_steps
        self.patience   = patience
        self._best_long, self._best_total = 0, 0
        self._best_state = copy.deepcopy(lin_net.state_dict())
        self._evals_since_improvement = 0
        self.should_stop = False
        self._call_count = 0
        self._header_shown = False

    def _header(self):
        print(f"\n{'P2 it':>5}  {'Loss':>9}  {'Track':>7}  {'fNorm':>7}  "
              f"{'Arr':>5}  {'Long':>5}  {'Total':>5}  {'BestL':>6}  {'BestT':>6}  {'Pat':>4}")
        print("─" * 90)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown: self._header()
        self._call_count += 1
        if self._call_count % self.eval_every == 0 or self._call_count == 1:
            with torch.no_grad():
                x_t, _ = train_module.rollout(
                    lin_net=self.lin_net, mpc=self.mpc,
                    x0=self.x0, x_goal=self.x_goal, num_steps=self.eval_steps,
                )
            arr, lng, tot = hold_metric(x_t.cpu().numpy(), self.x_goal.cpu().numpy())
            improved = (lng > self._best_long) or (tot > self._best_total)
            if improved:
                self._best_long  = max(self._best_long,  lng)
                self._best_total = max(self._best_total, tot)
                self._best_state = copy.deepcopy(self.lin_net.state_dict())
                self._evals_since_improvement = 0
            else:
                self._evals_since_improvement += 1
            if self._evals_since_improvement >= self.patience:
                self.should_stop = True
            arr_str = "—" if arr is None else f"{arr}"
            print(f"  {self._call_count:>3}  {loss:>9.3f}  "
                  f"{info.get('loss_track', 0):>7.3f}  "
                  f"{info.get('mean_f_extra_norm', 0):>7.3f}  "
                  f"{arr_str:>5}  {lng:>5}  {tot:>5}  "
                  f"{self._best_long:>6}  {self._best_total:>6}  "
                  f"{self._evals_since_improvement:>4}", flush=True)


# ──────────────────────────────────────────────────────────────────────
def phase1(device):
    print("\n" + "=" * 80)
    print("  PHASE 1: SWING-UP TRAINING (clean — no hold signal)")
    print("=" * 80)
    x0     = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_swingup_demo(P1_NUM_STEPS, device)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    mpc.Qf          = torch.diag(torch.tensor(P1_QF_DIAG, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=0.0,
    ).to(device).double()
    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, P1_Q_GATE_KICKSTART_BIAS)

    monitor = Phase1Monitor(P1_EPOCHS)
    recorder = network_module.NetworkOutputRecorder()

    t0 = time.time()
    train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=P1_NUM_STEPS,
        num_epochs=P1_EPOCHS, lr=P1_LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=False, track_mode="energy", w_terminal_anchor=0.0,
        w_q_profile=P1_W_Q_PROFILE,
        q_profile_pump=P1_Q_PROFILE_PUMP,
        q_profile_stable=P1_Q_PROFILE_STABLE,
        q_profile_state_phase=True,
        w_end_q_high=P1_W_END_Q_HIGH,
        end_phase_steps=P1_END_PHASE_STEPS,
    )
    print(f"  Phase 1 trained in {time.time()-t0:.0f}s; best GoalDist@170 = {monitor._best:.4f}")

    # Quick check: does it arrive?
    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=200)
    arr, lng, tot = hold_metric(x_t.cpu().numpy(), x_goal.cpu().numpy())
    print(f"  Phase 1 quick eval (200 steps): arr={arr} long={lng} total={tot}")
    return lin_net, mpc, x0, x_goal


def phase2(lin_net, mpc, x0, x_goal):
    print("\n" + "=" * 80)
    print("  PHASE 2: HOLD TRAINING (load Phase 1 model + hold reward only)")
    print("=" * 80)
    # Switch Qf to qf50 for braking
    mpc.Qf = torch.diag(torch.tensor(P2_QF_DIAG, device=mpc.device, dtype=torch.float64))
    print(f"  Qf bumped to {P2_QF_DIAG}")

    demo = make_swingup_hold_demo(P2_NUM_STEPS, P1_NUM_STEPS, mpc.device)

    monitor = Phase2Monitor(
        mpc, lin_net, x0, x_goal,
        eval_every=P2_HOLD_EVAL_EVERY,
        eval_steps=P2_HOLD_EVAL_STEPS,
        patience=P2_HOLD_PATIENCE,
    )
    recorder = network_module.NetworkOutputRecorder()
    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=P2_LR, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(P2_EPOCHS):
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0, x_goal=x_goal, demo=demo, num_steps=P2_NUM_STEPS,
            num_epochs=1, lr=P2_LR,
            debug_monitor=monitor, recorder=recorder, grad_debug=False,
            track_mode="energy", w_terminal_anchor=0.0,
            w_q_profile=P2_W_Q_PROFILE,                 # tiny
            q_profile_pump=P1_Q_PROFILE_PUMP,
            q_profile_stable=P1_Q_PROFILE_STABLE,
            q_profile_state_phase=True,
            w_end_q_high=0.0,                           # disabled in Phase 2
            end_phase_steps=20,
            w_hold_reward=P2_W_HOLD_REWARD,             # << the new signal
            hold_sigma=P2_HOLD_SIGMA,
            hold_start_step=P2_HOLD_START,
            external_optimizer=optimizer,
            restore_best=False,
        )
        if monitor.should_stop:
            print(f"\n  Phase 2 EarlyStop at outer iter {monitor._call_count}: "
                  f"hold metric (long={monitor._best_long}, total={monitor._best_total}) "
                  f"hasn't improved for {monitor.patience} evals.")
            break
    print(f"  Phase 2 trained in {time.time()-t0:.0f}s")

    # Load best
    lin_net.load_state_dict(monitor._best_state)
    return lin_net, mpc, x0, x_goal


PHASE1_PRETRAINED = "saved_models/stageD_phase1_20260429_165603/stageD_phase1_20260429_165603.pth"


def main():
    device = torch.device("cpu")

    # Phase 1 — skip if checkpoint exists, just load
    if os.path.exists(PHASE1_PRETRAINED):
        print(f"\n  Loading existing Phase 1 model: {PHASE1_PRETRAINED}")
        x0     = torch.tensor(X0, device=device, dtype=torch.float64)
        x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
        mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
        mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
        mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
        mpc.Qf          = torch.diag(torch.tensor(P1_QF_DIAG, device=device, dtype=torch.float64))
        lin_net = network_module.LinearizationNetwork.load(PHASE1_PRETRAINED, device="cpu").double()
        # Quick eval
        x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=200)
        arr, lng, tot = hold_metric(x_t.cpu().numpy(), x_goal.cpu().numpy())
        print(f"  Phase 1 loaded: arr={arr} long={lng} total={tot}")
    else:
        lin_net, mpc, x0, x_goal = phase1(device)

    # Save phase 1 model
    p1_name = f"stageD_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "phased_swingup_hold", "phase": 1, "qf_diag": P1_QF_DIAG},
        session_name=p1_name,
    )
    print(f"\n  Phase 1 saved → saved_models/{p1_name}/")

    # Phase 2
    lin_net, mpc, x0, x_goal = phase2(lin_net, mpc, x0, x_goal)

    p2_name = f"stageD_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "phased_swingup_hold", "phase": 2, "qf_diag": P2_QF_DIAG,
                         "w_hold_reward": P2_W_HOLD_REWARD, "hold_sigma": P2_HOLD_SIGMA},
        session_name=p2_name,
    )
    print(f"\n  Phase 2 saved → saved_models/{p2_name}/")

    # Final eval
    print("\n  FINAL eval (canonical x0=zero):")
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


if __name__ == "__main__":
    main()
