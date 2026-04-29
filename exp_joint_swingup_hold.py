"""exp_joint_swingup_hold.py — train from scratch with BOTH signals.

Phased fine-tune approach FAILED: every loaded swing-up policy
(0.0612 baseline AND qf50 v2) is in such a sharp local minimum that
ANY gradient step (even from tiny track loss alone) destroys the
arrival behavior. We can't add hold_reward to a trained model.

NEW APPROACH: train a fresh network from scratch with the joint
loss (track + q_profile + end_q_high + hold_reward) from epoch 1.
The network learns both signals SIMULTANEOUSLY in a single
optimization — no fine-tune fragility because the network never
develops a sharp local-min policy that subsequent gradients have
to fight.

NUM_STEPS = 350 (170 swing-up + 180 hold). The hold_reward only
fires for steps >= 170, so swing-up is unaffected during epochs
where the network can't reach upright. As the network learns the
swing-up, the hold portion of the rollout starts entering the
stable zone and providing additional gradient signal.

w_hold_reward starts SMALL (1.0) so swing-up training dominates
early. If swing-up converges, hold reward starts to matter.
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
NUM_STEPS = 350
DT        = 0.05
EPOCHS    = 150
LR        = 1e-3
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM, CONTROL_DIM = 4, 2
SAVE_DIR  = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
QF_DIAG     = [20.0, 50.0, 40.0, 30.0]   # qf50 (the working config)

W_Q_PROFILE      = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
W_END_Q_HIGH     = 80.0
END_PHASE_STEPS  = 20
Q_GATE_KICKSTART_BIAS = -3.0

# Hold reward
W_HOLD_REWARD = 5.0       # similar magnitude to other phase_pen contributions
HOLD_SIGMA    = 0.5
HOLD_START    = 170       # only fires after the swing-up phase

# Hold-eval cadence
HOLD_EVAL_EVERY = 5
HOLD_EVAL_STEPS = 600
HOLD_PATIENCE   = 20      # 20 evals * 5 epochs = 100 epochs


def apply_q1_kickstart(net, state_dim, horizon, bias):
    final = [m for m in net.q_head.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            final.bias[k * state_dim + 0] = bias
            final.bias[k * state_dim + 1] = bias


def make_demo(num_steps, swingup_steps, device):
    """Cosine-eased q1 ramp 0 → π over swingup_steps, then hold at π."""
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


class HoldMonitor:
    """Best metric: longest contiguous + total in zone (composite)."""
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
        self._header_shown = False
        # Use OUR own counter for OUTER iters (the train fn's internal
        # epoch is always 0 because num_epochs=1 per call).
        self._call_count = 0

    def _header(self):
        print(f"\n{'Epoch':>6}  {'Loss':>9}  {'Track':>7}  {'fNorm':>7}  "
              f"{'Arr':>5}  {'Long':>5}  {'Total':>5}  {'BestL':>6}  {'BestT':>6}  {'Pat':>4}")
        print("─" * 90)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown: self._header()
        self._call_count += 1
        do_eval = (self._call_count % self.eval_every == 0) or self._call_count == 1
        if do_eval:
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


def main():
    print("=" * 80)
    print("  EXP JOINT SWING-UP + HOLD (from scratch)")
    print(f"  NUM_STEPS={NUM_STEPS}  EPOCHS={EPOCHS}  LR={LR}")
    print(f"  Qf={QF_DIAG}  w_hold_reward={W_HOLD_REWARD}  σ={HOLD_SIGMA}  start={HOLD_START}")
    print("=" * 80)

    device = torch.device("cpu")
    x0     = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_demo(NUM_STEPS, 170, device)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    mpc.Qf          = torch.diag(torch.tensor(QF_DIAG, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=0.99, gate_range_r=0.20,
        f_extra_bound=3.0, f_kickstart_amp=0.0,
    ).to(device).double()
    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, Q_GATE_KICKSTART_BIAS)

    monitor = HoldMonitor(mpc, lin_net, x0, x_goal,
                          eval_every=HOLD_EVAL_EVERY,
                          eval_steps=HOLD_EVAL_STEPS,
                          patience=HOLD_PATIENCE)
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
            w_end_q_high=W_END_Q_HIGH,
            end_phase_steps=END_PHASE_STEPS,
            w_hold_reward=W_HOLD_REWARD,
            hold_sigma=HOLD_SIGMA,
            hold_start_step=HOLD_START,
            external_optimizer=optimizer,
            restore_best=False,
        )
        if monitor.should_stop:
            print(f"\n  EarlyStop at epoch {epoch+1}: hold metric (long={monitor._best_long}, total={monitor._best_total}) hasn't improved for {monitor.patience} evals.")
            break

    elapsed = time.time() - t0
    lin_net.load_state_dict(monitor._best_state)

    name = f"stageD_joint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "joint_swingup_hold",
                         "qf_diag": QF_DIAG, "w_hold_reward": W_HOLD_REWARD,
                         "hold_sigma": HOLD_SIGMA, "hold_start": HOLD_START,
                         "best_long": monitor._best_long,
                         "best_total": monitor._best_total},
        session_name=name,
    )
    print(f"\n  Saved → saved_models/{name}/  best_long={monitor._best_long}  best_total={monitor._best_total}")

    # Post-eval
    print(f"\n  Post-eval (canonical x0=zero):")
    for n in [170, 220, 300, 400, 600, 1000, 1500, 2000]:
        x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n)
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = math.sqrt(wrap_pi(last[0]-X_GOAL[0])**2 + last[1]**2 + last[2]**2 + last[3]**2)
        status = "STABLE" if wrp < 0.3 else ("CLOSE" if wrp < 1.0 else "FAIL")
        print(f"    {n:>4} steps ({n*DT:>5.1f}s): raw={raw:.4f}  wrap={wrp:.4f}  {status}")

    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=2000)
    arr, lng, tot = hold_metric(x_t.cpu().numpy(), x_goal.cpu().numpy())
    print(f"\n  Sustained hold (2000 steps):")
    print(f"    arrival: {'step '+str(arr)+' ('+f'{arr*DT:.2f}s'+')' if arr is not None else 'NEVER'}")
    print(f"    longest contiguous: {lng} steps ({lng*DT:.2f}s)")
    print(f"    total in zone: {tot} steps ({tot*DT:.2f}s)")

    print(f"\n  Reference qf50 v2 (1000 steps): arr=219 longest=14 total=96")
    print(f"  Training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
