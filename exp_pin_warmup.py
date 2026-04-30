"""exp_combined_hold.py — train fresh with EVERY ingredient that helped.

History of partial successes:
  - 0.0612 baseline: clean swing-up, arrives at goal (~step 167)
  - stab_state (w_f_stable=50): 3.3% time wrap<0.1 (best 'real' hold)
  - qf50 v2 (Qf q1d=50): wider perturbation basin
  - real_hold (w_stable_phase=30): force settle in last 130 steps

Combine ALL of them at once:
  Qf q1d=50              ← brake at horizon
  w_f_stable=50          ← state-conditional f_extra suppression near goal
  w_stable_phase=30      ← state-mode position pin in last 130 steps
  + 0.0612 base recipe   ← proven swing-up (track + q_profile + end_q_high)

Single-shot, from scratch, no fine-tuning. The ingredients address
different failure modes simultaneously:
  - Qf q1d=50: terminal cost penalises residual velocity
  - w_f_stable: network outputs ~0 f_extra near upright (no kicking out)
  - w_stable_phase: pin state in last 130 steps (force settling)

If this doesn't produce real holding, simple ingredients can't combine.
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
NUM_STEPS = 300
EPOCHS    = 120

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
QF_DIAG     = [20.0, 20.0, 40.0, 30.0]   # qf50 brake
W_Q_PROFILE = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
W_END_Q_HIGH     = 80.0
END_PHASE_STEPS  = 20
Q_GATE_KICKSTART_BIAS = -3.0

# Combined ingredients
W_F_STABLE         = 0.0   # stab_state's recipe: state-conditional f_extra penalty
W_STABLE_PHASE     = 30.0
W_F_POS_ONLY       = 0.0   # position-only penalty (no velocity gating)   # state-pin in last 130 steps
STABLE_PHASE_STEPS = 130


def apply_q1_kickstart(net, state_dim, horizon, bias):
    final = [m for m in net.q_head.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            final.bias[k * state_dim + 0] = bias
            final.bias[k * state_dim + 1] = bias


def make_demo(num_steps, swingup_steps, device):
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


def hold_metric(traj, x_goal):
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
    tight = float((wraps < 0.1).mean())
    return arr, longest, int(np.sum(in_zone)), tight


class Mon:
    def __init__(self, mpc, net, x0, xg):
        self.mpc, self.net = mpc, net
        self.x0, self.xg = x0, xg
        self._best_long = 0; self._best_tight = 0.0
        self._best_state = copy.deepcopy(net.state_dict())
        self._call = 0
    def log_epoch(self, epoch, num_epochs, loss, info):
        self._call += 1
        if self._call % 5 == 0 or self._call == 1:
            with torch.no_grad():
                x_t, _ = train_module.rollout(
                    lin_net=self.net, mpc=self.mpc, x0=self.x0, x_goal=self.xg, num_steps=600,
                )
            arr, lng, tot, tight = hold_metric(x_t.cpu().numpy(), self.xg.cpu().numpy())
            improved = lng > self._best_long or tight > self._best_tight
            if improved:
                self._best_long = max(self._best_long, lng)
                self._best_tight = max(self._best_tight, tight)
                self._best_state = copy.deepcopy(self.net.state_dict())
            arr_str = "—" if arr is None else f"{arr}"
            print(f"  ep {self._call:>3}  loss={loss:>9.3f}  track={info.get('loss_track',0):.3f}  "
                  f"fnorm={info.get('mean_f_extra_norm',0):.3f}  "
                  f"arr={arr_str:>4}  long={lng:>4}  tight={tight:.1%}  "
                  f"best_long={self._best_long}  best_tight={self._best_tight:.1%}",
                  flush=True)


def main():
    print("=" * 90)
    print(f"  EXP PIN WARMUP (state-pin only, no Qf bump no w_f_stable) (+ w_f_pos_only=20) — Qf q1d=50 + w_f_stable=50 + w_stable_phase=30")
    print(f"  EPOCHS={EPOCHS}  NUM_STEPS={NUM_STEPS}  LR={LR}")
    print("=" * 90)

    device = torch.device("cpu")
    x0     = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_demo(NUM_STEPS, 170, device)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    mpc.Qf = torch.diag(torch.tensor(QF_DIAG, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=0.99, gate_range_r=0.20,
        f_extra_bound=3.0, f_kickstart_amp=0.0,
    ).to(device).double()
    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, Q_GATE_KICKSTART_BIAS)

    monitor = Mon(mpc, lin_net, x0, x_goal)
    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=LR, weight_decay=1e-4)
    recorder = network_module.NetworkOutputRecorder()

    t0 = time.time()
    PIN_WARMUP = 30   # off for first 30 epochs (let swing-up develop)
    for epoch in range(EPOCHS):
        active_pin = W_STABLE_PHASE if epoch >= PIN_WARMUP else 0.0
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
            w_stable_phase=active_pin,
            w_f_pos_only=W_F_POS_ONLY,
            stable_phase_steps=STABLE_PHASE_STEPS,
            external_optimizer=optimizer, restore_best=False,
        )
    print(f"\n  Trained in {time.time()-t0:.0f}s. Best long={monitor._best_long}  best_tight(<0.1)={monitor._best_tight:.1%}", flush=True)
    lin_net.load_state_dict(monitor._best_state)

    name = f"stageD_pinwarmup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "combined_hold",
                         "qf_diag": QF_DIAG,
                         "w_f_stable": W_F_STABLE,
                         "w_stable_phase": W_STABLE_PHASE,
                         "best_long": monitor._best_long,
                         "best_tight": monitor._best_tight},
        session_name=name,
    )
    print(f"  Saved → saved_models/{name}/")

    # Final eval
    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=2000)
    traj = x_t.cpu().numpy()
    wraps = np.array([math.sqrt(wrap_pi(s[0]-X_GOAL[0])**2 + s[1]**2 + s[2]**2 + s[3]**2) for s in traj])
    print(f"\n  Final 2000-step eval:")
    for thr in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        print(f"    fraction wrap<{thr:.2f}: {(wraps<thr).mean():.1%}", flush=True)
    arr, lng, tot, tight = hold_metric(traj, np.array(X_GOAL))
    print(f"  arrival: {arr}  longest contiguous (<0.3): {lng}  tight time (<0.1): {tight:.1%}")


if __name__ == "__main__":
    main()
