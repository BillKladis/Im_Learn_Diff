"""exp_qf_head.py — Train from scratch with a learned Qf-head.

This is exp_no_demo.py's recipe (the one that produced the 0.0612
baseline) with one change: the network now has a fourth head, qf_head,
that outputs 4 gates scaling the diagonal of the MPC's terminal cost
Qf at every step. The training applies the same q_profile_state_phase
mechanism that successfully shaped the running-Q gates — but to the
new Qf gates. Everything else is identical.

Pump phase target  (q1 ≈ 0):  qf_pump   = [0.01, 0.01, 1, 1]
                              → suppress Qf q1 + q1d so terminal cost
                                doesn't fight the swing-up
Stable phase target (q1 ≈ π): qf_stable = [1, 1, 1, 1]
                              → full Qf cost to pin the upright

Per user direction: "do a similar style of training of our best
pipeline with Qf as well". One change at a time so we can isolate
what helps; runs in parallel with exp_qf50_progressive.
"""

import math, os, sys, time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
EPOCHS    = 100
LR        = 1e-3
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM, CONTROL_DIM = 4, 2
SAVE_DIR  = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

W_Q_PROFILE      = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
W_END_Q_HIGH     = 80.0
END_PHASE_STEPS  = 20
Q_GATE_KICKSTART_BIAS = -3.0

# NEW: Qf head config — same recipe applied to terminal cost gates
GATE_RANGE_QF      = 0.50          # gates_Qf = 1 + 0.50 * tanh(raw) ∈ [0.5, 1.5]
W_QF_PROFILE       = 100.0
QF_PROFILE_PUMP    = [0.01, 0.01, 1.0, 1.0]
QF_PROFILE_STABLE  = [1.0,  1.0,  1.0, 1.0]


def make_synthetic_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


def apply_q1_kickstart(net, state_dim, horizon, bias):
    """Initialise q_head's final bias so q1 gate starts low (suppressed
    pumping bias). Same as exp_no_demo.py."""
    with torch.no_grad():
        final = [m for m in net.q_head.modules() if isinstance(m, torch.nn.Linear)][-1]
        final.bias.zero_()
        # First entry of each horizon block is q1 gate
        for k in range(horizon - 1):
            final.bias[k * state_dim + 0] = bias  # q1
            final.bias[k * state_dim + 1] = bias  # q1d


class PrintMonitor:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self._best = float('inf')
        self._header_shown = False

    def _header(self):
        print(f"\n{'Epoch':>8}  {'Loss':>9}  {'Track':>8}  {'GoalD':>7}  "
              f"{'fNorm':>6}  {'qfP':>6}  {'Time':>5}  {'Best':>7}")
        print("─" * 70)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        d = info.get('pure_end_error', float('nan'))
        if d < self._best:
            self._best = d
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"  {epoch+1:>4}/{num_epochs:<3}"
                  f"  {loss:>9.3f}"
                  f"  {info.get('loss_track', float('nan')):>8.4f}"
                  f"  {d:>7.4f}"
                  f"  {info.get('mean_f_extra_norm', float('nan')):>6.3f}"
                  f"  {'—':>6}"
                  f"  {info.get('epoch_time', float('nan')):>4.1f}s"
                  f"  {self._best:>7.4f}",
                  flush=True)


def main():
    device = torch.device("cpu")
    x0     = torch.zeros(4, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_synthetic_demo(NUM_STEPS, device)

    print("=" * 80)
    print("  EXP QF-HEAD: train from scratch with learned Qf gates")
    print(f"  Architecture: state_dim={STATE_DIM}, hidden={HIDDEN_DIM}, horizon={HORIZON}")
    print(f"  gate_range_qf = {GATE_RANGE_QF}  (gates_Qf ∈ [1-{GATE_RANGE_QF}, 1+{GATE_RANGE_QF}])")
    print(f"  Q_BASE_DIAG = {Q_BASE_DIAG}")
    print(f"  EPOCHS={EPOCHS}  LR={LR}  NUM_STEPS={NUM_STEPS}")
    print(f"  q_profile (pump→stable):  {Q_PROFILE_PUMP} → {Q_PROFILE_STABLE}  w={W_Q_PROFILE}")
    print(f"  qf_profile (pump→stable): {QF_PROFILE_PUMP} → {QF_PROFILE_STABLE}  w={W_QF_PROFILE}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    # Note: Qf left at default (mpc_controller's diag(20,20,40,30)). The
    # network's gates_Qf will scale this diagonal at every step.

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_qf=GATE_RANGE_QF,    # << enables the qf_head output
    ).to(device).double()
    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, Q_GATE_KICKSTART_BIAS)

    # Sanity: initial qf gates at zeros input
    with torch.no_grad():
        sh = torch.zeros((5, 4), dtype=torch.float64, device=device)
        out = lin_net(sh)
        gates_qf = out[5]
        print(f"\n  Initial gates_Qf at x0=0: {gates_qf.tolist()}")

    monitor  = PrintMonitor(num_epochs=EPOCHS)
    recorder = network_module.NetworkOutputRecorder()

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
        w_qf_profile=W_QF_PROFILE,
        qf_profile_pump=QF_PROFILE_PUMP,
        qf_profile_stable=QF_PROFILE_STABLE,
        qf_profile_state_phase=True,
        w_end_q_high=W_END_Q_HIGH,
        end_phase_steps=END_PHASE_STEPS,
    )
    elapsed = time.time() - t0

    session_name = f"stageD_qfhead_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "experiment": "qf_head_from_scratch",
            "q_base_diag": Q_BASE_DIAG,
            "gate_range_qf": GATE_RANGE_QF,
            "w_qf_profile": W_QF_PROFILE,
            "qf_profile_pump": QF_PROFILE_PUMP,
            "qf_profile_stable": QF_PROFILE_STABLE,
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"\n  Saved → saved_models/{session_name}/")

    # Loss monotonicity diagnostic
    if len(loss_history) > 2:
        decreased = sum(1 for i in range(1, len(loss_history))
                        if loss_history[i] < loss_history[i-1])
        print(f"\n  Loss monotonicity:")
        print(f"    epoch 1: {loss_history[0]:.3f}")
        print(f"    epoch {len(loss_history)//2}: {loss_history[len(loss_history)//2]:.3f}")
        print(f"    epoch {len(loss_history)}: {loss_history[-1]:.3f}")
        print(f"    decreasing transitions: {decreased}/{len(loss_history)-1}")

    # Post-eval: compare to baseline 0.0612
    print(f"\n  Post-eval (canonical x0=zero):")

    def wrap_pi(x): return math.atan2(math.sin(x), math.cos(x))
    for n in [170, 220, 300, 400, 600, 1000, 1500]:
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n,
        )
        last = x_t.cpu().numpy()[-1]
        raw = float(np.linalg.norm(last - np.array(X_GOAL)))
        wrp = math.sqrt(wrap_pi(last[0]-X_GOAL[0])**2 + last[1]**2 + last[2]**2 + last[3]**2)
        status = "STABLE" if wrp < 0.3 else ("CLOSE" if wrp < 1.0 else "FAIL")
        print(f"    {n:>4} steps ({n*DT:>5.1f}s): raw={raw:.4f}  wrap={wrp:.4f}  {status}")

    # Sustained hold metric (longest contiguous wrap < 0.3 in 1000-step rollout)
    x_t, _ = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=1000,
    )
    arr = x_t.cpu().numpy()
    wraps = np.array([
        math.sqrt(wrap_pi(s[0]-X_GOAL[0])**2 + s[1]**2 + s[2]**2 + s[3]**2)
        for s in arr
    ])
    in_zone = wraps < 0.3
    arrival = next((i for i, v in enumerate(in_zone) if v), None)
    longest = 0; cur = 0
    for v in in_zone:
        cur = cur + 1 if v else 0
        if cur > longest: longest = cur
    print(f"\n  Sustained hold (1000-step rollout, wrap < 0.3):")
    print(f"    first arrival: {'step ' + str(arrival) + ' (' + f'{arrival*DT:.2f}s' + ')' if arrival is not None else 'NEVER'}")
    print(f"    longest contiguous: {longest} steps ({longest*DT:.2f}s)")
    print(f"    total in zone: {int(np.sum(in_zone))} steps ({int(np.sum(in_zone))*DT:.2f}s)")

    # Final qf gate behaviour
    with torch.no_grad():
        sh_zero = torch.zeros((5, 4), dtype=torch.float64, device=device)
        sh_up = torch.zeros((5, 4), dtype=torch.float64, device=device); sh_up[:, 0] = math.pi
        gqf_zero = lin_net(sh_zero)[5]
        gqf_up   = lin_net(sh_up)[5]
        print(f"\n  Final gates_Qf:")
        print(f"    at x=zero (pump):    {[f'{v:.3f}' for v in gqf_zero.tolist()]}")
        print(f"    at x=upright:        {[f'{v:.3f}' for v in gqf_up.tolist()]}")

    print(f"\n  Training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
