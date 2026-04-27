"""
main_imitation.py — Run trajectory-imitation training on the Stage D
network (no physics in controller, free f_extra term).

USAGE
=====
1. Run the physics-informed version of main.py to convergence on the
   target swing-up.  This produces a session directory at:
       saved_models/<session_name>/<session_name>_rollout_final.csv
   That CSV contains the demo trajectory we will imitate.

2. Set DEMO_CSV below to the path of that file.

3. Run this script.  It builds a fresh Stage D network with the kickstart
   bias DISABLED (since strong tracking gradient should make it
   unnecessary), loads the demo trajectory, and trains via squared-L2
   tracking against demo[t+1] at each step.
"""

import csv
import math
import os
import time
from datetime import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURE HERE
# ──────────────────────────────────────────────────────────────────────────

# Path to the demo trajectory CSV (produced by the physics version of main.py).
# Must contain columns q1_rad, q1_dot_rads, q2_rad, q2_dot_rads.
DEMO_CSV = "run_20260428_001459_rollout_final.csv"

X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05

EPOCHS      = 80
LR          = 1e-3
HORIZON     = 10
HIDDEN_DIM  = 128

# Imitation tracking mode:
#   "state"  — original rigid Euclidean ||x - demo[t+1]||² (saturated, locks)
#   "energy" — scalar (E(x) - E_demo[t+1])²/E_range² (monotone, escapes lock)
TRACK_MODE  = "energy"

# Final-step-only wrap(q1-π)² + 0.1·q1d² anchor.  In testing, anchor=1.0
# actually HURT performance because the wrap(q1-π)² gradient pulls toward
# q1=π immediately at every step, which the QP saturates against (recreating
# the original lock).  Energy tracking alone produces clean swing-up
# (goal_dist=0.14) so we keep this disabled.  Left as a knob in case
# stability becomes a concern; values like 0.05-0.2 may help.
W_TERMINAL_ANCHOR = 0.0

# Zero out the q1/q1_dot RUNNING costs in the QP.  Otherwise the QP's
# state-error pull (12·π² ≈ 118 in f-vector) saturates u at +3 Nm and
# f_extra (max ±3) cannot overcome it — the network gets stuck because
# its gradient vanishes against the active control bound.  With these
# zeroed, f_extra is the actual angular policy; the QP's job becomes
# (a) preventing q2 fold via q2/q2_dot running cost, (b) terminal pull
# via Qf, and (c) being a smoothness regulariser on the network's policy.
ZERO_Q1_COSTS = True

GATE_RANGE_Q     = 0.95
GATE_RANGE_R     = 0.20
F_EXTRA_BOUND    = 3.0
F_KICKSTART_AMP  = 0.0      # DISABLED for imitation runs — tracking signal is strong enough.

PRINT_EVERY      = 1
GRAD_DEBUG       = True
GRAD_DEBUG_EVERY = 1
GRAD_SMOKE_STEPS = 5

SAVE_DIR  = "saved_models"
SAVE_NAME = None

STATE_DIM   = 4
CONTROL_DIM = 2

# ──────────────────────────────────────────────────────────────────────────
# Rollout CSV export (same as main.py)
# ──────────────────────────────────────────────────────────────────────────

def save_rollout_csv(x_hist, u_hist, dt, x_goal_np, filepath):
    T = u_hist.shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    header = ["time_s", "q1_rad", "q1_dot_rads", "q2_rad", "q2_dot_rads",
              "tau1_Nm", "tau2_Nm", "goal_dist", "q1_err_rad"]
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(T):
            xs   = x_hist[i]
            us   = u_hist[i]
            dist = float(np.linalg.norm(xs - x_goal_np))
            q1_err = float(abs(xs[0] - x_goal_np[0]))
            w.writerow([
                round(i * dt,       4),
                round(float(xs[0]), 6), round(float(xs[1]), 6),
                round(float(xs[2]), 6), round(float(xs[3]), 6),
                round(float(us[0]), 6), round(float(us[1]), 6),
                round(dist,         6), round(q1_err,       6),
            ])
    print(f"  Saved rollout  → {filepath}  ({T} steps)")

# ──────────────────────────────────────────────────────────────────────────
# Comparison plot
# ──────────────────────────────────────────────────────────────────────────

def plot_comparison(
    x_first, x_final, u_first, u_final,
    x_demo, x0_np, x_goal_np, dt, filepath,
):
    T   = min(len(x_first), len(x_final)) - 1
    t   = np.arange(T + 1) * dt
    xf0 = x_first[:T+1];  xfn = x_final[:T+1]
    uf0 = u_first[:T];    ufn = u_final[:T]
    # Demo CSV stores T states (x_hist[0..T-1]); pad/trim to match plot length.
    xdemo = np.zeros_like(xfn)
    L = min(x_demo.shape[0], T + 1)
    xdemo[:L] = x_demo[:L]
    if L < T + 1:
        xdemo[L:] = x_demo[-1]

    C_FIRST = "#4488ff"; C_FINAL = "#ff8844"; C_DEMO = "#88ff88"; LW = 1.8

    fig = plt.figure(figsize=(17, 10), facecolor="#0f0f1a")
    fig.suptitle("Double Pendulum Swing-up — Stage D imitation",
                 color="white", fontsize=12, y=0.99)

    outer = gridspec.GridSpec(1, 3, figure=fig,
                              left=0.05, right=0.97,
                              top=0.91, bottom=0.10, wspace=0.30)

    def style(ax):
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        ax.grid(True, alpha=0.12, color="#aaaaaa")

    def leg(ax, **kw):
        ax.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444466",
                  labelcolor="white", framealpha=0.85, **kw)

    ax2d = fig.add_subplot(outer[0])
    style(ax2d)
    ax2d.plot(xdemo[:, 0], xdemo[:, 2], color=C_DEMO,  lw=LW,     alpha=0.7,  label="Demo")
    ax2d.plot(xf0[:, 0],   xf0[:, 2],   color=C_FIRST, lw=LW,     alpha=0.85, label="Epoch 1")
    ax2d.plot(xfn[:, 0],   xfn[:, 2],   color=C_FINAL, lw=LW+0.5, alpha=0.95, label="Trained")
    ax2d.plot(x0_np[0], x0_np[2], "s", color="#44ff88", ms=8, zorder=5, label="Start")
    ax2d.plot(x_goal_np[0], x_goal_np[2], "*", color="#ff4466", ms=14, zorder=6, label="Goal")
    ax2d.axvline(math.pi, color="#ff4466", lw=0.6, ls="--", alpha=0.4)
    ax2d.set_xlabel("q1 [rad]", color="#aaaaaa", fontsize=9)
    ax2d.set_ylabel("q2 [rad]", color="#aaaaaa", fontsize=9)
    ax2d.set_title("Phase portrait (q1 vs q2)", color="white", fontsize=10)
    leg(ax2d, loc="best")

    inner_s = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], hspace=0.50, wspace=0.40)
    slabels = ["q1 [rad]", "q1_dot [rad/s]", "q2 [rad]", "q2_dot [rad/s]"]
    for i in range(4):
        ax = fig.add_subplot(inner_s[i // 2, i % 2])
        style(ax)
        ax.axhline(x_goal_np[i], color="#ff4466", lw=0.8, ls="--", alpha=0.6)
        ax.plot(t, xdemo[:, i], color=C_DEMO,  lw=LW,  alpha=0.7)
        ax.plot(t, xf0[:, i],   color=C_FIRST, lw=LW,  alpha=0.7)
        ax.plot(t, xfn[:, i],   color=C_FINAL, lw=LW)
        ax.set_title(slabels[i], color="#cccccc", fontsize=8, pad=3)
        if i >= 2:
            ax.set_xlabel("t [s]", color="#aaaaaa", fontsize=8)
    fig.text(0.505, 0.935, "States  (green = demo target)", color="white", fontsize=9, ha="center")

    inner_c = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[2], hspace=0.55)
    tau_lim = 10.0
    for i, label in enumerate(["tau1 [Nm]", "tau2 [Nm]"]):
        ax = fig.add_subplot(inner_c[i])
        style(ax)
        ax.axhspan(-tau_lim, tau_lim, color="#333355", alpha=0.20)
        ax.axhline( tau_lim, color="#555577", lw=0.8, ls=":")
        ax.axhline(-tau_lim, color="#555577", lw=0.8, ls=":")
        ax.plot(np.arange(T) * dt, uf0[:, i], color=C_FIRST, lw=LW,     alpha=0.8)
        ax.plot(np.arange(T) * dt, ufn[:, i], color=C_FINAL, lw=LW+0.5, alpha=0.95)
        ax.set_title(label, color="#cccccc", fontsize=8, pad=3)
        ax.set_xlabel("t [s]", color="#aaaaaa", fontsize=8)

    ax_dist = fig.add_subplot(inner_c[2])
    style(ax_dist)
    dist0 = np.linalg.norm(xf0 - x_goal_np, axis=1)
    distn = np.linalg.norm(xfn - x_goal_np, axis=1)
    distd = np.linalg.norm(xdemo - x_goal_np, axis=1)
    ax_dist.plot(t, distd, color=C_DEMO,  lw=LW,     alpha=0.7,  label="Demo")
    ax_dist.plot(t, dist0, color=C_FIRST, lw=LW,     alpha=0.7,  label="Epoch 1")
    ax_dist.plot(t, distn, color=C_FINAL, lw=LW+0.5, alpha=0.95, label="Trained")
    ax_dist.axhline(0, color="#ff4466", lw=0.8, ls="--", alpha=0.5)
    ax_dist.set_title("Goal distance [state norm]", color="#cccccc", fontsize=8, pad=3)
    ax_dist.set_xlabel("t [s]", color="#aaaaaa", fontsize=8)
    leg(ax_dist, loc="upper right")
    fig.text(0.820, 0.935, "Controls + Progress", color="white", fontsize=9, ha="center")

    handles = [
        Line2D([0], [0], color=C_DEMO,  lw=2, label="Demo"),
        Line2D([0], [0], color=C_FIRST, lw=2, label="Epoch 1  (untrained)"),
        Line2D([0], [0], color=C_FINAL, lw=2, label="Final    (trained)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white",
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved comparison → {filepath}")

# ──────────────────────────────────────────────────────────────────────────
# Pretty printer
# ──────────────────────────────────────────────────────────────────────────

class PrintMonitor:
    def __init__(self, print_every, num_epochs):
        self.print_every   = print_every
        self.num_epochs    = num_epochs
        self._header_shown = False

    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Track':>9}  {'Term':>9}  "
              f"{'Q2':>8}  {'GoalDist':>9}  {'QDev':>7}  {'fNorm':>7}  "
              f"{'fτ1[0]':>8}  {'QPFail':>7}  {'LR':>9}  {'Time':>6}")
        print("─" * 144)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        if epoch == 0 or (epoch+1) % self.print_every == 0 or epoch == num_epochs-1:
            print(f"{epoch+1:>4}/{num_epochs:<4}"
                  f"  {loss:>10.3f}"
                  f"  {info.get('loss_track',        float('nan')):>9.3f}"
                  f"  {info.get('loss_terminal',     float('nan')):>9.3f}"
                  f"  {info.get('loss_q2',           float('nan')):>8.4f}"
                  f"  {info.get('pure_end_error',    float('nan')):>9.4f}"
                  f"  {info.get('mean_Q_gate_dev',   float('nan')):>7.4f}"
                  f"  {info.get('mean_f_extra_norm', float('nan')):>7.3f}"
                  f"  {info.get('mean_f_tau1_first', float('nan')):>8.3f}"
                  f"  {info.get('qp_fallbacks',      0):>7d}"
                  f"  {info.get('learning_rate',     float('nan')):>9.2e}"
                  f"  {info.get('epoch_time',        float('nan')):>5.2f}s")

# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(DEMO_CSV):
        raise FileNotFoundError(
            f"Demo CSV not found at {DEMO_CSV}. "
            f"Update DEMO_CSV in main_imitation.py to point at a "
            f"successful rollout CSV from the physics-informed run."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x0     = torch.tensor(X0,     device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    # Load demo trajectory.
    demo = train_module.load_demo_trajectory(
        DEMO_CSV, expected_length=NUM_STEPS, device=device,
    )

    print("=" * 76)
    print("  MPC LinearizationNetwork  —  Stage D + trajectory imitation")
    print("  Architecture: state encoder → trunk → {Q, R, F} heads")
    print("  QP solver:    cvxpylayers (DPP, SCS)")
    print(f"  Track mode  : {TRACK_MODE}  (state | energy)")
    print(f"  Anchor pull : {W_TERMINAL_ANCHOR}  (final-step wrap(q1-π)²)")
    print(f"  Zero q1 cost: {ZERO_Q1_COSTS}")
    print("  Init:         f_head kickstart DISABLED")
    print("=" * 76)
    print(f"  Demo CSV        : {DEMO_CSV}")
    print(f"  Demo length     : {demo.shape[0]} states")
    print(f"  Device          : {device}")
    print(f"  Epochs          : {EPOCHS}  |  LR : {LR}")
    print(f"  Horizon (N)     : {HORIZON}  |  Hidden : {HIDDEN_DIM}")
    print(f"  Gate ranges Q/R : ±{GATE_RANGE_Q} / ±{GATE_RANGE_R}")
    print(f"  f_extra bound   : ±{F_EXTRA_BOUND}")
    print(f"  f kickstart amp : {F_KICKSTART_AMP}  (0 = disabled)")
    print(f"  dt              : {DT*1000:.1f} ms  |  Steps : {NUM_STEPS}  ({NUM_STEPS*DT:.2f} s)")
    print(f"  x0              : q1={X0[0]:.3f}  q1d={X0[1]:.3f}  q2={X0[2]:.3f}  q2d={X0[3]:.3f}")
    print(f"  x_goal          : q1={X_GOAL[0]:.3f}  q1d={X_GOAL[1]:.3f}  q2={X_GOAL[2]:.3f}  q2d={X_GOAL[3]:.3f}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)

    if ZERO_Q1_COSTS:
        # See top of file for rationale.
        mpc.q_base_diag = torch.tensor(
            [0.0, 0.0, 50.0, 40.0], device=device, dtype=torch.float64,
        )

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q,
        gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND,
        f_kickstart_amp=F_KICKSTART_AMP,
    ).to(device).double()

    n_params = sum(p.numel() for p in lin_net.parameters() if p.requires_grad)
    def count(mod):
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)

    print(f"\n  Network params    : {n_params:,}")
    print(f"    state_encoder   : {count(lin_net.state_encoder):,}")
    print(f"    trunk           : {count(lin_net.trunk):,}")
    print(f"    Q_head          : {count(lin_net.q_head):,}")
    print(f"    R_head          : {count(lin_net.r_head):,}")
    print(f"    F_head          : {count(lin_net.f_head):,}")
    print()

    session_name = SAVE_NAME or f"stageD_imit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir  = os.path.join(SAVE_DIR, session_name)

    print("  Rolling out epoch-0 (untrained)...")
    x_first, u_first = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    x_first_np = x_first.cpu().numpy()
    u_first_np = u_first.cpu().numpy()
    x_goal_np  = x_goal.cpu().numpy()
    x0_np      = x0.cpu().numpy()
    demo_np    = demo.cpu().numpy()

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(print_every=PRINT_EVERY, num_epochs=EPOCHS)

    if GRAD_DEBUG:
        print("\n  Running gradient-flow smoke test...")
        grad_report = train_module.gradient_flow_smoke_test(
            lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal,
            demo=demo, num_steps=GRAD_SMOKE_STEPS,
            track_mode=TRACK_MODE,
        )
        mods = grad_report["module_norms"]
        print(f"    Smoke loss        : {grad_report['smoke_loss']:.6f}")
        print(f"    Total grad norm   : {grad_report['total_norm']:.3e}")
        print(
            "    Module grad norms : "
            f"trunk={mods['trunk']:.3e}, "
            f"q={mods['q_head']:.3e}, "
            f"r={mods['r_head']:.3e}, "
            f"f={mods['f_head']:.3e}"
        )
        print(f"    Missing grads     : {grad_report['missing_count']}")
        if grad_report["missing_count"] > 0:
            sample = ", ".join(grad_report["missing_names"][:5])
            print(f"    Missing sample    : {sample}")

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor,
        recorder=recorder,
        grad_debug=GRAD_DEBUG,
        grad_debug_every=GRAD_DEBUG_EVERY,
        track_mode=TRACK_MODE,
        w_terminal_anchor=W_TERMINAL_ANCHOR,
    )
    total_time = time.time() - t0

    print("\n" + "─" * 76)
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Initial loss : {loss_history[0]:.6f}")
    print(f"  Final loss   : {loss_history[-1]:.6f}")
    if len(loss_history) > 1 and abs(loss_history[0]) > 1e-12:
        pct = (loss_history[0] - loss_history[-1]) / abs(loss_history[0]) * 100
        print(f"  Improvement  : {pct:.1f}%")
    final_sum = recorder.epoch_summary(len(loss_history) - 1)
    print(f"  Final Q gate dev      : {final_sum.get('mean_Q_gate_dev',    float('nan')):.4f}")
    print(f"  Final R gate dev      : {final_sum.get('mean_R_gate_dev',    float('nan')):.4f}")
    print(f"  Final f_extra norm    : {final_sum.get('mean_f_extra_norm',  float('nan')):.4f}")
    print(f"  Final f_τ1 step-0 avg : {final_sum.get('mean_f_tau1_first',  float('nan')):.4f}")
    print("─" * 76)

    print("\n  Rolling out final (trained)...")
    x_final, u_final = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    x_final_np = x_final.cpu().numpy()
    u_final_np = u_final.cpu().numpy()
    dist_final = np.linalg.norm(x_final_np[-1] - x_goal_np)
    dist_first = np.linalg.norm(x_first_np[-1] - x_goal_np)
    print(f"  Trained final goal distance : {dist_final:.4f}  (was {dist_first:.4f})")
    print(f"  (Note: model restored to best-seen checkpoint before final rollout)")

    print()
    os.makedirs(session_dir, exist_ok=True)
    save_rollout_csv(x_first_np, u_first_np, DT, x_goal_np,
                     os.path.join(session_dir, f"{session_name}_rollout_epoch0.csv"))
    save_rollout_csv(x_final_np, u_final_np, DT, x_goal_np,
                     os.path.join(session_dir, f"{session_name}_rollout_final.csv"))
    plot_comparison(x_first_np, x_final_np, u_first_np, u_final_np,
                    demo_np, x0_np, x_goal_np, DT,
                    os.path.join(session_dir, f"{session_name}_comparison.png"))

    manager = network_module.ModelManager(base_dir=SAVE_DIR)
    manager.save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "epochs": EPOCHS, "lr": LR, "horizon": HORIZON,
            "hidden_dim": HIDDEN_DIM,
            "gate_range_q": GATE_RANGE_Q,
            "gate_range_r": GATE_RANGE_R,
            "f_extra_bound": F_EXTRA_BOUND,
            "f_kickstart_amp": F_KICKSTART_AMP,
            "state_dim": STATE_DIM, "control_dim": CONTROL_DIM,
            "dt": DT, "num_steps": NUM_STEPS,
            "x0": X0, "goal": X_GOAL,
            "demo_csv": DEMO_CSV,
            "task": "double_pendulum_swingup_imitation",
            "device": str(device),
            "architecture": "stage_d_three_head_qrf_imitation",
            "qp_solver": "cvxpylayers",
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"  All outputs → {session_dir}/")


if __name__ == "__main__":
    main()