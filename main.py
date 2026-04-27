"""
main.py — Train the LinearizationNetwork for MPC cost shaping & dense Qf learning.
           Double pendulum: stabilisation near upright [pi, 0, 0, 0].

State:   [q1, q1_dot, q2, q2_dot]
Control: [tau1, tau2]

After training:
  - First epoch rollout  → <session>_rollout_epoch0.csv
  - Last  epoch rollout  → <session>_rollout_final.csv
  - Side-by-side plot    → <session>_comparison.png
"""

import csv
import os
import time
import math
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import mpc_controller as mpc_module
import lin_net as network_module
import Simulate as train_module

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURE HERE
# ──────────────────────────────────────────────────────────────────────────────

X0        = [0.0, 0.0, 0.0, 0.0]   # [q1, q1_dot, q2, q2_dot] — perfectly down
X_GOAL    = [math.pi, 0.0, 0.0, 0.0] # [q1, q1_dot, q2, q2_dot] — perfectly upright
NUM_STEPS = 170
DT        = 0.05                      # seconds

EPOCHS       = 40
LR           = 5e-4
BPTT_WINDOW  = 20
E_PUMP_BOOST = 1.15
HORIZON      = 10
HIDDEN_DIM   = 128
GATE_RANGE_Q = 0.50
GATE_RANGE_R = 0.30
GATE_RANGE_E = 0.60
N_RES        = 5
PRINT_EVERY  = 1
GRAD_DEBUG   = True
GRAD_DEBUG_EVERY = 1
GRAD_SMOKE_STEPS = 5
SAVE_DIR    = "saved_models"
SAVE_NAME   = None

STATE_DIM   = 4   # [q1, q1_dot, q2, q2_dot]
CONTROL_DIM = 2   # [tau1, tau2]

# ──────────────────────────────────────────────────────────────────────────────
# Rollout CSV export
# ──────────────────────────────────────────────────────────────────────────────

def save_rollout_csv(x_hist, u_hist, dt, x_goal_np, filepath):
    T = u_hist.shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    header = ["time_s", "q1_rad", "q1_dot_rads", "q2_rad", "q2_dot_rads",
              "tau1_Nm", "tau2_Nm", "goal_dist", "q1_err_rad"]
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(T):
            xs     = x_hist[i]
            us     = u_hist[i]
            dist   = float(np.linalg.norm(xs - x_goal_np))
            q1_err = float(abs(xs[0] - x_goal_np[0]))
            w.writerow([
                round(i * dt,       4),
                round(float(xs[0]), 6), round(float(xs[1]), 6),
                round(float(xs[2]), 6), round(float(xs[3]), 6),
                round(float(us[0]), 6), round(float(us[1]), 6),
                round(dist,   6), round(q1_err, 6),
            ])
    print(f"  Saved rollout  → {filepath}  ({T} steps)")

# ──────────────────────────────────────────────────────────────────────────────
# Comparison plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison(x_first, x_final, u_first, u_final, x0_np, x_goal_np, dt, filepath):
    T    = min(len(x_first), len(x_final)) - 1
    t    = np.arange(T + 1) * dt
    xf0  = x_first[:T+1];  xfn  = x_final[:T+1]
    uf0  = u_first[:T];    ufn  = u_final[:T]

    C_FIRST = "#4488ff";  C_FINAL = "#ff8844";  LW = 1.8

    fig = plt.figure(figsize=(17, 10), facecolor="#0f0f1a")
    fig.suptitle("Double Pendulum Stabilisation — Epoch-0 vs Trained",
                 color="white", fontsize=12, y=0.99)

    outer = gridspec.GridSpec(1, 3, figure=fig,
                              left=0.05, right=0.97,
                              top=0.91, bottom=0.10,
                              wspace=0.30)

    def style(ax):
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        ax.grid(True, alpha=0.12, color="#aaaaaa")

    def leg(ax, **kw):
        ax.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444466",
                  labelcolor="white", framealpha=0.85, **kw)

    # Col 1: q1 vs q2 phase portrait
    ax2d = fig.add_subplot(outer[0])
    style(ax2d)
    ax2d.plot(xf0[:, 0], xf0[:, 2], color=C_FIRST, lw=LW,     alpha=0.85, label="Epoch 1")
    ax2d.plot(xfn[:, 0], xfn[:, 2], color=C_FINAL, lw=LW+0.5, alpha=0.95, label="Trained")
    ax2d.plot(x0_np[0],     x0_np[2],     "s", color="#44ff88", ms=8,  zorder=5, label="Start")
    ax2d.plot(x_goal_np[0], x_goal_np[2], "*", color="#ff4466", ms=14, zorder=6, label="Goal")
    ax2d.axvline(math.pi, color="#ff4466", lw=0.6, ls="--", alpha=0.4)
    ax2d.set_xlabel("q1 [rad]", color="#aaaaaa", fontsize=9)
    ax2d.set_ylabel("q2 [rad]", color="#aaaaaa", fontsize=9)
    ax2d.set_title("Phase portrait (q1 vs q2)", color="white", fontsize=10)
    leg(ax2d, loc="best")

    # Col 2: states over time
    inner_s = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], hspace=0.50, wspace=0.40)
    slabels = ["q1 [rad]", "q1_dot [rad/s]", "q2 [rad]", "q2_dot [rad/s]"]
    for i in range(4):
        ax = fig.add_subplot(inner_s[i // 2, i % 2])
        style(ax)
        ax.axhline(x_goal_np[i], color="#ff4466", lw=0.8, ls="--", alpha=0.6)
        ax.plot(t, xf0[:, i], color=C_FIRST, lw=LW,  alpha=0.8)
        ax.plot(t, xfn[:, i], color=C_FINAL, lw=LW)
        ax.set_title(slabels[i], color="#cccccc", fontsize=8, pad=3)
        if i >= 2:
            ax.set_xlabel("t [s]", color="#aaaaaa", fontsize=8)
    fig.text(0.505, 0.935, "States  (dashed = goal)", color="white", fontsize=9, ha="center")

    # Col 3: torques + goal distance
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
    ax_dist.plot(t, dist0, color=C_FIRST, lw=LW,     alpha=0.8,  label="Epoch 1")
    ax_dist.plot(t, distn, color=C_FINAL, lw=LW+0.5, alpha=0.95, label="Trained")
    ax_dist.axhline(0, color="#ff4466", lw=0.8, ls="--", alpha=0.5)
    ax_dist.set_title("Goal distance [state norm]", color="#cccccc", fontsize=8, pad=3)
    ax_dist.set_xlabel("t [s]", color="#aaaaaa", fontsize=8)
    leg(ax_dist, loc="upper right")
    fig.text(0.820, 0.935, "Controls + Progress", color="white", fontsize=9, ha="center")

    handles = [
        Line2D([0], [0], color=C_FIRST, lw=2, label="Epoch 1  (untrained)"),
        Line2D([0], [0], color=C_FINAL, lw=2, label="Final    (trained)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white",
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved comparison → {filepath}")

# ──────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ──────────────────────────────────────────────────────────────────────────────

class PrintMonitor:
    def __init__(self, print_every, num_epochs):
        self.print_every   = print_every
        self.num_epochs    = num_epochs
        self._header_shown = False

    def _header(self):
        print(f"\n{'Epoch':>8}  {'Total':>10}  {'Term':>9}  {'Pump':>9}  {'GoalDist':>9}  "
              f"{'QDev':>7}  {'EDev':>7}  {'u_lin':>7}  {'QfNorm':>7}  {'PumpW':>7}  {'QPFail':>7}  {'LR':>9}  {'Time':>6}")
        print("─" * 136)
        self._header_shown = True

    def log_epoch(self, epoch, num_epochs, loss, info):
        if not self._header_shown:
            self._header()
        if epoch == 0 or (epoch+1) % self.print_every == 0 or epoch == num_epochs-1:
            print(f"{epoch+1:>4}/{num_epochs:<4}"
                  f"  {loss:>10.3f}"
                  f"  {info.get('loss_terminal',   float('nan')):>9.3f}"
                  f"  {info.get('loss_pump',       float('nan')):>9.4f}"
                  f"  {info.get('pure_end_error',  float('nan')):>9.4f}"
                  f"  {info.get('mean_Q_gate_dev', float('nan')):>7.4f}"
                  f"  {info.get('mean_E_gate_dev', float('nan')):>7.4f}"
                  f"  {info.get('mean_u_lin_norm', float('nan')):>7.4f}"
                  f"  {info.get('mean_qf_norm',    float('nan')):>7.1f}"
                  f"  {info.get('pump_weight',     float('nan')):>7.3f}"
                  f"  {info.get('qp_fallbacks',    0):>7d}"
                  f"  {info.get('learning_rate',   float('nan')):>9.2e}"
                  f"  {info.get('epoch_time',      float('nan')):>5.2f}s")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x0     = torch.tensor(X0,     device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    print("=" * 76)
    print("  MPC LinearizationNetwork Training  —  Double Pendulum Stabilisation")
    print("  Architecture: five-branch affine-tanh gate + dense Qf + u_lin + energy")
    print("  State branch: LayerNorm → enc   |   Residual branch: LayerNorm → enc")
    print("  Q_head: (N-1, 4)  |  R_head: (N, 2)  |  E_head: (N,)")
    print("  u_lin_head: (N, 2) offset  |  qf_head: (10,) -> Cholesky L -> Qf (4x4)")
    print("=" * 76)
    print(f"  Device      : {device}")
    print(f"  Epochs      : {EPOCHS}  |  LR : {LR}  |  BPTT window : {BPTT_WINDOW}")
    print(f"  Horizon (N) : {HORIZON}  |  Hidden : {HIDDEN_DIM}  |  Gate ranges (Q/R/E) : +/-{GATE_RANGE_Q} / +/-{GATE_RANGE_R} / +/-{GATE_RANGE_E}")
    print(f"  n_res       : {N_RES}   (residual history steps)")
    print(f"  dt          : {DT*1000:.1f} ms  |  Steps : {NUM_STEPS}  ({NUM_STEPS*DT:.2f} s)")
    print(f"  x0          : q1={X0[0]:.3f}  q1d={X0[1]:.3f}  q2={X0[2]:.3f}  q2d={X0[3]:.3f}")
    print(f"  x_goal      : q1={X_GOAL[0]:.3f}  q1d={X_GOAL[1]:.3f}  q2={X_GOAL[2]:.3f}  q2d={X_GOAL[3]:.3f}")
    print("=" * 76)

    # MPC
    mpc    = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)

    # Network
    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R, gate_range_e=GATE_RANGE_E, n_res=N_RES,
    ).to(device).double()

    n_params = sum(p.numel() for p in lin_net.parameters() if p.requires_grad)

    def count(mod):
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)

    print(f"\n  Network params    : {n_params:,}")
    
    print(f"    state_encoder   : {count(lin_net.state_encoder):,}")
    
    print(f"    res_encoder     : {count(lin_net.res_encoder):,}")
    print(f"    trunk           : {count(lin_net.trunk):,}")
    print(f"    Q_head          : {count(lin_net.q_head):,}")
    print(f"    R_head          : {count(lin_net.r_head):,}")
    print(f"    e_head          : {count(lin_net.e_head):,}")
    print(f"    u_lin_head      : {count(lin_net.u_lin_head):,}")
    print(f"    qf_head         : {count(lin_net.qf_head):,}")
    print(f"  Q gate range      : ({1-GATE_RANGE_Q:.2f}, {1+GATE_RANGE_Q:.2f})")
    print(f"  R gate range      : ({1-GATE_RANGE_R:.2f}, {1+GATE_RANGE_R:.2f})")
    print(f"  E gate range      : ({1-GATE_RANGE_E:.2f}, {1+GATE_RANGE_E:.2f})\n")

    session_name = SAVE_NAME or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir  = os.path.join(SAVE_DIR, session_name)

    # Rollout before training
    print("  Rolling out epoch-0 (untrained)...")
    x_first, u_first = train_module.rollout(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
    )
    x_first_np = x_first.cpu().numpy()
    u_first_np = u_first.cpu().numpy()
    x_goal_np  = x_goal.cpu().numpy()
    x0_np      = x0.cpu().numpy()

    # Train
    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(print_every=PRINT_EVERY, num_epochs=EPOCHS)
    if GRAD_DEBUG:
        print("\n  Running gradient-flow smoke test...")
        grad_report = train_module.gradient_flow_smoke_test(
            lin_net=lin_net,
            mpc=mpc,
            x0=x0,
            x_goal=x_goal,
            num_steps=GRAD_SMOKE_STEPS,
        )
        mods = grad_report["module_norms"]
        print(f"    Smoke loss        : {grad_report['smoke_loss']:.6f}")
        print(f"    Total grad norm   : {grad_report['total_norm']:.3e}")
        print(
            "    Module grad norms : "
            f"trunk={mods['trunk']:.3e}, "
            f"q={mods['q_head']:.3e}, "
            f"r={mods['r_head']:.3e}, "
            f"e={mods['e_head']:.3e}, "
            f"u_lin={mods['u_lin_head']:.3e}, "
            f"qf={mods['qf_head']:.3e}"
        )
        print(f"    Missing grads     : {grad_report['missing_count']}")
        if grad_report["missing_count"] > 0:
            sample = ", ".join(grad_report["missing_names"][:5])
            print(f"    Missing sample    : {sample}")

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        bptt_window=BPTT_WINDOW,
        e_pump_boost=E_PUMP_BOOST,
        debug_monitor=monitor,
        recorder=recorder,
        grad_debug=GRAD_DEBUG,
        grad_debug_every=GRAD_DEBUG_EVERY,
    )
    total_time = time.time() - t0

    print("\n" + "─" * 76)
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Initial loss : {loss_history[0]:.6f}")
    print(f"  Final loss   : {loss_history[-1]:.6f}")
    if len(loss_history) > 1:
        pct = (loss_history[0] - loss_history[-1]) / (abs(loss_history[0]) + 1e-12) * 100
        print(f"  Improvement  : {pct:.1f}%")
    final_sum = recorder.epoch_summary(len(loss_history) - 1)
    print(f"  Final Q gate dev    : {final_sum.get('mean_Q_gate_dev',    float('nan')):.4f}")
    print(f"  Final R gate dev    : {final_sum.get('mean_R_gate_dev',    float('nan')):.4f}")
    print(f"  Final E gate dev    : {final_sum.get('mean_E_gate_dev',    float('nan')):.4f}")
    print(f"  Final u_lin norm    : {final_sum.get('mean_u_lin_norm',    float('nan')):.4f}")
    print(f"  Mean residual norm  : {final_sum.get('mean_residual_norm', float('nan')):.4f}")
    print("─" * 76)

    # Rollout after training
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

    # Save
    print()
    os.makedirs(session_dir, exist_ok=True)
    save_rollout_csv(x_first_np, u_first_np, DT, x_goal_np,
                     os.path.join(session_dir, f"{session_name}_rollout_epoch0.csv"))
    save_rollout_csv(x_final_np, u_final_np, DT, x_goal_np,
                     os.path.join(session_dir, f"{session_name}_rollout_final.csv"))
    plot_comparison(x_first_np, x_final_np, u_first_np, u_final_np,
                    x0_np, x_goal_np, DT,
                    os.path.join(session_dir, f"{session_name}_comparison.png"))

    manager = network_module.ModelManager(base_dir=SAVE_DIR)
    manager.save_training_session(
        model=lin_net, loss_history=loss_history,
        training_params={
            "epochs": EPOCHS, "lr": LR, "horizon": HORIZON,
            "hidden_dim": HIDDEN_DIM,
            "gate_range_q": GATE_RANGE_Q,
            "gate_range_r": GATE_RANGE_R,
            "gate_range_e": GATE_RANGE_E,
            "n_res": N_RES,
            "state_dim": STATE_DIM, "control_dim": CONTROL_DIM,
            "dt": DT, "num_steps": NUM_STEPS,
            "x0": X0, "goal": X_GOAL,
            "task": "double_pendulum_stabilisation",
            "device": str(device), "architecture": "five_branch_dense_qf_u_lin_energy",
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"  All outputs → {session_dir}/")


if __name__ == "__main__":
    main()

