"""
exp_q_modulation.py — Core-challenge experiment: can the NN learn to
suppress q1 running cost autonomously via the Q-gate, WITHOUT explicitly
zeroing q_base_diag?

SETUP:
  - default q_base_diag = [12, 5, 50, 40]  (ZERO_Q1_COSTS = False)
  - gate_range_q = 0.99  → q1 gate range [0.01, 1.99]
  - Q-gate kickstart: final linear layer of q_head biased to -4.0 on
    q1 dims → initial q1 gate ≈ 0.011 → effective q1 cost ≈ 0.13
  - f_kickstart_amp = 1.0  (sinusoidal f_extra kickstart re-enabled)
  - energy tracking loss (unchanged)

HYPOTHESIS:
  With q1 gate near 0 from init, the f-vector is no longer dominated by
  the 12*pi state-error pull, so f_extra can drive policy and gradients
  can flow through the unfrozen QP.  The network then learns BOTH:
    (a) keep q1 gate low during swing-up so f_extra can do work
    (b) set f_extra to the correct τ·q̇ pumping pattern

After training, dump q1-gate values per step to verify hypothesis (a).
"""

import csv
import math
import os
import sys
import time
from datetime import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────

DEMO_CSV = "run_20260428_001459_rollout_final.csv"

X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05

EPOCHS     = 80
LR         = 1e-3
HORIZON    = 10
HIDDEN_DIM = 128

TRACK_MODE        = "energy"
W_TERMINAL_ANCHOR = 0.0

# KEY DIFFERENCE: do NOT zero q1 costs; rely on Q-gate to suppress them.
ZERO_Q1_COSTS     = False

GATE_RANGE_Q    = 0.99    # wide enough that q1 gate can reach 0.01
GATE_RANGE_R    = 0.20
F_EXTRA_BOUND   = 3.0
F_KICKSTART_AMP = 1.0     # re-enable sinusoidal f-extra kickstart

# Q-gate kickstart: set q1 components of q_head final bias to this value.
# tanh(-4.0) ≈ -0.9993  →  gate = 1 + 0.99 * (-0.9993) ≈ 0.011
Q_GATE_KICKSTART_BIAS = -4.0

PRINT_EVERY      = 1
GRAD_DEBUG       = True
GRAD_DEBUG_EVERY = 1
GRAD_SMOKE_STEPS = 5

SAVE_DIR  = "saved_models"
STATE_DIM   = 4
CONTROL_DIM = 2

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


def apply_q1_gate_kickstart(lin_net, state_dim, horizon, bias_val):
    """Set q1 dims of q_head's final linear layer bias to bias_val."""
    q_final = [m for m in lin_net.q_head.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):   # q_head output spans horizon-1 steps
            q_final.bias[k * state_dim + 0] = bias_val  # q1 dim
    print(f"  Q-gate kickstart: q1 bias set to {bias_val:.1f} "
          f"→ init gate ≈ {1.0 + GATE_RANGE_Q * math.tanh(bias_val):.4f}")


def dump_q1_gate_stats(lin_net, mpc, x0, x_goal, num_steps, device):
    """Roll out the trained model and report q1 gate per step."""
    lin_net.eval()
    x = x0.clone()
    u_prev = torch.zeros(HORIZON, CONTROL_DIM, device=device, dtype=torch.float64)
    gates_q1 = []

    with torch.no_grad():
        for _ in range(num_steps):
            gQ, gR, fE, _, _ = lin_net(
                x.unsqueeze(0).expand(5, -1),
                mpc.q_base_diag, mpc.r_base_diag,
            )
            gates_q1.append(gQ[:, 0].mean().item())   # mean over horizon of q1 gate
            u_seq, _, _ = mpc.QP_formulation(
                x, u_prev, x_goal,
                diag_corrections_Q=gQ,
                diag_corrections_R=gR,
                extra_linear_control=fE.reshape(-1),
            )
            x = mpc.MPC_RK4_disc(x, u_seq[0], mpc.dt)
            u_prev = torch.cat([u_seq[1:], u_seq[-1:]])

    arr = np.array(gates_q1)
    print(f"\n  Q1 gate stats over {num_steps} steps:")
    print(f"    mean={arr.mean():.4f}  min={arr.min():.4f}  "
          f"max={arr.max():.4f}  std={arr.std():.4f}")
    print(f"    First 10: {' '.join(f'{v:.3f}' for v in arr[:10])}")
    lin_net.train()
    return arr


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
                  f"  {info.get('epoch_time',        float('nan')):>5.2f}s",
                  flush=True)


def main():
    if not os.path.exists(DEMO_CSV):
        raise FileNotFoundError(f"Demo CSV not found: {DEMO_CSV}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    demo = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)

    print("=" * 76)
    print("  EXP: Q-modulation — NN suppresses q1 gate, no hard zeroing")
    print("  q_base_diag = [12, 5, 50, 40]  (DEFAULT, not zeroed)")
    print(f"  gate_range_q = {GATE_RANGE_Q}  →  q1_gate_init ≈ "
          f"{1.0 + GATE_RANGE_Q * math.tanh(Q_GATE_KICKSTART_BIAS):.4f}")
    print(f"  f_kickstart_amp = {F_KICKSTART_AMP}")
    print(f"  track_mode = {TRACK_MODE}")
    print(f"  epochs={EPOCHS}  lr={LR}  horizon={HORIZON}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    # q_base_diag stays at default [12, 5, 50, 40]

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q,
        gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND,
        f_kickstart_amp=F_KICKSTART_AMP,
    ).to(device).double()

    # Apply Q-gate kickstart: q1 gates start near minimum
    apply_q1_gate_kickstart(lin_net, STATE_DIM, HORIZON, Q_GATE_KICKSTART_BIAS)

    n_params = sum(p.numel() for p in lin_net.parameters() if p.requires_grad)
    print(f"  Network params: {n_params:,}")

    session_name = f"stageD_qmod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir  = os.path.join(SAVE_DIR, session_name)

    # Epoch-0 rollout
    print("\n  Rolling out epoch-0 (untrained, with kickstart)...")
    x_first, u_first = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS)
    x_first_np = x_first.cpu().numpy()
    u_first_np = u_first.cpu().numpy()
    x_goal_np  = x_goal.cpu().numpy()
    demo_np    = demo.cpu().numpy()

    dist_first = np.linalg.norm(x_first_np[-1] - x_goal_np)
    print(f"  Epoch-0 goal_dist: {dist_first:.4f}")

    # Gradient smoke test
    if GRAD_DEBUG:
        print("\n  Gradient-flow smoke test...")
        grad_report = train_module.gradient_flow_smoke_test(
            lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal,
            demo=demo, num_steps=GRAD_SMOKE_STEPS, track_mode=TRACK_MODE,
        )
        mods = grad_report["module_norms"]
        print(f"    Smoke loss     : {grad_report['smoke_loss']:.6f}")
        print(f"    Total grad norm: {grad_report['total_norm']:.3e}")
        print(f"    Module norms   : trunk={mods['trunk']:.3e}  "
              f"q={mods['q_head']:.3e}  r={mods['r_head']:.3e}  f={mods['f_head']:.3e}")
        print(f"    Missing grads  : {grad_report['missing_count']}")

    recorder = network_module.NetworkOutputRecorder()
    monitor  = PrintMonitor(print_every=PRINT_EVERY, num_epochs=EPOCHS)

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=GRAD_DEBUG, grad_debug_every=GRAD_DEBUG_EVERY,
        track_mode=TRACK_MODE,
        w_terminal_anchor=W_TERMINAL_ANCHOR,
    )
    total_time = time.time() - t0

    print(f"\n  Training done in {total_time:.1f}s")
    print(f"  Loss: {loss_history[0]:.6f} → {loss_history[-1]:.6f}")

    # Final rollout
    print("\n  Rolling out final (trained)...")
    x_final, u_final = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS)
    x_final_np = x_final.cpu().numpy()
    u_final_np = u_final.cpu().numpy()
    dist_final = np.linalg.norm(x_final_np[-1] - x_goal_np)
    print(f"  Final goal_dist: {dist_final:.4f}  (epoch-0 was {dist_first:.4f})")
    print(f"  (model restored to best-seen checkpoint)")

    # Q1-gate analysis
    dump_q1_gate_stats(lin_net, mpc, x0, x_goal, NUM_STEPS, device)

    # Save outputs
    os.makedirs(session_dir, exist_ok=True)
    save_rollout_csv(x_first_np, u_first_np, DT, x_goal_np,
                     os.path.join(session_dir, f"{session_name}_rollout_epoch0.csv"))
    save_rollout_csv(x_final_np, u_final_np, DT, x_goal_np,
                     os.path.join(session_dir, f"{session_name}_rollout_final.csv"))

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
            "q_gate_kickstart_bias": Q_GATE_KICKSTART_BIAS,
            "zero_q1_costs": False,
            "state_dim": STATE_DIM, "control_dim": CONTROL_DIM,
            "dt": DT, "num_steps": NUM_STEPS,
            "x0": X0, "goal": X_GOAL,
            "experiment": "q_modulation_default_q_base",
        },
        session_name=session_name, recorder=recorder,
    )
    print(f"\n  All outputs → {session_dir}/")

    print("\n  === Q-MODULATION EXPERIMENT SUMMARY ===")
    print(f"  q_base_diag : [12, 5, 50, 40] (DEFAULT)")
    print(f"  gate_range_q: {GATE_RANGE_Q}  (q1 can reach {1 - GATE_RANGE_Q:.2f}x default)")
    print(f"  q1 kickstart: bias={Q_GATE_KICKSTART_BIAS}  "
          f"→ init gate≈{1 + GATE_RANGE_Q * math.tanh(Q_GATE_KICKSTART_BIAS):.4f}")
    print(f"  epochs      : {len(loss_history)}/{EPOCHS}  (early-stop?)")
    print(f"  goal_dist   : {dist_final:.4f}  {'SUCCESS <1.0' if dist_final < 1.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
