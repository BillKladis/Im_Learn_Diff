"""
Simulate_imitation.py — Stage D trajectory-imitation training loop.

GOAL OF THIS VARIANT
====================
The pure outcome-based Stage D variants (with bootstraps, with τ·q̇ reward,
with kickstart bias) all hit a stable local minimum where the f_extra head
locks to a constant direction and the pendulum never builds energy.

Trajectory imitation breaks the discovery problem at its root: instead of
asking the network to find the swing-up strategy from outcome signals,
we provide a successful state trajectory (recorded from the working
physics-informed version) and train the network to produce trajectories
that match it.  Tracking error is monotone in distance to the demo
trajectory — there is no flat region.  The network is free to discover
ANY internal cost shaping that achieves the demonstrated trajectory shape;
it does not see the demo's controls.

LOSS DESIGN (drastically simplified vs the outcome-based variants)
==================================================================
At each step t we have:
    current_state_detached    — our rollout's state at step t
    next_state                — our rollout's state at step t+1 (differentiable)
    demo[t+1]                 — the demo trajectory's state at step t+1

Tracking loss:
    L_track(t) = ||next_state − demo[t+1]||²   (squared L2 over all 4 dims)

Total loss:
    L_total = W_TRACK    × mean(L_track)         (the dominant term, W=5.0)
            + W_TERMINAL × terminal_loss          (still useful: arrive cleanly)
            + W_Q2_SHAPE × q2_shape_avg           (light anti-fold)
            + W_Q2_DOT   × q2_dot_avg

REMOVED compared to the previous Stage D loss:
    - pump reward, velocity bootstrap, acceleration bootstrap, τ·q̇ reward
    - pump-dominance gating, threshold curriculum, pure-pump epoch window
All of these existed to fight the flat region.  Tracking has no flat region.

CONTROLLER PATH (unchanged from Stage D)
========================================
    f_extra (network) → extra_linear_control → QP → u_mpc
No physics inside the controller.  The network must produce f_extra such
that the resulting closed-loop trajectory matches the demo.
"""

import copy
import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import lin_net as network_module
import mpc_controller


# ──────────────────────────────────────────────────────────────────────────
# Demo loader
# ──────────────────────────────────────────────────────────────────────────
def load_demo_trajectory(
    csv_path: str,
    expected_length: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Load a demo state trajectory from a rollout CSV produced by main.py.

    Reads columns q1_rad, q1_dot_rads, q2_rad, q2_dot_rads.
    Returns a (T, 4) torch tensor on the requested device.

    NOTE: The CSV produced by main.py's save_rollout_csv has T rows where
    T = u_hist.shape[0] = NUM_STEPS.  The state recorded at row i is the
    state at time step i (i.e. x_hist[i]).  So the CSV gives us
    x_hist[0..NUM_STEPS-1] — it does NOT include the final x_hist[NUM_STEPS].
    For tracking purposes we want demo[1..NUM_STEPS], i.e. the state we
    expect to reach after each step.  The training loop handles indexing
    such that we compare next_state against demo[step+1] directly using
    this T-row tensor — caller should ensure CSV has at least NUM_STEPS+1
    rows of states.  See _train_one_epoch for indexing.
    """
    import csv
    rows: List[List[float]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([
                float(row["q1_rad"]),
                float(row["q1_dot_rads"]),
                float(row["q2_rad"]),
                float(row["q2_dot_rads"]),
            ])
    if not rows:
        raise ValueError(f"Demo CSV {csv_path} is empty.")
    if expected_length is not None and len(rows) < expected_length:
        raise ValueError(
            f"Demo CSV has {len(rows)} rows but training expects at least "
            f"{expected_length} (NUM_STEPS+1).  Re-export from a longer rollout."
        )

    arr = np.asarray(rows, dtype=np.float64)
    return torch.tensor(arr, device=device, dtype=torch.float64)


# ──────────────────────────────────────────────────────────────────────────
# Gradient diagnostics
# ──────────────────────────────────────────────────────────────────────────
def _gradient_stats(lin_net: nn.Module) -> dict:
    tracked = ["state_encoder", "trunk", "q_head", "r_head", "f_head"]
    module_sq = {k: 0.0 for k in tracked}
    total_sq = 0.0
    missing = []

    for name, param in lin_net.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
            continue
        g2 = float(param.grad.detach().pow(2).sum().item())
        total_sq += g2
        for prefix in tracked:
            if name.startswith(prefix):
                module_sq[prefix] += g2
                break

    return {
        "total_norm":    math.sqrt(max(total_sq, 0.0)),
        "module_norms":  {k: math.sqrt(max(v, 0.0)) for k, v in module_sq.items()},
        "missing_count": len(missing),
        "missing_names": missing,
    }


def gradient_flow_smoke_test(
    lin_net:     nn.Module,
    mpc:         mpc_controller.MPC_controller,
    x0:          torch.Tensor,
    x_goal:      torch.Tensor,
    demo:        torch.Tensor,
    num_steps:   int = 5,
) -> dict:
    """Smoke test: ensure gradient flows from a tracking loss to all heads."""
    lin_net.train()
    lin_net.zero_grad(set_to_none=True)

    n_u = mpc.MPC_dynamics.u_min.shape[0]

    current_state = x0.detach().clone()
    state_history = [current_state.clone() for _ in range(5)]
    u_seq_guess = torch.zeros((mpc.N, n_u), device=mpc.device, dtype=torch.float64)

    step_losses = []
    for t in range(num_steps):
        gates_Q, gates_R, f_extra, _, _ = lin_net(
            torch.stack(state_history, dim=0),
            q_base_diag=mpc.q_base_diag,
            r_base_diag=mpc.r_base_diag,
        )

        x_lin_seq = current_state.unsqueeze(0).expand(mpc.N, -1).clone()
        u_lin_seq = torch.clamp(
            u_seq_guess.clone(),
            min=mpc.MPC_dynamics.u_min.unsqueeze(0),
            max=mpc.MPC_dynamics.u_max.unsqueeze(0),
        )
        extra_ctrl = f_extra.reshape(-1)

        u_mpc, U_opt_full = mpc.control(
            current_state, x_lin_seq, u_lin_seq, x_goal,
            diag_corrections_Q=gates_Q,
            diag_corrections_R=gates_R,
            extra_linear_control=extra_ctrl,
        )

        next_state = mpc.true_RK4_disc(current_state, u_mpc, mpc.dt)
        target = demo[min(t + 1, demo.shape[0] - 1)]
        step_losses.append(((next_state - target) ** 2).sum())

        current_state = next_state.detach()
        U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
        u_seq_guess = torch.cat([U_opt_reshaped[1:], U_opt_reshaped[-1:]], dim=0).clone()
        state_history.pop(0)
        state_history.append(current_state.clone())

    smoke_loss = torch.stack(step_losses).mean()
    smoke_loss.backward()
    stats = _gradient_stats(lin_net)
    stats["smoke_loss"] = float(smoke_loss.item())
    lin_net.zero_grad(set_to_none=True)
    return stats


# ──────────────────────────────────────────────────────────────────────────
# Main training loop (imitation)
# ──────────────────────────────────────────────────────────────────────────
def train_linearization_network(
    lin_net:    nn.Module,
    mpc:        mpc_controller.MPC_controller,
    x0:         torch.Tensor,
    x_goal:     torch.Tensor,
    demo:       torch.Tensor,
    num_steps:  int,
    num_epochs: int   = 30,
    lr:         float = 1e-4,
    debug_monitor       = None,
    recorder:           Optional[network_module.NetworkOutputRecorder] = None,
    grad_debug:         bool  = False,
    grad_debug_every:   int   = 1,
) -> Tuple[List[float], network_module.NetworkOutputRecorder]:

    # ── Loss weights ──────────────────────────────────────────────────────
    W_TRACK         = 5.0    # dominant — drives the network toward demo trajectory
    W_TERMINAL      = 0.0    # secondary — make sure we arrive cleanly at goal
    W_Q2_SHAPE      = 0.0    # light always-on anti-fold
    W_Q2_DOT        = 0.0
    STEP_LOSS_CLAMP = 200.0

    SKIP_UPDATE_GRAD_NORM = 5e7
    CLIP_OTHER = 2.0

    n_u = mpc.MPC_dynamics.u_min.shape[0]
    demo_T = demo.shape[0]   # number of demo states available

    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-5,
    )

    loss_history    = []
    best_goal_dist  = float("inf")
    best_state_dict = None

    if recorder is None:
        recorder = network_module.NetworkOutputRecorder()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        lin_net.train()
        optimizer.zero_grad()
        recorder.start_epoch()
        qp_fallback_start = int(getattr(mpc, "qp_fallback_count", 0))

        current_state_detached = x0.detach().clone()
        state_history = [current_state_detached.clone() for _ in range(5)]
        u_seq_guess = torch.zeros((mpc.N, n_u), device=mpc.device, dtype=torch.float64)

        track_step_terms    = []
        terminal_step_terms = []
        q2_step_terms       = []

        for step in range(num_steps):
            state_history_seq = torch.stack(state_history, dim=0)

            gates_Q, gates_R, f_extra, q_diags, r_diags = lin_net(
                state_history_seq,
                q_base_diag=mpc.q_base_diag,
                r_base_diag=mpc.r_base_diag,
            )

            x_lin_seq = current_state_detached.unsqueeze(0).expand(mpc.N, -1).clone()
            u_lin_seq = torch.clamp(
                u_seq_guess.clone(),
                min=mpc.MPC_dynamics.u_min.unsqueeze(0),
                max=mpc.MPC_dynamics.u_max.unsqueeze(0),
            )

            extra_ctrl = f_extra.reshape(-1)

            u_mpc, U_opt_full = mpc.control(
                current_state_detached, x_lin_seq, u_lin_seq, x_goal,
                diag_corrections_Q=gates_Q,
                diag_corrections_R=gates_R,
                extra_linear_control=extra_ctrl,
            )

            next_state = mpc.true_RK4_disc(current_state_detached, u_mpc, mpc.dt)

            # ── Tracking term (the main signal) ──────────────────────────
            target_idx = min(step + 1, demo_T - 1)
            target = demo[target_idx]
            track_step = ((next_state - target) ** 2).sum()
            track_step = torch.clamp(track_step, max=STEP_LOSS_CLAMP)
            track_step_terms.append(track_step)

            # ── Always-on q2 anti-fold ───────────────────────────────────
            q2_shape       = 1.0 - torch.cos(next_state[2])
            q2_dot_penalty = W_Q2_DOT * next_state[3] ** 2
            q2_step_terms.append(W_Q2_SHAPE * q2_shape + q2_dot_penalty)

            # ── Terminal style (per-step distance to goal) ──────────────
            err = next_state - x_goal
            q1_err_w = torch.atan2(
                torch.sin(next_state[0] - x_goal[0]),
                torch.cos(next_state[0] - x_goal[0]),
            )
            terminal_step = (
                3.0 * q1_err_w ** 2 + err[1] ** 2 + err[3] ** 2
                + W_Q2_SHAPE * q2_shape
                + q2_dot_penalty
            )
            terminal_step = torch.clamp(terminal_step, max=STEP_LOSS_CLAMP)
            terminal_step_terms.append(terminal_step)

            recorder.record_step(
                gates_Q=gates_Q, gates_R=gates_R, f_extra=f_extra,
                q_diags=q_diags, r_diags=r_diags,
                u_mpc=u_mpc,
                state_err=((next_state.detach() - x_goal) ** 2).sum(),
            )

            current_state_detached = next_state.detach()
            U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
            u_seq_guess = torch.cat(
                [U_opt_reshaped[1:], U_opt_reshaped[-1:]], dim=0,
            ).clone()
            state_history.pop(0)
            state_history.append(current_state_detached.clone())

        # ── Combine ──────────────────────────────────────────────────────
        track_loss    = torch.stack(track_step_terms).sum()    / num_steps
        terminal_loss = torch.stack(terminal_step_terms).sum() / num_steps
        q2_loss       = torch.stack(q2_step_terms).sum()       / num_steps

        total_loss = (
            W_TRACK    * track_loss
            + W_TERMINAL * terminal_loss
            + q2_loss
        )

        loss_history.append(total_loss.item())
        recorder.end_epoch(total_loss.item())

        total_loss.backward()
        grad_stats = None
        if grad_debug and ((epoch + 1) % max(1, grad_debug_every) == 0 or epoch == 0):
            grad_stats = _gradient_stats(lin_net)

        with torch.no_grad():
            goal_dist = torch.norm(current_state_detached - x_goal).item()
        if goal_dist < best_goal_dist:
            best_goal_dist = goal_dist
            best_state_dict = copy.deepcopy(lin_net.state_dict())

        if not torch.isfinite(total_loss):
            optimizer.zero_grad()
        else:
            is_bad = any(
                (not torch.isfinite(p.grad).all())
                for _, p in lin_net.named_parameters()
                if p.grad is not None
            )
            if is_bad:
                optimizer.zero_grad()
            elif grad_stats is not None and grad_stats["total_norm"] > SKIP_UPDATE_GRAD_NORM:
                optimizer.zero_grad()
            else:
                params = [p for p in lin_net.parameters() if p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=CLIP_OTHER)
                optimizer.step()

        scheduler.step(epoch + 1)

        if debug_monitor:
            with torch.no_grad():
                summary = recorder.epoch_summary(epoch)
                qp_fallbacks_epoch = (
                    int(getattr(mpc, "qp_fallback_count", 0)) - qp_fallback_start
                )
            debug_monitor.log_epoch(epoch, num_epochs, total_loss.item(), {
                "epoch_time":         time.time() - epoch_start_time,
                "learning_rate":      optimizer.param_groups[0]["lr"],
                "loss_track":         track_loss.item(),
                "loss_terminal":      terminal_loss.item(),
                "loss_q2":            q2_loss.item(),
                "qp_fallbacks":       qp_fallbacks_epoch,
                "pure_end_error":     goal_dist,
                "mean_Q_gate_dev":    summary.get("mean_Q_gate_dev",    float("nan")),
                "mean_f_extra_norm":  summary.get("mean_f_extra_norm",  float("nan")),
                "mean_f_tau1_first":  summary.get("mean_f_tau1_first",  float("nan")),
            })

        if grad_stats is not None:
            mn = grad_stats["module_norms"]
            print(
                "      GradFlow | "
                f"tot={grad_stats['total_norm']:.3e} "
                f"trunk={mn['trunk']:.3e} "
                f"q={mn['q_head']:.3e} "
                f"r={mn['r_head']:.3e} "
                f"f={mn['f_head']:.3e} "
                f"missing={grad_stats['missing_count']}"
            )

    if best_state_dict is not None:
        lin_net.load_state_dict(best_state_dict)

    return loss_history, recorder


# ──────────────────────────────────────────────────────────────────────────
# Rollout (no learning) — same as Simulate.py
# ──────────────────────────────────────────────────────────────────────────
def rollout(
    lin_net,
    mpc:       mpc_controller.MPC_controller,
    x0:        torch.Tensor,
    x_goal:    torch.Tensor,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    n_x = x0.shape[0]
    n_u = mpc.MPC_dynamics.u_min.shape[0]

    x_hist = torch.zeros(num_steps + 1, n_x, dtype=torch.float64, device=mpc.device)
    u_hist = torch.zeros(num_steps,     n_u, dtype=torch.float64, device=mpc.device)

    x = x0.clone().to(mpc.device)
    x_hist[0] = x
    u_seq_guess = torch.zeros((mpc.N, n_u), dtype=torch.float64, device=mpc.device)

    init_q1 = float(x[0].item())
    if abs(init_q1) > 0.01:
        gravity_torque = 2.0 * 9.81 * 0.5 * abs(math.sin(init_q1))
        wrapped_err = math.atan2(
            math.sin(float(x_goal[0].item()) - init_q1),
            math.cos(float(x_goal[0].item()) - init_q1),
        )
        goal_sign = 1.0 if wrapped_err > 0 else -1.0
        seed_tau1 = goal_sign * min(
            float(mpc.MPC_dynamics.u_max[0].item()),
            gravity_torque * 2.0,
        )
        u_seq_guess[:, 0] = seed_tau1

    state_history = [x.clone() for _ in range(5)]

    if lin_net is not None:
        lin_net.eval()

    for step in range(num_steps):
        with torch.no_grad():
            if lin_net is not None:
                gates_Q, gates_R, f_extra, _, _ = lin_net(
                    torch.stack(state_history, dim=0),
                    q_base_diag=mpc.q_base_diag,
                    r_base_diag=mpc.r_base_diag,
                )
            else:
                gates_Q, gates_R, f_extra = None, None, None

        x_lin_seq = x.unsqueeze(0).expand(mpc.N, -1).clone()
        u_lin_seq = torch.clamp(
            u_seq_guess.clone(),
            min=mpc.MPC_dynamics.u_min.unsqueeze(0),
            max=mpc.MPC_dynamics.u_max.unsqueeze(0),
        )

        extra_ctrl = f_extra.reshape(-1) if f_extra is not None else None

        u_opt, U_opt_full = mpc.control(
            x, x_lin_seq, u_lin_seq, x_goal,
            diag_corrections_Q=gates_Q,
            diag_corrections_R=gates_R,
            extra_linear_control=extra_ctrl,
        )

        x = mpc.true_RK4_disc(x, u_opt, mpc.dt)

        u_hist[step]     = u_opt.detach()
        x_hist[step + 1] = x.detach()

        U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
        u_seq_guess[:-1] = U_opt_reshaped[1:].clone()
        u_seq_guess[-1]  = U_opt_reshaped[-1].clone()

        state_history.pop(0)
        state_history.append(x.detach().clone())

    return x_hist, u_hist