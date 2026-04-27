import time
import math
import mpc_controller
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from torch.func import jacrev

import lin_net as network_module


# ── Task-specific energy shaping ─────────────────────────────────────────────
# These helpers live here, not in mpc_controller, so that swapping to a new
# system only requires editing Simulate.py (or a task-specific file).

def _compute_q1_energy(x: torch.Tensor) -> torch.Tensor:
    """
    Single-link effective energy for q1 — structurally zero gradient w.r.t. q2
    and q2_dot, so no projection is ever needed and q2 coupling is impossible.

    Treats the double pendulum as an effective pendulum of mass (m1+m2) at l1.
    Goal energy gap: V(π) − V(0) = 2·(m1+m2)·g·l1 = 19.62 J.
    """
    q1, q1_dot = x[0], x[1]
    m_eff, l_eff, g = 2.0, 0.5, 9.81
    return 0.5 * m_eff * (l_eff * q1_dot) ** 2 - m_eff * g * l_eff * torch.cos(q1)


def _build_energy_control_tau1(
    u_lin_seq: torch.Tensor,
    current_state: torch.Tensor,
    mpc: "mpc_controller.MPC_controller",
    gates_E: torch.Tensor,
    E_q1_goal: torch.Tensor,
    w_e_base: float = 20.0,
) -> torch.Tensor:
    """
    Compute extra_linear_control (N*nu,) for q1-only energy shaping via τ1 ONLY.

    Root-cause of q2 folding: the mass-matrix coupling M^{-1}[q2,τ1] ≈ -8 means
    applying B_big^T @ state_gradient biases τ2 in the WRONG direction (opposes
    q2 correction).  This function instead:
      1. Computes B_i^T @ ∇E_q1 to get the energy gradient in control space.
      2. Zeros the τ2 component — energy shaping only ever pushes τ1.
      3. Adds directly to the QP's f vector (extra_linear_control), bypassing
         the full B_big^T mapping entirely.

    The gradient path to the network flows through gates_E (the network's energy
    gate), preserving differentiability for training.
    """
    N, n_u = mpc.N, 2
    device = mpc.device

    # Nominal rollout + per-step B matrices (all detached).
    X_bar_seq, B_list = mpc.compute_nominal_rollout(current_state, u_lin_seq)

    E_curr = torch.stack([_compute_q1_energy(X_bar_seq[i]) for i in range(N)])
    ctrl_energy = torch.zeros(N * n_u, device=device, dtype=torch.float64)

    for i in range(N):
        idx = slice(i * n_u, (i + 1) * n_u)
        e_i  = E_curr[i] - E_q1_goal   # signed: negative = deficit
        gate = gates_E[i]               # network gate: only grad path

        # Deficit boost: drives pumping when energy is low.
        deficit_norm = torch.relu(-e_i) / (2.0 * E_q1_goal.abs() + 1.0)
        w_k = w_e_base * gate * (1.0 + 2.0 * deficit_norm ** 2)

        # Angle gate: turns off shaping as q1 → π so position tracking takes over.
        q1_i = X_bar_seq[i][0]
        dist_top = torch.abs(torch.atan2(
            torch.sin(q1_i - math.pi), torch.cos(q1_i - math.pi)
        ))
        angle_gate = 1.0 - torch.exp(-3.0 * dist_top ** 2)

        # State-space energy gradient (indices 2,3 are zero by construction).
        g_state = jacrev(_compute_q1_energy)(X_bar_seq[i])

        # Project to control space via one-step B matrix.
        B_i  = B_list[i].detach()   # (nx, nu)
        g_ctrl = B_i.T @ g_state    # (nu=2,): [grad_τ1, grad_τ2]

        # ── ZERO τ2 ──────────────────────────────────────────────────────────
        # B^T coupling creates a τ2 component that OPPOSES q2 correction
        # (M^{-1}[q2,τ1] ≈ −8 drives τ2 backwards).  Zeroing it means energy
        # shaping cannot disturb q2 — folding from this pathway is eliminated.
        g_ctrl_τ1 = g_ctrl.clone()
        g_ctrl_τ1[1] = 0.0

        ctrl_energy[idx] = w_k * angle_gate * e_i * g_ctrl_τ1

    return ctrl_energy
# ─────────────────────────────────────────────────────────────────────────────


def _gradient_stats(lin_net: nn.Module) -> dict:
    tracked_prefixes = [
        "state_encoder",
        "res_encoder",
        "trunk",
        "q_head",
        "r_head",
        "e_head",
        "u_lin_head",
        "qf_head",
    ]
    module_sq = {k: 0.0 for k in tracked_prefixes}
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
        for prefix in tracked_prefixes:
            if name.startswith(prefix):
                module_sq[prefix] += g2
                break

    return {
        "total_norm": math.sqrt(max(total_sq, 0.0)),
        "module_norms": {k: math.sqrt(max(v, 0.0)) for k, v in module_sq.items()},
        "missing_count": len(missing),
        "missing_names": missing,
    }


def gradient_flow_smoke_test(
    lin_net: nn.Module,
    mpc: mpc_controller.MPC_controller,
    x0: torch.Tensor,
    x_goal: torch.Tensor,
    num_steps: int = 5,
) -> dict:
    lin_net.train()
    lin_net.zero_grad(set_to_none=True)

    n_res = lin_net.n_res
    state_dim = lin_net.state_dim
    n_u = mpc.MPC_dynamics.u_min.shape[0]

    current_state = x0.detach().clone()
    state_history = [current_state.clone() for _ in range(5)]
    zero_residual = torch.zeros(state_dim, device=mpc.device, dtype=torch.float64)
    residual_history = [zero_residual.clone() for _ in range(n_res)]
    u_seq_guess = torch.zeros((mpc.N, n_u), device=mpc.device, dtype=torch.float64)

    E_q1_goal_smoke = _compute_q1_energy(x_goal).detach()

    step_losses = []
    for _ in range(num_steps):
        gates_Q, gates_R, gates_E, Qf_dense, _, _, u_lin_delta = lin_net(
            torch.stack(state_history, dim=0),
            torch.stack(residual_history, dim=0),
            q_base_diag=mpc.q_base_diag,
            r_base_diag=mpc.r_base_diag,
        )

        x_lin_seq = current_state.unsqueeze(0).expand(mpc.N, -1).clone()
        u_lin_seq = torch.clamp(
            u_seq_guess.clone() + u_lin_delta,
            min=mpc.MPC_dynamics.u_min.unsqueeze(0),
            max=mpc.MPC_dynamics.u_max.unsqueeze(0),
        )
        extra_ctrl = _build_energy_control_tau1(
            u_lin_seq, current_state, mpc, gates_E, E_q1_goal_smoke,
        )
        u_mpc, U_opt_full = mpc.control(
            current_state, x_lin_seq, u_lin_seq, x_goal,
            diag_corrections_Q=gates_Q,
            diag_corrections_R=gates_R,
            extra_linear_control=extra_ctrl,
            Qf_dense=Qf_dense,
        )

        next_state = mpc.true_RK4_disc(current_state, u_mpc, mpc.dt)
        step_losses.append(((next_state - x_goal) ** 2).sum())

        with torch.no_grad():
            delta = next_state.detach() - mpc.MPC_RK4_disc(current_state, u_mpc.detach(), mpc.dt)

        current_state = next_state
        U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
        u_seq_guess = torch.cat([U_opt_reshaped[1:], U_opt_reshaped[-1:]], dim=0).clone()
        state_history.pop(0)
        state_history.append(current_state.clone())
        residual_history.pop(0)
        residual_history.append(delta)

    smoke_loss = torch.stack(step_losses).mean()
    smoke_loss.backward()

    stats = _gradient_stats(lin_net)
    stats["smoke_loss"] = float(smoke_loss.item())
    lin_net.zero_grad(set_to_none=True)
    return stats


def train_linearization_network(
    lin_net: nn.Module,
    mpc: mpc_controller.MPC_controller,
    x0: torch.Tensor,
    x_goal: torch.Tensor,
    num_steps: int,
    num_epochs: int = 25,
    lr: float = 1e-3,
    debug_monitor=None,
    recorder: Optional[network_module.NetworkOutputRecorder] = None,
    grad_debug: bool = False,
    grad_debug_every: int = 1,
) -> Tuple[List[float], network_module.NetworkOutputRecorder]:

    # ── Loss weights ──────────────────────────────────────────────────────────
    # Energy shaping is FIXED (gates_E detached in the energy term below).
    # The network trains Q/R/Qf only — those gates have a direct, clean gradient
    # through the QP cost.  Energy-gate gradients go through H^{-1} and are too
    # weak to overcome the initialisation gap within a reasonable epoch count.
    W_TERMINAL = 1.0
    W_Q2_SHAPE = 1.5    # always-on anti-fold penalty in training loss
    W_Q2_DOT   = 0.3
    W_E_SHAPE  = 12.0   # fixed energy shaping strength (no gate ramp needed)
    W_QF_ANCHOR = 1e-3
    W_U_LIN_IMITATION = 0.05
    STEP_LOSS_CLAMP = 200.0
    CLIP_QF_HEAD = 5.0
    CLIP_U_LIN  = 2.0
    CLIP_OTHER  = 2.0
    SKIP_UPDATE_GRAD_NORM = 5e7

    # Fixed phase boundaries (fraction of trajectory length).
    # No epoch-level ramp: the loss floor must not rise with epoch count.
    ENERGY_PHASE_END  = 0.45
    POSITION_PHASE_IN = 0.60

    n_res = lin_net.n_res
    state_dim = lin_net.state_dim
    n_u = mpc.MPC_dynamics.u_min.shape[0]

    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=lr, weight_decay=1e-4)
    # Warm restarts let the model escape local optima that ReduceLROnPlateau
    # would freeze it in. T_0=5 → restart every 5 epochs, doubling each cycle.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-5,
    )
    loss_history = []
    best_goal_dist = float('inf')
    best_state_dict = None          # best model weights seen during training

    if recorder is None:
        recorder = network_module.NetworkOutputRecorder()

    # q1-only effective energy at the goal — used for MPC shaping (zero q2 coupling)
    E_q1_goal = _compute_q1_energy(x_goal).detach()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        lin_net.train()
        optimizer.zero_grad()
        recorder.start_epoch()
        qp_fallback_start = int(getattr(mpc, 'qp_fallback_count', 0))

        current_state_detached = x0.detach().clone()
        state_history = [current_state_detached.clone() for _ in range(5)]

        zero_residual = torch.zeros(state_dim, device=mpc.device, dtype=torch.float64)
        residual_history = [zero_residual.clone() for _ in range(n_res)]

        u_seq_guess = torch.zeros((mpc.N, n_u), device=mpc.device, dtype=torch.float64)
        step_losses = []
        qf_anchor_terms = []
        u_lin_imitation_terms = []

        for step in range(num_steps):
            state_history_seq = torch.stack(state_history, dim=0)
            residual_history_seq = torch.stack(residual_history, dim=0)

            gates_Q, gates_R, gates_E, Qf_dense, q_diags, r_diags, u_lin_delta = lin_net(
                state_history_seq,
                residual_history_seq,
                q_base_diag=mpc.q_base_diag,
                r_base_diag=mpc.r_base_diag,
            )

            x_lin_seq = current_state_detached.unsqueeze(0).expand(mpc.N, -1).clone()

            # Detach u_lin from the QP warm-start gradient path: backprop through
            # 170 QP solves w.r.t. the warm-start input accumulates explosively.
            # u_lin_head is trained separately via imitation loss below.
            u_lin_seq = torch.clamp(
                u_seq_guess.clone() + u_lin_delta.detach(),
                min=mpc.MPC_dynamics.u_min.unsqueeze(0),
                max=mpc.MPC_dynamics.u_max.unsqueeze(0),
            )

            # τ1-only energy shaping — gates_E is DETACHED so gradient only flows
            # through Q/R/Qf.  Energy shaping runs at fixed strength; the network
            # improves tracking quality, not energy building.
            extra_ctrl = _build_energy_control_tau1(
                u_lin_seq, current_state_detached, mpc, gates_E.detach(), E_q1_goal, W_E_SHAPE,
            )

            u_mpc, U_opt_full = mpc.control(
                current_state_detached,
                x_lin_seq,
                u_lin_seq,
                x_goal,
                diag_corrections_Q=gates_Q,
                diag_corrections_R=gates_R,
                extra_linear_control=extra_ctrl,
                Qf_dense=Qf_dense,
            )

            # Imitation loss: train u_lin_head to predict the first MPC action.
            # Gradient is simple (no QP), preventing the warm-start explosion.
            u_lin_imitation_terms.append(
                ((u_lin_delta[0] - u_mpc.detach()) ** 2).mean()
            )

            next_state = mpc.true_RK4_disc(current_state_detached, u_mpc, mpc.dt)

            # ── Phase weight (fixed boundaries, no epoch-level ramp) ──────────
            t_frac = step / max(num_steps - 1, 1)
            if t_frac < ENERGY_PHASE_END:
                pos_w = 0.05
            elif t_frac < POSITION_PHASE_IN:
                pos_w = 0.05 + 0.95 * (t_frac - ENERGY_PHASE_END) / (POSITION_PHASE_IN - ENERGY_PHASE_END)
            else:
                pos_w = 1.0

            err = next_state - x_goal
            q1_err_w = torch.atan2(torch.sin(next_state[0] - x_goal[0]),
                                   torch.cos(next_state[0] - x_goal[0]))
            q2_shape      = 1.0 - torch.cos(next_state[2])
            q2_dot_penalty = W_Q2_DOT * next_state[3] ** 2

            step_state_err = (pos_w * (3.0 * q1_err_w**2 + err[1]**2 + err[3]**2)
                              + W_Q2_SHAPE * q2_shape
                              + q2_dot_penalty)
            step_losses.append(torch.clamp(step_state_err, max=STEP_LOSS_CLAMP))

            qf_anchor_terms.append(((Qf_dense - mpc.Qf) ** 2).mean())

            with torch.no_grad():
                x_mpc_predicted = mpc.MPC_RK4_disc(current_state_detached, u_mpc.detach(), mpc.dt)
                delta = next_state.detach() - x_mpc_predicted

            recorder.record_step(
                gates_Q=gates_Q,
                gates_R=gates_R,
                gates_E=gates_E,
                q_diags=q_diags,
                r_diags=r_diags,
                u_mpc=u_mpc,
                state_err=((next_state.detach() - x_goal) ** 2).sum(),
                residual_norm=delta.norm().item(),
                Qf_dense=Qf_dense,
                u_lin_delta=u_lin_delta,
            )

            # Detach state to cut gradient accumulation across 170 QP steps.
            # Without this, each step's QP Jacobian multiplies into the previous,
            # producing gradient norms that grow exponentially with trajectory length.
            # Each step still contributes a clean one-step gradient to the network.
            current_state_detached = next_state.detach()

            U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
            u_seq_guess = torch.cat([U_opt_reshaped[1:], U_opt_reshaped[-1:]], dim=0).clone()

            state_history.pop(0)
            state_history.append(current_state_detached.clone())
            residual_history.pop(0)
            residual_history.append(delta)

        terminal_loss        = torch.stack(step_losses).sum() / num_steps
        qf_anchor_loss       = torch.stack(qf_anchor_terms).mean()
        u_lin_imitation_loss = torch.stack(u_lin_imitation_terms).mean()

        total_loss = (
            W_TERMINAL * terminal_loss
            + W_QF_ANCHOR * qf_anchor_loss
            + W_U_LIN_IMITATION * u_lin_imitation_loss
        )

        loss_history.append(total_loss.item())
        recorder.end_epoch(total_loss.item())

        total_loss.backward()
        grad_stats = None
        if grad_debug and ((epoch + 1) % max(1, grad_debug_every) == 0 or epoch == 0):
            grad_stats = _gradient_stats(lin_net)

        # Track best model BEFORE the optimizer step: goal_dist reflects the
        # trajectory produced by the CURRENT weights. Saving after step would
        # record the updated (different) weights instead.
        with torch.no_grad():
            goal_dist = torch.norm(current_state_detached - x_goal).item()
        if goal_dist < best_goal_dist:
            best_goal_dist = goal_dist
            import copy
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
            else:
                if grad_stats is not None and grad_stats["total_norm"] > SKIP_UPDATE_GRAD_NORM:
                    optimizer.zero_grad()
                else:
                    qf_params    = [p for n, p in lin_net.named_parameters() if n.startswith("qf_head")    and p.grad is not None]
                    u_lin_params = [p for n, p in lin_net.named_parameters() if n.startswith("u_lin_head") and p.grad is not None]
                    other_params = [p for n, p in lin_net.named_parameters() if (not n.startswith("qf_head")) and (not n.startswith("u_lin_head")) and p.grad is not None]
                    if qf_params:
                        torch.nn.utils.clip_grad_norm_(qf_params,    max_norm=CLIP_QF_HEAD)
                    if u_lin_params:
                        torch.nn.utils.clip_grad_norm_(u_lin_params, max_norm=CLIP_U_LIN)
                    if other_params:
                        torch.nn.utils.clip_grad_norm_(other_params, max_norm=CLIP_OTHER)
                    optimizer.step()

        scheduler.step(epoch + 1)

        if debug_monitor:
            with torch.no_grad():
                summary = recorder.epoch_summary(epoch)
                qp_fallbacks_epoch = int(getattr(mpc, 'qp_fallback_count', 0)) - qp_fallback_start

            debug_monitor.log_epoch(epoch, num_epochs, total_loss.item(), {
                'epoch_time': time.time() - epoch_start_time,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'loss_terminal': terminal_loss.item(),
                'loss_qf_anchor': qf_anchor_loss.item(),
                'qp_fallbacks': qp_fallbacks_epoch,
                'pure_end_error': goal_dist,
                'mean_Q_gate_dev': summary.get('mean_Q_gate_dev', float('nan')),
                'mean_E_gate_dev': summary.get('mean_E_gate_dev', float('nan')),
                'mean_u_lin_norm': summary.get('mean_u_lin_norm', float('nan')),
                'mean_qf_norm': summary.get('mean_qf_norm', float('nan')),
            })

        if grad_stats is not None:
            mn = grad_stats["module_norms"]
            print(
                "      GradFlow | "
                f"tot={grad_stats['total_norm']:.3e} "
                f"trunk={mn['trunk']:.3e} "
                f"q={mn['q_head']:.3e} "
                f"r={mn['r_head']:.3e} "
                f"e={mn['e_head']:.3e} "
                f"u_lin={mn['u_lin_head']:.3e} "
                f"qf={mn['qf_head']:.3e} "
                f"missing={grad_stats['missing_count']}"
            )
            if grad_stats["total_norm"] > SKIP_UPDATE_GRAD_NORM:
                print(f"      GradFlow | update skipped (norm>{SKIP_UPDATE_GRAD_NORM:.1e})")
            if grad_stats["missing_count"] > 0:
                sample = ", ".join(grad_stats["missing_names"][:5])
                print(f"      NoGrad sample: {sample}")

    # Restore best weights so the caller gets the best-seen model, not the last
    if best_state_dict is not None:
        lin_net.load_state_dict(best_state_dict)

    return loss_history, recorder


def rollout(
    lin_net,
    mpc: mpc_controller.MPC_controller,
    x0: torch.Tensor,
    x_goal: torch.Tensor,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    n_x = x0.shape[0]
    n_u = mpc.MPC_dynamics.u_min.shape[0]

    x_hist = torch.zeros(num_steps + 1, n_x, dtype=torch.float64, device=mpc.device)
    u_hist = torch.zeros(num_steps, n_u, dtype=torch.float64, device=mpc.device)

    x = x0.clone().to(mpc.device)
    x_hist[0] = x
    u_seq_guess = torch.zeros((mpc.N, n_u), dtype=torch.float64, device=mpc.device)

    # For non-zero initial tilt, the zero warm-start exposes gravity pulling
    # the nominal trajectory backward, which causes the velocity-gradient energy
    # shaping to push τ1 negative before any forward momentum is built.
    # Pre-seed u_seq_guess with enough τ1 to overcome gravity so the first
    # nominal rollout shows forward motion and the energy shaping fires correctly.
    init_q1 = float(x[0].item())
    if abs(init_q1) > 0.01:
        gravity_torque = 2.0 * 9.81 * 0.5 * abs(math.sin(init_q1))
        wrapped_err = math.atan2(
            math.sin(float(x_goal[0].item()) - init_q1),
            math.cos(float(x_goal[0].item()) - init_q1),
        )
        goal_sign = 1.0 if wrapped_err > 0 else -1.0
        seed_tau1 = goal_sign * min(float(mpc.MPC_dynamics.u_max[0].item()),
                                    gravity_torque * 2.0)
        u_seq_guess[:, 0] = seed_tau1

    state_history = [x.clone() for _ in range(5)]
    E_q1_goal_rollout = _compute_q1_energy(x_goal).detach()

    if lin_net is not None:
        lin_net.eval()
        residual_history = [
            torch.zeros(lin_net.state_dim, device=mpc.device, dtype=torch.float64)
            for _ in range(lin_net.n_res)
        ]

    for step in range(num_steps):
        with torch.no_grad():
            if lin_net is not None:
                gates_Q, gates_R, gates_E, Qf_dense, _, _, u_lin_delta = lin_net(
                    torch.stack(state_history, dim=0),
                    torch.stack(residual_history, dim=0),
                    q_base_diag=mpc.q_base_diag,
                    r_base_diag=mpc.r_base_diag,
                )
            else:
                gates_Q, gates_R, gates_E, Qf_dense = None, None, None, None
                u_lin_delta = torch.zeros((mpc.N, n_u), dtype=torch.float64, device=mpc.device)

        x_lin_seq = x.unsqueeze(0).expand(mpc.N, -1).clone()
        u_lin_seq = torch.clamp(
            u_seq_guess.clone() + u_lin_delta,
            min=mpc.MPC_dynamics.u_min.unsqueeze(0),
            max=mpc.MPC_dynamics.u_max.unsqueeze(0),
        )

        extra_ctrl = (
            _build_energy_control_tau1(u_lin_seq, x, mpc, gates_E, E_q1_goal_rollout)
            if gates_E is not None else None
        )

        u_opt, U_opt_full = mpc.control(
            x,
            x_lin_seq,
            u_lin_seq,
            x_goal,
            diag_corrections_Q=gates_Q,
            diag_corrections_R=gates_R,
            extra_linear_control=extra_ctrl,
            Qf_dense=Qf_dense,
        )

        x_prev = x.clone()
        x = mpc.true_RK4_disc(x, u_opt, mpc.dt)

        u_hist[step] = u_opt.detach()
        x_hist[step + 1] = x.detach()

        U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
        u_seq_guess[:-1] = U_opt_reshaped[1:].clone()
        u_seq_guess[-1] = U_opt_reshaped[-1].clone()

        state_history.pop(0)
        state_history.append(x.detach().clone())

        if lin_net is not None:
            with torch.no_grad():
                delta = x.detach() - mpc.MPC_RK4_disc(x_prev, u_opt.detach(), mpc.dt)
            residual_history.pop(0)
            residual_history.append(delta)

    return x_hist, u_hist
