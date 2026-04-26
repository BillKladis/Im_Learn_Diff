import time
import math
import mpc_controller
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

import lin_net as network_module


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
        u_mpc, U_opt_full = mpc.control(
            current_state, x_lin_seq, u_lin_seq, x_goal,
            diag_corrections_Q=gates_Q,
            diag_corrections_R=gates_R,
            gates_E=gates_E,
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

    W_TERMINAL = 1.0
    W_ENERGY = 1.5      # energy-deficit loss (replaces disabled control-effort loss)
    W_PUMP = 5.0        # stronger pump reward
    W_WAYPOINT = 2.0    # reward for passing through q1=π/2 waypoint (fades over training)
    PUMP_WARMUP_EPOCHS = 2
    W_QF_ANCHOR = 1e-3
    W_U_LIN_IMITATION = 0.05  # supervised loss: u_lin_head predicts MPC output
    STEP_LOSS_CLAMP = 200.0
    CLIP_QF_HEAD = 5.0
    CLIP_U_LIN = 2.0
    CLIP_OTHER = 2.0
    SKIP_UPDATE_GRAD_NORM = 5e7

    # Phase-aware curriculum boundaries (fraction of num_steps) — base values,
    # shrink epoch-by-epoch as the network learns to build energy faster.
    ENERGY_PHASE_END_BASE  = 0.50
    POSITION_PHASE_IN_BASE = 0.65

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

    E_goal_det = mpc.compute_energy_single(x_goal).detach()
    E_bottom   = mpc.compute_energy_single(x0).detach()
    E_span     = (E_goal_det - E_bottom).abs() + 1.0  # normalisation denominator

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        lin_net.train()
        optimizer.zero_grad()
        recorder.start_epoch()
        qp_fallback_start = int(getattr(mpc, 'qp_fallback_count', 0))

        # ── Epoch-level curriculum variables ─────────────────────────────
        epoch_frac = epoch / max(num_epochs - 1, 1)

        # Energy curriculum: ramp target from 40 % → 100 % of goal energy.
        # Starting with a reachable target makes the deficit gradient locally
        # convex before the full upright energy is demanded.
        E_target_frac = min(1.0, 0.4 + 0.6 * epoch_frac)
        E_goal_curr   = (E_bottom + E_target_frac * (E_goal_det - E_bottom)).detach()

        # Shrink energy-only phase as training progresses (network gets faster
        # at building energy so can afford to weight position sooner).
        energy_phase_end  = max(0.20, ENERGY_PHASE_END_BASE  - 0.25 * epoch_frac)
        position_phase_in = max(0.35, POSITION_PHASE_IN_BASE - 0.20 * epoch_frac)

        # Waypoint weight fades to 0 after the first 60 % of training.
        waypoint_weight = W_WAYPOINT * max(0.0, 1.0 - epoch_frac / 0.6)
        # ─────────────────────────────────────────────────────────────────

        current_state_detached = x0.detach().clone()
        state_history = [current_state_detached.clone() for _ in range(5)]

        zero_residual = torch.zeros(state_dim, device=mpc.device, dtype=torch.float64)
        residual_history = [zero_residual.clone() for _ in range(n_res)]

        u_seq_guess = torch.zeros((mpc.N, n_u), device=mpc.device, dtype=torch.float64)
        step_losses = []
        energy_terms = []
        pump_rewards = []
        waypoint_terms = []
        qf_anchor_terms = []
        u_lin_imitation_terms = []

        E_prev = mpc.compute_energy_single(current_state_detached).detach()

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

            u_mpc, U_opt_full = mpc.control(
                current_state_detached,
                x_lin_seq,
                u_lin_seq,
                x_goal,
                diag_corrections_Q=gates_Q,
                diag_corrections_R=gates_R,
                gates_E=gates_E,
                Qf_dense=Qf_dense,
            )

            # Imitation loss: train u_lin_head to predict the first MPC action.
            # Gradient is simple (no QP), preventing the warm-start explosion.
            u_lin_imitation_terms.append(
                ((u_lin_delta[0] - u_mpc.detach()) ** 2).mean()
            )

            next_state = mpc.true_RK4_disc(current_state_detached, u_mpc, mpc.dt)

            # ── Cosine position loss — smooth, bounded, no wrap discontinuity ──
            # (1-cos(Δ)) ∈ [0,2] has a single global min at Δ=0 and is C∞.
            # Avoids the cusp that atan2²  can have when Δ crosses ±π.
            err = next_state - x_goal
            q1_cost = 1.0 - torch.cos(next_state[0] - x_goal[0])
            q2_cost = 1.0 - torch.cos(next_state[2] - x_goal[2])
            step_state_err = 3.0 * q1_cost + err[1]**2 + 3.0 * q2_cost + err[3]**2

            # ── Phase weight: suppress position cost while energy still building ─
            t_frac = step / max(num_steps - 1, 1)
            if t_frac < energy_phase_end:
                pos_w = 0.05
            elif t_frac < position_phase_in:
                pos_w = 0.05 + 0.95 * (t_frac - energy_phase_end) / (position_phase_in - energy_phase_end)
            else:
                pos_w = 1.0

            step_losses.append(torch.clamp(pos_w * step_state_err, max=STEP_LOSS_CLAMP))

            # ── Waypoint: Gaussian reward centred at q1=π/2 (arm horizontal) ──
            # Breaks the non-convex leap 0→π into two easier sub-goals.
            # Fades out once the network consistently passes through it.
            waypoint_terms.append(torch.exp(-((next_state[0] - math.pi / 2) ** 2) / 0.4))

            qf_anchor_terms.append(((Qf_dense - mpc.Qf) ** 2).mean())

            # Energy deficit: use curriculum target (locally convex, gentler gradient).
            # Pump reward: always gate on the full goal so we never stop pumping.
            E_next = mpc.compute_energy_single(next_state)
            deficit_next = torch.relu(E_goal_curr - E_next) / E_span
            energy_terms.append(deficit_next ** 2)

            with torch.no_grad():
                deficit_w = torch.relu(E_goal_det - E_prev) / E_span
            pump_rewards.append(deficit_w * (E_next - E_prev))
            E_prev = E_next.detach()

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

        terminal_loss    = torch.stack(step_losses).sum() / num_steps
        energy_loss      = torch.stack(energy_terms).sum() / num_steps
        pump_loss        = -torch.stack(pump_rewards).sum() / num_steps
        waypoint_loss    = -torch.stack(waypoint_terms).sum() / num_steps  # negative = reward
        qf_anchor_loss   = torch.stack(qf_anchor_terms).mean()
        u_lin_imitation_loss = torch.stack(u_lin_imitation_terms).mean()

        pump_weight = W_PUMP * min(1.0, float(epoch + 1) / float(PUMP_WARMUP_EPOCHS))
        total_loss = (
            W_TERMINAL * terminal_loss
            + W_ENERGY * energy_loss
            + pump_weight * pump_loss
            + waypoint_weight * waypoint_loss
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
                'loss_pump': pump_loss.item(),
                'loss_waypoint': waypoint_loss.item(),
                'loss_qf_anchor': qf_anchor_loss.item(),
                'pump_weight': pump_weight,
                'waypoint_weight': waypoint_weight,
                'e_target_frac': E_target_frac,
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
    state_history = [x.clone() for _ in range(5)]

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

        u_opt, U_opt_full = mpc.control(
            x,
            x_lin_seq,
            u_lin_seq,
            x_goal,
            diag_corrections_Q=gates_Q,
            diag_corrections_R=gates_R,
            gates_E=gates_E,
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
