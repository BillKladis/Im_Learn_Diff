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

    u_min = mpc.MPC_dynamics.u_min
    u_max = mpc.MPC_dynamics.u_max

    step_losses = []
    for _ in range(num_steps):
        _, _, _, _, _, _, u_lin_delta = lin_net(
            torch.stack(state_history, dim=0),
            torch.stack(residual_history, dim=0),
            q_base_diag=mpc.q_base_diag,
            r_base_diag=mpc.r_base_diag,
        )

        x_lin_seq = current_state.unsqueeze(0).expand(mpc.N, -1).clone()
        # Match training: no energy shaping in QP, residual u_lin_delta on top
        u_mpc, U_opt_full = mpc.control(
            current_state, x_lin_seq, u_seq_guess, x_goal,
            diag_corrections_Q=None,
            diag_corrections_R=None,
            extra_linear_control=None,
            Qf_dense=None,
        )

        u_applied = torch.clamp(u_mpc.detach() + u_lin_delta[0], min=u_min, max=u_max)
        next_state = mpc.true_RK4_disc(current_state, u_applied, mpc.dt)
        step_losses.append(((next_state - x_goal) ** 2).sum())

        with torch.no_grad():
            delta = next_state.detach() - mpc.MPC_RK4_disc(current_state, u_mpc.detach(), mpc.dt)

        current_state = next_state.detach()
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
    lr: float = 2e-3,
    bptt_window: int = 10,
    e_pump_boost: float = 1.0,
    debug_monitor=None,
    recorder: Optional[network_module.NetworkOutputRecorder] = None,
    grad_debug: bool = False,
    grad_debug_every: int = 1,
) -> Tuple[List[float], network_module.NetworkOutputRecorder]:

    # ── Hyperparameters ───────────────────────────────────────────────────────
    # Truncated BPTT with energy-based loss.  With a 10-step window the
    # network can see that swinging *backward* with high velocity increases
    # kinetic energy → reduces energy deficit → lower loss.  This teaches
    # the pump rule without explicit supervision.
    #
    # QP has NO energy shaping — the network must learn it via u_lin_delta.
    BPTT_W      = bptt_window   # gradient window; N × 0.05s lookahead
    W_E_DIFF    = 8.0     # energy increase reward
    W_Q1        = 2.0     # position error toward π
    W_VEL       = 0.5     # velocity residual
    W_Q2_SHAPE  = 25.0    # q2² penalty — strong gradient to force τ2 compensation
    W_Q2_VEL    = 3.0     # q2_dot² penalty
    CLIP_GRAD   = 1.5
    STEP_LOSS_CLAMP = 50.0

    n_res     = lin_net.n_res
    state_dim = lin_net.state_dim
    n_u       = mpc.MPC_dynamics.u_min.shape[0]

    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5,
    )

    loss_history  = []
    best_goal_dist = float('inf')
    best_state_dict = None

    if recorder is None:
        recorder = network_module.NetworkOutputRecorder()

    E_q1_goal  = _compute_q1_energy(x_goal).detach()
    # Boosted goal keeps pump gradient alive near top (1.15x → 10x stronger at 170°)
    E_pump_goal = E_q1_goal * e_pump_boost
    u_min = mpc.MPC_dynamics.u_min
    u_max = mpc.MPC_dynamics.u_max

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        lin_net.train()
        optimizer.zero_grad()
        recorder.start_epoch()
        qp_fallback_start = int(getattr(mpc, 'qp_fallback_count', 0))

        x          = x0.detach().clone()
        state_hist = [x.clone() for _ in range(5)]
        zero_res   = torch.zeros(state_dim, device=mpc.device, dtype=torch.float64)
        res_hist   = [zero_res.clone() for _ in range(n_res)]
        u_seq_guess = torch.zeros((mpc.N, n_u), device=mpc.device, dtype=torch.float64)
        epoch_loss_sum  = 0.0
        n_windows_total = 0
        window_losses   = []
        u_hist_epoch    = []
        grad_stats      = None

        optimizer.zero_grad()  # clear at start

        for step in range(num_steps):
            # ── Network forward ───────────────────────────────────────────
            _, _, _, _, _, _, u_lin_delta = lin_net(
                torch.stack(state_hist, 0),
                torch.stack(res_hist,   0),
                q_base_diag=mpc.q_base_diag,
                r_base_diag=mpc.r_base_diag,
            )

            # ── QP: fixed costs, NO energy shaping ────────────────────────
            x_det = x.detach()
            x_lin = x_det.unsqueeze(0).expand(mpc.N, -1).clone()
            with torch.no_grad():
                u_mpc, U_opt_full = mpc.control(
                    x_det, x_lin, u_seq_guess, x_goal,
                    diag_corrections_Q=None,
                    diag_corrections_R=None,
                    extra_linear_control=None,
                    Qf_dense=None,
                )

            # ── Residual ──────────────────────────────────────────────────
            u_applied = torch.clamp(
                u_mpc.detach() + u_lin_delta[0],
                min=u_min, max=u_max,
            )

            # RK4: gradient flows within BPTT window
            x_next = mpc.true_RK4_disc(x, u_applied, mpc.dt)

            # ── Loss ──────────────────────────────────────────────────────
            # Energy-difference reward: penalise energy DECREASE when below
            # goal.  Gradient sign flips with velocity — if pushing τ1 slows
            # backward motion, kinetic energy falls → loss rises → network
            # learns to follow velocity sign (the pump rule).
            E_prev = _compute_q1_energy(x.detach())   # current KE+PE, no grad
            E_next_val = _compute_q1_energy(x_next)   # next step, grad flows
            deficit = torch.relu(E_pump_goal - E_prev).detach()
            # loss = -dE * deficit/scale  → minimised by increasing energy when deficit>0
            loss_pump = -W_E_DIFF * (E_next_val - E_prev) * deficit / (E_q1_goal.abs()**2 + 1.0)

            q1_err = torch.atan2(
                torch.sin(x_next[0] - x_goal[0]),
                torch.cos(x_next[0] - x_goal[0]),
            )
            err   = x_next - x_goal
            sloss = (loss_pump
                     + W_Q1  * q1_err**2
                     + W_VEL * err[1]**2
                     + W_Q2_SHAPE * x_next[2]**2
                     + W_Q2_VEL   * x_next[3]**2)
            window_losses.append(torch.clamp(sloss, max=STEP_LOSS_CLAMP))

            end_of_window = ((step + 1) % BPTT_W == 0) or (step == num_steps - 1)

            if end_of_window:
                # One backward + one optimizer step per window
                window_loss = torch.stack(window_losses).mean()
                epoch_loss_sum += window_loss.item()
                n_windows_total += 1
                window_loss.backward()
                if all(
                    (p.grad is None or torch.isfinite(p.grad).all())
                    for _, p in lin_net.named_parameters()
                ):
                    torch.nn.utils.clip_grad_norm_(lin_net.parameters(), max_norm=CLIP_GRAD)
                    optimizer.step()
                # Capture grad stats before clearing — last window of the epoch
                is_last_window = (step == num_steps - 1)
                if is_last_window and grad_debug and (
                    (epoch + 1) % max(1, grad_debug_every) == 0 or epoch == 0
                ):
                    grad_stats = _gradient_stats(lin_net)
                optimizer.zero_grad()
                window_losses = []
                x = x_next.detach()   # detach at window boundary
            else:
                x = x_next            # gradient flows within window

            # ── Bookkeeping ───────────────────────────────────────────────
            with torch.no_grad():
                delta = x.detach() - mpc.MPC_RK4_disc(x_det, u_mpc.detach(), mpc.dt)

            recorder.record_step(
                gates_Q=torch.ones(mpc.N-1, state_dim, device=mpc.device, dtype=torch.float64),
                gates_R=torch.ones(mpc.N,   n_u,        device=mpc.device, dtype=torch.float64),
                gates_E=torch.ones(mpc.N, device=mpc.device, dtype=torch.float64),
                q_diags=None, r_diags=None,
                u_mpc=u_applied,
                state_err=((x.detach() - x_goal)**2).sum(),
                residual_norm=delta.norm().item(),
                Qf_dense=None,
                u_lin_delta=u_lin_delta,
            )

            U_opt_r = U_opt_full.detach().view(mpc.N, n_u)
            u_seq_guess = torch.cat([U_opt_r[1:], U_opt_r[-1:]], 0).clone()
            state_hist.pop(0); state_hist.append(x.detach().clone())
            res_hist.pop(0);   res_hist.append(delta)

        avg_loss = epoch_loss_sum / max(n_windows_total, 1)
        loss_history.append(avg_loss)
        recorder.end_epoch(avg_loss)



        with torch.no_grad():
            goal_dist = float(torch.norm(x - x_goal).item())
        if goal_dist < best_goal_dist:
            best_goal_dist = goal_dist
            import copy
            best_state_dict = copy.deepcopy(lin_net.state_dict())

        # Optimizer steps already done inside the BPTT window loop above.
        # Just advance the LR scheduler with this epoch's average loss.
        scheduler.step(avg_loss)

        if debug_monitor:
            with torch.no_grad():
                summary = recorder.epoch_summary(epoch)
            qp_fallbacks_epoch = int(getattr(mpc, 'qp_fallback_count', 0)) - qp_fallback_start
            debug_monitor.log_epoch(epoch, num_epochs, avg_loss, {
                'epoch_time':      time.time() - epoch_start_time,
                'learning_rate':   optimizer.param_groups[0]['lr'],
                'loss_terminal':   avg_loss,
                'qp_fallbacks':    qp_fallbacks_epoch,
                'pure_end_error':  goal_dist,
                'mean_Q_gate_dev': 0.0,
                'mean_E_gate_dev': 0.0,
                'mean_u_lin_norm': summary.get('mean_u_lin_norm', float('nan')),
                'mean_qf_norm':    float('nan'),
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
    """
    Inference rollout — mirrors the residual training architecture exactly:
      u_applied = clamp(u_mpc_fixed_costs + u_lin_delta[0], limits)
    QP costs are fixed (no gate influence); only u_lin_delta adjusts the action.
    """
    n_x = x0.shape[0]
    n_u = mpc.MPC_dynamics.u_min.shape[0]

    x_hist = torch.zeros(num_steps + 1, n_x, dtype=torch.float64, device=mpc.device)
    u_hist = torch.zeros(num_steps, n_u, dtype=torch.float64, device=mpc.device)

    x = x0.clone().to(mpc.device)
    x_hist[0] = x
    u_seq_guess = torch.zeros((mpc.N, n_u), dtype=torch.float64, device=mpc.device)

    state_history = [x.clone() for _ in range(5)]
    u_min = mpc.MPC_dynamics.u_min
    u_max = mpc.MPC_dynamics.u_max

    if lin_net is not None:
        lin_net.eval()
        residual_history = [
            torch.zeros(lin_net.state_dim, device=mpc.device, dtype=torch.float64)
            for _ in range(lin_net.n_res)
        ]
    else:
        residual_history = None

    for step in range(num_steps):
        with torch.no_grad():
            if lin_net is not None:
                _, _, _, _, _, _, u_lin_delta = lin_net(
                    torch.stack(state_history, dim=0),
                    torch.stack(residual_history, dim=0),
                    q_base_diag=mpc.q_base_diag,
                    r_base_diag=mpc.r_base_diag,
                )
            else:
                u_lin_delta = torch.zeros((mpc.N, n_u), dtype=torch.float64, device=mpc.device)

            x_lin = x.unsqueeze(0).expand(mpc.N, -1).clone()
            u_opt, U_opt_full = mpc.control(
                x, x_lin, u_seq_guess, x_goal,
                diag_corrections_Q=None,
                diag_corrections_R=None,
                extra_linear_control=None,
                Qf_dense=None,
            )

            # Residual correction — same formula as training
            u_applied = torch.clamp(u_opt + u_lin_delta[0], min=u_min, max=u_max)

        x_prev = x.clone()
        x = mpc.true_RK4_disc(x, u_applied, mpc.dt)

        u_hist[step] = u_applied.detach()
        x_hist[step + 1] = x.detach()

        U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
        u_seq_guess[:-1] = U_opt_reshaped[1:].clone()
        u_seq_guess[-1]  = U_opt_reshaped[-1].clone()

        state_history.pop(0)
        state_history.append(x.detach().clone())

        if lin_net is not None:
            delta = x.detach() - mpc.MPC_RK4_disc(x_prev, u_opt.detach(), mpc.dt)
            residual_history.pop(0)
            residual_history.append(delta)

    return x_hist, u_hist
