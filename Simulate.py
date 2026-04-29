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
    track_mode:  str = "energy",
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
        gates_Q, gates_R, f_extra, _, _, gates_Qf = lin_net(
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
            diag_corrections_Qf=gates_Qf,
        )

        next_state = mpc.true_RK4_disc(current_state, u_mpc, mpc.dt)
        target_idx = min(t + 1, demo.shape[0] - 1)
        if track_mode == "energy":
            E_now = mpc.compute_energy_single(next_state)
            E_target = mpc.compute_energy_single(demo[target_idx])
            step_losses.append((E_now - E_target) ** 2)
        elif track_mode == "cos_q1":
            # Wrapped q1-angle + small velocity tracking.  Bounded [0, 4.2],
            # unique minimum at demo state — distinguishes upright from
            # spinning-bottom (same energy but different cos/sin).
            target = demo[target_idx]
            q1, q1d = next_state[0], next_state[1]
            q1_t, q1d_t = target[0], target[1]
            angle_err = (torch.cos(q1) - torch.cos(q1_t))**2 \
                      + (torch.sin(q1) - torch.sin(q1_t))**2
            vel_err = (q1d - q1d_t)**2 / 64.0   # normalised by 8^2
            step_losses.append(angle_err + 0.1 * vel_err)
        else:
            target = demo[target_idx]
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
    track_mode: str     = "energy",   # "state" (rigid Euclidean) or "energy" (scalar)
    w_terminal_anchor:  float = 0.0,  # ONLY at last step: small wrap(q1-π)² pull
    w_q1_gate_reg:      float = 0.0,  # penalty to keep q1 gate near q1_gate_reg_target
    q1_gate_reg_target: float = 0.01, # target q1 gate value when regularizing
    # Phase-conditional outer loss: during the first phase_split_frac portion
    # of the trajectory, penalise large q1 gate AND reward large |f_extra|.
    # Last (1-phase_split_frac) portion has no extra penalty (stabilisation).
    w_q1_phase_pen:     float = 0.0,
    w_f_phase_reward:   float = 0.0,
    phase_split_frac:   float = 0.6,
    # Optional Q-gate profile target during pump phase: explicit target
    # values for [q1, q1d, q2, q2d] gates.  ||gates_Q - target||² added to
    # loss with weight w_q_profile.  Set w_q_profile=0 to disable.
    w_q_profile:        float = 0.0,
    q_profile_pump:     Optional[List[float]] = None,    # default [0.01,1,1,1]
    q_profile_stable:   Optional[List[float]] = None,    # default [1,1,1,1]
    q_profile_state_phase: bool = False,   # blend pump↔stable from cos(q1)
    # Same idea for the network's Qf-head (terminal cost gates). Pump-phase
    # target should be small (don't penalise terminal state away from upright
    # while still pumping), stable-phase target should be ~1 (full Qf cost
    # to pin the upright). Requires the network to have gate_range_qf > 0.
    w_qf_profile:        float = 0.0,
    qf_profile_pump:     Optional[List[float]] = None,    # default [0.01,0.01,1,1]
    qf_profile_stable:   Optional[List[float]] = None,    # default [1,1,1,1]
    qf_profile_state_phase: bool = False,
    # End-phase Q-gate increase: in the last `end_phase_steps` steps,
    # add penalty (1 - gates_Q[:, 0])² + (1 - gates_Q[:, 1])² to push
    # q1 and q1d gates UP for stabilisation at the goal.
    w_end_q_high:       float = 0.0,
    end_phase_steps:    int   = 20,
    # Selective gradient training: detach gates_Q before passing to QP, so
    # the tracking loss gradient only trains f_head (and trunk/encoder via
    # f_head/r_head paths).  Q-head is then trained ONLY by the profile
    # penalty, which gives it a clean state-dependent supervision signal
    # without interference from the energy-tracking gradient trap.
    detach_gates_Q_for_qp: bool = False,
    # f_extra end-phase regularisation: in the last `f_end_reg_steps` steps,
    # penalise large feedforward output with w_f_end_reg * ||f_extra||².
    # Prevents the network from applying large pumping torques near the goal
    # ("overlearned energy pumping"), forcing the MPC to stabilise using its
    # own Q-weighted feedback instead of relying on f_extra.
    w_f_end_reg:          float = 0.0,
    f_end_reg_steps:      int   = 20,
    # State-conditional f_extra penalty: penalise ‖f_extra‖² weighted by
    # stable_zone(state) = near_goal * low_velocity, where
    #   near_goal     = (1 + cos(q1 - q1_goal)) / 2     ∈ [0, 1]
    #   low_velocity  = exp(-(q1d² + q2d²) / 2)         ∈ (0, 1]
    # The penalty is ZERO during energetic swing-up (low_velocity → 0) and
    # FULL when the pendulum has settled near the goal.  This teaches the
    # network to stop pumping ONLY in the stable zone — without breaking
    # the swing-up the way a time-window penalty did.
    # w_f_stable=0 → disabled (default).
    w_f_stable:           float = 0.0,
    # Stable-phase direct position tracking: in the last `stable_phase_steps`
    # steps, add a POSITION loss that directly drives the state toward the
    # goal (wrapped q1 error + normalised velocities/q2).  This is stronger
    # than energy-only tracking for stabilisation — energy can be correct even
    # when the pendulum is swinging through the goal rather than staying there.
    # w_stable_phase=0 → disabled (default).
    w_stable_phase:       float = 0.0,
    stable_phase_steps:   int   = 30,
    # Noise injection during training: add Gaussian noise to the state
    # observations the network sees (state_history).  This is data
    # augmentation that makes the trained model robust to sensor noise
    # at deployment time.  Pass a list/tensor of 4 σ values (one per
    # state dim).  Default: zeros = no noise.
    train_noise_sigma:    Optional[List[float]] = None,
    # Late-phase track penalty: in the last `track_late_phase_steps` steps,
    # add w_track_late_phase * track_step ON TOP of the normal track loss.
    # Heavily emphasises tracking near goal arrival — encourages the network
    # to raise Q/Qf gates so the QP locks the rollout to the demo trajectory
    # late. Use with a REALISTIC demo (states from a known-good rollout).
    w_track_late_phase:   float = 0.0,
    track_late_phase_steps: int = 50,
    # Optional initial state history.  Default behaviour: state_history is
    # initialised to 5 copies of x0 (with optional training noise on each).
    # Pass a (5, 4) tensor to seed the network with an in-distribution
    # 5-frame history — useful when x0 is picked from somewhere along a
    # successful trajectory (curriculum learning).  init_history[-1]
    # should equal x0 (the most recent frame is the current state).
    init_history:         Optional[torch.Tensor] = None,
    # Early-stop patience: epochs without best_goal_dist improvement
    # before stopping. Default 15 matches the original heuristic.
    early_stop_patience:  int  = 15,
    # Optional external optimizer.  Default: a fresh AdamW is created each
    # call (correct for one-shot training runs).  Curriculum experiments
    # that call this function many times with num_epochs=1 MUST pass an
    # external optimizer — otherwise Adam's momentum (m, v) is reset every
    # call and the per-iter gradient steps act as cold-start updates with
    # no adaptation, producing essentially no cumulative learning.
    external_optimizer:   Optional[torch.optim.Optimizer] = None,
    # Whether to restore best_state_dict at function exit.  Default True
    # is correct for one-shot training (returns the model from the best
    # epoch). MUST be set to False for curriculum-style use:
    # best_state_dict is captured BEFORE optimizer.step() each epoch, so
    # with num_epochs=1 the function would compute gradients, step the
    # optimizer, and then ROLL BACK to pre-step weights — making every
    # such call a no-op for the network parameters.
    restore_best:         bool = True,
) -> Tuple[List[float], network_module.NetworkOutputRecorder]:

    # ── Loss weights ──────────────────────────────────────────────────────
    # State-mode: dominant track term that compares full state to demo[t+1].
    # Energy-mode: dominant single-scalar energy track (no state mismatch).
    W_TRACK         = 5.0
    W_TERMINAL      = 0.0
    W_Q2_SHAPE      = 0.0
    W_Q2_DOT        = 0.0
    STEP_LOSS_CLAMP = 200.0
    # Terminal-anchor strength.  Energy tracking alone leaves position
    # under-constrained — many trajectories with the right energy curve
    # do NOT end at q1=π.  A small final-step-only wrap(q1-π)² pull
    # stabilises which "correct-energy" trajectory the network converges
    # to.  This is one extra term, not a generic stack.
    W_FINAL_ANCHOR  = float(w_terminal_anchor)

    SKIP_UPDATE_GRAD_NORM = 5e7
    CLIP_OTHER = 2.0

    n_u = mpc.MPC_dynamics.u_min.shape[0]
    demo_T = demo.shape[0]   # number of demo states available

    # Build Q-profile target tensors (used by phase-conditional outer loss).
    if q_profile_pump is None:
        q_profile_pump = [0.01, 1.0, 1.0, 1.0]
    if q_profile_stable is None:
        q_profile_stable = [1.0, 1.0, 1.0, 1.0]
    q_profile_pump_t = torch.tensor(q_profile_pump, device=mpc.device, dtype=torch.float64)
    q_profile_stable_t = torch.tensor(q_profile_stable, device=mpc.device, dtype=torch.float64)
    if qf_profile_pump is None:
        qf_profile_pump = [0.01, 0.01, 1.0, 1.0]
    if qf_profile_stable is None:
        qf_profile_stable = [1.0, 1.0, 1.0, 1.0]
    qf_profile_pump_t   = torch.tensor(qf_profile_pump,   device=mpc.device, dtype=torch.float64)
    qf_profile_stable_t = torch.tensor(qf_profile_stable, device=mpc.device, dtype=torch.float64)

    # Precompute demo's energy curve once (used by track_mode == "energy").
    with torch.no_grad():
        E_demo = torch.stack([mpc.compute_energy_single(demo[i]) for i in range(demo_T)])
    # Energy track loss is in (Joules)² which is much larger numerically
    # than (rad)² for state tracking — rescale by 1/E_range² so the
    # gradient is comparable in magnitude regardless of units.
    E_range = (E_demo.max() - E_demo.min()).clamp(min=1.0)

    if external_optimizer is not None:
        optimizer = external_optimizer
    else:
        optimizer = torch.optim.AdamW(lin_net.parameters(), lr=lr, weight_decay=1e-4)
    # Constant LR (cosine annealing was tested and didn't prevent the
    # post-swing-up drift — it just slowed it).  Early stopping on
    # best-goal-dist patience is the actual fix.
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=max(num_epochs, 1),
    )

    # Early-stopping bookkeeping: stop when best_goal_dist hasn't improved
    # for `patience` consecutive epochs.  Energy-tracking imitation finds
    # the swing-up basin in ~15-20 epochs and then drifts because the
    # loss landscape is path-degenerate; without early stopping the
    # final state is a different (worse) energy-matching trajectory.
    EARLY_STOP_PATIENCE = early_stop_patience
    epochs_since_improvement = 0

    loss_history    = []
    best_goal_dist  = float("inf")
    best_state_dict = None

    if recorder is None:
        recorder = network_module.NetworkOutputRecorder()

    # Build noise tensor for training-time noise injection (if any)
    if train_noise_sigma is not None and any(s > 0 for s in train_noise_sigma):
        train_noise_tensor = torch.tensor(
            train_noise_sigma, device=mpc.device, dtype=torch.float64,
        )
    else:
        train_noise_tensor = None

    def add_train_noise(state):
        if train_noise_tensor is None:
            return state.clone()
        return state + torch.randn_like(state) * train_noise_tensor

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        lin_net.train()
        optimizer.zero_grad()
        recorder.start_epoch()
        qp_fallback_start = int(getattr(mpc, "qp_fallback_count", 0))

        current_state_detached = x0.detach().clone()
        if init_history is not None:
            # Seed with the actual previous 5 states from a reference trajectory.
            # Each frame is detached (no grad) and not re-noised; init_history
            # is what the network would have observed at this point in a real
            # rollout. We do clone to avoid aliasing.
            state_history = [init_history[i].detach().clone()
                             for i in range(init_history.shape[0])]
        else:
            state_history = [add_train_noise(current_state_detached).detach() for _ in range(5)]
        u_seq_guess = torch.zeros((mpc.N, n_u), device=mpc.device, dtype=torch.float64)

        track_step_terms    = []
        terminal_step_terms = []
        q2_step_terms       = []
        q1_gate_reg_terms   = []
        phase_pen_terms     = []
        last_anchor_term    = torch.tensor(0.0, device=mpc.device, dtype=torch.float64)
        phase_split_step    = int(phase_split_frac * num_steps)
        f_extra_max_norm    = math.sqrt(mpc.N * n_u) * lin_net.f_extra_bound

        for step in range(num_steps):
            state_history_seq = torch.stack(state_history, dim=0)

            gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf = lin_net(
                state_history_seq,
                q_base_diag=mpc.q_base_diag,
                r_base_diag=mpc.r_base_diag,
            )

            # Q1-gate regularization: penalise q1 gate for deviating above
            # q1_gate_reg_target.  Only fires when w_q1_gate_reg > 0.
            if w_q1_gate_reg > 0.0:
                q1_dev = (gates_Q[:, 0].mean() - q1_gate_reg_target).clamp(min=0.0)
                q1_gate_reg_terms.append(q1_dev ** 2)

            # Phase-conditional outer loss: during pumping phase (first
            # phase_split_frac of trajectory) push q1 gate down and |f_extra|
            # up.  This directly counteracts the gradient trap by exposing
            # network outputs to the outer loss with phase-aware targets.
            if (w_q1_phase_pen > 0.0 or w_f_phase_reward > 0.0) and step < phase_split_step:
                q1_pen = gates_Q[:, 0].mean() ** 2
                # Reward: penalty for ||f_extra|| being below max possible.
                f_norm = torch.sqrt((f_extra ** 2).sum() + 1e-12)
                f_short = (f_extra_max_norm - f_norm).clamp(min=0.0) / f_extra_max_norm
                phase_pen_terms.append(
                    w_q1_phase_pen   * q1_pen
                    + w_f_phase_reward * (f_short ** 2)
                )

            # Q-gate profile target: explicit per-dim target for gates_Q.
            # During pump phase target is q_profile_pump (e.g. [0.01,0.01,1,1]),
            # during stabilise phase target is q_profile_stable ([1,1,1,1]).
            # If `state_phase=True`: blend factor uses state's cos(q1-q1_goal)
            # (0 at upright = stable, 1 far from upright = pump).
            if w_q_profile > 0.0:
                if q_profile_state_phase:
                    near_goal = (1.0 + torch.cos(current_state_detached[0] - x_goal[0])) / 2.0
                    near_goal = torch.clamp(near_goal, 0.0, 1.0)
                    target = (1.0 - near_goal) * q_profile_pump_t + near_goal * q_profile_stable_t
                else:
                    target = q_profile_pump_t if step < phase_split_step else q_profile_stable_t
                profile_dev = ((gates_Q - target.unsqueeze(0)) ** 2).mean()
                phase_pen_terms.append(w_q_profile * profile_dev)

            # Qf-gate profile target: same idea as q_profile but for the
            # network's terminal-cost head. Only active when the network has
            # gate_range_qf > 0 (so gates_Qf is not None).
            if w_qf_profile > 0.0 and gates_Qf is not None:
                if qf_profile_state_phase:
                    near_goal_qf = (1.0 + torch.cos(current_state_detached[0] - x_goal[0])) / 2.0
                    near_goal_qf = torch.clamp(near_goal_qf, 0.0, 1.0)
                    target_qf = (1.0 - near_goal_qf) * qf_profile_pump_t + near_goal_qf * qf_profile_stable_t
                else:
                    target_qf = qf_profile_pump_t if step < phase_split_step else qf_profile_stable_t
                qf_profile_dev = ((gates_Qf - target_qf) ** 2).mean()
                phase_pen_terms.append(w_qf_profile * qf_profile_dev)

            # End-phase Q-gate increase: in the last `end_phase_steps` steps,
            # explicitly push q1 and q1d gates UP (toward 1.0) regardless of
            # state.  This drives stabilisation by activating the QP's q1
            # cost near the end, tightening the final position to π.
            # Designed to ADD to (not replace) the state-based profile target
            # so it stacks: gates get pulled up only when this fires.
            if w_end_q_high > 0.0 and step >= num_steps - end_phase_steps:
                end_dev = ((1.0 - gates_Q[:, 0]) ** 2).mean() + \
                          ((1.0 - gates_Q[:, 1]) ** 2).mean()
                phase_pen_terms.append(w_end_q_high * end_dev)

            # f_extra end-phase regularisation: suppress feedforward output
            # when close to end of trajectory.  Prevents the network from
            # issuing large pumping torques at the goal (overlearned pumping).
            if w_f_end_reg > 0.0 and step >= num_steps - f_end_reg_steps:
                f_reg = w_f_end_reg * (f_extra ** 2).mean()
                phase_pen_terms.append(f_reg)

            # State-conditional f_extra penalty: stable_zone(state) gates
            # the penalty so it only fires near the goal AND at low velocity.
            # Verified by inference-time test (verify_smart_gate.py): the
            # corresponding output massage takes the 0.0612 model from
            # wrapped_dist=5.5 at 600 steps to 0.077.
            if w_f_stable > 0.0:
                q1_d  = current_state_detached[0]
                q1d_d = current_state_detached[1]
                q2d_d = current_state_detached[3]
                near_goal = (1.0 + torch.cos(q1_d - x_goal[0])) / 2.0
                low_vel   = torch.exp(-(q1d_d ** 2 + q2d_d ** 2) / 2.0)
                stable_zone = torch.clamp(near_goal * low_vel, 0.0, 1.0)
                f_stable_pen = w_f_stable * stable_zone * (f_extra ** 2).mean()
                phase_pen_terms.append(f_stable_pen)

            # Stable-phase direct position tracking: explicit state-to-goal
            # loss for the last N steps.  Uses wrapped q1 error so the loss
            # has a valid gradient through ±π.  Velocities normalised so they
            # contribute equally.  Together with f_end_reg this teaches the
            # network to HOLD the position, not just reach it.
            if w_stable_phase > 0.0 and step >= num_steps - stable_phase_steps:
                pass  # applied after next_state is computed below

            x_lin_seq = current_state_detached.unsqueeze(0).expand(mpc.N, -1).clone()
            u_lin_seq = torch.clamp(
                u_seq_guess.clone(),
                min=mpc.MPC_dynamics.u_min.unsqueeze(0),
                max=mpc.MPC_dynamics.u_max.unsqueeze(0),
            )

            extra_ctrl = f_extra.reshape(-1)

            # Optionally detach gates_Q so QP gradient doesn't flow back to
            # q_head (which is trained only by the profile penalty).
            gates_Q_qp = gates_Q.detach() if detach_gates_Q_for_qp else gates_Q

            u_mpc, U_opt_full = mpc.control(
                current_state_detached, x_lin_seq, u_lin_seq, x_goal,
                diag_corrections_Q=gates_Q_qp,
                diag_corrections_R=gates_R,
                extra_linear_control=extra_ctrl,
                diag_corrections_Qf=gates_Qf,
            )

            next_state = mpc.true_RK4_disc(current_state_detached, u_mpc, mpc.dt)

            # ── Tracking term (the main signal) ──────────────────────────
            target_idx = min(step + 1, demo_T - 1)
            target = demo[target_idx]
            if track_mode == "energy":
                # Match the demo's energy curve at this time index. Energy
                # is monotone over the swing-up (-14.7 → +14.7) so this
                # gives a clean scalar progress signal whose gradient
                # ∂E/∂q̇ is nonzero whenever there is motion — exactly the
                # τ·q̇ pumping signal.  Normalised by demo energy range so
                # numerical scale matches state tracking.
                E_now = mpc.compute_energy_single(next_state)
                track_step = ((E_now - E_demo[target_idx]) / E_range) ** 2
            elif track_mode == "phase_aware":
                # Pump phase: energy tracking (smooth gradient for pumping).
                # Stable phase: WRAPPED q1 angle² + small velocity damping.
                # The wrapped angle has non-vanishing gradient everywhere
                # except at the goal (q1=π), unlike (cos(q1)+1)² which has
                # zero gradient at q1=0.  This drives the pendulum reliably
                # toward upright in the stable phase.
                if step < phase_split_step:
                    E_now = mpc.compute_energy_single(next_state)
                    track_step = ((E_now - E_demo[target_idx]) / E_range) ** 2
                else:
                    q1, q1d = next_state[0], next_state[1]
                    q2, q2d = next_state[2], next_state[3]
                    q1_err_w = torch.atan2(
                        torch.sin(q1 - x_goal[0]),
                        torch.cos(q1 - x_goal[0]),
                    )
                    track_step = q1_err_w ** 2 + 0.1 * (q1d ** 2 + q2 ** 2 + q2d ** 2) / 64.0
            elif track_mode == "cos_q1":
                # Wrapped q1-angle tracking: bounded [0, 4.2], unique
                # minimum at demo state.  Breaks the spinning degeneracy
                # of pure energy tracking (spinning ≠ upright in cos/sin).
                q1, q1d = next_state[0], next_state[1]
                q1_t, q1d_t = target[0], target[1]
                angle_err = (torch.cos(q1) - torch.cos(q1_t))**2 \
                          + (torch.sin(q1) - torch.sin(q1_t))**2
                vel_err = (q1d - q1d_t)**2 / 64.0
                track_step = angle_err + 0.1 * vel_err
            else:
                track_step = ((next_state - target) ** 2).sum()
            track_step = torch.clamp(track_step, max=STEP_LOSS_CLAMP)
            track_step_terms.append(track_step)

            # Late-phase track penalty: in the last `track_late_phase_steps`
            # steps, add an EXTRA copy of the track loss scaled by
            # w_track_late_phase. Heavily emphasises matching the energy/
            # state target near goal arrival — pressures the network to
            # raise Q/Qf gates so the QP enforces tight tracking late.
            if w_track_late_phase > 0.0 and step >= num_steps - track_late_phase_steps:
                phase_pen_terms.append(w_track_late_phase * track_step)

            # ── Stable-phase direct position tracking ────────────────────
            if w_stable_phase > 0.0 and step >= num_steps - stable_phase_steps:
                q1s, q1ds = next_state[0], next_state[1]
                q2s, q2ds = next_state[2], next_state[3]
                q1_err_s = torch.atan2(
                    torch.sin(q1s - x_goal[0]),
                    torch.cos(q1s - x_goal[0]),
                )
                stable_loss = w_stable_phase * (
                    q1_err_s ** 2 +
                    (q1ds / 8.0) ** 2 +
                    (q2s / math.pi) ** 2 +
                    (q2ds / 8.0) ** 2
                )
                phase_pen_terms.append(stable_loss)

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

            # Capture the FINAL-step's wrapped q1 anchor while next_state
            # is still differentiable (it's about to be detached below).
            if step == num_steps - 1 and W_FINAL_ANCHOR > 0.0:
                q1_wrap = torch.atan2(
                    torch.sin(next_state[0] - x_goal[0]),
                    torch.cos(next_state[0] - x_goal[0]),
                )
                last_anchor_term = q1_wrap ** 2 + 0.1 * next_state[1] ** 2

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
            # Inject training noise into the observation if configured.
            state_history.append(add_train_noise(current_state_detached).detach().clone())

        # ── Combine ──────────────────────────────────────────────────────
        track_loss    = torch.stack(track_step_terms).sum()    / num_steps
        terminal_loss = torch.stack(terminal_step_terms).sum() / num_steps
        q2_loss       = torch.stack(q2_step_terms).sum()       / num_steps

        anchor_loss = last_anchor_term  # captured during the final step

        q1_gate_reg_loss = (
            torch.stack(q1_gate_reg_terms).mean()
            if q1_gate_reg_terms else
            torch.tensor(0.0, device=mpc.device, dtype=torch.float64)
        )
        phase_pen_loss = (
            torch.stack(phase_pen_terms).mean()
            if phase_pen_terms else
            torch.tensor(0.0, device=mpc.device, dtype=torch.float64)
        )

        total_loss = (
            W_TRACK    * track_loss
            + W_TERMINAL * terminal_loss
            + W_FINAL_ANCHOR * anchor_loss
            + q2_loss
            + w_q1_gate_reg * q1_gate_reg_loss
            + phase_pen_loss
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
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

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
                "loss_terminal":      float(anchor_loss.detach().item()) if W_FINAL_ANCHOR > 0 else terminal_loss.item(),
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

        # Early stop: best goal-dist hasn't improved for too many epochs
        # AND we've already found a near-goal solution worth keeping.
        if (
            epochs_since_improvement >= EARLY_STOP_PATIENCE
            and best_goal_dist < 1.0
        ):
            print(
                f"      EarlyStop after epoch {epoch+1}: "
                f"best_goal_dist={best_goal_dist:.4f} hasn't improved "
                f"for {epochs_since_improvement} epochs."
            )
            break

    if restore_best and best_state_dict is not None:
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
                gates_Q, gates_R, f_extra, _, _, _ = lin_net(
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