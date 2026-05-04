"""eval_comprehensive.py — Comprehensive robustness evaluation.

Tests both v14m (single-start champion) and v15 BESTMEAN (generalized).

Tests:
  1. Long 2000-step clean rollout — oscillation amplitude, stability
  2. Near-top balance starts: [π±δ, 0, 0, 0] for δ in {0.1, 0.3, 0.5, 1.0}
  3. Mid-rollout impulse disturbance (kick at arrival + 50 steps)
  4. Observation noise: Gaussian σ=0.05 on state before network
  5. Control noise: Gaussian σ=0.1 on u* before dynamics
  6. Combined obs+ctrl noise
"""

import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")
DT     = 0.05
N      = 10

X0_FIXED   = [0.0, 0.0, 0.0, 0.0]
X_GOAL_VAL = [math.pi, 0.0, 0.0, 0.0]

CHECKPOINTS = {
    "v14m": "saved_models/stageF_mixed_v14m_20260503_102608_ep50/stageF_mixed_v14m_20260503_102608_ep50.pth",
    "v15bm": "saved_models/stageF_gen_v15_BESTMEAN_20260503_225753_ep150/stageF_gen_v15_BESTMEAN_20260503_225753_ep150.pth",
}

MODEL_KWARGS = dict(
    state_dim=4, control_dim=2, horizon=10, hidden_dim=128,
    gate_range_q=0.99, gate_range_r=0.20, f_extra_bound=2.5, f_kickstart_amp=1.0,
)

WRAP_THRESH  = 0.10
ARR_THRESH   = 0.30

# Noise levels
OBS_SIGMA  = 0.05   # Gaussian obs noise std (added to state before network)
CTRL_SIGMA = 0.10   # Gaussian ctrl noise std (added to u_opt before dynamics)

# Disturbance: impulse kick magnitudes
KICK_Q1D   = 1.5    # rad/s impulse on q1_dot after arrival
KICK_Q2D   = 1.0    # rad/s impulse on q2_dot after arrival

SEED = 42


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    data = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = data.get("model_state_dict", data)
    model = network_module.SeparatedLinearizationNetwork(**MODEL_KWARGS).double()
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── Custom rollout with noise and disturbance support ─────────────────────────
def rollout_custom(
    model, mpc, x0, x_goal, num_steps,
    obs_sigma=0.0,    # Gaussian std added to state before network sees it
    ctrl_sigma=0.0,   # Gaussian std added to u* before dynamics
    kick_at_step=None, kick_delta=None,  # state += kick_delta at kick_at_step
    f_gate_thresh=0.0,
):
    """Rollout with optional obs/ctrl noise and mid-trajectory state kicks."""
    rng = torch.Generator(device=DEVICE)
    rng.manual_seed(SEED)

    n_x = x0.shape[0]
    n_u = mpc.MPC_dynamics.u_min.shape[0]

    x_hist = torch.zeros(num_steps + 1, n_x, dtype=torch.float64, device=DEVICE)
    u_hist = torch.zeros(num_steps,     n_u, dtype=torch.float64, device=DEVICE)

    x = x0.clone().to(DEVICE)
    x_hist[0] = x
    u_seq_guess = torch.zeros((mpc.N, n_u), dtype=torch.float64, device=DEVICE)

    init_q1 = float(x[0].item())
    if abs(init_q1) > 0.01 and abs(abs(init_q1) - math.pi) > 0.01:
        gravity_torque = 2.0 * 9.81 * 0.5 * abs(math.sin(init_q1))
        wrapped_err = math.atan2(
            math.sin(float(x_goal[0].item()) - init_q1),
            math.cos(float(x_goal[0].item()) - init_q1),
        )
        goal_sign = 1.0 if wrapped_err > 0 else -1.0
        seed_tau1 = goal_sign * min(float(mpc.MPC_dynamics.u_max[0].item()), gravity_torque * 2.0)
        u_seq_guess[:, 0] = seed_tau1

    state_history = [x.clone() for _ in range(5)]
    model.eval()

    for step in range(num_steps):
        # Optional impulse kick
        if kick_at_step is not None and step == kick_at_step and kick_delta is not None:
            x = x + kick_delta.to(DEVICE)
            state_history[-1] = x.clone()

        with torch.no_grad():
            # Observation noise: noisy copy seen by network
            x_noisy = x.clone()
            if obs_sigma > 0.0:
                x_noisy = x + torch.randn(n_x, generator=rng, dtype=torch.float64, device=DEVICE) * obs_sigma
            hist_noisy = [s.clone() for s in state_history]
            if obs_sigma > 0.0:
                hist_noisy = [s + torch.randn(n_x, generator=rng, dtype=torch.float64, device=DEVICE) * obs_sigma
                              for s in state_history]

            gates_Q, gates_R, f_extra, _, _, gates_Qf = model(
                torch.stack(hist_noisy, dim=0),
                q_base_diag=mpc.q_base_diag,
                r_base_diag=mpc.r_base_diag,
            )

        if f_gate_thresh > 0.0:
            _near_pi = (1.0 + torch.cos(x[0] - x_goal[0])) / 2.0
            _zf = ((_near_pi - f_gate_thresh) / max(1e-8, 1.0 - f_gate_thresh)).clamp(0.0, 1.0)
            f_extra = f_extra * (1.0 - _zf)

        x_lin_seq = x.unsqueeze(0).expand(mpc.N, -1).clone()
        u_lin_seq = torch.clamp(
            u_seq_guess.clone(),
            min=mpc.MPC_dynamics.u_min.unsqueeze(0),
            max=mpc.MPC_dynamics.u_max.unsqueeze(0),
        )

        u_opt, U_opt_full = mpc.control(
            x, x_lin_seq, u_lin_seq, x_goal,
            diag_corrections_Q=gates_Q,
            diag_corrections_R=gates_R,
            extra_linear_control=f_extra.reshape(-1),
            diag_corrections_Qf=gates_Qf,
        )

        # Control noise
        if ctrl_sigma > 0.0:
            with torch.no_grad():
                u_opt = u_opt + torch.randn(n_u, generator=rng, dtype=torch.float64, device=DEVICE) * ctrl_sigma
                u_opt = torch.clamp(u_opt,
                                    min=mpc.MPC_dynamics.u_min,
                                    max=mpc.MPC_dynamics.u_max)

        x = mpc.true_RK4_disc(x, u_opt, mpc.dt)

        u_hist[step]     = u_opt.detach()
        x_hist[step + 1] = x.detach()

        U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
        u_seq_guess[:-1] = U_opt_reshaped[1:].clone()
        u_seq_guess[-1]  = U_opt_reshaped[-1].clone()

        state_history.pop(0)
        state_history.append(x.detach().clone())

    return x_hist.detach(), u_hist.detach()


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(x_hist, x_goal, wrap_thresh=WRAP_THRESH, arr_thresh=ARR_THRESH):
    traj = x_hist.cpu().numpy()
    n    = len(traj)
    goal = float(x_goal[0].item())

    wraps = np.array([
        math.sqrt(
            math.atan2(math.sin(s[0] - goal), math.cos(s[0] - goal)) ** 2
            + s[1] ** 2 + s[2] ** 2 + s[3] ** 2
        )
        for s in traj
    ])

    arr  = next((i for i, w in enumerate(wraps) if w < arr_thresh), None)
    post = float((wraps[arr:] < wrap_thresh).mean()) if arr is not None else 0.0
    f01  = float((wraps < wrap_thresh).mean())

    # Oscillation stats: after arrival, measure q1 deviation from π
    if arr is not None:
        post_q1_err = np.array([
            math.atan2(math.sin(s[0] - goal), math.cos(s[0] - goal))
            for s in traj[arr:]
        ])
        osc_std = float(np.std(post_q1_err))
        osc_max = float(np.max(np.abs(post_q1_err)))
        post_wrap = wraps[arr:]
        wrap_std  = float(np.std(post_wrap))
        wrap_max  = float(np.max(post_wrap))
    else:
        osc_std = osc_max = wrap_std = wrap_max = float("nan")

    return dict(f01=f01, arr=arr, post=post,
                osc_std=osc_std, osc_max=osc_max,
                wrap_std=wrap_std, wrap_max=wrap_max)


def fmt(m):
    arr_s  = f"{m['arr']:4d}" if m['arr'] is not None else " ---"
    post_s = f"{m['post']:.1%}" if m['post'] > 0 else "  ---"
    return (
        f"f01={m['f01']:.1%}  arr={arr_s}  post={post_s}"
        f"  |  osc_std={m['osc_std']:.4f}  osc_max={m['osc_max']:.4f}"
        f"  wrap_std={m['wrap_std']:.4f}  wrap_max={m['wrap_max']:.4f}"
    )


# ── Test runner ───────────────────────────────────────────────────────────────
def run_all_tests(model_name, model, mpc, x0_fixed, x_goal):
    print(f"\n{'='*70}")
    print(f" MODEL: {model_name}")
    print(f"{'='*70}")

    # ── Test 1: Long clean rollout from [0,0,0,0] ──────────────────────────
    print("\n[1] 2000-step clean rollout from hanging [0,0,0,0]")
    x_hist, _ = rollout_custom(model, mpc, x0_fixed, x_goal, num_steps=2000)
    m = compute_metrics(x_hist, x_goal)
    print(f"    {fmt(m)}")

    # ── Test 2: Near-top balance starts ────────────────────────────────────
    print("\n[2] Near-top balance starts  [π+δ, 0, 0, 0]")
    for delta in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
        for sign_name, sign in [("+", 1), ("-", -1)]:
            x0_top = torch.tensor(
                [math.pi + sign * delta, 0.0, 0.0, 0.0],
                dtype=torch.float64, device=DEVICE,
            )
            x_hist, _ = rollout_custom(model, mpc, x0_top, x_goal, num_steps=2000)
            m = compute_metrics(x_hist, x_goal)
            print(f"    δ={sign_name}{delta:.2f}:  {fmt(m)}")

    # ── Test 3: Near-top with velocity ─────────────────────────────────────
    print("\n[3] Near-top start with initial velocity  [π±0.1, ±v, 0, 0]")
    for v in [0.5, 1.0, 2.0, 3.0]:
        for sign_name, sign in [("+", 1), ("-", -1)]:
            x0_kick = torch.tensor(
                [math.pi + 0.1, sign * v, 0.0, 0.0],
                dtype=torch.float64, device=DEVICE,
            )
            x_hist, _ = rollout_custom(model, mpc, x0_kick, x_goal, num_steps=2000)
            m = compute_metrics(x_hist, x_goal)
            print(f"    q1d={sign_name}{v:.1f}: {fmt(m)}")

    # ── Test 4: Impulse disturbance mid-hold ───────────────────────────────
    print("\n[4] Impulse disturbance after arrival (kick q1d, q2d at arr+50)")
    # First do a clean rollout to find arr
    x_hist_clean, _ = rollout_custom(model, mpc, x0_fixed, x_goal, num_steps=2000)
    m_clean = compute_metrics(x_hist_clean, x_goal)
    arr0 = m_clean['arr']

    if arr0 is not None:
        kick_step = min(arr0 + 50, 1900)
        for kq1d, kq2d in [(0.5, 0.3), (1.0, 0.5), (1.5, 1.0), (2.5, 1.5)]:
            kick_delta = torch.tensor([0.0, kq1d, 0.0, kq2d], dtype=torch.float64)
            x_hist, _ = rollout_custom(
                model, mpc, x0_fixed, x_goal, num_steps=2000,
                kick_at_step=kick_step, kick_delta=kick_delta,
            )
            m = compute_metrics(x_hist, x_goal)
            print(f"    kick q1d={kq1d:.1f} q2d={kq2d:.1f} @step {kick_step}:  {fmt(m)}")
    else:
        print("    (no arrival — skipping disturbance test)")

    # ── Test 5: Observation noise ──────────────────────────────────────────
    print(f"\n[5] Observation noise  σ_obs={OBS_SIGMA}")
    x_hist, _ = rollout_custom(
        model, mpc, x0_fixed, x_goal, num_steps=2000, obs_sigma=OBS_SIGMA,
    )
    m = compute_metrics(x_hist, x_goal)
    print(f"    {fmt(m)}")

    print(f"\n    σ_obs={OBS_SIGMA*2:.3f} (2×)")
    x_hist, _ = rollout_custom(
        model, mpc, x0_fixed, x_goal, num_steps=2000, obs_sigma=OBS_SIGMA * 2,
    )
    m = compute_metrics(x_hist, x_goal)
    print(f"    {fmt(m)}")

    # ── Test 6: Control noise ──────────────────────────────────────────────
    print(f"\n[6] Control noise  σ_ctrl={CTRL_SIGMA}")
    x_hist, _ = rollout_custom(
        model, mpc, x0_fixed, x_goal, num_steps=2000, ctrl_sigma=CTRL_SIGMA,
    )
    m = compute_metrics(x_hist, x_goal)
    print(f"    {fmt(m)}")

    print(f"\n    σ_ctrl={CTRL_SIGMA*2:.3f} (2×)")
    x_hist, _ = rollout_custom(
        model, mpc, x0_fixed, x_goal, num_steps=2000, ctrl_sigma=CTRL_SIGMA * 2,
    )
    m = compute_metrics(x_hist, x_goal)
    print(f"    {fmt(m)}")

    # ── Test 7: Combined obs + ctrl noise ──────────────────────────────────
    print(f"\n[7] Combined obs+ctrl noise  σ_obs={OBS_SIGMA}  σ_ctrl={CTRL_SIGMA}")
    x_hist, _ = rollout_custom(
        model, mpc, x0_fixed, x_goal, num_steps=2000,
        obs_sigma=OBS_SIGMA, ctrl_sigma=CTRL_SIGMA,
    )
    m = compute_metrics(x_hist, x_goal)
    print(f"    {fmt(m)}")

    print(f"\n    2× both  σ_obs={OBS_SIGMA*2:.3f}  σ_ctrl={CTRL_SIGMA*2:.3f}")
    x_hist, _ = rollout_custom(
        model, mpc, x0_fixed, x_goal, num_steps=2000,
        obs_sigma=OBS_SIGMA * 2, ctrl_sigma=CTRL_SIGMA * 2,
    )
    m = compute_metrics(x_hist, x_goal)
    print(f"    {fmt(m)}")

    # ── Test 8: Generalization sweep (bottom pert starts) ──────────────────
    print("\n[8] Generalization: perturbed hanging starts (5 trials each pert level)")
    for q1_pert in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        f01s = []
        for trial in range(5):
            random.seed(SEED + trial * 17)
            q1  = random.uniform(-q1_pert, q1_pert)
            q2  = random.uniform(-q1_pert * 0.8, q1_pert * 0.8)
            q1d = random.uniform(-0.3, 0.3)
            q2d = random.uniform(-0.3, 0.3)
            x0_pert = torch.tensor([q1, q1d, q2, q2d], dtype=torch.float64, device=DEVICE)
            x_hist, _ = rollout_custom(model, mpc, x0_pert, x_goal, num_steps=2000)
            m = compute_metrics(x_hist, x_goal)
            f01s.append(m['f01'])
        mean_f01 = float(np.mean(f01s))
        min_f01  = float(np.min(f01s))
        print(f"    q1_pert=±{q1_pert:.2f}:  mean_f01={mean_f01:.1%}  min_f01={min_f01:.1%}  trials={f01s}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    x0_fixed = torch.tensor(X0_FIXED,   dtype=torch.float64, device=DEVICE)
    x_goal   = torch.tensor(X_GOAL_VAL, dtype=torch.float64, device=DEVICE)

    for model_name, ckpt_path in CHECKPOINTS.items():
        print(f"\nLoading {model_name} from {ckpt_path}")
        model = load_model(ckpt_path)
        mpc   = mpc_module.MPC_controller(x0=x0_fixed, x_goal=x_goal, N=N, device=DEVICE)
        run_all_tests(model_name, model, mpc, x0_fixed, x_goal)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
