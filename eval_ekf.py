"""eval_ekf.py — Comprehensive EKF evaluation on v14m.

Tests EKF4 (obs filter) and EKF6 (state + bias estimation) against:
  1.  Clean baseline
  2.  Obs noise σ=0.10
  3.  Ctrl white noise σ=0.10
  4.  Ctrl white noise σ=0.20
  5.  Constant torque bias  [0.5, 0.5] Nm  (EKF6 expected to cancel)
  6.  Constant torque bias  [1.0, 1.0] Nm
  7.  Step disturbance: bias jumps from 0 to 1.0 at step 400
  8.  Slow ramp bias: 0→1.0 over first 500 steps
  9.  Combined: obs σ=0.10 + ctrl white σ=0.10
  10. Combined: obs σ=0.10 + constant bias 0.5

For each test, three rollouts are compared:
  RAW   — no filter, noisy state passed directly to MPC
  EKF4  — 4-state EKF filters obs; no bias estimation
  EKF6  — 6-state augmented EKF; filtered state + bias cancellation

EKF tuning: Q_state small (model is accurate), Q_bias moderate,
            R matched to obs noise level.
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
from ekf import EKF4, EKF6
from eval_comprehensive import compute_metrics, fmt

# ── Config ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")
X0 = [0.0, 0.0, 0.0, 0.0]
X_GOAL = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 2000
SEED = 42
DT = 0.05

MODEL_KWARGS = dict(
    state_dim=4, control_dim=2, horizon=10, hidden_dim=128,
    gate_range_q=0.99, gate_range_r=0.20, f_extra_bound=2.5, f_kickstart_amp=1.0,
)
CHECKPOINT = (
    "saved_models/stageF_mixed_v14m_20260503_102608_ep50/"
    "stageF_mixed_v14m_20260503_102608_ep50.pth"
)


# ── EKF tuning ────────────────────────────────────────────────────────────
# Q_state: model error per step (very small — dynamics are well-known)
Q_STATE = torch.diag(torch.tensor([1e-6, 1e-4, 1e-6, 1e-4], dtype=torch.float64))

# R_obs: measurement noise (small sensor → small R)
R_CLEAN  = torch.diag(torch.tensor([1e-6, 1e-4, 1e-6, 1e-4], dtype=torch.float64))
R_NOISY  = torch.diag(torch.tensor([0.01, 0.25, 0.01, 0.25], dtype=torch.float64))  # σ=0.10

# Q_bias: how fast bias estimate tracks changes
#   fast  → tracks quickly but noisy, bad for white noise
#   slow  → good for constant/slow bias, ignores fast noise
Q_BIAS_FAST = torch.eye(2, dtype=torch.float64) * 1e-2
Q_BIAS_MED  = torch.eye(2, dtype=torch.float64) * 1e-3
Q_BIAS_SLOW = torch.eye(2, dtype=torch.float64) * 1e-4


# ── Rollout with optional EKF ─────────────────────────────────────────────
def rollout_ekf(
    model, mpc, x0, x_goal, num_steps,
    obs_sigma=0.0,
    ctrl_sigma=0.0,       # white noise on torque
    ctrl_bias=None,       # constant bias tensor (2,), or callable(step) → (2,)
    ekf=None,             # EKF4 or EKF6 instance (already reset), or None
    cancel_bias=True,     # if True and EKF6: subtract estimated bias from u_mpc
):
    torch.manual_seed(SEED)
    rng = torch.Generator(device=DEVICE)
    rng.manual_seed(SEED)

    n_x, n_u = x0.shape[0], mpc.MPC_dynamics.u_min.shape[0]
    x_hist = torch.zeros(num_steps + 1, n_x, dtype=torch.float64)
    x = x0.clone()
    x_hist[0] = x
    u_seq_guess = torch.zeros((mpc.N, n_u), dtype=torch.float64)

    # Gravity-torque warm-start
    init_q1 = float(x[0].item())
    if abs(init_q1) > 0.01 and abs(abs(init_q1) - math.pi) > 0.01:
        grav = 2.0 * 9.81 * 0.5 * abs(math.sin(init_q1))
        sign = 1.0 if math.atan2(math.sin(float(x_goal[0]) - init_q1),
                                  math.cos(float(x_goal[0]) - init_q1)) > 0 else -1.0
        u_seq_guess[:, 0] = sign * min(float(mpc.MPC_dynamics.u_max[0]), grav * 2.0)

    state_history = [x.clone() for _ in range(5)]

    for step in range(num_steps):
        # ── True state + measurement noise ──────────────────────────────
        if obs_sigma > 0.0:
            y = x + torch.randn(n_x, generator=rng, dtype=torch.float64) * obs_sigma
        else:
            y = x.clone()

        # ── EKF filtering ───────────────────────────────────────────────
        if ekf is not None and step > 0:
            u_prev = u_seq_guess[0]  # approximate prev commanded u
            x_est, bias_est = ekf.step(y, u_prev)
        else:
            x_est = y
            bias_est = torch.zeros(n_u, dtype=torch.float64)
            if ekf is not None:
                ekf.reset(x_est)

        # Use filtered state for history / network input
        x_for_net = x_est.clone()
        state_history_use = [x_for_net.clone() if i == len(state_history) - 1
                             else state_history[i].clone()
                             for i in range(len(state_history))]

        # ── Network + MPC ────────────────────────────────────────────────
        with torch.no_grad():
            gates_Q, gates_R, f_extra, _, _, gates_Qf = model(
                torch.stack(state_history_use, dim=0),
                q_base_diag=mpc.q_base_diag,
                r_base_diag=mpc.r_base_diag,
            )

        x_lin_seq = x_est.unsqueeze(0).expand(mpc.N, -1).clone()
        u_lin_seq = torch.clamp(u_seq_guess.clone(),
                                min=mpc.MPC_dynamics.u_min.unsqueeze(0),
                                max=mpc.MPC_dynamics.u_max.unsqueeze(0))

        u_opt, U_opt_full = mpc.control(
            x_est, x_lin_seq, u_lin_seq, x_goal,
            diag_corrections_Q=gates_Q,
            diag_corrections_R=gates_R,
            extra_linear_control=f_extra.reshape(-1),
            diag_corrections_Qf=gates_Qf,
        )

        u_apply = u_opt.detach().clone()

        # ── Bias cancellation ────────────────────────────────────────────
        if cancel_bias and isinstance(ekf, EKF6) and step > 10:
            u_apply = u_apply - bias_est.detach()

        # ── Apply disturbances to actual torque ──────────────────────────
        if ctrl_sigma > 0.0:
            with torch.no_grad():
                u_apply = u_apply + torch.randn(n_u, generator=rng,
                                                dtype=torch.float64) * ctrl_sigma

        if ctrl_bias is not None:
            b = ctrl_bias(step) if callable(ctrl_bias) else ctrl_bias
            u_apply = u_apply + b.to(DEVICE)

        u_apply = torch.clamp(u_apply, min=mpc.MPC_dynamics.u_min,
                              max=mpc.MPC_dynamics.u_max)

        x = mpc.true_RK4_disc(x, u_apply, mpc.dt)
        x_hist[step + 1] = x.detach()

        U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
        u_seq_guess[:-1] = U_opt_reshaped[1:].clone()
        u_seq_guess[-1]  = U_opt_reshaped[-1].clone()

        state_history.pop(0)
        state_history.append(x.detach().clone())

    return x_hist.detach()


def make_ekf4(mpc, R=None):
    r = R if R is not None else R_CLEAN
    f = EKF4(mpc, Q_STATE, r)
    return f


def make_ekf6(mpc, Q_bias=None, R=None):
    qb = Q_bias if Q_bias is not None else Q_BIAS_MED
    r  = R if R is not None else R_CLEAN
    f = EKF6(mpc, Q_STATE, qb, r)
    return f


def run_test(label, model, mpc, x0, x_goal, **rollout_kwargs):
    """Run RAW / EKF4 / EKF6-med / EKF6-fast / EKF6-slow variants."""
    results = {}

    # RAW (no filter)
    xh = rollout_ekf(model, mpc, x0, x_goal, NUM_STEPS, ekf=None, **rollout_kwargs)
    results["RAW    "] = compute_metrics(xh, x_goal)

    # EKF4 (obs filter only)
    obs_s = rollout_kwargs.get("obs_sigma", 0.0)
    R_use = R_NOISY if obs_s >= 0.05 else R_CLEAN
    f4 = make_ekf4(mpc, R=R_use)
    f4.reset(x0)
    xh = rollout_ekf(model, mpc, x0, x_goal, NUM_STEPS, ekf=f4, cancel_bias=False, **rollout_kwargs)
    results["EKF4   "] = compute_metrics(xh, x_goal)

    # EKF6 med Q_bias
    f6m = make_ekf6(mpc, Q_bias=Q_BIAS_MED, R=R_use)
    f6m.reset(x0)
    xh = rollout_ekf(model, mpc, x0, x_goal, NUM_STEPS, ekf=f6m, cancel_bias=True, **rollout_kwargs)
    results["EKF6-med"] = compute_metrics(xh, x_goal)

    # EKF6 fast Q_bias
    f6f = make_ekf6(mpc, Q_bias=Q_BIAS_FAST, R=R_use)
    f6f.reset(x0)
    xh = rollout_ekf(model, mpc, x0, x_goal, NUM_STEPS, ekf=f6f, cancel_bias=True, **rollout_kwargs)
    results["EKF6-fast"] = compute_metrics(xh, x_goal)

    # EKF6 slow Q_bias
    f6s = make_ekf6(mpc, Q_bias=Q_BIAS_SLOW, R=R_use)
    f6s.reset(x0)
    xh = rollout_ekf(model, mpc, x0, x_goal, NUM_STEPS, ekf=f6s, cancel_bias=True, **rollout_kwargs)
    results["EKF6-slow"] = compute_metrics(xh, x_goal)

    print(f"\n{'─'*100}")
    print(f"  TEST: {label}")
    print(f"{'─'*100}")
    for variant, m in results.items():
        print(f"  {variant:<12}  {fmt(m)}")
    return results


def main():
    x0     = torch.tensor(X0,     dtype=torch.float64, device=DEVICE)
    x_goal = torch.tensor(X_GOAL, dtype=torch.float64, device=DEVICE)

    print("Loading v14m ...")
    data = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = network_module.SeparatedLinearizationNetwork(**MODEL_KWARGS).double()
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=10, device=DEVICE)

    all_results = {}

    # 1. Clean baseline
    all_results["1.clean"] = run_test(
        "Clean (no noise)", model, mpc, x0, x_goal)

    # 2. Obs noise σ=0.10
    all_results["2.obs0.10"] = run_test(
        "Obs noise σ=0.10", model, mpc, x0, x_goal,
        obs_sigma=0.10)

    # 3. Ctrl white noise σ=0.10
    all_results["3.ctrl0.10"] = run_test(
        "Ctrl white noise σ=0.10", model, mpc, x0, x_goal,
        ctrl_sigma=0.10)

    # 4. Ctrl white noise σ=0.20
    all_results["4.ctrl0.20"] = run_test(
        "Ctrl white noise σ=0.20", model, mpc, x0, x_goal,
        ctrl_sigma=0.20)

    # 5. Constant bias [0.5, 0.5] Nm
    bias_05 = torch.tensor([0.5, 0.5], dtype=torch.float64)
    all_results["5.bias0.5"] = run_test(
        "Constant torque bias [0.5, 0.5] Nm", model, mpc, x0, x_goal,
        ctrl_bias=bias_05)

    # 6. Constant bias [1.0, 1.0] Nm
    bias_10 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    all_results["6.bias1.0"] = run_test(
        "Constant torque bias [1.0, 1.0] Nm", model, mpc, x0, x_goal,
        ctrl_bias=bias_10)

    # 7. Step disturbance: 0 → 1.0 at step 400
    def step_bias(step):
        return torch.tensor([1.0, 1.0] if step >= 400 else [0.0, 0.0],
                            dtype=torch.float64)
    all_results["7.step@400"] = run_test(
        "Step disturbance: bias 0→1.0 at step 400", model, mpc, x0, x_goal,
        ctrl_bias=step_bias)

    # 8. Slow ramp bias: 0→1.0 over 500 steps
    def ramp_bias(step):
        alpha = min(1.0, step / 500.0)
        return torch.tensor([alpha, alpha], dtype=torch.float64)
    all_results["8.ramp"] = run_test(
        "Slow ramp bias: 0→1.0 over 500 steps", model, mpc, x0, x_goal,
        ctrl_bias=ramp_bias)

    # 9. Combined: obs σ=0.10 + ctrl white σ=0.10
    all_results["9.combined"] = run_test(
        "Combined: obs σ=0.10 + ctrl white σ=0.10", model, mpc, x0, x_goal,
        obs_sigma=0.10, ctrl_sigma=0.10)

    # 10. Combined: obs σ=0.10 + constant bias 0.5
    all_results["10.obs+bias"] = run_test(
        "Combined: obs σ=0.10 + constant bias 0.5", model, mpc, x0, x_goal,
        obs_sigma=0.10, ctrl_bias=bias_05)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n\n" + "=" * 110)
    print("  SUMMARY: f01 by test and variant")
    print("=" * 110)
    variants = ["RAW    ", "EKF4   ", "EKF6-med", "EKF6-fast", "EKF6-slow"]
    print(f"  {'Test':<32}", end="")
    for v in variants:
        print(f"  {v:<12}", end="")
    print()
    print("  " + "─" * 104)
    for test_label, res in all_results.items():
        print(f"  {test_label:<32}", end="")
        for v in variants:
            m = res.get(v)
            if m:
                arr = m['arr'] if m['arr'] is not None else 9999
                print(f"  {m['f01']:.1%} a={arr:<4}", end="")
            else:
                print(f"  {'—':<12}", end="")
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()
