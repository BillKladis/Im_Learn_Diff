"""exp_noise_test.py — Test the trained no-demo model under observation noise.

Mirrors Simulate.rollout exactly (incl. seed_tau1 warm-start logic) but
injects Gaussian noise into the state_history that the network observes.
True dynamics evolve cleanly.
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

X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
HORIZON   = 10
N_TRIALS  = 5

MODEL_PATH = "saved_models/stageD_nodemo_20260428_123448/stageD_nodemo_20260428_123448.pth"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

NOISE_LEVELS = [
    ("clean",  [0.0,  0.0,  0.0,  0.0 ]),
    ("low",    [0.01, 0.05, 0.01, 0.05]),
    ("medium", [0.05, 0.20, 0.05, 0.20]),
    ("high",   [0.10, 0.50, 0.10, 0.50]),
]


def noisy_rollout(lin_net, mpc, x0, x_goal, num_steps, noise_sigma, seed):
    """Mirror Simulate.rollout but add noise to state_history observations."""
    rng = torch.Generator(device=mpc.device).manual_seed(seed)
    sigma = torch.tensor(noise_sigma, device=mpc.device, dtype=torch.float64)
    sigma_zero = bool(sigma.abs().sum() == 0)

    n_x = x0.shape[0]
    n_u = mpc.MPC_dynamics.u_min.shape[0]

    x = x0.clone().to(mpc.device)
    u_seq_guess = torch.zeros((mpc.N, n_u), dtype=torch.float64, device=mpc.device)

    # Warm-start logic from Simulate.rollout (skipped for q1=0).
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

    def add_noise(s):
        if sigma_zero:
            return s.clone()
        return s + torch.randn(4, generator=rng, device=mpc.device, dtype=torch.float64) * sigma

    state_history = [add_noise(x) for _ in range(5)]

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

        U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
        u_seq_guess[:-1] = U_opt_reshaped[1:].clone()
        u_seq_guess[-1]  = U_opt_reshaped[-1].clone()

        state_history.pop(0)
        state_history.append(add_noise(x.detach()))

    return float(torch.norm(x - x_goal).item())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(MODEL_PATH, device=str(device)).double()

    print("=" * 76)
    print(f"  Noise robustness — model: {os.path.basename(MODEL_PATH)}")
    print(f"  q_base_diag = {Q_BASE_DIAG}  ||  {N_TRIALS} trials per noise level")
    print("=" * 76)

    summary = []
    for label, sigma in NOISE_LEVELS:
        print(f"\n--- {label}  σ = {sigma} ---")
        dists = []
        for trial in range(N_TRIALS):
            d = noisy_rollout(lin_net, mpc, x0, x_goal, NUM_STEPS, sigma, seed=trial)
            dists.append(d)
            print(f"  trial {trial}: goal_dist = {d:.4f}", flush=True)
        d_arr = np.array(dists)
        sr = (d_arr < 1.0).mean()
        print(f"  → mean={d_arr.mean():.4f}  std={d_arr.std():.4f}  "
              f"min={d_arr.min():.4f}  max={d_arr.max():.4f}  success={sr*100:.0f}%")
        summary.append((label, sigma, d_arr))

    print("\n" + "=" * 76)
    print("  SUMMARY")
    print(f"  {'level':>8}  {'σ_q':>6}  {'σ_qd':>6}  {'mean':>8}  {'std':>8}  {'success':>8}")
    for label, sigma, d_arr in summary:
        print(f"  {label:>8}  {sigma[0]:>6.2f}  {sigma[1]:>6.2f}  "
              f"{d_arr.mean():>8.4f}  {d_arr.std():>8.4f}  "
              f"{(d_arr<1.0).mean()*100:>7.0f}%")


if __name__ == "__main__":
    main()
