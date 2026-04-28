"""exp_diag_failure.py — Diagnose why q1=+0.2 fails (goal_dist=6.04).

The best model (0.0612) fails completely at q1=+0.2 initial condition.
This script runs a rollout and records what the network outputs at each step
to understand the failure mode:
  - Is f_extra pointing in the wrong direction?
  - Do Q-gates stay in pump mode instead of switching to stable?
  - Does the pendulum not have enough energy?
  - Does the pendulum overshoot and oscillate?
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
import Simulate as train_module

MODEL_PATH = "saved_models/stageD_nodemo_20260428_123448/stageD_nodemo_20260428_123448.pth"
X_GOAL = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT = 0.05
HORIZON = 10
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]


def diag_rollout(lin_net, mpc, x0, x_goal, num_steps):
    """Run rollout, collect per-step diagnostics."""
    x = x0.clone()
    n_u = mpc.MPC_dynamics.u_min.shape[0]
    u_seq_guess = torch.zeros((mpc.N, n_u), dtype=torch.float64, device=mpc.device)

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
    lin_net.eval()

    records = []
    for step in range(num_steps):
        with torch.no_grad():
            gates_Q, gates_R, f_extra, _, _ = lin_net(
                torch.stack(state_history, dim=0),
                q_base_diag=mpc.q_base_diag,
                r_base_diag=mpc.r_base_diag,
            )

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
        )

        E = float(mpc.compute_energy_single(x).item())
        q1_err = float(torch.atan2(
            torch.sin(x[0] - x_goal[0]),
            torch.cos(x[0] - x_goal[0]),
        ).item())

        records.append({
            "step": step,
            "q1": float(x[0].item()),
            "q1d": float(x[1].item()),
            "q2": float(x[2].item()),
            "q2d": float(x[3].item()),
            "energy": E,
            "q1_err_deg": math.degrees(q1_err),
            "u_tau1": float(u_opt[0].item()),
            "u_tau2": float(u_opt[1].item()),
            "f_extra_tau1": float(f_extra[0, 0].item()),
            "f_extra_norm": float(f_extra.norm().item()),
            "Q_gate_q1": float(gates_Q[0, 0].item()),
            "Q_gate_q1d": float(gates_Q[0, 1].item()),
        })

        x = mpc.true_RK4_disc(x, u_opt, mpc.dt)
        U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
        u_seq_guess[:-1] = U_opt_reshaped[1:].clone()
        u_seq_guess[-1]  = U_opt_reshaped[-1].clone()
        state_history.pop(0)
        state_history.append(x.detach())

    return records, float(torch.norm(x - x_goal).item())


def print_summary(records, x0_label, final_dist):
    print(f"\n=== {x0_label}: final goal_dist = {final_dist:.4f} ===")
    print(f"{'Step':>5} {'q1°':>8} {'Energy':>8} {'q1err°':>8} "
          f"{'τ1':>7} {'f_τ1':>7} {'fNorm':>7} {'Qq1':>7} {'Qq1d':>7}")
    # Print first 10, middle 10, last 10
    indices = list(range(10)) + list(range(80, 90)) + list(range(160, 170))
    for i in indices:
        if i >= len(records):
            break
        r = records[i]
        print(f"{r['step']:>5} {math.degrees(r['q1']):>8.2f} {r['energy']:>8.2f} "
              f"{r['q1_err_deg']:>8.2f} {r['u_tau1']:>7.3f} {r['f_extra_tau1']:>7.3f} "
              f"{r['f_extra_norm']:>7.3f} {r['Q_gate_q1']:>7.3f} {r['Q_gate_q1d']:>7.3f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(
        x0=torch.zeros(4, device=device, dtype=torch.float64),
        x_goal=x_goal, N=HORIZON, device=device,
    )
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(MODEL_PATH, device=str(device)).double()

    test_cases = [
        ("x0=zero (reference)",   [0.0,  0.0, 0.0, 0.0]),
        ("x0=q1=+0.2 (FAIL)",     [0.2,  0.0, 0.0, 0.0]),
        ("x0=q1=-0.2 (FAIL)",     [-0.2, 0.0, 0.0, 0.0]),
        ("x0=q1d=+0.5 (??)",      [0.0,  0.5, 0.0, 0.0]),
    ]

    print("=" * 76)
    print("  FAILURE DIAGNOSIS: why does q1=+0.2 give goal_dist=6.04?")
    print("=" * 76)

    for label, x0_list in test_cases:
        x0 = torch.tensor(x0_list, device=device, dtype=torch.float64)
        records, final_dist = diag_rollout(lin_net, mpc, x0, x_goal, NUM_STEPS)
        print_summary(records, label, final_dist)


if __name__ == "__main__":
    main()
