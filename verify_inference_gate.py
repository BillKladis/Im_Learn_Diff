"""verify_inference_gate.py — Inference-time f_extra suppression test.

Hypothesis: the 0.0612 model swings up correctly but keeps pumping at the
goal (fNorm=6.8 even at q1=π).  If we simply SCALE f_extra by (1 - near_goal)
at inference time — where near_goal = (1+cos(q1-π))/2 — the pendulum should
hold the upright without ANY retraining.

near_goal:
  q1 = 0 (bottom):     cos(-π) = -1   → near_goal = 0  → f_extra fully active
  q1 = π/2:            cos(-π/2) = 0  → near_goal = 0.5 → f_extra at half
  q1 = π (upright):    cos(0)   = +1  → near_goal = 1.0 → f_extra silenced

Run two rollouts (with and without the gate) and compare goal_dist at
multiple step counts.  No training, just inference policy modification.
"""

import math
import os
import sys
import glob

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module

X0     = [0.0, 0.0, 0.0, 0.0]
X_GOAL = [math.pi, 0.0, 0.0, 0.0]
DT = 0.05
HORIZON = 10
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
MODEL_PATH = "saved_models/stageD_nodemo_20260428_123448/stageD_nodemo_20260428_123448.pth"


def wrapped_goal_dist(x_state, x_goal):
    q1_err = math.atan2(math.sin(x_state[0] - x_goal[0]),
                        math.cos(x_state[0] - x_goal[0]))
    return math.sqrt(q1_err**2 + x_state[1]**2 + x_state[2]**2 + x_state[3]**2)


def rollout(lin_net, mpc, x0, x_goal, num_steps, gate_f_extra=False, gate_q_up=False):
    """Custom rollout with optional inference-time gates.

    gate_f_extra: scale f_extra by (1 - near_goal) at each step.
    gate_q_up:    scale gates_Q toward 1.0 by near_goal at each step.
    """
    n_u = mpc.MPC_dynamics.u_min.shape[0]
    x = x0.clone()
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
    traj = []
    f_norms = []

    for step in range(num_steps):
        with torch.no_grad():
            gates_Q, gates_R, f_extra, _, _, _ = lin_net(
                torch.stack(state_history, dim=0),
                q_base_diag=mpc.q_base_diag,
                r_base_diag=mpc.r_base_diag,
            )

            # Compute near_goal blend factor
            near_goal = float((1.0 + torch.cos(x[0] - x_goal[0])).item()) / 2.0
            near_goal = max(0.0, min(1.0, near_goal))

            if gate_f_extra:
                f_extra = f_extra * (1.0 - near_goal)

            if gate_q_up:
                # Pull gates_Q[:,0:2] toward 1.0 with weight near_goal
                gates_Q_new = gates_Q.clone()
                gates_Q_new[:, 0] = gates_Q[:, 0] * (1.0 - near_goal) + 1.0 * near_goal
                gates_Q_new[:, 1] = gates_Q[:, 1] * (1.0 - near_goal) + 1.0 * near_goal
                gates_Q = gates_Q_new

            f_norms.append(float(f_extra.norm().item()))

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

        x = mpc.true_RK4_disc(x, u_opt, mpc.dt)

        U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
        u_seq_guess[:-1] = U_opt_reshaped[1:].clone()
        u_seq_guess[-1]  = U_opt_reshaped[-1].clone()
        state_history.pop(0)
        state_history.append(x.detach())
        traj.append(x.detach().cpu().numpy())

    return np.array(traj), f_norms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(MODEL_PATH, device=str(device)).double()

    print("=" * 76)
    print("  INFERENCE-TIME f_EXTRA / Q-GATE GATING TEST")
    print(f"  Model: {os.path.basename(MODEL_PATH)}  (best 0.0612 baseline)")
    print("=" * 76)

    configs = [
        ("baseline",      False, False),
        ("gate_f",        True,  False),
        ("gate_Q",        False, True),
        ("gate_f+Q",      True,  True),
    ]

    print(f"\n  {'config':>12s}  {'170r':>8s}  {'170w':>8s}  "
          f"{'300r':>8s}  {'300w':>8s}  {'600r':>8s}  {'600w':>8s}  {'avg_fN':>8s}")
    print("  " + "─" * 80)
    for name, gf, gq in configs:
        results = []
        for n in [170, 300, 600]:
            traj, fnorms = rollout(lin_net, mpc, x0, x_goal, n,
                                   gate_f_extra=gf, gate_q_up=gq)
            last = traj[-1]
            r = float(np.linalg.norm(last - np.array(X_GOAL)))
            w = wrapped_goal_dist(last, X_GOAL)
            results.append((r, w, np.mean(fnorms)))
        avg_fn = results[-1][2]
        print(f"  {name:>12s}  "
              f"{results[0][0]:>8.3f}  {results[0][1]:>8.3f}  "
              f"{results[1][0]:>8.3f}  {results[1][1]:>8.3f}  "
              f"{results[2][0]:>8.3f}  {results[2][1]:>8.3f}  "
              f"{avg_fn:>8.3f}")


if __name__ == "__main__":
    main()
