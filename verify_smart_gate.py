"""verify_smart_gate.py — Smarter inference-time gating.

Previous test showed naive (1-near_goal) gates suppress f_extra too early
during the approach phase, hurting swing-up.

Smart gate: stable_zone = near_goal × low_velocity
  - near_goal = (1 + cos(q1-π))/2  → 1 at goal, 0 at bottom
  - low_velocity = exp(-q1d²/2)    → 1 at q1d=0, decays as |q1d| grows

Gate is fully OFF (f_extra full strength) during the energetic swing-up
(when velocity is high), and fully ON only when the pendulum has actually
settled near the goal.

Also tests a "switch" approach: pure MPC stabilizer (f_extra=0, gates_Q=1)
once wrap_dist < 0.5.
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


def rollout(lin_net, mpc, x0, x_goal, num_steps, mode="baseline"):
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
    switch_step = -1

    for step in range(num_steps):
        with torch.no_grad():
            gates_Q, gates_R, f_extra, _, _ = lin_net(
                torch.stack(state_history, dim=0),
                q_base_diag=mpc.q_base_diag,
                r_base_diag=mpc.r_base_diag,
            )

            q1, q1d = float(x[0].item()), float(x[1].item())
            q2d = float(x[3].item())
            wrap_q1 = math.atan2(math.sin(q1 - x_goal[0]),
                                 math.cos(q1 - x_goal[0]))
            wrap_dist = math.sqrt(wrap_q1**2 + q1d**2 + 0.0 + q2d**2)

            near_goal_pos = (1.0 + math.cos(q1 - x_goal[0])) / 2.0
            low_vel = math.exp(-(q1d**2 + q2d**2) / 2.0)
            stable_zone = near_goal_pos * low_vel  # 0..1

            if mode == "baseline":
                pass
            elif mode == "smart_gate":
                # Suppress f_extra and boost Q-gates only when in stable zone
                f_extra = f_extra * (1.0 - stable_zone)
                gates_Q_new = gates_Q.clone()
                gates_Q_new[:, 0] = gates_Q[:, 0] * (1 - stable_zone) + 1.0 * stable_zone
                gates_Q_new[:, 1] = gates_Q[:, 1] * (1 - stable_zone) + 1.0 * stable_zone
                gates_Q = gates_Q_new
            elif mode == "switch":
                # Pure MPC stabilizer once close enough to goal
                if wrap_dist < 0.5 or switch_step >= 0:
                    if switch_step < 0:
                        switch_step = step
                    f_extra = torch.zeros_like(f_extra)
                    gates_Q = torch.ones_like(gates_Q)
            elif mode == "switch_smart":
                # Smart gate during approach, hard switch once stable
                if wrap_dist < 0.3 or switch_step >= 0:
                    if switch_step < 0:
                        switch_step = step
                    f_extra = torch.zeros_like(f_extra)
                    gates_Q = torch.ones_like(gates_Q)
                else:
                    # Still gating during approach (suppress f_extra by stable_zone)
                    f_extra = f_extra * (1.0 - stable_zone)
            elif mode == "f_only_smart":
                # Only suppress f_extra (no Q-gate change)
                f_extra = f_extra * (1.0 - stable_zone)

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

    return np.array(traj), f_norms, switch_step


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(MODEL_PATH, device=str(device)).double()

    print("=" * 86)
    print("  SMART INFERENCE-TIME GATING")
    print(f"  Model: {os.path.basename(MODEL_PATH)}")
    print("=" * 86)

    modes = ["baseline", "f_only_smart", "smart_gate", "switch", "switch_smart"]
    print(f"\n  {'mode':>14s}  {'170r':>8s}  {'170w':>8s}  "
          f"{'300r':>8s}  {'300w':>8s}  {'600r':>8s}  {'600w':>8s}  "
          f"{'switch_at':>10s}  {'avg_fN_600':>10s}")
    print("  " + "─" * 90)
    for mode in modes:
        results = []
        switch_at = -1
        for n in [170, 300, 600]:
            traj, fnorms, sw = rollout(lin_net, mpc, x0, x_goal, n, mode=mode)
            last = traj[-1]
            r = float(np.linalg.norm(last - np.array(X_GOAL)))
            w = wrapped_goal_dist(last, X_GOAL)
            results.append((r, w, np.mean(fnorms)))
            if n == 600:
                switch_at = sw
        avg_fn = results[-1][2]
        print(f"  {mode:>14s}  "
              f"{results[0][0]:>8.3f}  {results[0][1]:>8.3f}  "
              f"{results[1][0]:>8.3f}  {results[1][1]:>8.3f}  "
              f"{results[2][0]:>8.3f}  {results[2][1]:>8.3f}  "
              f"{switch_at:>10d}  {avg_fn:>10.3f}")


if __name__ == "__main__":
    main()
