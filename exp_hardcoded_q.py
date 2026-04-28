"""
exp_hardcoded_q.py — DIAGNOSTIC: bypass q_head entirely with a
state-dependent q_base_diag schedule.  Tests whether the right
Q-modulation enables swing-up given the f_head training problem.

q_base_diag becomes time-varying based on cos(q1):
  near upright:  [12, 5, 50, 40]   (full default Q for stabilization)
  at bottom:     [0.12, 5, 50, 40] (suppressed q1 for pump)
  blend factor:  (1 + cos(q1 - π))/2  ∈ [0,1]

This is "the perfect Q-gate schedule" applied directly to MPC.  The
network only needs to learn f_extra (which we proved possible at
q1_cost=0.5 ablation).

If this swings up: confirms Q-modulation is the right answer, the
issue is just the gradient trap on q-head training.
If this fails: there's something deeper than Q-modulation needed.
"""

import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

DEMO_CSV  = "run_20260428_001459_rollout_final.csv"
X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
EPOCHS    = 50
LR        = 1e-3
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM   = 4
CONTROL_DIM = 2
SAVE_DIR    = "saved_models"

GATE_RANGE_Q   = 0.99
GATE_RANGE_R   = 0.20
F_EXTRA_BOUND  = 3.0


# Wrap the MPC controller to use a state-dependent q_base_diag.
class StateDependentMPC:
    """Wrapper that intercepts mpc.QP_formulation to inject state-based q_base."""
    def __init__(self, mpc, x_goal_q1, q1_pump=0.12, q1_full=12.0,
                 q1d_const=5.0, q2_const=50.0, q2d_const=40.0):
        self.mpc = mpc
        self.x_goal_q1 = x_goal_q1
        self.q1_pump = q1_pump
        self.q1_full = q1_full
        self.q1d_const = q1d_const
        self.q2_const = q2_const
        self.q2d_const = q2d_const
        # Original q_base_diag for forward()
        self.original_q_base = mpc.q_base_diag.clone()

    def state_dep_q_base(self, current_state):
        # near_goal in [0,1]: 1 at upright (q1=π), 0 at bottom (q1=0)
        q1 = current_state[0]
        near = 0.5 * (1.0 + torch.cos(q1 - self.x_goal_q1))
        q1_eff = self.q1_pump + (self.q1_full - self.q1_pump) * near
        return torch.tensor(
            [q1_eff.item(), self.q1d_const, self.q2_const, self.q2d_const],
            device=self.mpc.device, dtype=torch.float64,
        )


def hardcoded_rollout(lin_net, mpc, x0, x_goal, num_steps, sd_mpc):
    """Rollout with state-dependent q_base_diag, network only provides f_extra."""
    n_u = mpc.MPC_dynamics.u_min.shape[0]
    x = x0.clone()
    state_history = [x.clone() for _ in range(5)]
    u_seq_guess = torch.zeros((mpc.N, n_u), device=mpc.device, dtype=torch.float64)
    x_hist = [x.clone()]
    u_hist = []

    with torch.no_grad():
        for step in range(num_steps):
            sh = torch.stack(state_history, dim=0)
            # Set state-dependent q_base BEFORE network call
            mpc.q_base_diag = sd_mpc.state_dep_q_base(x)
            gQ, gR, fE, _, _ = lin_net(sh, mpc.q_base_diag, mpc.r_base_diag)

            x_lin_seq = x.unsqueeze(0).expand(mpc.N, -1).clone()
            u_lin_seq = torch.clamp(
                u_seq_guess.clone(),
                min=mpc.MPC_dynamics.u_min.unsqueeze(0),
                max=mpc.MPC_dynamics.u_max.unsqueeze(0),
            )
            u_mpc, U_full = mpc.control(
                x, x_lin_seq, u_lin_seq, x_goal,
                diag_corrections_Q=gQ, diag_corrections_R=gR,
                extra_linear_control=fE.reshape(-1),
            )
            x = mpc.true_RK4_disc(x, u_mpc, mpc.dt)
            U_re = U_full.detach().view(mpc.N, n_u)
            u_seq_guess = torch.cat([U_re[1:], U_re[-1:]], dim=0).clone()
            state_history.pop(0); state_history.append(x.clone())
            x_hist.append(x.clone())
            u_hist.append(u_mpc.clone())

    return torch.stack(x_hist), torch.stack(u_hist)


def custom_train(lin_net, mpc, sd_mpc, x0, x_goal, demo, num_steps, num_epochs, lr):
    """Custom training loop with state-dependent q_base_diag."""
    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=lr, weight_decay=1e-4)
    n_u = mpc.MPC_dynamics.u_min.shape[0]

    # Precompute demo energies (with cleaned demo)
    with torch.no_grad():
        E_demo = torch.stack([mpc.compute_energy_single(demo[i]) for i in range(demo.shape[0])])
    E_range = (E_demo.max() - E_demo.min()).clamp(min=1.0)

    best_dist = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        t0 = time.time()
        lin_net.train()
        optimizer.zero_grad()
        current = x0.detach().clone()
        sh = [current.clone() for _ in range(5)]
        u_guess = torch.zeros((mpc.N, n_u), device=mpc.device, dtype=torch.float64)

        track_terms = []
        for step in range(num_steps):
            sh_t = torch.stack(sh, dim=0)
            mpc.q_base_diag = sd_mpc.state_dep_q_base(current)
            gQ, gR, fE, _, _ = lin_net(sh_t, mpc.q_base_diag, mpc.r_base_diag)

            x_lin = current.unsqueeze(0).expand(mpc.N, -1).clone()
            u_lin = torch.clamp(u_guess.clone(),
                                min=mpc.MPC_dynamics.u_min.unsqueeze(0),
                                max=mpc.MPC_dynamics.u_max.unsqueeze(0))
            u_mpc, U_full = mpc.control(
                current, x_lin, u_lin, x_goal,
                diag_corrections_Q=gQ, diag_corrections_R=gR,
                extra_linear_control=fE.reshape(-1),
            )
            next_state = mpc.true_RK4_disc(current, u_mpc, mpc.dt)

            # Energy tracking
            target_idx = min(step + 1, demo.shape[0] - 1)
            E_now = mpc.compute_energy_single(next_state)
            track_step = ((E_now - E_demo[target_idx]) / E_range) ** 2
            track_terms.append(track_step)

            current = next_state.detach()
            U_re = U_full.detach().view(mpc.N, n_u)
            u_guess = torch.cat([U_re[1:], U_re[-1:]], dim=0).clone()
            sh.pop(0); sh.append(current.clone())

        loss = torch.stack(track_terms).mean() * 5.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lin_net.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        dist = float(torch.norm(current - x_goal).item())
        if dist < best_dist:
            best_dist = dist
            best_state = {k: v.clone() for k, v in lin_net.state_dict().items()}

        elapsed = time.time() - t0
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"  [{epoch+1:>3}/{num_epochs}] loss={loss.item():.4f} "
                  f"goal_dist={dist:.4f} best={best_dist:.4f} time={elapsed:.1f}s", flush=True)

    if best_state is not None:
        lin_net.load_state_dict(best_state)
    return best_dist


def clean_demo_tail(demo, num_smooth_steps=20):
    demo_clean = demo.clone()
    T = demo_clean.shape[0]
    goal = torch.tensor(X_GOAL, device=demo.device, dtype=demo.dtype)
    start_idx = T - num_smooth_steps
    start = demo_clean[start_idx].clone()
    for k in range(num_smooth_steps):
        alpha = (k + 1) / num_smooth_steps
        a = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo_clean[start_idx + k] = (1.0 - a) * start + a * goal
    return demo_clean


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo_raw = train_module.load_demo_trajectory(DEMO_CSV, expected_length=NUM_STEPS, device=device)
    demo = clean_demo_tail(demo_raw, num_smooth_steps=20)

    print("=" * 76)
    print("  EXP: HARD-CODED state-dependent q_base_diag schedule")
    print("  q1_cost = 0.12 at bottom, 12.0 at upright, blend = (1+cos(q1-π))/2")
    print("  Network only learns f_extra (q-head trained but its output ignored)")
    print(f"  EPOCHS = {EPOCHS}  LR = {LR}")
    print("=" * 76)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
    sd_mpc = StateDependentMPC(mpc, x_goal_q1=x_goal[0])

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=0.0,
    ).to(device).double()

    t0 = time.time()
    best_dist = custom_train(lin_net, mpc, sd_mpc, x0, x_goal, demo, NUM_STEPS, EPOCHS, LR)
    elapsed = time.time() - t0

    # Final evaluation
    x_final, u_final = hardcoded_rollout(lin_net, mpc, x0, x_goal, NUM_STEPS, sd_mpc)
    dist_final = float(torch.norm(x_final[-1] - x_goal).item())
    result = "SUCCESS" if dist_final < 1.0 else "FAIL"

    print(f"\n  best_dist (best ckpt): {best_dist:.4f}")
    print(f"  final rollout dist   : {dist_final:.4f}  {result}")
    print(f"  time                 : {elapsed:.0f}s")


if __name__ == "__main__":
    main()
