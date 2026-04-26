import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from torch.func import jacrev, vmap

try:
    from qpth.qp import QPFunction
except ImportError as e:
    raise ImportError("qpth not found.") from e

import MPC_dynamics
import true_dynamics


class MPC_controller:
    def __init__(self, x0: torch.Tensor, x_goal: torch.Tensor, N: int, device: torch.device):
        self.device = device
        self.x0     = x0.detach().clone().to(device=device, dtype=torch.float64)
        self.x_goal = x_goal.detach().clone().to(device=device, dtype=torch.float64)
        self.N      = N
        self.dt     = torch.tensor(0.05, device=device, dtype=torch.float64)

        self.q_base_diag = torch.tensor([12.0, 5.0, 10.0, 5.0], device=device, dtype=torch.float64)
        self.r_base_diag = torch.tensor([1.0, 1.0], device=device, dtype=torch.float64)
        self.Qf = torch.diag(torch.tensor([20.0, 20.0, 20.0, 20.0], device=device, dtype=torch.float64))
        
        # Base penalty weight for energy shaping
        self.w_e_base = 25.0

        self.true_dynamics = true_dynamics.DoublePendulumDynamics(device=device)
        self.MPC_dynamics  = MPC_dynamics.DoublePendulumDynamics(device=device)

        self.n_u_total = self.N * self.MPC_dynamics.u_min.shape[0]
        self.qp_fallback_count = 0
        I = torch.eye(self.n_u_total, device=device, dtype=torch.float64)
        self.G_box = torch.vstack([-I, I]).unsqueeze(0)
        self.qp_A = torch.empty(0, self.n_u_total, device=device, dtype=torch.float64).unsqueeze(0)
        self.qp_b = torch.empty(0,            device=device, dtype=torch.float64).unsqueeze(0)

    def build_cost_matrices(
        self,
        diag_corrections_Q: Optional[torch.Tensor] = None,
        diag_corrections_R: Optional[torch.Tensor] = None,
        Qf_dense: Optional[torch.Tensor] = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if diag_corrections_Q is not None:
            q_k = self.q_base_diag * diag_corrections_Q
            Q_run = q_k.flatten()
        else:
            Q_run = self.q_base_diag.repeat(self.N - 1)

        Q_bar = torch.diag(torch.cat([Q_run, torch.zeros(4, device=self.device, dtype=torch.float64)]))
        
        if Qf_dense is not None:
            Q_bar[-4:, -4:] = Qf_dense
        else:
            Q_bar[-4:, -4:] = self.Qf

        if diag_corrections_R is not None:
            r_k = self.r_base_diag * diag_corrections_R
            R_diag = r_k.flatten()
        else:
            R_diag = self.r_base_diag.repeat(self.N)

        return Q_bar, R_diag

    # ── ENERGY SHAPING FUNCTIONS ──────────────────────────────────────
    def compute_energy_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates total mechanical energy (T + V) for a single state vector (4,).
        Matched exactly to the point-mass DoublePendulumDynamics.
        """
        m1, m2 = 1.0, 1.0     
        l1, l2 = 0.5, 0.5     
        g      = 9.81

        q1, q1_dot, q2, q2_dot = x[0], x[1], x[2], x[3]

        # Potential Energy (V)
        # Point masses are at the exact ends of l1 and l2
        y1 = -l1 * torch.cos(q1)
        y2 = -l1 * torch.cos(q1) - l2 * torch.cos(q1 + q2)
        V = m1 * g * y1 + m2 * g * y2

        # Kinetic Energy (T)
        v1_sq = (l1 * q1_dot)**2
        
        vx2 = l1 * q1_dot * torch.cos(q1) + l2 * (q1_dot + q2_dot) * torch.cos(q1 + q2)
        vy2 = l1 * q1_dot * torch.sin(q1) + l2 * (q1_dot + q2_dot) * torch.sin(q1 + q2)
        v2_sq = vx2**2 + vy2**2

        # No I1 or I2 terms since they are modeled as point masses
        T = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq

        return T + V

    # ──────────────────────────────────────────────────────────────────

    def true_RK4_disc(self, x: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        t0 = dt.new_zeros(())
        tH = 0.5 * dt
        t1 = dt
        f1 = self.true_dynamics.deriv(t0, x, u)
        f2 = self.true_dynamics.deriv(tH, x + 0.5 * dt * f1, u)
        f3 = self.true_dynamics.deriv(tH, x + 0.5 * dt * f2, u)
        f4 = self.true_dynamics.deriv(t1, x + dt * f3, u)
        return x + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

    def MPC_RK4_disc(self, x: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        t0 = dt.new_zeros(())
        tH = 0.5 * dt
        t1 = dt
        f1 = self.MPC_dynamics.deriv(t0, x, u)
        f2 = self.MPC_dynamics.deriv(tH, x + 0.5 * dt * f1, u)
        f3 = self.MPC_dynamics.deriv(tH, x + 0.5 * dt * f2, u)
        f4 = self.MPC_dynamics.deriv(t1, x + dt * f3, u)
        return x + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

    def linearize_discrete(self, x_batch: torch.Tensor, u_batch: torch.Tensor, dt: torch.Tensor):
        def step_fn(x, u): return self.MPC_RK4_disc(x, u, dt)
        jac_x_fn = vmap(jacrev(step_fn, argnums=0))
        jac_u_fn = vmap(jacrev(step_fn, argnums=1))
        A_batch = jac_x_fn(x_batch.detach(), u_batch.detach())
        B_batch = jac_u_fn(x_batch.detach(), u_batch.detach())
        return list(A_batch), list(B_batch)

    def linearize_horizon(self, x_lin_seq, u_lin_seq):
        return self.linearize_discrete(x_lin_seq, u_lin_seq, self.dt)

    def build_prediction_matrices(self, A_seq, B_seq):
        N = len(A_seq); n_x = A_seq[0].shape[0]; n_u = B_seq[0].shape[1]
        Phi_rows = []; Phi_prev = A_seq[0]; Phi_rows.append(Phi_prev)
        for i in range(1, N):
            Phi_prev = A_seq[i] @ Phi_prev
            Phi_rows.append(Phi_prev)
        A_big = torch.cat(Phi_rows, dim=0)

        zero_block = torch.zeros((n_x, n_u), device=self.device, dtype=torch.float64)
        B_cols = []
        for j in range(N):
            col_blocks = [zero_block] * j
            col = B_seq[j]
            col_blocks.append(col)
            for i in range(j + 1, N):
                col = A_seq[i - 1] @ col
                col_blocks.append(col)
            B_cols.append(torch.cat(col_blocks, dim=0))
        B_big = torch.cat(B_cols, dim=1)
        return A_big, B_big

    def build_qp_matrices_delta(
        self,
        B_big: torch.Tensor,
        X_bar: torch.Tensor,
        X_bar_seq: torch.Tensor,
        U_bar: torch.Tensor,
        x_goal: torch.Tensor,
        Q_bar: torch.Tensor, 
        R_diag: torch.Tensor,
        gates_E: Optional[torch.Tensor] = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        X_ref = x_goal.unsqueeze(0).expand(self.N, -1).reshape(-1)
        E_raw = X_bar - X_ref

        # ── ANGLE WRAPPING for q1 (idx 0 mod 4) and q2 (idx 2 mod 4) ────
        # Prevents misdirected cost gradients when predicted angles cross ±π.
        # atan2(sin(Δ), cos(Δ)) maps any error to (−π, π] without changing
        # the sign or magnitude for small errors, so the QP stays valid.
        q1_idx = torch.arange(0, 4 * self.N, 4, device=self.device)
        q2_idx = torch.arange(2, 4 * self.N, 4, device=self.device)
        angle_idx = torch.cat([q1_idx, q2_idx])
        E = E_raw.clone()
        E[angle_idx] = torch.atan2(torch.sin(E_raw[angle_idx]),
                                   torch.cos(E_raw[angle_idx]))

        # ── ENERGY SHAPING ──────────────────────────────────────────
        E_goal = self.compute_energy_single(x_goal)
        E_curr = vmap(self.compute_energy_single)(X_bar_seq)   # (N,)

        linear_E = torch.zeros_like(E)

        for i in range(self.N):
            idx = slice(i*4, (i+1)*4)
            e_i = E_curr[i] - E_goal          # scalar energy error (negative = deficit)
            gate = gates_E[i] if gates_E is not None else 1.0

            # Adaptive weight: scale up quadratically when energy deficit is large
            # so energy pumping dominates position error during swing-up.
            deficit_norm = torch.relu(-e_i) / (2.0 * E_goal.abs() + 1.0)  # ∈ [0, 1]
            w_k = self.w_e_base * gate * (1.0 + 3.0 * deficit_norm ** 2)

            # dE/dx evaluated at predicted state X_bar_seq[i]
            g_i = jacrev(self.compute_energy_single)(X_bar_seq[i])

            # Project out q2 and q2_dot from the energy gradient.
            # dE/dq2 = m2·g·l2·sin(q1+q2) is non-zero during swing-up and pushes the
            # QP to fold q2 to build energy. Zeroing these forces energy pumping to
            # happen only through q1, producing a smooth single-arc swing-up.
            g_i = g_i.clone()
            g_i[2] = 0.0   # q2 component
            g_i[3] = 0.0   # q2_dot component

            # Linear term only: pushes the QP toward controls that increase energy.
            # No rank-1 Hessian block — Q_bar stays well-conditioned.
            linear_E[idx] = w_k * e_i * g_i

        # Q_bar_total is just Q_bar — no Q_energy added to Hessian
        H = 2.0 * (B_big.T @ Q_bar @ B_big) + torch.diag(2.0 * R_diag)
        H = 0.5 * (H + H.T)
        H = H + 1e-4 * torch.eye(H.shape[0], device=self.device, dtype=torch.float64)

        state_linear_term = 2.0 * (Q_bar @ E) + linear_E
        f = B_big.T @ state_linear_term + 2.0 * (R_diag * U_bar)

        return H, f
        
    def build_constraints_delta(self, U_bar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lb_base = self.MPC_dynamics.u_min.repeat(self.N)
        ub_base = self.MPC_dynamics.u_max.repeat(self.N)
        return lb_base - U_bar, ub_base - U_bar

    def QP_formulation(
        self,
        current_state: torch.Tensor,
        u_guess_seq: torch.Tensor, 
        x_goal: Optional[torch.Tensor] = None,
        diag_corrections_Q: Optional[torch.Tensor] = None,
        diag_corrections_R: Optional[torch.Tensor] = None,
        gates_E: Optional[torch.Tensor] = None,
        Qf_dense: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x_op_list = []
        X_bar_list = []
        curr_x = current_state
        
        for i in range(self.N):
            x_op_list.append(curr_x) 
            curr_x = self.MPC_RK4_disc(curr_x, u_guess_seq[i], self.dt)
            X_bar_list.append(curr_x)
            
        x_op_seq = torch.stack(x_op_list)
        X_bar_seq = torch.stack(X_bar_list)
        X_bar = torch.cat(X_bar_list)
        U_bar = u_guess_seq.reshape(-1)   

        A_list, B_list = self.linearize_horizon(x_op_seq, u_guess_seq)
        _, B_big = self.build_prediction_matrices(A_list, B_list)
        
        Q_bar, R_diag = self.build_cost_matrices(
            diag_corrections_Q=diag_corrections_Q,
            diag_corrections_R=diag_corrections_R,
            Qf_dense=Qf_dense, 
        )
        
        H, f = self.build_qp_matrices_delta(
            B_big, X_bar, X_bar_seq, U_bar, x_goal, Q_bar, R_diag, gates_E=gates_E,
        )
        
        return H, f, U_bar

    def solve_mpc_qp(self, H, f, lb, ub):
        def _fallback_zero():
            self.qp_fallback_count += 1
            return torch.zeros(self.n_u_total, device=self.device, dtype=torch.float64)

        h_box = torch.cat([-lb, ub])
        if not torch.isfinite(H).all() or not torch.isfinite(f).all():
            return _fallback_zero()

        p = f.unsqueeze(0); G = self.G_box; h = h_box.unsqueeze(0)
        A = self.qp_A; b = self.qp_b

        for attempt in range(3):
            try:
                reg = 1e-3 * (10 ** attempt) 
                Q_current = H.unsqueeze(0) + reg * torch.eye(H.shape[0], device=self.device).unsqueeze(0)
                
                import warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    U_opt = QPFunction(verbose=False, check_Q_spd=False, maxIter=100000, eps=1e-5)(Q_current, p, G, h, A, b)
                    if torch.isnan(U_opt).any() or not torch.isfinite(U_opt).all():
                        if attempt < 2: continue
                        return _fallback_zero()
                    
                    U_opt_final = U_opt.squeeze(0).clamp(lb, ub)
                    if U_opt_final.requires_grad:
                        U_opt_final.register_hook(lambda grad: torch.nan_to_num(grad, 0.0, 0.0, 0.0))
                    return U_opt_final
            except Exception:
                if attempt == 2: return _fallback_zero()
        return _fallback_zero()

    def control(
        self,
        current_state: torch.Tensor,
        x_lin_seq: torch.Tensor,
        u_lin_seq: torch.Tensor,
        x_goal: torch.Tensor,
        diag_corrections_Q: Optional[torch.Tensor] = None,
        diag_corrections_R: Optional[torch.Tensor] = None,
        gates_E: Optional[torch.Tensor] = None,
        Qf_dense: Optional[torch.Tensor] = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        H, f, U_bar = self.QP_formulation(
            current_state, u_lin_seq, x_goal,
            diag_corrections_Q=diag_corrections_Q,
            diag_corrections_R=diag_corrections_R,
            gates_E=gates_E,
            Qf_dense=Qf_dense,
        )
        
        lb_delta, ub_delta = self.build_constraints_delta(U_bar)
        Delta_U_opt = self.solve_mpc_qp(H, f, lb_delta, ub_delta)
        U_opt = U_bar + Delta_U_opt
        
        n_u = self.MPC_dynamics.u_min.shape[0]
        u_opt = torch.nan_to_num(U_opt[:n_u], nan=0.0, posinf=0.0, neginf=0.0)
        
        return u_opt, U_opt
