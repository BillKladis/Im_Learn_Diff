"""
mpc_controller.py — Linearised receding-horizon MPC with switchable QP backend.

Two QP solvers, sharing the same cost-matrix construction:

  solver_backend="cvx"  (default)
      cvxpylayers + SCS.  Slower (~50–300 ms per solve) but DIFFERENTIABLE
      via implicit-diff through KKT.  Required for training.

  solver_backend="osqp"
      OSQP direct C solver.  Fast (~1–5 ms per solve) but NOT differentiable.
      Use at deployment time when no gradients are needed.

QP form (control-space, delta-u parameterisation):
    min  ½ ΔUᵀ H ΔU + fᵀ ΔU
    s.t. lb ≤ ΔU ≤ ub

The cvxpylayers DPP-compliant formulation passes H via its Cholesky-like
square root H_sqrt where H_sqrt.T @ H_sqrt = H.  OSQP takes H directly
in upper-triangular CSC form.
"""

from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from torch.func import jacrev, vmap

import MPC_dynamics
import true_dynamics


class MPC_controller:
    def __init__(
        self,
        x0:     torch.Tensor,
        x_goal: torch.Tensor,
        N:      int,
        device: torch.device,
        u_lim:  float = 0.15,
        solver_backend: str = "cvx",   # "cvx" (differentiable) or "osqp" (fast)
        qp_eps: float = 1e-3,           # OSQP convergence tolerance
        qp_max_iters: int = 200,        # OSQP iteration cap (also used by SCS)
    ):
        self.device = device
        self.x0     = x0.detach().clone().to(device=device, dtype=torch.float64)
        self.x_goal = x_goal.detach().clone().to(device=device, dtype=torch.float64)
        self.N      = N
        self.dt     = torch.tensor(0.05, device=device, dtype=torch.float64)

        # Baseline cost diagonals — scaled for real hardware (m~0.1kg, l=0.05m, u_max=0.15Nm).
        # B_vel ≈ 55-90 per step (small inertia → large acceleration response).
        # Q_vel must stay ≤ 0.0001 to keep H well-conditioned at all linearisation points.
        # Q_pos = 0.1 gives strong position drive (cost ratio Q*π²/R ≈ 1 >> R*u_max²=0.02).
        # Top conditioning: cond ≈ 2.3e6 (acceptable for SCS with eps=1e-6).
        # State order: [q1, q1_dot, q2, q2_dot].  Hardware order: [q1, q2, q1_dot, q2_dot].
        self.q_base_diag = torch.tensor(
            [0.1, 0.0001, 0.1, 0.0001], device=device, dtype=torch.float64
        )
        self.r_base_diag = torch.tensor(
            [1.0, 1.0], device=device, dtype=torch.float64
        )
        self.Qf = torch.diag(torch.tensor(
            [0.2, 0.0002, 0.2, 0.0002], device=device, dtype=torch.float64
        ))

        self.true_dynamics = true_dynamics.DoublePendulumDynamics(device=device, u_lim=u_lim)
        self.MPC_dynamics  = MPC_dynamics.DoublePendulumDynamics(device=device, u_lim=u_lim)

        self.n_u_total = self.N * self.MPC_dynamics.u_min.shape[0]
        self.qp_fallback_count = 0

        # Stash solver-tuning parameters for both backends.
        self.qp_eps = float(qp_eps)
        self.qp_max_iters = int(qp_max_iters)

        # Build the chosen backend.  cvx is the default (covers training).
        if solver_backend not in ("cvx", "osqp"):
            raise ValueError(
                f"solver_backend must be 'cvx' or 'osqp', got {solver_backend!r}"
            )
        self.solver_backend = solver_backend
        if solver_backend == "cvx":
            self._build_qp_layer()
        else:
            self._build_osqp_workspace()

    # ──────────────────────────────────────────────────────────────────────
    # cvxpylayers QP construction
    # ──────────────────────────────────────────────────────────────────────
    def _build_qp_layer(self):
        n = self.n_u_total
        DU = cp.Variable(n)

        H_sqrt = cp.Parameter((n, n))   # H = H_sqrt.T @ H_sqrt
        f_par  = cp.Parameter(n)
        lb_par = cp.Parameter(n)
        ub_par = cp.Parameter(n)

        objective = cp.Minimize(
            0.5 * cp.sum_squares(H_sqrt @ DU) + f_par @ DU
        )
        constraints = [DU >= lb_par, DU <= ub_par]

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), "QP is not DPP-compliant"

        self.qp_layer = CvxpyLayer(
            problem,
            parameters=[H_sqrt, f_par, lb_par, ub_par],
            variables=[DU],
        )

    # ──────────────────────────────────────────────────────────────────────
    # OSQP workspace (fast direct solver, no autograd)
    # ──────────────────────────────────────────────────────────────────────
    def _build_osqp_workspace(self):
        """Pre-build a reusable OSQP workspace.

        OSQP wants P in upper-triangular CSC form.  We allocate a fully
        dense upper-triangular pattern at setup so that on every solve the
        pattern (nnz indices) is unchanged — only the .data values move.
        That lets us call `update(Px=...)` instead of `setup(...)` each
        step, which is the difference between ~1 ms and ~10 ms per solve.
        """
        import osqp
        import scipy.sparse as sp

        n = self.n_u_total

        # Fully-dense upper-triangular pattern (n*(n+1)/2 nonzeros).
        # Setting all to 1.0 first guarantees no entries get dropped.
        P_dense_init = np.triu(np.ones((n, n), dtype=np.float64))
        P_csc = sp.csc_matrix(P_dense_init)
        # Constraint matrix for box constraints is just identity.
        A_csc = sp.eye(n, format="csc", dtype=np.float64)

        self.osqp_prob = osqp.OSQP()
        self.osqp_prob.setup(
            P_csc, np.zeros(n), A_csc,
            np.full(n, -1e6), np.full(n, 1e6),
            eps_abs=self.qp_eps,
            eps_rel=self.qp_eps,
            max_iter=self.qp_max_iters,
            verbose=False,
            polish=False,         # polishing adds ~0.5 ms; box QPs rarely need it
            warm_start=True,      # carry over previous solution as warm start
            adaptive_rho=False,   # adaptive_rho costs Python overhead per solve
            check_termination=25, # check convergence every 25 iters (cheap)
            scaling=10,           # equilibration helps conditioning (~free)
        )

        # Cache the (row, col) index arrays so we can extract H values from a
        # dense numpy array in CSC column-major order.  scipy.sparse.find
        # returns (rows, cols, data) — we want them in the same order OSQP's
        # internal Px array expects, which is the order P_csc.data was built.
        rows = []
        cols = []
        for j in range(n):
            for i in range(j + 1):
                rows.append(i)
                cols.append(j)
        self._osqp_P_rows = np.asarray(rows, dtype=np.int64)
        self._osqp_P_cols = np.asarray(cols, dtype=np.int64)

    # ──────────────────────────────────────────────────────────────────────
    # Cost matrices
    # ──────────────────────────────────────────────────────────────────────
    def build_cost_matrices(
        self,
        diag_corrections_Q:  Optional[torch.Tensor] = None,
        diag_corrections_R:  Optional[torch.Tensor] = None,
        diag_corrections_Qf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if diag_corrections_Q is not None:
            q_k = self.q_base_diag * diag_corrections_Q
            Q_run = q_k.flatten()
        else:
            Q_run = self.q_base_diag.repeat(self.N - 1)

        Q_bar = torch.diag(torch.cat([
            Q_run,
            torch.zeros(4, device=self.device, dtype=torch.float64),
        ]))
        if diag_corrections_Qf is not None:
            Q_bar[-4:, -4:] = torch.diag(self.Qf.diag() * diag_corrections_Qf)
        else:
            Q_bar[-4:, -4:] = self.Qf

        if diag_corrections_R is not None:
            r_k = self.r_base_diag * diag_corrections_R
            R_diag = r_k.flatten()
        else:
            R_diag = self.r_base_diag.repeat(self.N)

        return Q_bar, R_diag

    # ──────────────────────────────────────────────────────────────────────
    # Energy helper
    # ──────────────────────────────────────────────────────────────────────
    def compute_energy_single(self, x: torch.Tensor) -> torch.Tensor:
        """Total mechanical energy (T + V) for the real rigid-body hardware."""
        dyn = self.true_dynamics
        m1, m2 = dyn.m1, dyn.m2
        l1, r1, r2 = dyn.l1, dyn.r1, dyn.r2
        I1, I2 = dyn.I1, dyn.I2
        g = dyn.g

        q1, q1_dot, q2, q2_dot = x[0], x[1], x[2], x[3]

        # Potential energy (reference: both links fully down)
        V = -m1*g*r1*torch.cos(q1) - m2*g*(l1*torch.cos(q1) + r2*torch.cos(q1 + q2))

        # Kinetic energy: rotational + translational CoM
        # CoM1 velocity magnitude squared: (r1*q1_dot)^2
        v1_sq = (r1 * q1_dot) ** 2
        KE1 = 0.5*m1*v1_sq + 0.5*I1*q1_dot**2

        # CoM2 velocity
        vx2 = l1*torch.cos(q1)*q1_dot + r2*torch.cos(q1+q2)*(q1_dot+q2_dot)
        vy2 = l1*torch.sin(q1)*q1_dot + r2*torch.sin(q1+q2)*(q1_dot+q2_dot)
        v2_sq = vx2**2 + vy2**2
        KE2 = 0.5*m2*v2_sq + 0.5*I2*(q1_dot+q2_dot)**2

        return KE1 + KE2 + V

    # ──────────────────────────────────────────────────────────────────────
    # Discretisation
    # ──────────────────────────────────────────────────────────────────────
    def true_RK4_disc(self, x: torch.Tensor, u: torch.Tensor, dt: torch.Tensor,
                      n_sub: int = 10) -> torch.Tensor:
        # 10 sub-steps of h=dt/10=0.005s prevent Coriolis overflow (M22^{-1}≈2944).
        h = dt / n_sub
        for _ in range(n_sub):
            t0 = h.new_zeros(())
            tH = 0.5 * h
            t1 = h
            f1 = self.true_dynamics.deriv(t0, x, u)
            f2 = self.true_dynamics.deriv(tH, x + 0.5 * h * f1, u)
            f3 = self.true_dynamics.deriv(tH, x + 0.5 * h * f2, u)
            f4 = self.true_dynamics.deriv(t1, x + h * f3, u)
            x = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        return x

    def MPC_RK4_disc(self, x: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        t0 = dt.new_zeros(())
        tH = 0.5 * dt
        t1 = dt
        f1 = self.MPC_dynamics.deriv(t0, x, u)
        f2 = self.MPC_dynamics.deriv(tH, x + 0.5 * dt * f1, u)
        f3 = self.MPC_dynamics.deriv(tH, x + 0.5 * dt * f2, u)
        f4 = self.MPC_dynamics.deriv(t1, x + dt * f3, u)
        return x + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

    # ──────────────────────────────────────────────────────────────────────
    # Linearisation
    # ──────────────────────────────────────────────────────────────────────
    def linearize_discrete(
        self,
        x_batch: torch.Tensor,
        u_batch: torch.Tensor,
        dt:      torch.Tensor,
    ):
        def step_fn(x, u):
            return self.MPC_RK4_disc(x, u, dt)

        jac_x_fn = vmap(jacrev(step_fn, argnums=0))
        jac_u_fn = vmap(jacrev(step_fn, argnums=1))
        A_batch = jac_x_fn(x_batch.detach(), u_batch.detach())
        B_batch = jac_u_fn(x_batch.detach(), u_batch.detach())
        return list(A_batch), list(B_batch)

    def linearize_horizon(self, x_lin_seq, u_lin_seq):
        return self.linearize_discrete(x_lin_seq, u_lin_seq, self.dt)

    # ──────────────────────────────────────────────────────────────────────
    # Nominal rollout helper (used by Simulate.py for energy shaping)
    # ──────────────────────────────────────────────────────────────────────
    def compute_nominal_rollout(
        self,
        current_state: torch.Tensor,
        u_guess_seq:   torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_op_list:  list = []
        X_bar_list: list = []
        curr = current_state.detach()
        for i in range(self.N):
            x_op_list.append(curr)
            curr = self.MPC_RK4_disc(curr, u_guess_seq[i].detach(), self.dt)
            X_bar_list.append(curr)
        x_op_seq  = torch.stack(x_op_list)
        X_bar_seq = torch.stack(X_bar_list)
        _, B_list = self.linearize_horizon(x_op_seq, u_guess_seq.detach())
        return X_bar_seq, B_list

    # ──────────────────────────────────────────────────────────────────────
    # Prediction matrices
    # ──────────────────────────────────────────────────────────────────────
    def build_prediction_matrices(self, A_seq, B_seq):
        N = len(A_seq)
        n_x = A_seq[0].shape[0]
        n_u = B_seq[0].shape[1]

        Phi_rows = []
        Phi_prev = A_seq[0]
        Phi_rows.append(Phi_prev)
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

    # ──────────────────────────────────────────────────────────────────────
    # QP cost in delta-u form
    # ──────────────────────────────────────────────────────────────────────
    def build_qp_matrices_delta(
        self,
        B_big:                torch.Tensor,
        X_bar:                torch.Tensor,
        U_bar:                torch.Tensor,
        x_goal:               torch.Tensor,
        Q_bar:                torch.Tensor,
        R_diag:               torch.Tensor,
        extra_linear_control: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the QP cost matrices H, f.

        extra_linear_control: (N*nu,) added DIRECTLY to f in control space,
                              bypassing B̄ᵀ mapping. Used for τ1-only energy
                              shaping (see Simulate.py).
        """
        X_ref = x_goal.unsqueeze(0).expand(self.N, -1).reshape(-1)
        E_raw = X_bar - X_ref

        # Angle-wrap q1 and q2 errors into (−π, π].
        q1_idx = torch.arange(0, 4 * self.N, 4, device=self.device)
        q2_idx = torch.arange(2, 4 * self.N, 4, device=self.device)
        angle_idx = torch.cat([q1_idx, q2_idx])
        E = E_raw.clone()
        E[angle_idx] = torch.atan2(
            torch.sin(E_raw[angle_idx]),
            torch.cos(E_raw[angle_idx]),
        )

        H = 2.0 * (B_big.T @ Q_bar @ B_big) + torch.diag(2.0 * R_diag)
        H = 0.5 * (H + H.T)
        H = H + 1e-4 * torch.eye(H.shape[0], device=self.device, dtype=torch.float64)

        state_linear_term = 2.0 * (Q_bar @ E)
        f = B_big.T @ state_linear_term + 2.0 * (R_diag * U_bar)
        if extra_linear_control is not None:
            f = f + extra_linear_control

        return H, f

    def build_constraints_delta(self, U_bar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lb_base = self.MPC_dynamics.u_min.repeat(self.N)
        ub_base = self.MPC_dynamics.u_max.repeat(self.N)
        return lb_base - U_bar, ub_base - U_bar

    # ──────────────────────────────────────────────────────────────────────
    # QP formulation orchestration
    # ──────────────────────────────────────────────────────────────────────
    def QP_formulation(
        self,
        current_state:        torch.Tensor,
        u_guess_seq:          torch.Tensor,
        x_goal:               torch.Tensor,
        diag_corrections_Q:   Optional[torch.Tensor] = None,
        diag_corrections_R:   Optional[torch.Tensor] = None,
        extra_linear_control: Optional[torch.Tensor] = None,
        diag_corrections_Qf:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x_op_list  = []
        X_bar_list = []
        curr_x = current_state

        for i in range(self.N):
            x_op_list.append(curr_x)
            curr_x = self.MPC_RK4_disc(curr_x, u_guess_seq[i], self.dt)
            X_bar_list.append(curr_x)

        x_op_seq = torch.stack(x_op_list)
        X_bar    = torch.cat(X_bar_list)
        U_bar    = u_guess_seq.reshape(-1)

        A_list, B_list = self.linearize_horizon(x_op_seq, u_guess_seq)
        _, B_big = self.build_prediction_matrices(A_list, B_list)

        Q_bar, R_diag = self.build_cost_matrices(
            diag_corrections_Q=diag_corrections_Q,
            diag_corrections_R=diag_corrections_R,
            diag_corrections_Qf=diag_corrections_Qf,
        )

        H, f = self.build_qp_matrices_delta(
            B_big, X_bar, U_bar, x_goal, Q_bar, R_diag,
            extra_linear_control=extra_linear_control,
        )

        return H, f, U_bar

    # ──────────────────────────────────────────────────────────────────────
    # QP solve via cvxpylayers
    # ──────────────────────────────────────────────────────────────────────
    def solve_mpc_qp(
        self,
        H:  torch.Tensor,
        f:  torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the QP and return ΔU* (n,).  Dispatches to the chosen backend."""
        if self.solver_backend == "osqp":
            return self._solve_mpc_qp_osqp(H, f, lb, ub)
        return self._solve_mpc_qp_cvx(H, f, lb, ub)

    def _solve_mpc_qp_cvx(
        self,
        H:  torch.Tensor,
        f:  torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ) -> torch.Tensor:
        """cvxpylayers backend — slower but differentiable.  Used in training."""

        def _fallback_zero():
            self.qp_fallback_count += 1
            return torch.zeros(self.n_u_total, device=self.device, dtype=torch.float64)

        if not torch.isfinite(H).all() or not torch.isfinite(f).all():
            return _fallback_zero()

        # Cholesky of H to get H_sqrt such that H_sqrt.T @ H_sqrt = H.
        try:
            H_reg = H + 1e-6 * torch.eye(H.shape[0], device=self.device, dtype=torch.float64)
            L = torch.linalg.cholesky(H_reg)   # L @ L.T = H_reg
            H_sqrt = L.T                        # H_sqrt.T @ H_sqrt = L @ L.T = H_reg
        except Exception:
            return _fallback_zero()

        try:
            (DU_opt,) = self.qp_layer(
                H_sqrt, f, lb, ub,
                solver_args={
                    "solve_method": "SCS",
                    "eps": self.qp_eps,
                    "max_iters": self.qp_max_iters,
                },
            )
        except Exception:
            return _fallback_zero()

        if not torch.isfinite(DU_opt).all():
            return _fallback_zero()

        DU_opt = DU_opt.clamp(lb, ub)
        if DU_opt.requires_grad:
            DU_opt.register_hook(
                lambda grad: torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            )
        return DU_opt

    def _solve_mpc_qp_osqp(
        self,
        H:  torch.Tensor,
        f:  torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ) -> torch.Tensor:
        """OSQP backend — direct C QP solver, ~10–100× faster than cvx.

        Not differentiable.  Use only at deployment / eval time.  Updates
        the pre-built workspace's P (upper-tri values), q, l, u in place.
        """
        def _fallback_zero():
            self.qp_fallback_count += 1
            return torch.zeros(self.n_u_total, device=self.device, dtype=torch.float64)

        if not torch.isfinite(H).all() or not torch.isfinite(f).all():
            return _fallback_zero()

        # Symmetric ridge for numerical safety (matches cvx path).
        H_np = H.detach().cpu().numpy().astype(np.float64, copy=False)
        n = H_np.shape[0]
        H_np = H_np + 1e-6 * np.eye(n, dtype=np.float64)

        # Extract upper-triangular values in the same column-major CSC order
        # OSQP saw at setup time.
        Px = H_np[self._osqp_P_rows, self._osqp_P_cols]

        f_np  = f.detach().cpu().numpy().astype(np.float64, copy=False)
        lb_np = lb.detach().cpu().numpy().astype(np.float64, copy=False)
        ub_np = ub.detach().cpu().numpy().astype(np.float64, copy=False)

        try:
            self.osqp_prob.update(Px=Px, q=f_np, l=lb_np, u=ub_np)
            result = self.osqp_prob.solve()
        except Exception:
            return _fallback_zero()

        # OSQP status: 'solved' / 'solved_inaccurate' are both usable.
        # Anything else (max_iter_reached, primal_infeasible, ...) → fallback.
        status = getattr(result.info, "status", "")
        if status not in ("solved", "solved inaccurate", "solved_inaccurate"):
            return _fallback_zero()

        DU = np.asarray(result.x, dtype=np.float64)
        if not np.isfinite(DU).all():
            return _fallback_zero()

        DU_t = torch.from_numpy(DU).to(device=self.device, dtype=torch.float64)
        DU_t = DU_t.clamp(lb, ub)
        return DU_t

    # ──────────────────────────────────────────────────────────────────────
    # Top-level control entry point
    # ──────────────────────────────────────────────────────────────────────
    def control(
        self,
        current_state:        torch.Tensor,
        x_lin_seq:            torch.Tensor,   # kept for API compatibility
        u_lin_seq:            torch.Tensor,
        x_goal:               torch.Tensor,
        diag_corrections_Q:   Optional[torch.Tensor] = None,
        diag_corrections_R:   Optional[torch.Tensor] = None,
        extra_linear_control: Optional[torch.Tensor] = None,
        diag_corrections_Qf:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        H, f, U_bar = self.QP_formulation(
            current_state, u_lin_seq, x_goal,
            diag_corrections_Q=diag_corrections_Q,
            diag_corrections_R=diag_corrections_R,
            extra_linear_control=extra_linear_control,
            diag_corrections_Qf=diag_corrections_Qf,
        )

        lb_delta, ub_delta = self.build_constraints_delta(U_bar)
        Delta_U_opt = self.solve_mpc_qp(H, f, lb_delta, ub_delta)
        U_opt = U_bar + Delta_U_opt

        n_u = self.MPC_dynamics.u_min.shape[0]
        u_opt = torch.nan_to_num(U_opt[:n_u], nan=0.0, posinf=0.0, neginf=0.0)

        return u_opt, U_opt