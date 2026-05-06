"""ekf.py — Extended Kalman Filter for double-pendulum state + disturbance estimation.

Two variants:
  EKF4   — standard 4-state EKF; filters measurement noise only.
  EKF6   — augmented 6-state EKF; estimates state + unknown constant torque bias.
           Enables online disturbance cancellation for systematic errors.

Usage:
    filter = EKF6(mpc, Q_state, Q_bias, R_obs)
    filter.reset(x0)
    for step in range(N):
        x_est, bias_est = filter.step(y_measured, u_commanded)
        u_apply = u_from_mpc - bias_est   # cancel estimated bias
"""

import torch


class EKF4:
    """Standard 4-state EKF: filters noisy state observations."""

    def __init__(self, mpc, Q: torch.Tensor, R: torch.Tensor,
                 jac_cache: int = 1):
        """
        mpc        : MPC_controller (provides true_RK4_disc)
        Q          : (4,4) process noise covariance
        R          : (4,4) measurement noise covariance
        jac_cache  : recompute the Jacobian F every K steps (1 = every step,
                     no cache).  Higher K trades estimation precision for
                     speed: F is the dominant cost (~60ms via autograd).
                     Reasonable: K∈{1,3,5}.  Risky: K>10.
        """
        self.mpc = mpc
        self.Q = Q.double()
        self.R = R.double()
        self.dt = mpc.dt
        self.x_est = None
        self.P = None
        self.jac_cache = max(1, int(jac_cache))
        self._jac_step = 0
        self._F_cached = None

    def reset(self, x0: torch.Tensor, P0: torch.Tensor = None):
        self.x_est = x0.detach().clone().double()
        self.P = P0 if P0 is not None else torch.eye(4, dtype=torch.float64) * 0.01
        self._jac_step = 0
        self._F_cached = None

    def _compute_jacobian(self, x, u):
        """Compute F = d(RK4)/dx via autograd."""
        x_in = x.detach().clone().requires_grad_(True)
        x_next = self.mpc.true_RK4_disc(x_in, u.detach(), self.dt)
        F = torch.zeros(4, 4, dtype=torch.float64)
        for i in range(4):
            grad = torch.autograd.grad(x_next[i], x_in, retain_graph=(i < 3))[0]
            F[i] = grad.detach()
        return F

    def _jacobian(self, x, u):
        """Cached Jacobian: recompute every jac_cache steps, reuse otherwise."""
        if self._F_cached is None or (self._jac_step % self.jac_cache == 0):
            self._F_cached = self._compute_jacobian(x, u)
        self._jac_step += 1
        return self._F_cached

    def step(self, y: torch.Tensor, u: torch.Tensor):
        """
        Predict + update one step.
        y : (4,) measured state (noisy)
        u : (4,) commanded control
        Returns: x_est (4,), bias_est (always zeros for EKF4)
        """
        x = self.x_est.detach()
        P = self.P

        # ── Predict ────────────────────────────────────────────────────────
        # x_pred is ALWAYS computed fresh from current x (nonlinear RK4).
        # Only F (linearisation for covariance propagation) may be cached.
        x_pred = self.mpc.true_RK4_disc(x, u.detach(), self.dt).detach()
        F = self._jacobian(x, u)
        P_pred = F @ P @ F.t() + self.Q

        # ── Update ─────────────────────────────────────────────────────────
        S = P_pred + self.R                          # H=I, so S = P_pred + R
        K = torch.linalg.solve(S.t(), P_pred.t()).t()  # K = P_pred S^{-1}
        innov = y.detach() - x_pred
        self.x_est = x_pred + K @ innov
        self.P = (torch.eye(4, dtype=torch.float64) - K) @ P_pred

        return self.x_est, torch.zeros(2, dtype=torch.float64)


class EKF6:
    """Augmented 6-state EKF: estimates state x (4D) + torque bias d (2D).

    State: [q1, q1d, q2, q2d, d1, d2]
    Process:
        x_{k+1} = RK4(x_k, u_k + d_k)
        d_{k+1} = d_k   (bias constant; random walk via Q_bias)
    Measurement:  y = x[0:4]  (observe state, not bias)
    """

    def __init__(self, mpc, Q_state: torch.Tensor, Q_bias: torch.Tensor,
                 R: torch.Tensor, bias_clamp: float = 3.0,
                 jac_cache: int = 1):
        """
        Q_state   : (4,4) state process noise covariance
        Q_bias    : (2,2) bias random-walk covariance (controls tracking speed)
        R         : (4,4) measurement noise covariance
        bias_clamp: saturate |d_est| to this value (actuator-limit-scale)
        jac_cache : recompute (F, G) every K steps; see EKF4.__init__.  The
                    Jacobians are the cost (~60ms via autograd).
        """
        self.mpc = mpc
        self.Q_state = Q_state.double()
        self.Q_bias  = Q_bias.double()
        self.R = R.double()
        self.dt = mpc.dt
        self.bias_clamp = bias_clamp
        self.x_aug = None   # (6,)
        self.P = None       # (6,6)
        self.jac_cache = max(1, int(jac_cache))
        self._jac_step = 0
        self._FG_cached = None

        # Measurement Jacobian H = [I_4 | 0_{4x2}]
        self.H = torch.cat([torch.eye(4, dtype=torch.float64),
                            torch.zeros(4, 2, dtype=torch.float64)], dim=1)

    def reset(self, x0: torch.Tensor, d0: torch.Tensor = None, P0: torch.Tensor = None):
        x = x0.detach().clone().double()
        d = d0.detach().clone().double() if d0 is not None else torch.zeros(2, dtype=torch.float64)
        self.x_aug = torch.cat([x, d])
        if P0 is not None:
            self.P = P0
        else:
            P = torch.zeros(6, 6, dtype=torch.float64)
            P[:4, :4] = torch.eye(4) * 0.01
            P[4:, 4:] = torch.eye(2) * 0.1
            self.P = P
        self._jac_step = 0
        self._FG_cached = None

    def _compute_jacobians(self, x, u_eff):
        """F = d(RK4)/dx, G = d(RK4)/du, both (4,4) and (4,2)."""
        x_in = x.detach().clone().requires_grad_(True)
        u_in = u_eff.detach().clone().requires_grad_(True)
        x_next = self.mpc.true_RK4_disc(x_in, u_in, self.dt)
        F = torch.zeros(4, 4, dtype=torch.float64)
        G = torch.zeros(4, 2, dtype=torch.float64)
        for i in range(4):
            retain = (i < 3)
            grads = torch.autograd.grad(x_next[i], [x_in, u_in], retain_graph=retain)
            F[i] = grads[0].detach()
            G[i] = grads[1].detach()
        return F, G

    def _jacobians(self, x, u_eff):
        """Cached Jacobians: recompute every jac_cache steps, reuse otherwise."""
        if self._FG_cached is None or (self._jac_step % self.jac_cache == 0):
            self._FG_cached = self._compute_jacobians(x, u_eff)
        self._jac_step += 1
        return self._FG_cached

    def step(self, y: torch.Tensor, u: torch.Tensor):
        """
        Predict + update.
        y : (4,) noisy state measurement
        u : (2,) commanded control (BEFORE disturbance compensation)
        Returns: x_est (4,), bias_est (2,)
        """
        x    = self.x_aug[:4]
        d    = self.x_aug[4:]
        P    = self.P

        u_eff = (u.detach() + d).clamp(-3.0, 3.0)

        # ── Predict ────────────────────────────────────────────────────────
        x_pred = self.mpc.true_RK4_disc(x.detach(), u_eff.detach(), self.dt).detach()
        d_pred = d.detach()
        x_aug_pred = torch.cat([x_pred, d_pred])

        F, G = self._jacobians(x, u_eff)

        # Augmented Jacobian F_aug = [[F, G], [0, I]]
        F_aug = torch.zeros(6, 6, dtype=torch.float64)
        F_aug[:4, :4] = F
        F_aug[:4, 4:] = G
        F_aug[4:, 4:] = torch.eye(2, dtype=torch.float64)

        # Augmented process noise Q_aug = diag(Q_state, Q_bias)
        Q_aug = torch.zeros(6, 6, dtype=torch.float64)
        Q_aug[:4, :4] = self.Q_state
        Q_aug[4:, 4:] = self.Q_bias

        P_pred = F_aug @ P @ F_aug.t() + Q_aug

        # ── Update ─────────────────────────────────────────────────────────
        H = self.H
        S = H @ P_pred @ H.t() + self.R              # (4,4)
        K = P_pred @ H.t() @ torch.linalg.inv(S)     # (6,4)
        innov = y.detach() - x_aug_pred[:4]
        x_aug_new = x_aug_pred + K @ innov

        # Clamp bias estimate to physical range
        x_aug_new[4:] = x_aug_new[4:].clamp(-self.bias_clamp, self.bias_clamp)

        I_KH = torch.eye(6, dtype=torch.float64) - K @ H
        self.P = I_KH @ P_pred
        self.x_aug = x_aug_new

        return x_aug_new[:4], x_aug_new[4:]
