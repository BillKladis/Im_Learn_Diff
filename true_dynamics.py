import torch

class DoublePendulumDynamics:
    """
    Relative-angle double pendulum dynamics.

    State x format:
        x = [q1, q1_dot, q2, q2_dot]
    where:
        q1      = absolute angle of link1 (world frame)
        q1_dot  = angular velocity of link1
        q2      = RELATIVE angle of link2 w.r.t. link1
        q2_dot  = angular velocity of link2

    Control:
        tau = [u1, u2]  (joint torques)

    Returns:
        xdot = [q1_dot, q1_ddot, q2_dot, q2_ddot]
    """
    def __init__(self, device=None, dtype=torch.float64):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.dtype  = dtype

        self.l1 = torch.tensor(0.5,  device=self.device, dtype=self.dtype)
        self.l2 = torch.tensor(0.5,  device=self.device, dtype=self.dtype)
        self.m1 = torch.tensor(1.0,  device=self.device, dtype=self.dtype)
        self.m2 = torch.tensor(1.0,  device=self.device, dtype=self.dtype)
        self.g  = torch.tensor(9.81, device=self.device, dtype=self.dtype)

        self.u_min = torch.tensor([-3.0, -3.0], device=self.device, dtype=self.dtype)
        self.u_max = torch.tensor([ 3.0,  3.0], device=self.device, dtype=self.dtype)

    def compute_M_C_G(self, x: torch.Tensor):
        x = x.to(device=self.device, dtype=self.dtype)
        
        # Restored to original PyTorch ordering
        q1, q1_dot, q2, q2_dot = x[0], x[1], x[2], x[3]

        # JAX formulation ported to PyTorch
        M11 = self.l1**2 * self.m1 + self.l2**2 * self.m2 + self.l1**2 * self.m2 + \
            2 * self.l1 * self.m2 * self.l2 * torch.cos(q2)
        M12 = self.l2**2 * self.m2 + self.l1 * self.m2 * self.l2 * torch.cos(q2)
        M21 = M12
        M22 = self.l2**2 * self.m2
        
        M = torch.stack([
            torch.stack([M11, M12]),
            torch.stack([M21, M22])
        ])

        C11 = -2 * q2_dot * self.l1 * self.m2 * self.l2 * torch.sin(q2)
        C12 = -q2_dot * self.l1 * self.m2 * self.l2 * torch.sin(q2)
        C21 = q1_dot * self.l1 * self.m2 * self.l2 * torch.sin(q2)
        C22 = torch.zeros((), device=self.device, dtype=self.dtype)
        
        C = torch.stack([
            torch.stack([C11, C12]),
            torch.stack([C21, C22])
        ])

        G1 = -self.g * self.m1 * self.l1 * torch.sin(q1) - \
            self.g * self.m2 * (self.l1 * torch.sin(q1) + self.l2 * torch.sin(q1 + q2))
        G2 = -self.g * self.m2 * self.l2 * torch.sin(q1 + q2)
        
        G = torch.stack([G1, G2])

        return M, C, G

    def deriv(self, t, x, tau=None):
        x = x.to(device=self.device, dtype=self.dtype)

        if tau is None:
            tau = torch.zeros(2, device=self.device, dtype=self.dtype)
        else:
            tau = tau.to(device=self.device, dtype=self.dtype)

        # Restored to original PyTorch ordering
        q1, q1_dot, q2, q2_dot = x[0], x[1], x[2], x[3]
        q_dot = torch.stack([q1_dot, q2_dot])

        M, C, G = self.compute_M_C_G(x)
        
        right_hand_side = tau - (C @ q_dot) + G
        q_ddot = torch.linalg.solve(M, right_hand_side)

        # Output restored to original PyTorch derivative order
        return torch.stack([q1_dot, q_ddot[0], q2_dot, q_ddot[1]])

    def rk4_step(self, x, dt, tau_func=None):
        x  = x.to(device=self.device, dtype=self.dtype)
        dt = torch.as_tensor(dt, device=self.device, dtype=self.dtype)

        if tau_func is None:
            tau = torch.zeros(2, device=self.device, dtype=self.dtype)
        else:
            tau = tau_func(x).to(device=self.device, dtype=self.dtype)

        k1 = self.deriv(0.0, x,                 tau)
        k2 = self.deriv(0.0, x + 0.5 * dt * k1, tau)
        k3 = self.deriv(0.0, x + 0.5 * dt * k2, tau)
        k4 = self.deriv(0.0, x +       dt * k3, tau)

        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def step(self, x0, dt, tau_func=None, n_steps=10):
        x  = x0.clone().to(device=self.device, dtype=self.dtype)
        dt = torch.as_tensor(dt, device=self.device, dtype=self.dtype)

        traj = [x.clone()]
        h = dt / n_steps
        for _ in range(n_steps):
            x = self.rk4_step(x, h, tau_func)
            traj.append(x.clone())
        return torch.stack(traj, dim=0)