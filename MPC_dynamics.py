import torch


# Real hardware parameters (MAB Robotics double pendulum)
_M1 = 0.10548177618443695
_M2 = 0.07619744360415454
_L1 = 0.05    # link 1 length
_L2 = 0.05    # link 2 length
_R1 = 0.05    # CoM distance link 1
_R2 = 0.03670036749567022   # CoM distance link 2
_I1 = 0.00046166221821039165
_I2 = 0.00023702395072092597
_G  = 9.81
_U_LIM = 0.15   # Nm

# Joint viscous friction — must match true_dynamics so MPC predictions agree.
_BV1 = 0.005
_BV2 = 0.005


class DoublePendulumDynamics:
    """
    Rigid-body double pendulum matching the real hardware.

    State x format:  [q1, q1_dot, q2, q2_dot]
        q1      = absolute angle of link 1 (world frame, 0=down)
        q1_dot  = angular velocity of link 1
        q2      = RELATIVE angle of link 2 w.r.t. link 1
        q2_dot  = angular velocity of link 2

    Hardware state format is [q1, q2, q1_dot, q2_dot].
    Interface permutation: x_ours = x_hw[[0, 2, 1, 3]]

    Control:  tau = [u1, u2]  (joint torques, Nm)
    """

    def __init__(self, device=None, dtype=torch.float64, u_lim=_U_LIM):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.dtype  = dtype

        self.m1 = torch.tensor(_M1, device=self.device, dtype=self.dtype)
        self.m2 = torch.tensor(_M2, device=self.device, dtype=self.dtype)
        self.l1 = torch.tensor(_L1, device=self.device, dtype=self.dtype)
        self.l2 = torch.tensor(_L2, device=self.device, dtype=self.dtype)
        self.r1 = torch.tensor(_R1, device=self.device, dtype=self.dtype)
        self.r2 = torch.tensor(_R2, device=self.device, dtype=self.dtype)
        self.I1 = torch.tensor(_I1, device=self.device, dtype=self.dtype)
        self.I2 = torch.tensor(_I2, device=self.device, dtype=self.dtype)
        self.g  = torch.tensor(_G,  device=self.device, dtype=self.dtype)

        self.u_min = torch.tensor([-u_lim, -u_lim], device=self.device, dtype=self.dtype)
        self.u_max = torch.tensor([ u_lim,  u_lim], device=self.device, dtype=self.dtype)

        self.bv = torch.tensor([_BV1, _BV2], device=self.device, dtype=self.dtype)

    def compute_M_C_G(self, x: torch.Tensor):
        x = x.to(device=self.device, dtype=self.dtype)
        q1, q1_dot, q2, q2_dot = x[0], x[1], x[2], x[3]

        # Inertia matrix
        H   = self.m2 * self.l1 * self.r2 * torch.cos(q2)
        M11 = self.I1 + self.m1*self.r1**2 + self.I2 + self.m2*(self.l1**2 + self.r2**2) + 2*H
        M12 = self.I2 + self.m2*self.r2**2 + H
        M22 = self.I2 + self.m2*self.r2**2
        M = torch.stack([torch.stack([M11, M12]),
                         torch.stack([M12, M22])])

        # Coriolis / centrifugal matrix (C@qdot identical to hardware formulation)
        h   = self.m2 * self.l1 * self.r2 * torch.sin(q2)
        C11 = -2 * h * q2_dot
        C12 = -h * q2_dot
        C21 =  h * q1_dot
        C22 = torch.zeros((), device=self.device, dtype=self.dtype)
        C = torch.stack([torch.stack([C11, C12]),
                         torch.stack([C21, C22])])

        # Gravity torque vector
        G1 = -(self.m1*self.r1 + self.m2*self.l1)*self.g*torch.sin(q1) \
             - self.m2*self.g*self.r2*torch.sin(q1 + q2)
        G2 = -self.m2*self.g*self.r2*torch.sin(q1 + q2)
        G = torch.stack([G1, G2])

        return M, C, G

    def deriv(self, t, x, tau=None):
        x = x.to(device=self.device, dtype=self.dtype)
        if tau is None:
            tau = torch.zeros(2, device=self.device, dtype=self.dtype)
        else:
            tau = tau.to(device=self.device, dtype=self.dtype)

        q1, q1_dot, q2, q2_dot = x[0], x[1], x[2], x[3]
        q_dot = torch.stack([q1_dot, q2_dot])

        # Viscous friction.
        tau_eff = tau - self.bv * q_dot

        M, C, G = self.compute_M_C_G(x)
        q_ddot = torch.linalg.solve(M, tau_eff - C @ q_dot + G)

        return torch.stack([q1_dot, q_ddot[0], q2_dot, q_ddot[1]])

    def rk4_step(self, x, dt, tau_func=None):
        x  = x.to(device=self.device, dtype=self.dtype)
        dt = torch.as_tensor(dt, device=self.device, dtype=self.dtype)
        tau = torch.zeros(2, device=self.device, dtype=self.dtype) \
              if tau_func is None else tau_func(x).to(device=self.device, dtype=self.dtype)
        k1 = self.deriv(0.0, x,                  tau)
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
