"""Quick test: can the QP stabilize near upright if f_extra is forced to zero?

This tests whether the MPC alone (without network f_extra) can hold the
pendulum at the upright when starting slightly off-center. If yes, then
teaching the network to output f_extra=0 near the top is sufficient.
"""
import math, sys, os
import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

def wrap_pi(x): return math.atan2(math.sin(x), math.cos(x))

class ZeroFNetWrapper:
    """Wraps a LinearizationNetwork but zeroes f_extra when near upright."""
    def __init__(self, net, zero_f_near_pi=False, pi_thresh=0.5):
        self.net = net
        self.zero_f_near_pi = zero_f_near_pi
        self.pi_thresh = pi_thresh

    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def eval(self): self.net.eval()
    def train(self): self.net.train()

    def forward(self, x_seq, **kwargs):
        gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf = self.net(x_seq, **kwargs)
        if self.zero_f_near_pi:
            q1 = float(x_seq[-1, 0].item())
            near_pi = (1.0 + math.cos(q1 - math.pi)) / 2.0
            if near_pi > 0.5:  # within ~π/2 of upright
                f_extra = f_extra * 0.0
        return gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf


# Monkey-patch approach: just override the network's forward at inference
class ManualNet(torch.nn.Module):
    def __init__(self, base_net, force_zero_f=False, zero_threshold=0.5):
        super().__init__()
        self.base_net = base_net
        self.force_zero_f = force_zero_f
        self.zero_threshold = zero_threshold
        self.f_extra_bound = base_net.f_extra_bound

    def forward(self, x_seq, q_base_diag=None, r_base_diag=None):
        gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf = \
            self.base_net(x_seq, q_base_diag=q_base_diag, r_base_diag=r_base_diag)
        if self.force_zero_f:
            q1 = float(x_seq[-1, 0].item())
            near_pi = (1.0 + math.cos(q1 - math.pi)) / 2.0
            if near_pi > self.zero_threshold:
                f_extra = torch.zeros_like(f_extra)
        return gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf

    def eval(self): return self.base_net.eval()


def eval_net(model, mpc, x0, x_goal, steps=600, label=""):
    x_t, _ = train_module.rollout(
        lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps
    )
    traj = x_t.cpu().numpy()
    wraps = np.array([
        math.sqrt(wrap_pi(s[0]-math.pi)**2 + s[1]**2 + s[2]**2 + s[3]**2)
        for s in traj
    ])
    arr = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    f01 = float((wraps < 0.10).mean())
    f03 = float((wraps < 0.30).mean())
    print(f"  {label}: arr={arr}  frac<0.10={f01:.1%}  frac<0.30={f03:.1%}  "
          f"wrap@end={wraps[-1]:.4f}")


def main():
    device = torch.device("cpu")
    x0 = torch.zeros(4, dtype=torch.float64)
    x_goal = torch.tensor([math.pi, 0., 0., 0.], dtype=torch.float64)
    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=10, device=device)
    mpc.dt = torch.tensor(0.05, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor([12., 5., 50., 40.], dtype=torch.float64)

    # Load best available model (ep30 checkpoint)
    import glob
    ckpts = sorted(glob.glob("saved_models/stageD_posonly_ft_*_ep030/*.pth"))
    if not ckpts:
        ckpts = sorted(glob.glob("saved_models/stageD_nodemo_20260428_123448/*.pth"))
        label = "0.0612 baseline"
    else:
        label = "posonly_ft ep30"
    ckpt = ckpts[-1]
    print(f"Loading: {ckpt}")

    base_net = network_module.LinearizationNetwork.load(ckpt, device="cpu").double()

    print(f"\n600-step rollouts from x0=zero:")
    eval_net(base_net, mpc, x0, x_goal, 600, f"{label} (original)")

    # Same model but zero f_extra when near π
    for thresh in [0.9, 0.7, 0.5, 0.3]:
        net_zerof = ManualNet(base_net, force_zero_f=True, zero_threshold=thresh)
        eval_net(net_zerof, mpc, x0, x_goal, 600, f"{label} (f=0 when near_pi>{thresh:.1f})")

    print(f"\nStarting near upright (stability test):")
    for q1_err, q1d_err in [(0.1, 0.3), (0.2, 0.5), (0.3, 0.8), (0.5, 0.0)]:
        x_pert = torch.tensor([math.pi - q1_err, q1d_err, 0., 0.], dtype=torch.float64)
        # Test with f=0 always (pure MPC stability)
        class ZeroFNet(torch.nn.Module):
            def __init__(self, n): super().__init__(); self.net=n; self.f_extra_bound=n.f_extra_bound
            def forward(self, x, q_base_diag=None, r_base_diag=None):
                gQ, gR, f, qd, rd, gQf = self.net(x, q_base_diag=q_base_diag, r_base_diag=r_base_diag)
                return gQ, gR, torch.zeros_like(f), qd, rd, gQf
            def eval(self): return self.net.eval()
        net_z = ZeroFNet(base_net)
        x_t, _ = train_module.rollout(net_z, mpc, x_pert, x_goal, 200)
        traj = x_t.cpu().numpy()
        wraps = np.array([math.sqrt(wrap_pi(s[0]-math.pi)**2 + s[1]**2 + s[2]**2 + s[3]**2) for s in traj])
        f01 = float((wraps < 0.10).mean())
        print(f"  pert=[q1-{q1_err:.1f}, q1d={q1d_err:.1f}]: frac<0.10={f01:.1%}  wrap@200steps={wraps[-1]:.4f}")


if __name__ == "__main__":
    main()
