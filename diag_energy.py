"""Diagnostic: what's qf50 v2's energy during its 'oscillating hold'?"""
import math, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

PRETRAINED = "saved_models/stageD_nodemo_qf50_20260429_111711/stageD_nodemo_qf50_20260429_111711.pth"

device = torch.device("cpu")
x_goal = torch.tensor([math.pi, 0., 0., 0.], dtype=torch.float64)
x0 = torch.zeros(4, dtype=torch.float64)
mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=10, device=device)
mpc.dt = torch.tensor(0.05, dtype=torch.float64)
mpc.q_base_diag = torch.tensor([12., 5., 50., 40.], dtype=torch.float64)
mpc.Qf = torch.diag(torch.tensor([20., 50., 40., 30.], dtype=torch.float64))
lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device="cpu").double()

x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=600)
traj = x_t.cpu().numpy()


def wrap_pi(x): return math.atan2(math.sin(x), math.cos(x))

print(f"{'step':>4}  {'time':>5}  {'q1':>7}  {'q1_w':>7}  {'q1d':>7}  {'E':>7}  {'E_kin':>7}  {'E_pot':>7}")
for i in range(170, 600, 10):
    s = traj[i]
    state_t = torch.tensor(s, dtype=torch.float64)
    e_total = mpc.compute_energy_single(state_t).item()
    # Manually compute kinetic (linear approx) and potential
    e_kin = 0.5 * (s[1]**2 + s[3]**2)
    e_pot = e_total - e_kin   # crude
    q1w = wrap_pi(s[0])
    print(f"{i:>4}  {i*0.05:>4.1f}s  {s[0]:>7.3f}  {q1w:>7.3f}  {s[1]:>+7.3f}  {e_total:>+7.3f}  {e_kin:>7.3f}  {e_pot:>+7.3f}")

# Demo target energy at q1=π is 2mgL = ?
# E_demo = E_pot at q1=π, q1d=0 = mpc.compute_energy_single([π, 0, 0, 0])
demo_E = mpc.compute_energy_single(torch.tensor([math.pi, 0., 0., 0.], dtype=torch.float64)).item()
print(f"\nDemo's target energy at upright (q1=π, all v=0): {demo_E:.3f}")
print(f"This is what the network is being trained to MATCH.")
