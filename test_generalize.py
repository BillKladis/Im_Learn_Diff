import sys, os; sys.path.insert(0, "/home/user/Im_Learn_Diff"); os.chdir("/home/user/Im_Learn_Diff")
import math, torch, numpy as np
import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

session = "saved_models/stageD_imit_20260427_231837/stageD_imit_20260427_231837.pth"
device = torch.device('cpu')
x_goal = torch.tensor([math.pi,0,0,0], device=device, dtype=torch.float64)

# Build mpc with same config
mpc = mpc_module.MPC_controller(x0=torch.tensor([0.]*4, device=device, dtype=torch.float64),
                                  x_goal=x_goal, N=10, device=device)
mpc.q_base_diag = torch.tensor([0.0,0.0,50.0,40.0], device=device, dtype=torch.float64)

net = network_module.LinearizationNetwork.load(session, device='cpu')

# Test 5 starting conditions
tests = [
    ("rest (training x0)", [0.0, 0.0, 0.0, 0.0]),
    ("upright (q1=π)",      [math.pi, 0.0, 0.0, 0.0]),
    ("small perturbation +", [0.1, 0.0, 0.0, 0.0]),
    ("small perturbation -", [-0.1, 0.0, 0.0, 0.0]),
    ("moving forward",       [0.0, 0.5, 0.0, 0.0]),
    ("near upright +",       [math.pi - 0.3, 0.0, 0.1, 0.0]),
    ("upside-down q1=π/2",   [math.pi/2, 0.0, 0.0, 0.0]),
]

print(f"Trained model: {os.path.basename(session)}")
print(f"  ZERO_Q1_COSTS = True (q_base_diag = [0, 0, 50, 40])")
print()
print(f"{'Initial state':<30} | final q1   q1d    q2    q2d   | goal_dist | E_final")
print("-"*100)
for name, x0_list in tests:
    x0 = torch.tensor(x0_list, device=device, dtype=torch.float64)
    xh, uh = train_module.rollout(lin_net=net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=170)
    xf = xh[-1].cpu().numpy()
    gd = float(torch.norm(xh[-1]-x_goal).item())
    Ef = mpc.compute_energy_single(xh[-1]).item()
    print(f"{name:<30} | {xf[0]:+.3f} {xf[1]:+.3f} {xf[2]:+.3f} {xf[3]:+.3f} | {gd:.4f}    | {Ef:+.3f}")
