"""diag_grad.py — Verify training is actually moving the network.

After a single train_linearization_network call with init_history,
check whether ANY network parameters changed.  If parameters are
identical pre/post call, we have a fundamental bug (gradients
zeroed, network frozen, or call is a no-op).
"""

import math, os, sys, copy
import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

PRETRAINED = "saved_models/stageD_stabstate_20260428_224856/stageD_stabstate_20260428_224856.pth"

device = torch.device("cpu")
x_goal  = torch.tensor([math.pi, 0., 0., 0.], dtype=torch.float64)
x0_zero = torch.zeros(4, dtype=torch.float64)

mpc = mpc_module.MPC_controller(x0=x0_zero, x_goal=x_goal, N=10, device=device)
mpc.dt          = torch.tensor(0.05, dtype=torch.float64)
mpc.q_base_diag = torch.tensor([12.0, 5.0, 50.0, 40.0], dtype=torch.float64)

lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device="cpu").double()

# Snapshot params before
def snapshot(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}

def diff(s1, s2):
    out = {}
    for k in s1:
        out[k] = (s2[k] - s1[k]).abs().max().item()
    return out

# Reference trajectory
print("Generating reference trajectory...")
x_ref, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=200)
traj = x_ref.cpu().numpy()
print(f"  done, shape={traj.shape}")

# Pick state at t=160 (close to arrival 167)
t = 160
init_history = torch.tensor(traj[t-4:t+1], dtype=torch.float64)
x0_pick = init_history[-1].clone()
print(f"\nx0_pick = {x0_pick.numpy()}")
print(f"init_history[0] = {init_history[0].numpy()}")
print(f"init_history[-1] = {init_history[-1].numpy()}")

# Flat upright demo
demo_flat = torch.zeros((80, 4), dtype=torch.float64)
demo_flat[:, 0] = math.pi

# Snapshot before
before = snapshot(lin_net)

# Build a persistent optimizer
optimizer = torch.optim.AdamW(lin_net.parameters(), lr=1e-4, weight_decay=1e-4)
print(f"\nOptimizer: {type(optimizer).__name__}, lr={optimizer.param_groups[0]['lr']}")

# Train ONE epoch via train_linearization_network with external_optimizer
print("\nCall 1 (curriculum from t=160)...")
loss_hist, _ = train_module.train_linearization_network(
    lin_net=lin_net, mpc=mpc, x0=x0_pick, x_goal=x_goal,
    demo=demo_flat, num_steps=80, num_epochs=1,
    init_history=init_history, external_optimizer=optimizer,
    restore_best=False,
    track_mode="energy", w_q_profile=100.0,
    q_profile_pump=[0.01, 0.01, 1.0, 1.0],
    q_profile_stable=[1.0, 1.0, 1.0, 1.0],
    q_profile_state_phase=True,
    w_end_q_high=80.0, end_phase_steps=20,
    w_f_stable=50.0,
)
print(f"  loss_hist: {loss_hist}")

after = snapshot(lin_net)
d = diff(before, after)

print(f"\nMax abs diff per parameter:")
total_change = 0.0
for k, v in sorted(d.items()):
    if v > 1e-12:
        total_change += v
    print(f"  {k:40s}  max|Δ|={v:.6e}")
print(f"\nSum of max-diffs: {total_change:.6e}")

# Adam state
print(f"\nAdam state:")
for p in lin_net.parameters():
    if p in optimizer.state:
        st = optimizer.state[p]
        if 'step' in st:
            print(f"  param shape={tuple(p.shape)}  step={st['step']}  "
                  f"exp_avg max={st['exp_avg'].abs().max().item():.4e}  "
                  f"exp_avg_sq max={st['exp_avg_sq'].abs().max().item():.4e}")
            break
