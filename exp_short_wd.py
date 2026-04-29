"""exp_short_wd.py — quick WD test: 1 seed, 50 epochs only.

Fast test to see if high weight_decay (1e-2) produces the swing-up
faster/more reliably than baseline (1e-4).
"""

import math, os, sys, time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

torch.manual_seed(0); np.random.seed(0)
device = torch.device("cpu")
x0     = torch.zeros(4, dtype=torch.float64)
x_goal = torch.tensor([math.pi, 0., 0., 0.], dtype=torch.float64)

demo = torch.zeros((170, 4), dtype=torch.float64)
for i in range(170):
    alpha = i / 169
    demo[i, 0] = math.pi * 0.5 * (1.0 - math.cos(math.pi * alpha))

mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=10, device=device)
mpc.dt = torch.tensor(0.05, dtype=torch.float64)
mpc.q_base_diag = torch.tensor([12, 5, 50, 40], dtype=torch.float64)
mpc.Qf = torch.diag(torch.tensor([20., 50., 40., 30.], dtype=torch.float64))

net = network_module.LinearizationNetwork(
    state_dim=4, control_dim=2, horizon=10, hidden_dim=128,
    gate_range_q=0.99, gate_range_r=0.20, f_extra_bound=3.0, f_kickstart_amp=0.0,
).double()
# Apply q1 kickstart
final = list(net.q_head.modules())[-1]
with torch.no_grad():
    for k in range(9):
        final.bias[k * 4 + 0] = -3.0
        final.bias[k * 4 + 1] = -3.0

# HIGH weight decay
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

class M:
    def __init__(self): self._best = float('inf')
    def log_epoch(self, epoch, num_epochs, loss, info):
        d = info.get('pure_end_error', float('nan'))
        if d < self._best: self._best = d
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  ep {epoch+1:>3}  loss={loss:>7.3f}  goal_d={d:.3f}  best={self._best:.3f}", flush=True)

mon = M()

print("WD short test: seed=0, weight_decay=1e-2, 50 epochs, 170 steps")
t0 = time.time()
for epoch in range(50):
    train_module.train_linearization_network(
        lin_net=net, mpc=mpc, x0=x0, x_goal=x_goal, demo=demo, num_steps=170,
        num_epochs=1, lr=1e-3,
        debug_monitor=mon, recorder=network_module.NetworkOutputRecorder(),
        track_mode="energy",
        w_q_profile=100.0, q_profile_pump=[0.01, 0.01, 1, 1],
        q_profile_stable=[1, 1, 1, 1], q_profile_state_phase=True,
        w_end_q_high=80.0, end_phase_steps=20,
        external_optimizer=optimizer, restore_best=False,
    )
elapsed = time.time() - t0
print(f"  Trained in {elapsed:.0f}s. Best={mon._best:.4f}")

# Save
name = f"stageD_shortWD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
network_module.ModelManager(base_dir="saved_models").save_training_session(
    model=net, loss_history=[],
    training_params={"experiment": "short_wd", "weight_decay": 1e-2, "best": mon._best},
    session_name=name,
)
print(f"  Saved → saved_models/{name}/")

# Quick gen
print(f"\n  Quick gen (6 ICs):")
test_x0s = [
    ("canonical",  [0.0, 0.0, 0.0, 0.0]),
    ("q1=+0.2",    [0.2, 0.0, 0.0, 0.0]),
    ("q1=-0.2",    [-0.2, 0.0, 0.0, 0.0]),
    ("q1d=+0.5",   [0.0, 0.5, 0.0, 0.0]),
    ("q1d=-0.5",   [0.0, -0.5, 0.0, 0.0]),
    ("combined+",  [0.15, 0.4, 0.1, 0.2]),
]


def wrap_pi(x): return math.atan2(math.sin(x), math.cos(x))

succ = 0
for label, x0_list in test_x0s:
    x0_t = torch.tensor(x0_list, dtype=torch.float64)
    x_t, _ = train_module.rollout(lin_net=net, mpc=mpc, x0=x0_t, x_goal=x_goal, num_steps=1000)
    arr = traj = x_t.cpu().numpy()
    wraps = np.array([math.sqrt(wrap_pi(s[0]-x_goal[0].item())**2 + s[1]**2 + s[2]**2 + s[3]**2) for s in traj])
    in_zone = wraps < 0.3
    arrival = next((i for i, v in enumerate(in_zone) if v), None)
    longest = 0; cur = 0
    for v in in_zone:
        cur = cur + 1 if v else 0
        if cur > longest: longest = cur
    total = int(np.sum(in_zone))
    ok = "OK" if total >= 50 else ("WEAK" if total > 0 else "FAIL")
    if total >= 50: succ += 1
    print(f"    {label:<11}  arr={'-' if arrival is None else str(arrival):>4}  long={longest:>3}  tot={total:>3}  {ok}", flush=True)
print(f"  >>> success={succ}/{len(test_x0s)}")
