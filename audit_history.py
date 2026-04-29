"""audit_history.py — Verify the network sees EXACTLY the same input
when we seed init_history from a real rollout vs when it observed
those states naturally.

Test: in a 600-step rollout from x0=zero, capture the network's
input at step 167 (goal arrival). Compare to what the network sees
when we seed init_history with traj[163..167] and ask it for an
output. They MUST be bit-identical.

Also verifies trajectory CSV ↔ generated trajectory equivalence.
"""

import math, os, sys, csv
import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

PRETRAINED = "saved_models/stageD_stabstate_20260428_224856/stageD_stabstate_20260428_224856.pth"
CSV_PATH   = "saved_models/stageD_trajcurr_20260429_075636/reference_trajectory.csv"
TARGET_STEP = 167  # goal-arrival step

device = torch.device("cpu")
x_goal  = torch.tensor([math.pi, 0., 0., 0.], dtype=torch.float64)
x0_zero = torch.zeros(4, dtype=torch.float64)

mpc = mpc_module.MPC_controller(x0=x0_zero, x_goal=x_goal, N=10, device=device)
mpc.dt          = torch.tensor(0.05, dtype=torch.float64)
mpc.q_base_diag = torch.tensor([12.0, 5.0, 50.0, 40.0], dtype=torch.float64)

lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device="cpu").double()
lin_net.eval()

# ---------- Method 1: regenerate trajectory in-memory ----------
print("=" * 70)
print("Method 1: regenerate trajectory in-memory (current trajcurr)")
print("=" * 70)
x_ref, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=200)
traj_mem = x_ref.cpu().numpy()
print(f"  Shape: {traj_mem.shape}")
for s in [TARGET_STEP - 4, TARGET_STEP, TARGET_STEP + 1]:
    print(f"  traj_mem[{s}] = {traj_mem[s]}")

# ---------- Method 2: load from CSV ----------
print()
print("=" * 70)
print("Method 2: load from CSV")
print("=" * 70)
csv_rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_rows.append([float(row['q1']), float(row['q1d']),
                         float(row['q2']), float(row['q2d'])])
traj_csv = np.array(csv_rows)
print(f"  Shape: {traj_csv.shape}")
for s in [TARGET_STEP - 4, TARGET_STEP, TARGET_STEP + 1]:
    print(f"  traj_csv[{s}] = {traj_csv[s]}")

# Equivalence?
diff = np.abs(traj_mem[:200] - traj_csv[:200]).max()
print(f"\n  Max diff (mem vs CSV): {diff:.6e}")
print(f"  Match (zero diff = OK): {diff == 0.0}")

# ---------- Method 3: capture network input during natural rollout ----------
print()
print("=" * 70)
print("Method 3: hook network forward to capture input at step 167")
print("=" * 70)

captured_inputs = []
captured_outputs = []
orig_forward = type(lin_net).forward

def hooked_forward(self, x, *args, **kwargs):
    out = orig_forward(self, x, *args, **kwargs)
    captured_inputs.append(x.detach().clone())
    captured_outputs.append(tuple(t.detach().clone() if torch.is_tensor(t) else t for t in out))
    return out

type(lin_net).forward = hooked_forward
try:
    x_ref2, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0_zero, x_goal=x_goal, num_steps=200)
finally:
    type(lin_net).forward = orig_forward

print(f"  Captured {len(captured_inputs)} forward calls during 200-step rollout")
input_at_target = captured_inputs[TARGET_STEP]
print(f"  input_at_step[{TARGET_STEP}].shape = {tuple(input_at_target.shape)}")
print(f"  input_at_step[{TARGET_STEP}][0] = {input_at_target[0]}")  # First frame
print(f"  input_at_step[{TARGET_STEP}][-1] = {input_at_target[-1]}")  # Last frame

# ---------- Method 4: init_history seeded ----------
print()
print("=" * 70)
print(f"Method 4: seed init_history with traj_csv[{TARGET_STEP-4}..{TARGET_STEP}]")
print("=" * 70)
hist = torch.tensor(traj_csv[TARGET_STEP-4:TARGET_STEP+1], dtype=torch.float64)
print(f"  init_history.shape = {tuple(hist.shape)}")
print(f"  init_history[0] = {hist[0]}")
print(f"  init_history[-1] = {hist[-1]}")

# Forward the network with this stacked history (matching how rollout stacks state_history)
with torch.no_grad():
    out_seeded = lin_net(hist)

# ---------- Compare seeded vs natural ----------
print()
print("=" * 70)
print("CRITICAL CHECK: do seeded and natural inputs match?")
print("=" * 70)

input_diff = (input_at_target - hist).abs().max().item()
print(f"  Input max-diff (natural vs seeded): {input_diff:.6e}")
print(f"  Match (zero = correct): {input_diff < 1e-12}")

# Compare outputs
nat_out = captured_outputs[TARGET_STEP]
print(f"\n  Output comparison:")
for i, (n, s) in enumerate(zip(nat_out, out_seeded)):
    if torch.is_tensor(n) and torch.is_tensor(s):
        d = (n - s).abs().max().item()
        print(f"    out[{i}] shape={tuple(n.shape)}  max-diff={d:.6e}  match={d < 1e-12}")

# ---------- Method 5: also verify what train_linearization_network does ----------
print()
print("=" * 70)
print("Method 5: trace state_history initialization in train_linearization_network")
print("=" * 70)
# Read it ourselves
import inspect
src = inspect.getsource(train_module.train_linearization_network)
# Find the state_history init lines
in_block = False
shown = 0
for line in src.split('\n'):
    if 'state_history' in line and ('init_history' in line or 'add_train_noise' in line or '[' in line or 'pop' in line or 'append' in line):
        if shown < 12:
            print(f"  {line.rstrip()}")
            shown += 1
