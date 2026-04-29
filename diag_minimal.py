"""Minimal: does manual backprop + step modify the network params?"""
import math, os, sys
import torch
sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module

PRETRAINED = "saved_models/stageD_stabstate_20260428_224856/stageD_stabstate_20260428_224856.pth"
lin_net = network_module.LinearizationNetwork.load(PRETRAINED, device="cpu").double()

# Reference to first param BEFORE optimizer creation
first_param = list(lin_net.parameters())[0]
print(f"First param shape: {tuple(first_param.shape)}")
print(f"First param[0,0:5] BEFORE: {first_param.data.flatten()[:5]}")
before_first = first_param.data.clone()

# Optimizer with AGGRESSIVE LR so any change is obvious
optim = torch.optim.AdamW(lin_net.parameters(), lr=1.0)

# Manual training step: forward, backprop, step
state_history = torch.zeros((5, 4), dtype=torch.float64)
state_history[:, 0] = math.pi  # near upright
gates_Q, gates_R, f_extra, _, _, _ = lin_net(state_history)
loss = (gates_Q.sum() + gates_R.sum() + f_extra.sum()) ** 2
print(f"\nLoss: {loss.item():.4f}")

optim.zero_grad()
loss.backward()

# Check gradient
print(f"First param grad: max|grad| = {first_param.grad.abs().max().item():.6e}")

optim.step()

after_first = first_param.data.clone()
diff = (after_first - before_first).abs().max().item()
print(f"\nAfter manual optim.step():")
print(f"First param[0,0:5] AFTER: {first_param.data.flatten()[:5]}")
print(f"Max diff first param: {diff:.6e}")
print(f"Match (zero diff = bug): {diff == 0.0}")

# ALSO check by re-pulling from list(parameters())
new_first_param = list(lin_net.parameters())[0]
print(f"\nIs same tensor object? {new_first_param is first_param}")
print(f"Is same .data?         {new_first_param.data is first_param.data}")

# state_dict comparison
sd_before = {k: v.detach().clone() for k, v in lin_net.state_dict().items()}
# Take ANOTHER step
optim.zero_grad()
gQ, gR, fE, _, _, _ = lin_net(state_history); loss = (gQ.sum()) ** 2
loss.backward()
optim.step()
sd_after = {k: v.detach().clone() for k, v in lin_net.state_dict().items()}
sd_diffs = {k: (sd_after[k] - sd_before[k]).abs().max().item() for k in sd_before}
nonzero = {k: v for k, v in sd_diffs.items() if v > 0}
print(f"\nstate_dict() diff after 2nd step:")
print(f"  Non-zero diffs: {len(nonzero)} / {len(sd_diffs)}")
for k, v in list(nonzero.items())[:5]:
    print(f"  {k}: {v:.6e}")
