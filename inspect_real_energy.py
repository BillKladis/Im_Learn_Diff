"""inspect_real_energy.py — what does a real successful swing-up's
energy curve actually look like?

Compute the energy along the stab_state model's reference trajectory
and report key statistics so we can design a realistic demo.
"""

import math, os, sys, csv
import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import mpc_controller as mpc_module

CSV_PATH = "saved_models/stageD_trajcurr_20260429_075636/reference_trajectory.csv"
DT = 0.05

device = torch.device("cpu")
mpc = mpc_module.MPC_controller(
    x0=torch.zeros(4, dtype=torch.float64),
    x_goal=torch.tensor([math.pi, 0., 0., 0.], dtype=torch.float64),
    N=10, device=device,
)
mpc.dt = torch.tensor(DT, dtype=torch.float64)

states = []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        states.append([float(row['q1']), float(row['q1d']),
                       float(row['q2']), float(row['q2d'])])
traj = np.array(states)

# Compute energy along trajectory
energies = []
kinetic_only = []
potential_only = []
for s in traj:
    state_t = torch.tensor(s, dtype=torch.float64)
    e = mpc.compute_energy_single(state_t).item()
    energies.append(e)
    # Approximate split: K = (1/2)(q1d² + q2d²), V from q1, q2
    kinetic_only.append(0.5 * (s[1]**2 + s[3]**2))
energies = np.array(energies)
kinetic_only = np.array(kinetic_only)

# Print energy at key timesteps
print(f"Trajectory length: {len(traj)} steps  ({len(traj)*DT:.2f} s)")
print(f"\nEnergy curve along the real trajectory:")
print(f"  {'step':>5}  {'time':>5}  {'q1':>7}  {'q1d':>7}  {'E_total':>9}  {'K_lin':>7}")
for s in [0, 30, 60, 90, 120, 140, 150, 160, 167, 170, 180, 200, 250, 300, 400, 500, 600]:
    if s < len(traj):
        st = traj[s]
        print(f"  {s:>5}  {s*DT:>4.1f}s  "
              f"{st[0]:>7.3f}  {st[1]:>7.3f}  "
              f"{energies[s]:>9.3f}  {kinetic_only[s]:>7.3f}")

print(f"\nKey statistics for first 200 steps (covers swing-up + early hold):")
e200 = energies[:200]
k200 = kinetic_only[:200]
print(f"  E_min: {e200.min():.3f} @ step {int(np.argmin(e200))}")
print(f"  E_max: {e200.max():.3f} @ step {int(np.argmax(e200))}")
print(f"  K_max (linear approx): {k200.max():.3f} @ step {int(np.argmax(k200))}")
print(f"  E at arrival step (167): {energies[167]:.3f}")
print(f"  E at step 170:            {energies[170]:.3f}")

# Also report where energy crosses 0 (going from negative to positive)
zero_cross = next((i for i in range(len(energies)-1)
                   if energies[i] < 0 <= energies[i+1]), None)
print(f"\nE = 0 crossing: step {zero_cross}  ({zero_cross*DT:.2f}s)" if zero_cross else "no E=0 crossing")
