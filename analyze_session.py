"""Analyse a training session to summarise swing-up quality."""
import sys, os, glob, json
sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")
import math, csv, numpy as np
import torch
import lin_net as network_module
import mpc_controller as mpc_module

def latest_session(base="saved_models"):
    sessions = sorted(glob.glob(os.path.join(base, "stageD_imit_*")))
    return sessions[-1] if sessions else None

if len(sys.argv) > 1:
    session_dir = sys.argv[1]
else:
    session_dir = latest_session()

if not session_dir:
    print("No session found"); sys.exit(1)

name = os.path.basename(session_dir)
print(f"Session: {name}")
print(f"Path:    {session_dir}")
print()

# Load metrics from rollouts
def read_rollout(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append([float(r[k]) for k in ["q1_rad","q1_dot_rads","q2_rad","q2_dot_rads","tau1_Nm","tau2_Nm","goal_dist"]])
    return np.array(rows)

ep0 = read_rollout(os.path.join(session_dir, f"{name}_rollout_epoch0.csv"))
fin = read_rollout(os.path.join(session_dir, f"{name}_rollout_final.csv"))
demo_path = "run_20260428_001459_rollout_final.csv"
demo = read_rollout(demo_path)

device = torch.device("cpu")
mpc = mpc_module.MPC_controller(
    x0=torch.tensor([0.0]*4, device=device, dtype=torch.float64),
    x_goal=torch.tensor([math.pi,0,0,0], device=device, dtype=torch.float64),
    N=10, device=device,
)

def E_array(arr):
    Es = []
    for r in arr:
        x = torch.tensor(r[:4], dtype=torch.float64)
        Es.append(mpc.compute_energy_single(x).item())
    return np.array(Es)

E_ep0 = E_array(ep0); E_fin = E_array(fin); E_demo = E_array(demo)

print(f"=== Initial (epoch 0, untrained) ===")
print(f"  q1 range:        [{ep0[:,0].min():+.3f}, {ep0[:,0].max():+.3f}]")
print(f"  q1_dot range:    [{ep0[:,1].min():+.3f}, {ep0[:,1].max():+.3f}]")
print(f"  q2 range:        [{ep0[:,2].min():+.3f}, {ep0[:,2].max():+.3f}]")
print(f"  tau1 range:      [{ep0[:,4].min():+.2f}, {ep0[:,4].max():+.2f}]")
print(f"  Final q1:         {ep0[-1,0]:+.3f}  (goal=π=3.142)")
print(f"  Final goal_dist:  {ep0[-1,6]:.4f}")
print(f"  Final E:          {E_ep0[-1]:+.3f}  (goal=14.715)")
print(f"  Max E:            {E_ep0.max():+.3f}")
print()
print(f"=== Final (trained) ===")
print(f"  q1 range:        [{fin[:,0].min():+.3f}, {fin[:,0].max():+.3f}]")
print(f"  q1_dot range:    [{fin[:,1].min():+.3f}, {fin[:,1].max():+.3f}]")
print(f"  q2 range:        [{fin[:,2].min():+.3f}, {fin[:,2].max():+.3f}]")
print(f"  tau1 range:      [{fin[:,4].min():+.2f}, {fin[:,4].max():+.2f}]")
print(f"  Final q1:         {fin[-1,0]:+.3f}  (goal=π=3.142)")
print(f"  Final goal_dist:  {fin[-1,6]:.4f}")
print(f"  Final E:          {E_fin[-1]:+.3f}  (goal=14.715)")
print(f"  Max E:            {E_fin.max():+.3f}")
print()
print(f"  tau1 sign changes: {(np.sign(fin[:-1,4])*np.sign(fin[1:,4])<0).sum()} (demo: {(np.sign(demo[:-1,4])*np.sign(demo[1:,4])<0).sum()})")
print(f"  Energy curve RMSE vs demo: {np.sqrt(((E_fin[:len(E_demo)] - E_demo[:len(E_fin)])**2).mean()):.3f}")
print()
print(f"=== Demo (target) ===")
print(f"  Final q1:         {demo[-1,0]:+.3f}")
print(f"  Final goal_dist:  {demo[-1,6]:.4f}")
print(f"  Final E:          {E_demo[-1]:+.3f}")
