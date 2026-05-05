"""diag_traj.py — Print step-by-step trajectory of latest checkpoint.

Loads newest hw_v1*_diag*/*.pth, runs 200-step rollout from x=0, prints
state, energy, control input each step. Use to see whether pendulum
swings up at all and what controls the model is producing.
"""
import glob, math, sys, torch
sys.path.insert(0, "/home/user/Im_Learn_Diff")
import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

device = torch.device("cpu")
x0 = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
x_goal = torch.tensor([math.pi, 0.0, 0.0, 0.0], dtype=torch.float64)

MODEL_KWARGS = dict(
    state_dim=4, control_dim=2, horizon=10, hidden_dim=128,
    gate_range_q=0.99, gate_range_r=0.20, f_extra_bound=1.5, f_kickstart_amp=0.01,
)

ckpts = sorted(glob.glob("/home/user/Im_Learn_Diff/saved_models/hw_v1*/*.pth"))
if not ckpts:
    print("No hw_v1 checkpoint found.")
    sys.exit(1)
ckpt = ckpts[-1]
print(f"Checkpoint: {ckpt}")
data = torch.load(ckpt, map_location=device, weights_only=False)
state_dict = data.get("model_state_dict", data)
model = network_module.SeparatedLinearizationNetwork(**MODEL_KWARGS).double()
model.load_state_dict(state_dict)
model.eval()

mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=10, device=device)
mpc.dt = torch.tensor(0.05, dtype=torch.float64)

NSTEPS = 200
x = x0.clone()
state_history = [x.clone() for _ in range(5)]
u_seq = torch.zeros((10, 2), dtype=torch.float64)

print(f"\n{'step':>4} {'q1':>7} {'q1d':>7} {'q2':>7} {'q2d':>7} {'E':>8} {'u1':>7} {'fenorm':>7} {'gQ_q1':>6}")
with torch.no_grad():
    for t in range(NSTEPS):
        hist = torch.stack(state_history)
        gQ, gR, fe, _, _, _ = model(hist, mpc.q_base_diag, mpc.r_base_diag)
        u_lin = u_seq.clamp(mpc.MPC_dynamics.u_min, mpc.MPC_dynamics.u_max)
        x_lin = x.unsqueeze(0).expand(10, -1).clone()
        u_mpc, U_full = mpc.control(x, x_lin, u_lin, x_goal,
                                     diag_corrections_Q=gQ, diag_corrections_R=gR,
                                     extra_linear_control=fe.reshape(-1))
        E = mpc.compute_energy_single(x)
        if t < 30 or t % 10 == 0:
            print(f"{t:>4} {x[0].item():>7.3f} {x[1].item():>7.3f} {x[2].item():>7.3f} {x[3].item():>7.3f} "
                  f"{E.item():>8.4f} {u_mpc[0].item():>7.4f} {fe.norm().item():>7.3f} {gQ[:,0].mean().item():>6.3f}")
        x = mpc.true_RK4_disc(x, u_mpc, mpc.dt).detach()
        u_seq = torch.cat([U_full.detach().view(10, 2)[1:], U_full.detach().view(10, 2)[-1:]])
        state_history.pop(0); state_history.append(x.clone())

# Final check
q1_final = float(x[0].item())
q1_wrap = math.atan2(math.sin(q1_final - math.pi), math.cos(q1_final - math.pi))
print(f"\nFinal: q1={q1_final:.3f} (wrap from π = {q1_wrap:.3f}), q1d={x[1].item():.3f}, |E|={abs(mpc.compute_energy_single(x).item()):.4f}")
