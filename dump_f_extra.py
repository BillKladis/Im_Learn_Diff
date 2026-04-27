import sys, os; sys.path.insert(0, "/home/user/Im_Learn_Diff"); os.chdir("/home/user/Im_Learn_Diff")
import math, torch, numpy as np
import lin_net as network_module
import mpc_controller as mpc_module

session = "saved_models/stageD_imit_20260427_231837/stageD_imit_20260427_231837.pth"
device = torch.device('cpu')
x0 = torch.tensor([0.0]*4, device=device, dtype=torch.float64)
x_goal = torch.tensor([math.pi,0,0,0], device=device, dtype=torch.float64)
mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=10, device=device)
mpc.q_base_diag = torch.tensor([0.0,0.0,50.0,40.0], device=device, dtype=torch.float64)

net = network_module.LinearizationNetwork.load(session, device='cpu')
print(f"Loaded net params: {sum(p.numel() for p in net.parameters()):,}")

# Rollout with the trained net, capturing f_extra and gates each step
net.eval()
n_u = 2
x = x0.clone()
state_history = [x.clone() for _ in range(5)]
u_seq_guess = torch.zeros((mpc.N, n_u), dtype=torch.float64, device=device)

f_series = []
u_series = []
gQ_series = []
gR_series = []
x_series = [x.clone()]

for step in range(170):
    with torch.no_grad():
        gQ, gR, fE, _, _ = net(torch.stack(state_history,0),
                                q_base_diag=mpc.q_base_diag, r_base_diag=mpc.r_base_diag)
    f_series.append(fE.cpu().numpy())
    gQ_series.append(gQ.cpu().numpy())
    gR_series.append(gR.cpu().numpy())
    x_lin = x.unsqueeze(0).expand(mpc.N,-1).clone()
    u_lin = torch.clamp(u_seq_guess.clone(),
                        min=mpc.MPC_dynamics.u_min.unsqueeze(0),
                        max=mpc.MPC_dynamics.u_max.unsqueeze(0))
    u_opt, U_full = mpc.control(x, x_lin, u_lin, x_goal,
                                 diag_corrections_Q=gQ, diag_corrections_R=gR,
                                 extra_linear_control=fE.reshape(-1))
    x = mpc.true_RK4_disc(x, u_opt, mpc.dt)
    u_series.append(u_opt.cpu().numpy())
    x_series.append(x.clone())
    Ur = U_full.detach().view(mpc.N, n_u)
    u_seq_guess = torch.cat([Ur[1:], Ur[-1:]], 0).clone()
    state_history.pop(0); state_history.append(x.detach().clone())

xs = torch.stack(x_series).cpu().numpy()
us = np.array(u_series)
fs = np.array(f_series)  # (170, N, 2) — first horizon step is fs[t,0]
gQs = np.array(gQ_series)  # (170, N-1, 4) — Q gates per horizon step per state dim
gRs = np.array(gR_series)  # (170, N, 2) — R gates per horizon step per control dim

print(f"\nRollout final goal_dist: {np.linalg.norm(xs[-1]-x_goal.cpu().numpy()):.4f}")
print(f"Trained net dims: f_extra shape per step = {fs.shape[1:]}, gQ shape = {gQs.shape[1:]}")
print()
print("=== f_extra at horizon-step=0 (the τ1, τ2 the QP sees as a bias for the next applied control) ===")
print("step| q1     q1d    | tau1  tau2 | f_extra[0,τ1] f_extra[0,τ2] | sign(f·q1d)")
for i in range(0, 170, 12):
    f_dot_q1d = np.sign(fs[i,0,0]) * np.sign(xs[i,1])
    print(f"{i:3d} | {xs[i,0]:+.3f} {xs[i,1]:+.3f} | {us[i,0]:+.2f} {us[i,1]:+.2f} | {fs[i,0,0]:+.3f}      {fs[i,0,1]:+.3f}     | {f_dot_q1d:+.0f}")

print()
print(f"f_extra stats over rollout:")
print(f"  τ1[0] mean={fs[:,0,0].mean():+.3f}, std={fs[:,0,0].std():.3f}, range=[{fs[:,0,0].min():+.3f}, {fs[:,0,0].max():+.3f}]")
print(f"  τ2[0] mean={fs[:,0,1].mean():+.3f}, std={fs[:,0,1].std():.3f}")
print(f"  τ1[0] sign changes (per-step): {(np.sign(fs[:-1,0,0])*np.sign(fs[1:,0,0])<0).sum()}")
print()
print(f"Q-gate (Q running cost scaling) stats:")
print(f"  Mean per-state-dim across rollout (averaging horizon and time):")
print(f"    q1:     {gQs[:,:,0].mean():.3f} (range {gQs[:,:,0].min():.3f}-{gQs[:,:,0].max():.3f})")
print(f"    q1_dot: {gQs[:,:,1].mean():.3f}")
print(f"    q2:     {gQs[:,:,2].mean():.3f}")
print(f"    q2_dot: {gQs[:,:,3].mean():.3f}")
print(f"  (Note: q_base_diag = [0, 0, 50, 40] — q1/q1d gates are multiplied by zero anyway)")
print()
print(f"R-gate (control cost scaling) stats:")
print(f"  τ1: mean={gRs[:,:,0].mean():.3f}, range=[{gRs[:,:,0].min():.3f}, {gRs[:,:,0].max():.3f}]")
print(f"  τ2: mean={gRs[:,:,1].mean():.3f}")
