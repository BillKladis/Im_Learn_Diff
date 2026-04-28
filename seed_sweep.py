import sys, os; sys.path.insert(0, "/home/user/Im_Learn_Diff"); os.chdir("/home/user/Im_Learn_Diff")
import time, math, torch
import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

# Quick multi-seed sweep with the working config to characterise variance.
device = torch.device('cpu')
x0     = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64)
x_goal = torch.tensor([math.pi, 0.0, 0.0, 0.0], device=device, dtype=torch.float64)
demo = train_module.load_demo_trajectory('run_20260428_001459_rollout_final.csv', expected_length=170, device=device)

LR = 1e-3
EPOCHS = 80
results = []

for seed in [1, 7, 13, 42, 99, 256]:
    torch.manual_seed(seed)
    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=10, device=device)
    mpc.q_base_diag = torch.tensor([0.0, 0.0, 50.0, 40.0], device=device, dtype=torch.float64)
    lin_net = network_module.LinearizationNetwork(
        state_dim=4, control_dim=2, horizon=10, hidden_dim=128,
        f_extra_bound=3.0, f_kickstart_amp=0.0,
    ).to(device).double()

    print(f"\n=== SEED {seed} ===", flush=True)
    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, demo=demo,
        num_steps=170, num_epochs=EPOCHS, lr=LR,
        debug_monitor=None, grad_debug=False, track_mode='energy',
        w_terminal_anchor=0.0,
    )
    elapsed = time.time()-t0
    xh, uh = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=170)
    gd = torch.norm(xh[-1]-x_goal).item()
    Ef = mpc.compute_energy_single(xh[-1]).item()
    epochs_run = len(loss_history)
    print(f"  seed {seed}: epochs={epochs_run}/{EPOCHS} goal_dist={gd:.4f} E={Ef:.3f} elapsed={elapsed:.0f}s", flush=True)
    results.append((seed, gd, Ef, epochs_run))

print("\n\n=== MULTI-SEED SUMMARY ===")
print(f"{'seed':>5} | {'epochs':>6} | {'goal_dist':>9} | {'E_final':>8} | swing-up?")
print("-"*55)
for s, gd, e, ep in results:
    print(f"{s:>5} | {ep:>6} | {gd:>9.4f} | {e:>+.3f} | {'YES' if gd < 1.0 else ('PARTIAL' if gd < 3.0 else 'NO')}")
gds = [r[1] for r in results]
success = [g < 1.0 for g in gds]
print(f"\n  successful (<1.0): {sum(success)}/{len(success)}")
print(f"  mean goal_dist:    {sum(gds)/len(gds):.4f}")
print(f"  best goal_dist:    {min(gds):.4f}")
print(f"  worst goal_dist:   {max(gds):.4f}")
