"""exp_nodemo_sweep.py — Verify the no-demo result is robust across seeds."""

import math, os, sys, time
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
NUM_STEPS = 170
DT        = 0.05
EPOCHS    = 80
LR        = 1e-3
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM, CONTROL_DIM = 4, 2
SAVE_DIR    = "saved_models"

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]


def apply_q1_kickstart(lin_net, state_dim, horizon, bias_val):
    q_final = [m for m in lin_net.q_head.modules()
               if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            q_final.bias[k * state_dim + 0] = bias_val
            q_final.bias[k * state_dim + 1] = bias_val


def make_synthetic_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


class QuietMonitor:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
    def log_epoch(self, epoch, num_epochs, loss, info):
        if epoch == 0 or (epoch+1) % 5 == 0 or epoch == num_epochs-1:
            gd = info.get('pure_end_error', float('nan'))
            print(f"    [{epoch+1:>3}/{num_epochs}] loss={loss:.3f} goal_dist={gd:.4f}", flush=True)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,    device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo = make_synthetic_demo(NUM_STEPS, device)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=0.99, gate_range_r=0.20,
        f_extra_bound=3.0, f_kickstart_amp=0.0,
    ).to(device).double()
    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, -3.0)

    recorder = network_module.NetworkOutputRecorder()
    monitor  = QuietMonitor(num_epochs=EPOCHS)

    t0 = time.time()
    loss_history, recorder = train_module.train_linearization_network(
        lin_net=lin_net, mpc=mpc,
        x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
        num_epochs=EPOCHS, lr=LR,
        debug_monitor=monitor, recorder=recorder,
        grad_debug=False, track_mode="energy", w_terminal_anchor=0.0,
        w_q_profile=100.0,
        q_profile_pump=[0.01, 0.01, 1.0, 1.0],
        q_profile_stable=[1.0, 1.0, 1.0, 1.0],
        q_profile_state_phase=True,
        w_end_q_high=80.0,
        end_phase_steps=20,
    )
    elapsed = time.time() - t0
    x_final, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=NUM_STEPS)
    dist = float(np.linalg.norm(x_final.cpu().numpy()[-1] - np.array(X_GOAL)))
    return dist, len(loss_history), elapsed


if __name__ == "__main__":
    SEEDS = [0, 1, 7, 13, 42]
    results = []
    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        d, eps, et = run_seed(seed)
        success = "SUCCESS" if d < 1.0 else "FAIL"
        print(f"  seed={seed}: goal_dist={d:.4f} epochs={eps} time={et:.0f}s {success}")
        results.append((seed, d, eps, success))

    print("\n=== NO-DEMO SEED SWEEP SUMMARY ===")
    print(f"  q_base_diag = [12, 5, 50, 40]  (DEFAULT)  ||  no reference trajectory")
    print(f"  {'seed':>5}  {'goal_dist':>10}  {'epochs':>7}  result")
    for s, d, e, r in results:
        print(f"  {s:>5}  {d:>10.4f}  {e:>7}  {r}")
    succ = [d for s, d, e, r in results if r == "SUCCESS"]
    print(f"\n  successful: {len(succ)}/{len(SEEDS)}  ||  "
          f"best={min(d for s,d,e,r in results):.4f}  "
          f"mean(success)={np.mean(succ) if succ else float('nan'):.4f}")
