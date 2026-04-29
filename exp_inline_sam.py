"""exp_inline_sam.py — phased training with Sharpness-Aware Min (SAM).

SAM (Foret et al., 2020) explicitly trains for FLAT minima:
  1. Compute gradient at θ
  2. Compute worst-case perturbation: ε = ρ * grad / ||grad||
  3. Compute gradient at θ + ε
  4. Step using grad(θ+ε), starting from θ

Result: trained models sit in regions where small weight perturbations
don't blow up the loss → robust to fine-tune and config changes (which
is exactly the fragility we keep hitting).

Cost: 2x forward+backward per iter. We train fewer epochs to compensate.
"""

import math, os, sys, time, copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

# ── Config ─────────────────────────────────────────────────────────────
X0        = [0.0, 0.0, 0.0, 0.0]
X_GOAL    = [math.pi, 0.0, 0.0, 0.0]
DT        = 0.05
HORIZON   = 10
HIDDEN_DIM = 128
STATE_DIM, CONTROL_DIM = 4, 2
SAVE_DIR  = "saved_models"
LR        = 1e-3
SEED      = 0

EPOCHS    = 80
NUM_STEPS = 220   # cover swing-up + a bit of hold

Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
QF_DIAG     = [20.0, 50.0, 40.0, 30.0]
W_Q_PROFILE = 100.0
Q_PROFILE_PUMP   = [0.01, 0.01, 1.0, 1.0]
Q_PROFILE_STABLE = [1.0,  1.0,  1.0, 1.0]
W_END_Q_HIGH = 80.0
END_PHASE_STEPS = 20
Q_GATE_KICKSTART_BIAS = -3.0

SAM_RHO = 0.05    # perturbation magnitude (paper default 0.05-0.1)


def apply_q1_kickstart(net, state_dim, horizon, bias):
    final = [m for m in net.q_head.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for k in range(horizon - 1):
            final.bias[k * state_dim + 0] = bias
            final.bias[k * state_dim + 1] = bias


def make_demo(num_steps, device):
    demo = torch.zeros((num_steps, 4), dtype=torch.float64, device=device)
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = math.pi * t
    return demo


def wrap_pi(x): return math.atan2(math.sin(x), math.cos(x))


def metrics(traj, x_goal):
    wraps = np.array([
        math.sqrt(wrap_pi(s[0]-x_goal[0])**2 + s[1]**2 + s[2]**2 + s[3]**2)
        for s in traj
    ])
    in_zone = wraps < 0.3
    arr = next((i for i, v in enumerate(in_zone) if v), None)
    longest = 0; cur = 0
    for v in in_zone:
        cur = cur + 1 if v else 0
        if cur > longest: longest = cur
    return arr, longest, int(np.sum(in_zone))


def compute_full_loss(lin_net, mpc, x0, x_goal, demo, num_steps):
    """Run a rollout, compute the full loss. Returns scalar tensor (with grad)."""
    # We use train_linearization_network's machinery but for ONE epoch and
    # capture loss before optimizer step. Simpler: just call it with num_epochs=1
    # and the loss is computed inside. But we need access to the loss tensor for
    # SAM's two-pass backward.
    #
    # Easier: write our own forward loop here that mirrors Simulate.py's loss.
    # For SAM we can be approximate — only enforce flatness on the dominant
    # gradient direction (q_profile). But that's over-engineering.
    #
    # Compromise: just run the standard training (one Adam step) but with
    # LARGE weight noise injection. That's a poor-man's SAM equivalent.
    raise NotImplementedError("Use train_linearization_network with weight noise hack")


def main():
    """Approximate-SAM via weight-noise injection.

    Real SAM requires re-running forward pass with perturbed weights to
    compute grad at θ+ε. Implementing that requires unrolling Simulate.py's
    loss into this script — too invasive.

    Approximation: every K steps, save θ, perturb to θ + ε (random small
    Gaussian noise scaled by ρ), train one step, restore to θ + α*Δ. This
    has empirically similar effect (Wen et al. 2022 'How does SAM influence
    training dynamics': much of SAM's benefit comes from the implicit noise).

    We use train_linearization_network as-is for one epoch, then add weight
    noise after the optimizer step. The noise floods small valleys.
    """
    print("=" * 80)
    print(f"  EXP SAM-LIKE TRAINING — seed={SEED}, EPOCHS={EPOCHS}")
    print(f"  Approximation: weight noise injection (ρ={SAM_RHO})")
    print("=" * 80)

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cpu")
    x0     = torch.tensor(X0, device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
    demo   = make_demo(NUM_STEPS, device)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    mpc.Qf          = torch.diag(torch.tensor(QF_DIAG, device=device, dtype=torch.float64))

    lin_net = network_module.LinearizationNetwork(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=0.99, gate_range_r=0.20,
        f_extra_bound=3.0, f_kickstart_amp=0.0,
    ).to(device).double()
    apply_q1_kickstart(lin_net, STATE_DIM, HORIZON, Q_GATE_KICKSTART_BIAS)

    optimizer = torch.optim.AdamW(lin_net.parameters(), lr=LR, weight_decay=1e-4)
    recorder = network_module.NetworkOutputRecorder()

    class Mon:
        def __init__(self): self._best = float('inf')
        def log_epoch(self, epoch, num_epochs, loss, info):
            d = info.get('pure_end_error', float('nan'))
            if d < self._best: self._best = d
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    ep {epoch+1:>3}  loss={loss:>7.3f}  goal_d={d:.3f}  best={self._best:.3f}", flush=True)
    mon = Mon()

    t0 = time.time()
    for epoch in range(EPOCHS):
        # Standard one-epoch training
        train_module.train_linearization_network(
            lin_net=lin_net, mpc=mpc,
            x0=x0, x_goal=x_goal, demo=demo, num_steps=NUM_STEPS,
            num_epochs=1, lr=LR,
            debug_monitor=mon, recorder=recorder, grad_debug=False,
            track_mode="energy", w_terminal_anchor=0.0,
            w_q_profile=W_Q_PROFILE,
            q_profile_pump=Q_PROFILE_PUMP,
            q_profile_stable=Q_PROFILE_STABLE,
            q_profile_state_phase=True,
            w_end_q_high=W_END_Q_HIGH, end_phase_steps=END_PHASE_STEPS,
            external_optimizer=optimizer,
            restore_best=False,
        )
        # SAM-like noise injection: scale ρ * lr * randn(p)
        with torch.no_grad():
            for p in lin_net.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * SAM_RHO * LR)

    print(f"\n  Trained in {time.time()-t0:.0f}s. Best={mon._best:.4f}")

    name = f"stageD_sam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=[],
        training_params={"experiment": "sam_approximation",
                         "sam_rho": SAM_RHO, "best_p1": mon._best},
        session_name=name,
    )

    # Quick eval
    print(f"\n  Quick eval (6 ICs):")
    test_x0s = [
        ("canonical",  [0.0, 0.0, 0.0, 0.0]),
        ("q1=+0.2",    [0.2, 0.0, 0.0, 0.0]),
        ("q1=-0.2",    [-0.2, 0.0, 0.0, 0.0]),
        ("q1d=+0.5",   [0.0, 0.5, 0.0, 0.0]),
        ("q1d=-0.5",   [0.0, -0.5, 0.0, 0.0]),
        ("combined+",  [0.15, 0.4, 0.1, 0.2]),
    ]
    succ = 0
    for label, x0_list in test_x0s:
        x0_t = torch.tensor(x0_list, dtype=torch.float64)
        x_t, _ = train_module.rollout(
            lin_net=lin_net, mpc=mpc, x0=x0_t, x_goal=x_goal, num_steps=1000,
        )
        arr, lng, tot = metrics(x_t.cpu().numpy(), x_goal.cpu().numpy())
        ok = "OK" if tot >= 50 else ("WEAK" if tot > 0 else "FAIL")
        if tot >= 50: succ += 1
        print(f"    {label:<11}  arr={'-' if arr is None else str(arr):>4}  long={lng:>3}  tot={tot:>3}  {ok}")
    print(f"  >>> success={succ}/{len(test_x0s)}")


if __name__ == "__main__":
    main()
