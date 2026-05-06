"""exp_hardware_v6_sa010.py — Single-actuated shoulder-only, u_max=0.10 Nm.

Physical setup:
  - Joint 1 (shoulder): controlled by MPC / lin_net, limit U_LIM=0.10 Nm
  - Joint 2 (elbow): held rigid via PD stiffness, limit SA_U_LIM_ELBOW=2.0 Nm

See exp_hardware_v5_sa015.py for rationale on the separate elbow PD cap.

Starting from hw_v1_ep50. u_max=0.10 Nm shoulder authority.
Note: gravity at q1=π/2 is ~0.089 Nm ≈ u_lim — swing-up is borderline;
energy pumping may be needed.
"""

import glob
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

# ── Config ─────────────────────────────────────────────────────────────────────
X0     = [0.0, 0.0, 0.0, 0.0]
X_GOAL = [math.pi, 0.0, 0.0, 0.0]
DT     = 0.05
HORIZON = 10
U_LIM   = 0.10   # Nm — full shoulder authority

# PD stiffness for elbow joint (joint 2), holding q2=0
# No PD parameters needed — SA mode freezes q2=0 (rigid link)

STATE_DIM   = 4
CONTROL_DIM = 2
HIDDEN_DIM      = 128
GATE_RANGE_Q    = 0.99
GATE_RANGE_R    = 0.20
F_EXTRA_BOUND   = 1.5
F_KICKSTART_AMP = 0.01

W_F_END_REG      = 1.0
F_END_REG_STEPS  = 10
Q_NEAR_PI_POWER  = 4

META_EPOCHS      = 150
N_BOTTOM_PER_TOP = 3
N_BOTTOM         = 25
N_TOP            = 100
LR               = 5e-4
WEIGHT_DECAY     = 1e-4

W_Q_PROFILE  = 100.0
PUMP         = [1.0, 1.0, 1.0, 1.0]
STABLE       = [2.0, 1.0, 2.0, 1.0]

W_STABLE_PHASE     = 3.0
STABLE_PHASE_STEPS = N_TOP

W_F_POS_ONLY_TOP = 0.3
F_GATE_THRESH_TOP  = 0.8
DETACH_F_EXTRA_TOP = True

W_F_POS_ONLY_FE = 0.5
N_FE_STEPS      = 5

W_Q_PROFILE_BOT   = 10.0
N_Q_PROFILE_STEPS = 5

TOP_PERT_Q1  = 0.30
TOP_PERT_Q1D = 0.30
TOP_PERT_Q2  = 0.05   # tight: joint 2 is held near 0 by PD
TOP_PERT_Q2D = 0.05

EVAL_EVERY      = 10
SAVE_EVERY      = 50
DIAG_SAVE_EVERY = 20
SAVE_DIR        = "saved_models"
LOG_FILE        = "/tmp/hw_v6_sa010.log"


# ── Single-actuated dynamics wrapper ──────────────────────────────────────────

def wrap_sa_dynamics(mpc):
    """Simulate single-actuated mode by freezing the elbow (q2 = 0 always).

    See exp_hardware_v5_sa015.py for full rationale.
    """
    orig_rk4 = mpc.true_RK4_disc

    def sa_rk4(x, u, dt, n_sub=10):
        x_in = torch.cat([x[:2], torch.zeros(2, dtype=x.dtype, device=x.device)])
        u_in = torch.cat([u[:1], torch.zeros(1, dtype=u.dtype, device=u.device)])
        x_out = orig_rk4(x_in, u_in, dt, n_sub)
        return torch.cat([x_out[:2], torch.zeros(2, dtype=x_out.dtype, device=x_out.device)])

    mpc.true_RK4_disc = sa_rk4
    return mpc


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_energy_demo(n, device, q1_start=0.0):
    demo = torch.zeros((n, 4), dtype=torch.float64, device=device)
    span = math.pi - q1_start
    for i in range(n):
        alpha = i / max(n - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = q1_start + span * t
    return demo


def make_hold_demo(n, device):
    demo = torch.zeros((n, 4), dtype=torch.float64, device=device)
    demo[:, 0] = math.pi
    return demo


def sample_top(device):
    return torch.tensor([
        math.pi + random.uniform(-TOP_PERT_Q1,  TOP_PERT_Q1),
        random.uniform(-TOP_PERT_Q1D, TOP_PERT_Q1D),
        random.uniform(-TOP_PERT_Q2,  TOP_PERT_Q2),
        random.uniform(-TOP_PERT_Q2D, TOP_PERT_Q2D),
    ], dtype=torch.float64, device=device)


def probe_network(model, mpc, device):
    model.eval()
    results = {}
    with torch.no_grad():
        for name, q1 in [("bot", 0.0), ("mid", math.pi / 2), ("top", math.pi)]:
            s    = torch.tensor([q1, 0.0, 0.0, 0.0], dtype=torch.float64, device=device)
            hist = s.unsqueeze(0).expand(5, -1).contiguous()
            gQ, _, fe, _, _, _ = model(hist, mpc.q_base_diag, mpc.r_base_diag)
            results[name] = {"Q_q1": float(gQ[:, 0].mean()), "fe_norm": float(fe.norm())}
    model.train()
    return results


def eval2k(model, mpc, x0, x_goal, steps=2000):
    model.eval()
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0,
                                   x_goal=x_goal, num_steps=steps)
    traj  = x_t.cpu().numpy()
    wraps = np.array([
        math.sqrt(
            math.atan2(math.sin(s[0] - math.pi), math.cos(s[0] - math.pi)) ** 2
            + s[1] ** 2 + s[2] ** 2 + s[3] ** 2
        )
        for s in traj
    ])
    arr  = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    f01  = float((wraps < 0.10).mean())
    model.train()
    return f01, arr, post


def save_checkpoint(model_kwargs, state_dict, meta, label, save_dir, tag=""):
    name = f"hw_v6_sa010{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ep{meta}"
    m = network_module.SeparatedLinearizationNetwork(**model_kwargs).double()
    m.load_state_dict(state_dict)
    network_module.ModelManager(base_dir=save_dir).save_training_session(
        model=m, loss_history=[],
        training_params={
            "experiment": "hardware_v6_sa010",
            "meta_epoch": meta,
            "label": label,
            "u_lim": U_LIM,
            "single_actuated": True,
            "single_actuated": True,
            "rigid_elbow": True,
        },
        session_name=name,
    )
    return name


def main():
    log = open(LOG_FILE, "w", buffering=1)

    def out(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     dtype=torch.float64, device=device)
    x_goal = torch.tensor(X_GOAL, dtype=torch.float64, device=device)

    out("=" * 80)
    out("  EXP: HARDWARE v6 — single-actuated (shoulder only), u_max=0.10 Nm")
    out(f"  device: {device}")
    out(f"  U_LIM={U_LIM}  (single-actuated: elbow frozen at q2=0)")
    out("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON,
                                     device=device, u_lim=U_LIM)
    mpc.dt = torch.tensor(DT, dtype=torch.float64, device=device)

    # Apply single-actuated dynamics wrapper
    wrap_sa_dynamics(mpc)
    out(f"  SA dynamics: elbow frozen at q2=0 (rigid link approximation)")

    demo_bottom = make_energy_demo(N_BOTTOM, device)
    demo_top    = make_hold_demo(N_TOP, device)

    model_kwargs = dict(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=F_KICKSTART_AMP,
    )

    # Load best hw_v1 checkpoint
    ckpt_paths = glob.glob("saved_models/hw_v1*/*.pth")
    if not ckpt_paths:
        out("ERROR: No hw_v1 checkpoint found. Run exp_hardware_v1.py first.")
        return
    ckpt = max(ckpt_paths, key=os.path.getmtime)
    out(f"  Loading from: {ckpt}")
    data = torch.load(ckpt, map_location=device, weights_only=False)
    state_dict = data.get("model_state_dict", data)
    model = network_module.SeparatedLinearizationNetwork(**model_kwargs).to(device).double()
    model.load_state_dict(state_dict)

    optimizer_f = torch.optim.AdamW(model.f_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer_q = torch.optim.AdamW(model.q_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    p = probe_network(model, mpc, device)
    out(f"\n  Init: Q[q1] bot={p['bot']['Q_q1']:.3f}  mid={p['mid']['Q_q1']:.3f}  top={p['top']['Q_q1']:.3f}")
    out(f"        fe   bot={p['bot']['fe_norm']:.3f}  mid={p['mid']['fe_norm']:.3f}  top={p['top']['fe_norm']:.3f}")
    out(f"\n  Starting single-actuated fine-tuning ...\n")

    hdr = (f"  {'Meta':>5}  {'L_bot':>8}  {'L_top':>8}  "
           f"{'Q@bot':>6}  {'Q@top':>6}  "
           f"{'fe@bot':>7}  {'f01':>6}  {'arr':>5}  {'post':>6}")
    out(hdr)
    out("  " + "-" * len(hdr.rstrip()))

    best_f01   = 0.0
    best_state = None
    t0 = time.time()

    for meta in range(META_EPOCHS):
        L_bot_last = float("nan")
        for _ in range(N_BOTTOM_PER_TOP):
            loss_b, _ = train_module.train_linearization_network(
                lin_net=model, mpc=mpc,
                x0=x0, x_goal=x_goal, demo=demo_bottom,
                num_steps=N_BOTTOM, num_epochs=1, lr=LR,
                track_mode="energy",
                detach_gates_Q_for_qp=True,
                w_f_end_reg=W_F_END_REG,
                f_end_reg_steps=F_END_REG_STEPS,
                external_optimizer=optimizer_f,
                restore_best=False,
            )
            L_bot_last = loss_b[0] if loss_b else float("nan")

            x0_fe = sample_top(device)
            train_module.train_linearization_network(
                lin_net=model, mpc=mpc,
                x0=x0_fe, x_goal=x_goal, demo=demo_top,
                num_steps=N_FE_STEPS, num_epochs=1, lr=LR,
                track_mode="cos_q1",
                detach_gates_Q_for_qp=True,
                detach_f_extra_for_qp=True,
                w_f_pos_only=W_F_POS_ONLY_FE,
                external_optimizer=optimizer_f,
                restore_best=False,
            )

            train_module.train_linearization_network(
                lin_net=model, mpc=mpc,
                x0=x0, x_goal=x_goal, demo=demo_bottom,
                num_steps=N_Q_PROFILE_STEPS, num_epochs=1, lr=LR,
                track_mode="energy",
                detach_gates_Q_for_qp=True,
                w_q_profile=W_Q_PROFILE_BOT,
                q_profile_pump=PUMP,
                q_profile_stable=PUMP,
                q_profile_state_phase=True,
                external_optimizer=optimizer_q,
                restore_best=False,
            )

        x0_top = sample_top(device)
        loss_t, _ = train_module.train_linearization_network(
            lin_net=model, mpc=mpc,
            x0=x0_top, x_goal=x_goal, demo=demo_top,
            num_steps=N_TOP, num_epochs=1, lr=LR,
            track_mode="cos_q1",
            w_q_profile=W_Q_PROFILE,
            q_profile_pump=PUMP,
            q_profile_stable=STABLE,
            q_profile_state_phase=True,
            q_profile_near_pi_power=Q_NEAR_PI_POWER,
            w_stable_phase=W_STABLE_PHASE,
            stable_phase_steps=STABLE_PHASE_STEPS,
            w_f_pos_only=W_F_POS_ONLY_TOP,
            f_gate_thresh=F_GATE_THRESH_TOP,
            detach_f_extra_for_qp=DETACH_F_EXTRA_TOP,
            external_optimizer=optimizer_q,
            restore_best=False,
        )
        L_top = loss_t[0] if loss_t else float("nan")

        p = probe_network(model, mpc, device)

        f01_str = arr_str = post_str = "—"
        mark = ""
        if (meta + 1) % EVAL_EVERY == 0:
            f01, arr, post = eval2k(model, mpc, x0, x_goal)
            f01_str  = f"{f01:.1%}"
            arr_str  = str(arr) if arr is not None else "None"
            post_str = f"{post:.1%}" if post is not None else "N/A"
            if f01 > best_f01:
                best_f01   = f01
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                mark = " ★"

        out(f"  [{meta+1:>3}]  {L_bot_last:>8.3f}  {L_top:>8.3f}  "
            f"  {p['bot']['Q_q1']:.3f}  {p['top']['Q_q1']:.3f}  "
            f"  {p['bot']['fe_norm']:.3f}  "
            f"  {f01_str}  {arr_str}  {post_str}{mark}")

        if (meta + 1) % SAVE_EVERY == 0 and best_state is not None:
            name = save_checkpoint(model_kwargs, best_state, meta + 1,
                                   f"best_f01={best_f01:.1%}", SAVE_DIR)
            out(f"  → Saved: {name}")

        if (meta + 1) % DIAG_SAVE_EVERY == 0:
            cur_state = {k: v.clone() for k, v in model.state_dict().items()}
            name = save_checkpoint(model_kwargs, cur_state, meta + 1,
                                   f"diag_ep{meta+1}", SAVE_DIR, tag="_diag")
            out(f"  → Diag snapshot: {name}")
            # Also persist best_state so regression doesn't lose it between SAVE_EVERY intervals
            if best_state is not None:
                bname = save_checkpoint(model_kwargs, best_state, meta + 1,
                                        f"best_f01={best_f01:.1%}_diag{meta+1}",
                                        SAVE_DIR, tag="_best")
                out(f"  → Best state: {bname}  (f01={best_f01:.1%})")

    elapsed = time.time() - t0
    if best_state is not None:
        name = save_checkpoint(model_kwargs, best_state, META_EPOCHS,
                               f"best_f01={best_f01:.1%}", SAVE_DIR, tag="_FINAL")
        out(f"\n  FINAL best f01={best_f01:.1%}  saved: {name}")
    out(f"  Total time: {elapsed/60:.1f} min")
    log.close()


if __name__ == "__main__":
    main()
