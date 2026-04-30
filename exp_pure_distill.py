"""exp_pure_distill.py — Pure MSE distillation into f_head.

CLEANEST POSSIBLE APPROACH:
  No training loop, no energy tracking, no conflicting signals.

  Goal: teach f_head to output f_extra * (1 - zerofnet_gate) for all states.

  ZeroFNet gate: zeros f_extra when near_pi > 0.9 (the 25.9% frac<0.10 variant).

  Method:
    1. Generate diverse synthetic states covering the full state space
    2. Get f_extra_original from frozen original network
    3. Compute f_target = f_extra_original * (1 - gate(near_pi))
    4. Train f_head to minimize MSE(f_current, f_target)
    5. All other layers frozen: trunk, encoder, q_head, r_head

  No conflicting signals: distillation IS the loss, nothing else.
  f_head just learns: near top → output 0, elsewhere → match original.

STATE COVERAGE:
  - Near-top states (q1 ≈ π): should output 0
  - Swing-up states (q1 ≈ 0..π): should match original
  - Full velocity range: q1d, q2, q2d ∈ [-3, 3]
"""

import glob
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

# ── Config ──────────────────────────────────────────────────────────────────
POSONLY_FINAL = "saved_models/stageD_posonly_ft_20260430_083618/stageD_posonly_ft_20260430_083618.pth"
BASELINE      = "saved_models/stageD_nodemo_20260428_123448/stageD_nodemo_20260428_123448.pth"

X0       = [0.0, 0.0, 0.0, 0.0]
X_GOAL   = [math.pi, 0.0, 0.0, 0.0]
DT       = 0.05
HORIZON  = 10
SAVE_DIR = "saved_models"
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]

# Distillation parameters
ZEROFNET_THRESH = 0.9     # match the best ZeroFNet variant (near_pi > 0.9 → f=0)
LR              = 1e-3    # high LR: simple regression, converges fast
BATCH_SIZE      = 256
N_BATCHES_EPOCH = 100     # batches per epoch
EPOCHS          = 60
SAVE_EVERY      = 10

# State sampling ranges for distillation dataset
Q1_RANGE   = (0.0, 2 * math.pi)  # full q1 range
QD_RANGE   = (-4.0, 4.0)          # velocities
Q2_RANGE   = (-math.pi, math.pi)  # q2 full range


def _find_pretrained():
    if os.path.isfile(POSONLY_FINAL):
        return POSONLY_FINAL, "posonly_ft_final"
    ckpts = sorted(glob.glob("saved_models/stageD_posonly_ft_20260430_083618_ep*/*.pth"))
    if ckpts:
        return ckpts[-1], "posonly_ep" + os.path.basename(os.path.dirname(ckpts[-1]))[-3:]
    return BASELINE, "0.0612_baseline"


def zerofnet_gate(q1, x_goal_q1, thresh=0.9):
    """ZeroFNet gate: 1.0 when near top (suppress f_extra), 0 when far."""
    near_pi = (1.0 + torch.cos(q1 - x_goal_q1)) / 2.0
    gate = ((near_pi - thresh) / max(1e-8, 1.0 - thresh)).clamp(0.0, 1.0)
    return gate


def sample_state_batch(batch_size, device, near_top_frac=0.5):
    """Sample states: half near top, half general.

    near_top_frac: fraction of batch that should be near the top.
    """
    n_top  = int(batch_size * near_top_frac)
    n_gen  = batch_size - n_top

    # Near-top states: q1 close to π
    q1_top = math.pi + torch.zeros(n_top, device=device, dtype=torch.float64).uniform_(-1.0, 1.0)
    qd_top = torch.zeros(n_top, device=device, dtype=torch.float64).uniform_(-3.0, 3.0)
    q2_top = torch.zeros(n_top, device=device, dtype=torch.float64).uniform_(-2.0, 2.0)
    q2d_top = torch.zeros(n_top, device=device, dtype=torch.float64).uniform_(-3.0, 3.0)
    top_states = torch.stack([q1_top, qd_top, q2_top, q2d_top], dim=1)  # (n_top, 4)

    # General states: full range
    q1_gen = torch.zeros(n_gen, device=device, dtype=torch.float64).uniform_(0.0, 2 * math.pi)
    qd_gen = torch.zeros(n_gen, device=device, dtype=torch.float64).uniform_(-4.0, 4.0)
    q2_gen = torch.zeros(n_gen, device=device, dtype=torch.float64).uniform_(-math.pi, math.pi)
    q2d_gen = torch.zeros(n_gen, device=device, dtype=torch.float64).uniform_(-4.0, 4.0)
    gen_states = torch.stack([q1_gen, qd_gen, q2_gen, q2d_gen], dim=1)  # (n_gen, 4)

    return torch.cat([top_states, gen_states], dim=0)  # (batch_size, 4)


def states_to_histories(states):
    """Convert (B, 4) states to (B, 5, 4) histories (repeat current state 5 times)."""
    return states.unsqueeze(1).expand(-1, 5, -1).contiguous()


def eval_hold_quality(lin_net, mpc, x0, x_goal, steps=600):
    x_t, _ = train_module.rollout(lin_net=lin_net, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([
        math.sqrt(
            math.atan2(math.sin(s[0] - math.pi), math.cos(s[0] - math.pi)) ** 2
            + s[1] ** 2 + s[2] ** 2 + s[3] ** 2
        )
        for s in traj
    ])
    arr_idx = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    return {
        "arr_idx": arr_idx,
        "frac_01": float((wraps < 0.10).mean()),
        "frac_03": float((wraps < 0.30).mean()),
        "wrap_final": float(wraps[-1]),
        "wraps": wraps,
    }


def batch_features(encoder, trunk, state_scale, states):
    """Run encoder+trunk on (B, 4) states (repeated 5× as history), no grad."""
    B = states.shape[0]
    x_flat   = states.unsqueeze(1).expand(-1, 5, -1).reshape(B, -1)  # (B, 20)
    x_normed = x_flat / state_scale.unsqueeze(0)  # (B, 20), state_scale is (20,)
    state_emb = encoder(x_normed)  # (B, enc_out)
    features  = trunk(state_emb)   # (B, trunk_out)
    return features


def batch_f_head(f_head, features, f_extra_bound, horizon, control_dim):
    """Run f_head on (B, feat) features, return (B, horizon, control_dim)."""
    raw_F   = f_head(features)         # (B, horizon*control_dim)
    f_extra = f_extra_bound * torch.tanh(raw_F)
    return f_extra.reshape(-1, horizon, control_dim)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    pretrained, src_label = _find_pretrained()
    print("=" * 80)
    print("  EXP PURE-DISTILL: MSE distillation into f_head only (batched)")
    print(f"  Pretrained: {src_label}")
    print(f"  FROZEN: state_encoder, trunk, q_head, r_head")
    print(f"  TRAINED: f_head only via pure MSE distillation")
    print(f"  ZeroFNet thresh={ZEROFNET_THRESH}  (matches 25.9% frac<0.10 variant)")
    print(f"  LR={LR}  batch={BATCH_SIZE}  batches/ep={N_BATCHES_EPOCH}  epochs={EPOCHS}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(pretrained, device=str(device)).double()
    lin_net_orig = network_module.LinearizationNetwork.load(pretrained, device=str(device)).double()

    # Freeze EVERYTHING in teacher network
    for param in lin_net_orig.parameters():
        param.requires_grad = False

    # In student network: freeze everything except f_head
    for param in lin_net.state_encoder.parameters():
        param.requires_grad = False
    for param in lin_net.trunk.parameters():
        param.requires_grad = False
    for param in lin_net.q_head.parameters():
        param.requires_grad = False
    for param in lin_net.r_head.parameters():
        param.requires_grad = False

    frozen_params = sum(p.numel() for p in lin_net.parameters() if not p.requires_grad)
    total_params  = sum(p.numel() for p in lin_net.parameters())
    f_head_params = sum(p.numel() for p in lin_net.f_head.parameters())
    print(f"\n  Frozen: {frozen_params}/{total_params} params")
    print(f"  f_head trainable: {f_head_params} params")

    # Cache network components for batched access
    enc_s   = lin_net.state_encoder
    trk_s   = lin_net.trunk
    fhd_s   = lin_net.f_head
    scale_s = lin_net.state_scale      # (20,) registered buffer
    enc_t   = lin_net_orig.state_encoder
    trk_t   = lin_net_orig.trunk
    fhd_t   = lin_net_orig.f_head
    scale_t = lin_net_orig.state_scale
    bound   = lin_net.f_extra_bound
    H       = lin_net.horizon
    U       = lin_net.control_dim

    session_name = f"stageD_pure_distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    optimizer = torch.optim.Adam(lin_net.f_head.parameters(), lr=LR)

    print("\n  Pre-eval:")
    r = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=600)
    print(f"    600 steps: frac<0.10={r['frac_01']:.1%}  frac<0.30={r['frac_03']:.1%}  "
          f"arr={r['arr_idx']}  wrap_final={r['wrap_final']:.4f}")

    best_frac01 = 0.0
    best_state  = {k: v.clone() for k, v in lin_net.state_dict().items()}
    t0 = time.time()
    all_losses = []

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        lin_net.train()

        for _ in range(N_BATCHES_EPOCH):
            states = sample_state_batch(BATCH_SIZE, device, near_top_frac=0.5)  # (B, 4)

            # Teacher features + f_extra (no grad)
            with torch.no_grad():
                feat_t  = batch_features(enc_t, trk_t, scale_t, states)         # (B, feat)
                f_orig  = batch_f_head(fhd_t, feat_t, bound, H, U)              # (B, H, U)

            # ZeroFNet gate: (B,) → (B, 1, 1) for broadcasting
            gate = zerofnet_gate(states[:, 0], x_goal[0], thresh=ZEROFNET_THRESH)
            gate = gate.reshape(-1, 1, 1)                                        # (B, 1, 1)
            f_target = f_orig * (1.0 - gate)                                    # (B, H, U)

            # Student features (no grad through encoder/trunk, frozen)
            with torch.no_grad():
                feat_s = batch_features(enc_s, trk_s, scale_s, states)          # (B, feat)

            # f_head forward WITH grad
            f_pred = batch_f_head(fhd_s, feat_s, bound, H, U)                  # (B, H, U)

            loss = F.mse_loss(f_pred, f_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / N_BATCHES_EPOCH
        all_losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  ep={epoch:>3}  mse_loss={avg_loss:.4f}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            lin_net.eval()
            r = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=600)
            print(f"  [ep={epoch}] frac<0.10={r['frac_01']:.1%}  "
                  f"frac<0.30={r['frac_03']:.1%}  arr={r['arr_idx']}  "
                  f"wrap_final={r['wrap_final']:.4f}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

            if r['frac_01'] > best_frac01:
                best_frac01 = r['frac_01']
                best_state  = {k: v.clone() for k, v in lin_net.state_dict().items()}

            ckpt_name = f"{session_name}_ep{epoch:03d}"
            network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
                model=lin_net, loss_history=all_losses,
                training_params={
                    "experiment": "pure_distill",
                    "src": src_label,
                    "zerofnet_thresh": ZEROFNET_THRESH,
                    "frozen": ["state_encoder", "trunk", "q_head", "r_head"],
                    "checkpoint_epoch": epoch,
                },
                session_name=ckpt_name,
            )
            print(f"  Checkpoint → saved_models/{ckpt_name}/", flush=True)

            if r['frac_01'] > 0.5:
                print(f"  EXCELLENT HOLD ({r['frac_01']:.1%}) — stopping early.")
                break

    lin_net.load_state_dict(best_state)
    network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
        model=lin_net, loss_history=all_losses,
        training_params={
            "experiment": "pure_distill_FINAL",
            "src": src_label,
            "best_frac01": best_frac01,
        },
        session_name=session_name,
    )
    print(f"\n  Final → saved_models/{session_name}/  best_frac01={best_frac01:.1%}")

    lin_net.eval()
    print(f"\n  Post-eval:")
    for n in [280, 600, 1000, 2000]:
        r = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=n)
        tag = "HOLD" if r['frac_01'] > 0.3 else ("PART" if r['frac_01'] > 0.05 else "FAIL")
        print(f"    {n:>4} steps: frac<0.10={r['frac_01']:.1%}  "
              f"frac<0.30={r['frac_03']:.1%}  [{tag}]")

    r2k = eval_hold_quality(lin_net, mpc, x0, x_goal, steps=2000)
    arr = r2k['arr_idx']
    if arr is not None:
        post = r2k['wraps'][arr:]
        print(f"\n  2000-step post-arrival (step {arr}):")
        for thresh in [0.05, 0.10, 0.30]:
            n_in = int(np.sum(post < thresh))
            print(f"    wrap<{thresh:.2f}: {n_in/len(post):.1%} ({n_in}/{len(post)} steps)")
    else:
        print("\n  Never arrived in 2000 steps!")


if __name__ == "__main__":
    main()
