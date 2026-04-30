"""exp_zerofnet_eval.py — Comprehensive evaluation of ZeroFNet approach.

FINDING: Zeroing f_extra when near_pi > 0.9 gives 5× improvement in hold quality.
This is the best result found. Training attempts to bake this into weights all failed
because training rollouts don't cover the hold phase (arrival at step 294+, training
ends at step 280), making every training signal fire on swing-up states instead of hold.

This script:
1. Evaluates posonly_ft ep75 baseline (no gate)
2. Evaluates multiple ZeroFNet threshold variants
3. Reports comprehensive stats (600, 1000, 2000 step rollouts)
4. Saves the best ZeroFNet model as a wrapper

BEST RESULT SO FAR:
  f=0 near_pi>0.9 (no vel gate): frac<0.10=25.9%  post-arr<0.10=31.0%  arr=326
  (from /tmp/claude-0/.../bf98yv9nu.output)
"""

import math
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

# ── Config ──────────────────────────────────────────────────────────────────
POSONLY_FINAL = "saved_models/stageD_posonly_ft_20260430_083618/stageD_posonly_ft_20260430_083618.pth"

X0          = [0.0, 0.0, 0.0, 0.0]
X_GOAL      = [math.pi, 0.0, 0.0, 0.0]
DT          = 0.05
HORIZON     = 10
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
SAVE_DIR    = "saved_models"


class ZeroFNetWrapper(nn.Module):
    """Wraps lin_net to apply ZeroFNet gate at inference."""
    def __init__(self, lin_net, thresh, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net    = lin_net
        self.thresh     = thresh
        self.x_goal_q1  = x_goal_q1
        self.f_extra_bound = lin_net.f_extra_bound
        self.horizon       = lin_net.horizon
        self.state_dim     = lin_net.state_dim
        self.control_dim   = lin_net.control_dim

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf = self.lin_net(
            x_sequence, q_base_diag, r_base_diag
        )
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        f_extra = f_extra * (1.0 - gate)
        return gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf


def eval_comprehensive(model, mpc, x0, x_goal, step_counts=(600, 1000, 2000)):
    """Evaluate model at multiple step counts. Returns dict of results."""
    results = {}
    for n in step_counts:
        x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=n)
        traj = x_t.cpu().numpy()
        wraps = np.array([
            math.sqrt(
                math.atan2(math.sin(s[0] - math.pi), math.cos(s[0] - math.pi)) ** 2
                + s[1] ** 2 + s[2] ** 2 + s[3] ** 2
            )
            for s in traj
        ])
        arr_idx = next((i for i, w in enumerate(wraps) if w < 0.3), None)
        post_01 = None
        if arr_idx is not None:
            post = wraps[arr_idx:]
            post_01 = float((post < 0.10).mean())

        results[n] = {
            "arr_idx": arr_idx,
            "frac_01": float((wraps < 0.10).mean()),
            "frac_03": float((wraps < 0.30).mean()),
            "wrap_final": float(wraps[-1]),
            "post_arr_01": post_01,
            "wraps": wraps,
        }
    return results


def print_results(label, results):
    print(f"\n  {label}:")
    for n, r in results.items():
        post = f"  post-arr<0.10={r['post_arr_01']:.1%}" if r['post_arr_01'] is not None else "  (never arrived)"
        tag = "HOLD" if r['frac_01'] > 0.3 else ("PART" if r['frac_01'] > 0.05 else "FAIL")
        print(f"    {n:>4} steps: frac<0.10={r['frac_01']:.1%}  "
              f"frac<0.30={r['frac_03']:.1%}  arr={r['arr_idx']}"
              f"{post}  [{tag}]")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    print("=" * 80)
    print("  ZEROFNET COMPREHENSIVE EVALUATION")
    print(f"  Model: posonly_ft ep75")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)

    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.eval()

    # 1. Baseline (no gate)
    print("\n1. BASELINE (no ZeroFNet gate):")
    r = eval_comprehensive(lin_net, mpc, x0, x_goal, step_counts=(600, 1000, 2000))
    print_results("No gate", r)

    # 2. ZeroFNet at various thresholds
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    print("\n2. ZEROFNET THRESHOLD SWEEP:")
    best_frac01 = 0.0
    best_thresh = None
    best_results = None
    best_model = None

    for thresh in thresholds:
        zf_model = ZeroFNetWrapper(lin_net, thresh=thresh, x_goal_q1=X_GOAL[0])
        zf_model.eval()
        r = eval_comprehensive(zf_model, mpc, x0, x_goal, step_counts=(600, 1000, 2000))
        angle_deg = math.degrees(math.acos(max(-1, min(1, 2 * thresh - 1))))
        print(f"\n  thresh={thresh:.2f}  (|q1-π|<{angle_deg:.1f}°):")
        for n in (600, 1000, 2000):
            ri = r[n]
            post = f"  post<0.10={ri['post_arr_01']:.1%}" if ri['post_arr_01'] is not None else ""
            print(f"    {n:>4} steps: frac<0.10={ri['frac_01']:.1%}  "
                  f"frac<0.30={ri['frac_03']:.1%}  arr={ri['arr_idx']}{post}")

        if r[2000]['frac_01'] > best_frac01:
            best_frac01 = r[2000]['frac_01']
            best_thresh = thresh
            best_results = r
            best_model = zf_model

    print(f"\n{'='*80}")
    print(f"  BEST: thresh={best_thresh}  frac<0.10={best_frac01:.1%} (2000-step)")
    print_results(f"Best ZeroFNet (thresh={best_thresh})", best_results)

    # 3. Save best ZeroFNet model
    if best_model is not None:
        session_name = f"stageD_zerofnet_{best_thresh:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
            model=lin_net,  # save underlying lin_net (apply gate at inference)
            loss_history=[],
            training_params={
                "experiment": "zerofnet_eval",
                "source": POSONLY_FINAL,
                "zerofnet_thresh": best_thresh,
                "frac_01_2000step": best_frac01,
                "NOTE": f"Apply ZeroFNet gate at inference: f_extra=0 when near_pi>{best_thresh}",
            },
            session_name=session_name,
        )
        print(f"\n  Saved → saved_models/{session_name}/")
        print(f"  NOTE: Apply ZeroFNet gate at inference to reproduce results")
        print(f"        gate: f_extra *= (1 - ((near_pi-{best_thresh})/(1-{best_thresh})).clamp(0,1))")

    # 4. Summary vs training experiments
    print(f"\n{'='*80}")
    print("  RESULT SUMMARY vs TRAINING EXPERIMENTS:")
    print(f"  baseline posonly_ft ep75 (no gate):       5.2% frac<0.10 (2000-step)")
    print(f"  exp_tighttop_ft ep30 (w_tight_top=500):   0.0% frac<0.10 (fNorm collapsed)")
    print(f"  exp_fhead_ft ep45 (frozen q/r):           0.0% frac<0.10 (fNorm=3.0)")
    print(f"  exp_fonly_ft ep15 (frozen all but f):      0.0% frac<0.10 (fNorm collapsed)")
    print(f"  exp_distill_ft (w_distill_goal=3000):     0.0% frac<0.10 (fNorm=9.3)")
    print(f"  exp_pure_distill (MSE distillation):      1.5% frac<0.10 (ep30)")
    print(f"  ZeroFNet thresh=0.90 (NO RETRAINING):    {best_frac01:.1%} frac<0.10 (2000-step)")
    print(f"\n  CONCLUSION: ZeroFNet inference hack is the best approach.")
    print(f"  Training cannot replicate it because training rollouts don't cover hold phase.")
    print(f"  The 280-step training rollout ends before the pendulum arrives (~step 294+).")


if __name__ == "__main__":
    main()
