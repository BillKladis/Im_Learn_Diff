"""exp_zerofnet_hardzero.py — Hard zero gate variant of ZeroFNet.

Soft ramp (exp_zerofnet_eval.py best): thresh=0.80 → 26.2% frac<0.10 (2000-step)
Old test result (hard zero): thresh=0.90 → 25.9% frac<0.10 (2000-step)

This sweeps hard zero thresholds to find the optimum.
Hard zero: f_extra = 0 when near_pi > thresh (binary, not ramp).
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

POSONLY_FINAL = "saved_models/stageD_posonly_ft_20260430_083618/stageD_posonly_ft_20260430_083618.pth"
X0          = [0.0, 0.0, 0.0, 0.0]
X_GOAL      = [math.pi, 0.0, 0.0, 0.0]
DT          = 0.05
HORIZON     = 10
Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]
SAVE_DIR    = "saved_models"


class HardZeroNet(nn.Module):
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
        if near_pi > self.thresh:
            f_extra = torch.zeros_like(f_extra)
        return gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf


def eval_comprehensive(model, mpc, x0, x_goal, step_counts=(600, 1000, 2000)):
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
        }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     device=device, dtype=torch.float64)
    x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)

    print("=" * 80)
    print("  ZEROFNET HARD ZERO GATE SWEEP")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
    mpc.dt          = torch.tensor(DT, device=device, dtype=torch.float64)
    mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
    lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
    lin_net.eval()

    thresholds = [0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]
    best_frac01 = 0.0
    best_thresh = None

    for thresh in thresholds:
        model = HardZeroNet(lin_net, thresh=thresh)
        model.eval()
        r = eval_comprehensive(model, mpc, x0, x_goal, step_counts=(600, 1000, 2000))
        angle_deg = math.degrees(math.acos(max(-1, min(1, 2*thresh - 1))))
        print(f"\n  thresh={thresh:.2f}  (|q1-π|<{angle_deg:.1f}°):")
        for n in (600, 1000, 2000):
            ri = r[n]
            post = f"  post<0.10={ri['post_arr_01']:.1%}" if ri['post_arr_01'] is not None else ""
            print(f"    {n:>4} steps: frac<0.10={ri['frac_01']:.1%}  "
                  f"frac<0.30={ri['frac_03']:.1%}  arr={ri['arr_idx']}{post}")
        if r[2000]['frac_01'] > best_frac01:
            best_frac01 = r[2000]['frac_01']
            best_thresh = thresh

    print(f"\n{'='*80}")
    print(f"  BEST hard-zero: thresh={best_thresh}  frac<0.10={best_frac01:.1%} (2000-step)")
    print(f"  Soft-ramp best: thresh=0.80  frac<0.10=26.2% (2000-step)")
    print(f"  Use whichever is higher.")

    if best_frac01 > 0.262:
        print(f"\n  Hard zero BEATS soft ramp! Saving best hard-zero model.")
        best_model = HardZeroNet(lin_net, thresh=best_thresh)
        session_name = f"stageD_zerofnet_hard_{best_thresh:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        network_module.ModelManager(base_dir=SAVE_DIR).save_training_session(
            model=lin_net,
            loss_history=[],
            training_params={
                "experiment": "zerofnet_hardzero",
                "source": POSONLY_FINAL,
                "gate_type": "hard_zero",
                "zerofnet_thresh": best_thresh,
                "frac_01_2000step": best_frac01,
                "NOTE": f"f_extra=0 when near_pi>{best_thresh}",
            },
            session_name=session_name,
        )
        print(f"  Saved → saved_models/{session_name}/")


if __name__ == "__main__":
    main()
