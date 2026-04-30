"""exp_save_scaled_ckpt.py — Save a scaled version of the best checkpoint.

Scales the trained delta_Q by a constant factor and saves as a new checkpoint.
Used to persist the 87.0% result found at scale=3.0.

USAGE:
  python exp_save_scaled_ckpt.py --scale 3.0          # save scale=3.0 as checkpoint
  python exp_save_scaled_ckpt.py --scale 3.0 --eval   # also evaluate before saving
"""

import argparse, math, os, sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

POSONLY_FINAL = "saved_models/stageD_posonly_ft_20260430_083618/stageD_posonly_ft_20260430_083618.pth"
BEST_CKPT     = "saved_models/stageD_optinit_holdboost_dq0.99x0.99_20260430_165519/stageD_optinit_holdboost_dq0.99x0.99_20260430_165519.pth"
X0            = [0.0, 0.0, 0.0, 0.0]
X_GOAL        = [math.pi, 0.0, 0.0, 0.0]
DT            = 0.05; HORIZON = 10; Q_BASE_DIAG = [12.0, 5.0, 50.0, 40.0]; THRESH = 0.8


class HoldBoostWrapper(nn.Module):
    def __init__(self, lin_net, thresh, dQ_init, dR_init=None, x_goal_q1=math.pi):
        super().__init__()
        self.lin_net = lin_net; self.thresh = thresh; self.x_goal_q1 = x_goal_q1
        self.f_extra_bound = lin_net.f_extra_bound; self.horizon = lin_net.horizon
        self.state_dim = lin_net.state_dim; self.control_dim = lin_net.control_dim
        self.delta_Q = nn.Parameter(dQ_init.clone().double())
        r_shape = (lin_net.horizon, lin_net.control_dim)
        self.delta_R = nn.Parameter(
            dR_init.clone().double() if dR_init is not None
            else torch.zeros(r_shape, dtype=torch.float64)
        )

    def forward(self, x_sequence, q_base_diag=None, r_base_diag=None):
        gQ, gR, fe, qd, rd, gQf = self.lin_net(x_sequence, q_base_diag, r_base_diag)
        q1 = x_sequence[-1, 0]
        near_pi = (1.0 + torch.cos(q1 - self.x_goal_q1)) / 2.0
        gate = ((near_pi - self.thresh) / max(1e-8, 1.0 - self.thresh)).clamp(0.0, 1.0)
        fe = fe * (1.0 - gate)
        gQ = gQ + gate * self.delta_Q
        gR = gR + gate * self.delta_R
        return gQ, gR, fe, qd, rd, gQf


def eval2k(model, mpc, x0, x_goal, steps=2000):
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0, x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([math.sqrt(
        math.atan2(math.sin(s[0]-math.pi), math.cos(s[0]-math.pi))**2
        + s[1]**2 + s[2]**2 + s[3]**2) for s in traj])
    arr = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    return float((wraps < 0.10).mean()), float((wraps < 0.30).mean()), arr, post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_ckpt", default=BEST_CKPT)
    parser.add_argument("--scale", type=float, required=True,
                        help="Scale factor for delta_Q and delta_R (e.g. 3.0)")
    parser.add_argument("--eval", action="store_true",
                        help="Run 2000-step evaluation before saving (requires CVXPY)")
    args = parser.parse_args()

    ckpt = torch.load(args.from_ckpt, map_location='cpu', weights_only=False)
    tp = ckpt['metadata'].get('training_params', {})
    dQ = torch.tensor(tp['best_delta_Q'], dtype=torch.float64)
    dR = torch.tensor(tp.get('best_delta_R', [[0]*2]*10), dtype=torch.float64)

    dQ_scaled = dQ * args.scale
    dR_scaled = dR * args.scale

    print(f"  Source: {args.from_ckpt}")
    print(f"  Scale factor: {args.scale}")
    print(f"  Original dQ mean: {dQ.mean(0).tolist()}")
    print(f"  Scaled dQ mean:   {dQ_scaled.mean(0).tolist()}")

    result_info = {}
    if args.eval:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x0 = torch.tensor(X0, device=device, dtype=torch.float64)
        x_goal = torch.tensor(X_GOAL, device=device, dtype=torch.float64)
        mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=HORIZON, device=device)
        mpc.dt = torch.tensor(DT, device=device, dtype=torch.float64)
        mpc.q_base_diag = torch.tensor(Q_BASE_DIAG, device=device, dtype=torch.float64)
        lin_net = network_module.LinearizationNetwork.load(POSONLY_FINAL, device=str(device)).double()
        lin_net.eval()

        boost = HoldBoostWrapper(lin_net, thresh=THRESH, dQ_init=dQ_scaled, dR_init=dR_scaled)
        f01, f03, arr, post = eval2k(boost, mpc, x0, x_goal)
        print(f"  Eval: frac<0.10={f01:.1%}  f<0.30={f03:.1%}  arr={arr}  "
              f"post={f'{post:.1%}' if post else 'N/A'}")
        result_info = {
            "best_frac01_2000step": f01,
            "eval_f03": f03,
            "eval_arr": arr,
            "eval_post_arr": post,
        }

        lin_net_for_save = lin_net
    else:
        lin_net_for_save = network_module.LinearizationNetwork.load(
            POSONLY_FINAL, map_location='cpu')

    from datetime import datetime
    session_name = f"stageD_scale{args.scale:.1f}x_dQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    network_module.ModelManager(base_dir="saved_models").save_training_session(
        model=lin_net_for_save if args.eval else network_module.LinearizationNetwork.load(POSONLY_FINAL),
        loss_history=[],
        training_params={
            "experiment": f"scale_{args.scale}x_dQ",
            "source_ckpt": args.from_ckpt,
            "scale_factor": args.scale,
            "best_delta_Q": dQ_scaled.tolist(),
            "best_delta_R": dR_scaled.tolist(),
            **result_info,
        },
        session_name=session_name,
    )
    print(f"  Saved as: {session_name}")


if __name__ == "__main__":
    main()
