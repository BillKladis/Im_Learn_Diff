"""hardware_deploy.py — Hardware abstraction for MAB double pendulum deployment.

Wraps pyCandle I/O, model loading (auto-detected architecture), EKF state
estimation, and single/double actuation selection into one clean interface.

Usage
-----
  # Simulation test (no hardware):
  python hardware_deploy.py --model latest_v1 --sim

  # Real hardware, double-actuated, EKF6:
  python hardware_deploy.py --model latest_v1 --ekf ekf6

  # Single-actuated (shoulder only), u_lim override:
  python hardware_deploy.py --model latest_v5 --actuate single

  # Quick model-load check:
  python hardware_deploy.py --model latest_v3 --check

State ordering
--------------
  Hardware (pyCandle): [q1, q2,     q1_dot, q2_dot]
  Network (ours):      [q1, q1_dot, q2,     q2_dot]
  Permutation (self-inverse): x_ours = x_hw[[0, 2, 1, 3]]

Joint mapping
-------------
  Joint 1 (shoulder): MAB ID 402
  Joint 2 (elbow):    MAB ID 382
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
from ekf import EKF4, EKF6

# ── Hardware IDs ───────────────────────────────────────────────────────────────
SHOULDER_ID = 402
ELBOW_ID    = 382

# State permutation: hardware [q1,q2,q1d,q2d] ↔ ours [q1,q1d,q2,q2d]
HW_TO_OURS = [0, 2, 1, 3]   # self-inverse

# Default EKF noise tuning (MAB double pendulum)
Q_STATE = torch.diag(torch.tensor([1e-6, 1e-4, 1e-6, 1e-4], dtype=torch.float64))
Q_BIAS  = torch.eye(2, dtype=torch.float64) * 1e-3
R_OBS   = torch.diag(torch.tensor([4e-6, 1e-3, 4e-6, 1e-3], dtype=torch.float64))

# Single-actuated elbow PD (holds q2 ≈ 0 when shoulder is torque-controlled)
SA_KP = 5.0   # Nm/rad
SA_KD = 0.5   # Nm·s/rad

# Safety
ESTOP_VEL_LIMIT = 20.0   # rad/s


# ── SA dynamics helper ────────────────────────────────────────────────────────

def wrap_sa_dynamics(mpc) -> None:
    """Replace mpc.true_RK4_disc with a rigid-elbow version (q2=q2d=0).

    Used for both sim and hardware SA mode so MPC planning matches the
    rigid-elbow dynamics the model was trained with.
    """
    orig = mpc.true_RK4_disc

    def _sa_rk4(x, u, dt, n_sub=10):
        x_in = torch.cat([x[:2], torch.zeros(2, dtype=x.dtype, device=x.device)])
        u_in = torch.cat([u[:1], torch.zeros(1, dtype=u.dtype, device=u.device)])
        x_out = orig(x_in, u_in, dt, n_sub)
        return torch.cat([x_out[:2], torch.zeros(2, dtype=x_out.dtype, device=x_out.device)])

    mpc.true_RK4_disc = _sa_rk4


# ── Model loading ──────────────────────────────────────────────────────────────

def resolve_model_path(spec: str) -> str:
    """Resolve shorthand like 'latest_v1' to an actual .pth file path."""
    shorthands = {
        "latest_v1":    "saved_models/hw_v1*/*.pth",
        "latest_v2":    "saved_models/hw_v2_nr_*FINAL*/*.pth",
        "latest_v2diag":"saved_models/hw_v2_nr_diag*/*.pth",
        "latest_v3":    "saved_models/hw_v3_u010*/*.pth",
        "latest_v4":    "saved_models/hw_v4_u007*/*.pth",
        "latest_v5":    "saved_models/hw_v5_sa015*/*.pth",
        "latest_v6":    "saved_models/hw_v6_sa010*/*.pth",
    }
    pattern = shorthands.get(spec, spec)
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No checkpoint found for {spec!r} (pattern: {pattern})")
    return max(paths, key=os.path.getmtime)


def detect_architecture(state_dict: dict) -> str:
    """Infer architecture from state_dict key structure."""
    keys = set(state_dict.keys())
    if any(k.startswith("f_net.") for k in keys):
        return "SeparatedLinearizationNetwork"
    # SinCos variant: encoder input dim = 30 (6 history × 5 sincos features)
    enc = state_dict.get("encoder.0.weight")
    if enc is not None and enc.shape[1] == 30:
        return "LinearizationNetworkSC"
    return "LinearizationNetwork"


def load_model(ckpt_path: str) -> Tuple[object, dict, float]:
    """Load checkpoint → (model, info_dict, u_lim).

    info_dict contains 'arch', 'u_lim', and model kwargs.
    Model is returned in eval mode on CPU.
    """
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        state_dict = data.get("model_state_dict", data)
        metadata   = data.get("metadata", {}) or {}
        # training_params may sit at top level or nested inside metadata
        tp = data.get("training_params") or metadata.get("training_params") or {}
        u_lim = tp.get("u_lim", 0.15)
    else:
        state_dict = data
        metadata   = {}
        u_lim      = 0.15

    arch = detect_architecture(state_dict)

    # Infer model kwargs (fall back to standard defaults)
    def _from(key, default):
        return metadata.get(key, default)

    kwargs = dict(
        state_dim       = _from("state_dim",       4),
        control_dim     = _from("control_dim",      2),
        horizon         = _from("horizon",          10),
        hidden_dim      = _from("hidden_dim",       128),
        gate_range_q    = _from("gate_range_q",     0.99),
        gate_range_r    = _from("gate_range_r",     0.20),
        f_extra_bound   = _from("f_extra_bound",    1.5),
        f_kickstart_amp = _from("f_kickstart_amp",  0.01),
    )

    arch_cls = getattr(network_module, arch)
    model = arch_cls(**kwargs).double()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected}")

    model.eval()
    info = {"arch": arch, "u_lim": u_lim, **kwargs}
    return model, info, u_lim


# ── Hardware interface ─────────────────────────────────────────────────────────

class HardwareInterface:
    """Thin wrapper around pyCandle for the MAB double pendulum.

    In simulation mode (sim=True) the hardware is replaced by the MPC
    dynamics model so the same control loop can be tested without hardware.
    """

    def __init__(self, mpc, sim: bool = False):
        self.sim = sim
        self.mpc = mpc
        self._x  = torch.zeros(4, dtype=torch.float64)  # sim state (our ordering)

        if not sim:
            import pyCandle
            self._pc = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True, pyCandle.USB)
            ids = self._pc.ping()
            if SHOULDER_ID not in ids or ELBOW_ID not in ids:
                raise RuntimeError(
                    f"Expected motors {SHOULDER_ID},{ELBOW_ID}, found {ids}"
                )
            self._sh = next(m for m in self._pc.getMd80List() if m.getId() == SHOULDER_ID)
            self._el = next(m for m in self._pc.getMd80List() if m.getId() == ELBOW_ID)
            for md in (self._sh, self._el):
                self._pc.controlMd80Mode(md.getId(), pyCandle.IMPEDANCE)
                md.setImpedanceControllerParams(0.0, 0.0)  # pure torque mode
                self._pc.controlMd80Enable(md.getId(), True)
            self._pc.begin()
            print(f"  Hardware ready: shoulder={SHOULDER_ID}, elbow={ELBOW_ID}")

    def read(self) -> np.ndarray:
        """Return state in hardware ordering: [q1, q2, q1_dot, q2_dot]."""
        if self.sim:
            x = self._x.numpy()
            return np.array([x[0], x[2], x[1], x[3]])  # ours→hw permutation
        self._pc.waitForAnswer()
        return np.array([
            self._sh.getPosition(), self._el.getPosition(),
            self._sh.getVelocity(), self._el.getVelocity(),
        ])

    def write(self, tau: np.ndarray):
        """Apply torques [tau1, tau2] (Nm). tau1=shoulder, tau2=elbow."""
        if self.sim:
            u = torch.tensor(tau, dtype=torch.float64)
            self._x = self.mpc.true_RK4_disc(self._x, u, self.mpc.dt)
            return
        self._sh.setTargetTorque(float(tau[0]))
        self._el.setTargetTorque(float(tau[1]))
        self._pc.transmit()

    def zero(self):
        """Send zero torques (safe shutdown)."""
        self.write(np.zeros(2))

    def close(self):
        self.zero()
        if not self.sim:
            for md in (self._sh, self._el):
                self._pc.controlMd80Enable(md.getId(), False)
            self._pc.end()


# ── Control loop ───────────────────────────────────────────────────────────────

class ControlLoop:
    """
    One-step controller: read hardware → EKF → lin_net + QP → apply torque.

    Parameters
    ----------
    model     : loaded lin_net (SeparatedLinearizationNetwork etc.)
    mpc       : MPC_controller instance (carries QP and dynamics)
    hw        : HardwareInterface
    x_goal    : goal state in our ordering [q1, q1d, q2, q2d]
    ekf_mode  : 'none' | 'ekf4' | 'ekf6'
    actuate   : 'double' | 'single'  (single = shoulder only, elbow PD-held)
    u_lim     : torque limit (Nm) — used for single-actuated elbow clamp
    monitor   : verbosity 0=none 1=minimal 2=standard
    """

    def __init__(
        self,
        model,
        mpc,
        hw:        HardwareInterface,
        x_goal:    torch.Tensor,
        ekf_mode:  str   = "ekf6",
        actuate:   str   = "double",
        u_lim:     float = 0.15,
        monitor:   int   = 1,
    ):
        self.model    = model
        self.mpc      = mpc
        self.hw       = hw
        self.x_goal   = x_goal
        self.actuate  = actuate
        self.u_lim    = u_lim
        self.monitor  = monitor
        self.step_n   = 0

        # Build EKF
        if ekf_mode == "ekf6":
            self.ekf = EKF6(mpc, Q_STATE, Q_BIAS, R_OBS)
        elif ekf_mode == "ekf4":
            self.ekf = EKF4(mpc, Q_STATE, R_OBS)
        else:
            self.ekf = None

        # State history for lin_net (5 frames)
        self._hist       = torch.zeros((5, 4), dtype=torch.float64)
        # Warm-start control sequence for QP (N × 2)
        self._u_seq      = torch.zeros((mpc.N, 2), dtype=torch.float64)
        # Previous torque sent to hardware (for EKF predict step)
        self._u_prev     = torch.zeros(2, dtype=torch.float64)

    def reset(self, x_hw: np.ndarray):
        x_t = torch.tensor(np.array(x_hw)[HW_TO_OURS], dtype=torch.float64)
        self._hist[:] = x_t
        self._u_seq[:]  = 0.0
        self._u_prev[:] = 0.0
        if self.ekf is not None:
            self.ekf.reset(x_t)
        self.step_n = 0

    def step(self, x_hw: np.ndarray) -> np.ndarray:
        """One control step. Returns [tau1, tau2] actually applied."""
        t0 = time.perf_counter()

        # 1. Permute hardware ordering → network ordering
        x_raw = torch.tensor(np.array(x_hw)[HW_TO_OURS], dtype=torch.float64)

        # 2. EKF predict+update (skip on first step — no previous u yet)
        if self.ekf is not None and self.step_n > 0:
            x_est, bias = self.ekf.step(x_raw, self._u_prev)
        else:
            x_est = x_raw.clone()
            bias  = torch.zeros(2, dtype=torch.float64)
            if self.ekf is not None and self.step_n == 0:
                self.ekf.reset(x_est)

        # 3. Roll state history and run lin_net
        # For SA mode: zero out elbow states (q2, q2d) before model inference —
        # the model was trained with rigid-link freeze and has never seen q2≠0.
        x_model = x_est.clone()
        if self.actuate == "single":
            x_model[2] = 0.0
            x_model[3] = 0.0
        self._hist = torch.roll(self._hist, -1, dims=0)
        self._hist[-1] = x_model
        with torch.no_grad():
            gQ, gR, f_extra, _, _, gQf = self.model(
                self._hist, self.mpc.q_base_diag, self.mpc.r_base_diag
            )

        # 4. Build linearisation sequences and call QP
        x_lin_seq = x_model.unsqueeze(0).expand(self.mpc.N, -1).clone()
        u_lin_seq = torch.clamp(self._u_seq.clone(),
                                min=self.mpc.MPC_dynamics.u_min.unsqueeze(0),
                                max=self.mpc.MPC_dynamics.u_max.unsqueeze(0))

        u_opt, U_opt_full = self.mpc.control(
            x_model, x_lin_seq, u_lin_seq, self.x_goal,
            diag_corrections_Q=gQ,
            diag_corrections_R=gR,
            extra_linear_control=f_extra.reshape(-1),
            diag_corrections_Qf=gQf,
        )

        # Shift warm-start for next step
        U_reshaped = U_opt_full.detach().view(self.mpc.N, 2)
        self._u_seq[:-1] = U_reshaped[1:].clone()
        self._u_seq[-1]  = U_reshaped[-1].clone()

        # 5. Bias cancellation (EKF6 only; skip first 20 steps for warmup)
        if isinstance(self.ekf, EKF6) and self.step_n >= 20:
            u_cmd = torch.clamp(u_opt.detach() - bias,
                                self.mpc.MPC_dynamics.u_min,
                                self.mpc.MPC_dynamics.u_max)
        else:
            u_cmd = torch.clamp(u_opt.detach(),
                                self.mpc.MPC_dynamics.u_min,
                                self.mpc.MPC_dynamics.u_max)

        tau = u_cmd.numpy().copy()

        # 6. Single-actuated: replace elbow torque with PD to hold q2 ≈ 0
        if self.actuate == "single":
            q2  = float(x_est[2])
            q2d = float(x_est[3])
            tau[1] = float(np.clip(-SA_KP * q2 - SA_KD * q2d, -self.u_lim, self.u_lim))

        # 7. Safety: e-stop on excessive velocity
        vel = float(np.linalg.norm([float(x_est[1]), float(x_est[3])]))
        if vel > ESTOP_VEL_LIMIT:
            tau = np.zeros(2)
            if self.monitor >= 1:
                print(f"  [ESTOP] step={self.step_n}  vel={vel:.1f} rad/s", flush=True)

        # 8. Send torques; record for next EKF predict
        self.hw.write(tau)
        self._u_prev = torch.tensor(tau, dtype=torch.float64)
        self.step_n += 1

        # 9. Periodic status print
        if self.monitor >= 2 and self.step_n % 50 == 0:
            err = float(np.linalg.norm(x_est.numpy() - self.x_goal.numpy()))
            dt_ms = (time.perf_counter() - t0) * 1000
            print(f"  step={self.step_n:>5}  u=[{tau[0]:+.3f},{tau[1]:+.3f}]"
                  f"  err={err:.3f}  loop={dt_ms:.1f}ms", flush=True)

        return tau


# ── Run loop ───────────────────────────────────────────────────────────────────

def run(
    model,
    mpc,
    x_goal:   torch.Tensor,
    sim:      bool  = False,
    ekf_mode: str   = "ekf6",
    actuate:  str   = "double",
    u_lim:    float = 0.15,
    monitor:  int   = 1,
    dt:       float = 0.05,
    max_steps: int  = 4000,
):
    hw   = HardwareInterface(mpc, sim=sim)
    ctrl = ControlLoop(model, mpc, hw, x_goal,
                       ekf_mode=ekf_mode, actuate=actuate,
                       u_lim=u_lim, monitor=monitor)

    print(f"  Starting control loop  sim={sim}  ekf={ekf_mode}  "
          f"actuate={actuate}  u_lim={u_lim}  dt={dt}s", flush=True)
    print("  Press Ctrl+C to stop.", flush=True)

    x_hw = hw.read()
    ctrl.reset(x_hw)

    try:
        for _ in range(max_steps):
            t_loop = time.perf_counter()
            x_hw   = hw.read()
            ctrl.step(x_hw)
            elapsed = time.perf_counter() - t_loop
            sleep   = max(0.0, dt - elapsed)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        hw.close()
        print("  Torques zeroed. Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="MAB double pendulum hardware deployment")
    p.add_argument("--model",   default="latest_v1",
                   help="Checkpoint path or shorthand: latest_v1/v2/v3/v4/v5/v6")
    p.add_argument("--ekf",     default="ekf6", choices=["none", "ekf4", "ekf6"])
    p.add_argument("--actuate", default="double", choices=["double", "single"])
    p.add_argument("--u_lim",   type=float, default=None,
                   help="Override torque limit from checkpoint")
    p.add_argument("--sim",     action="store_true",
                   help="Simulation mode (no hardware)")
    p.add_argument("--check",   action="store_true",
                   help="Just load and print model info, then exit")
    p.add_argument("--steps",   type=int, default=4000)
    p.add_argument("--monitor", type=int, default=1, choices=[0, 1, 2],
                   help="0=silent 1=minimal 2=standard")
    args = p.parse_args()

    ckpt = resolve_model_path(args.model)
    print(f"  Loading: {ckpt}")
    model, info, u_lim_ckpt = load_model(ckpt)
    u_lim = args.u_lim if args.u_lim is not None else u_lim_ckpt
    print(f"  arch={info['arch']}  u_lim={u_lim}  horizon={info['horizon']}")

    if args.check:
        print("  Model loaded OK.")
        return

    x0     = torch.zeros(4, dtype=torch.float64)
    x_goal = torch.tensor([math.pi, 0.0, 0.0, 0.0], dtype=torch.float64)

    mpc = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=info["horizon"],
                                     device=torch.device("cpu"), u_lim=u_lim)
    mpc.dt = torch.tensor(0.05, dtype=torch.float64)

    # SA mode: MPC plans with rigid-elbow dynamics (matches SA training)
    if args.actuate == "single":
        wrap_sa_dynamics(mpc)
        print("  SA mode: rigid-elbow dynamics in MPC planner")

    run(model, mpc, x_goal,
        sim=args.sim, ekf_mode=args.ekf, actuate=args.actuate,
        u_lim=u_lim, monitor=args.monitor, max_steps=args.steps)


if __name__ == "__main__":
    main()
