"""hardware_deploy.py — Plug-and-play deployment pipeline for MAB double pendulum.

Minimal-latency control loop with configurable EKF, TVLQR stabiliser,
auto-detected network architecture, single/double actuation, and
per-step diagnostics at selectable verbosity levels.

Usage examples
--------------
# Simulation (no hardware):
python hardware_deploy.py --mode sim --model latest_v1

# Real hardware, full diagnostics, EKF6, TVLQR stabiliser:
python hardware_deploy.py --mode hw --model latest_v1 --ekf ekf6 \
    --stabilizer tvlqr --tvlqr_policy tvlqr_policy_1.npz \
    --monitor diagnostic --log /tmp/deploy.log

# Single-actuated shoulder-only, sim test:
python hardware_deploy.py --mode sim --model latest_v3 --actuate single \
    --u_lim 0.10 --monitor verbose

# Latency benchmark:
python hardware_deploy.py --mode bench --model saved_models/hw_v1_*/hw_v1_*.pth

State ordering
--------------
  Hardware (pyCandle):  [q1, q2,     q1_dot, q2_dot]
  Network (ours):       [q1, q1_dot, q2,     q2_dot]
  Permutation:  x_ours = x_hw[[0, 2, 1, 3]]   (same inverse)
  TVLQR uses hardware ordering internally.

Joint mapping
-------------
  SHOULDER = joint 1 (q1),  MAB ID 402
  ELBOW    = joint 2 (q2),  MAB ID 382
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
from ekf import EKF4, EKF6

# ── Constants ─────────────────────────────────────────────────────────────────
SHOULDER_ID = 402
ELBOW_ID    = 382

# Permutation between hardware and our state ordering (self-inverse)
HW_TO_OURS  = [0, 2, 1, 3]   # x_ours = x_hw[[0,2,1,3]]
OURS_TO_HW  = [0, 2, 1, 3]   # same permutation

# Default EKF noise matrices (tuned for MAB double pendulum)
DEFAULT_Q_STATE = torch.diag(torch.tensor([1e-6, 1e-4, 1e-6, 1e-4], dtype=torch.float64))
DEFAULT_Q_BIAS  = torch.eye(2, dtype=torch.float64) * 1e-3
DEFAULT_R_OBS   = torch.diag(torch.tensor([4e-6, 1e-3, 4e-6, 1e-3], dtype=torch.float64))

# Single-actuated PD stiffness on joint 2 (hold elbow rigid at q2=0)
SA_KP = 5.0   # Nm/rad
SA_KD = 0.5   # Nm·s/rad

# Safety limits
SAFETY_ERR_LIMIT = 2.0   # rad — max angle error before E-stop (angle components only)
SAFETY_VEL_LIMIT = 20.0  # rad/s — max velocity before E-stop
RESET_DELAY      = 0.5   # seconds after E-stop before re-arm

# Monitor verbosity levels
MONITOR_NONE       = 0
MONITOR_MINIMAL    = 1
MONITOR_STANDARD   = 2
MONITOR_VERBOSE    = 3
MONITOR_DIAGNOSTIC = 4

MONITOR_LEVELS = {
    "none":       MONITOR_NONE,
    "minimal":    MONITOR_MINIMAL,
    "standard":   MONITOR_STANDARD,
    "verbose":    MONITOR_VERBOSE,
    "diagnostic": MONITOR_DIAGNOSTIC,
}


# ── Diagnostics ────────────────────────────────────────────────────────────────

@dataclass
class StepMetrics:
    step:          int
    t_read_ms:     float = 0.0
    t_ekf_ms:      float = 0.0
    t_model_ms:    float = 0.0
    t_qp_ms:       float = 0.0
    t_tvlqr_ms:    float = 0.0
    t_write_ms:    float = 0.0
    t_total_ms:    float = 0.0
    t_sleep_ms:    float = 0.0
    jitter_ms:     float = 0.0
    qp_fallback:   bool  = False
    tvlqr_active:  bool  = False
    safety_trip:   bool  = False
    u_mpc:         np.ndarray = field(default_factory=lambda: np.zeros(2))
    u_tvlqr:       np.ndarray = field(default_factory=lambda: np.zeros(2))
    u_applied:     np.ndarray = field(default_factory=lambda: np.zeros(2))
    bias_est:      np.ndarray = field(default_factory=lambda: np.zeros(2))
    ekf_innov:     np.ndarray = field(default_factory=lambda: np.zeros(4))
    x_raw:         np.ndarray = field(default_factory=lambda: np.zeros(4))
    x_est:         np.ndarray = field(default_factory=lambda: np.zeros(4))
    err_norm:      float = 0.0


class DiagnosticsLogger:
    """Ring-buffer metric logger with selectable verbosity."""

    _HDR = (
        f"{'step':>6}  {'t_rd':>6}  {'t_ekf':>6}  {'t_mod':>6}  {'t_qp':>6}  "
        f"{'t_tvlqr':>7}  {'t_wr':>6}  {'t_tot':>6}  {'jit':>6}  "
        f"{'tvlqr':>6}  {'fb':>3}  {'safe':>4}  "
        f"{'u1':>7}  {'u2':>7}  {'b1':>7}  {'b2':>7}  "
        f"{'q1':>7}  {'q1d':>7}  {'q2':>7}  {'q2d':>7}  {'err':>6}"
    )

    def __init__(
        self,
        level:       int             = MONITOR_STANDARD,
        window:      int             = 300,
        print_every: int             = 50,
        log_file:    Optional[str]   = None,
    ):
        self.level       = level
        self.print_every = print_every
        self.buf: deque  = deque(maxlen=window)
        self._fh         = open(log_file, "w", buffering=1) if log_file else None

        if level >= MONITOR_VERBOSE:
            self._write(self._HDR)
            self._write("-" * len(self._HDR))

    # ── public ────────────────────────────────────────────────────────────────

    def log(self, m: StepMetrics):
        if self.level == MONITOR_NONE:
            return
        self.buf.append(m)

        if self.level >= MONITOR_VERBOSE:
            self._write(self._row(m))

        if m.step > 0 and m.step % self.print_every == 0:
            if self.level >= MONITOR_STANDARD:
                self._print_summary(m.step)
            elif self.level == MONITOR_MINIMAL:
                print(f"  step={m.step:>5}  u=[{m.u_applied[0]:+.3f},{m.u_applied[1]:+.3f}]"
                      f"  err={m.err_norm:.3f}  loop={m.t_total_ms:.1f}ms", flush=True)

    def final_summary(self):
        if not self.buf or self.level == MONITOR_NONE:
            return
        ms = list(self.buf)
        self._banner("FINAL DIAGNOSTICS SUMMARY")
        self._timing_table(ms)
        fb_rate = sum(1 for m in ms if m.qp_fallback) / len(ms) * 100
        tvlqr_rate = sum(1 for m in ms if m.tvlqr_active) / len(ms) * 100
        safe_trips = sum(1 for m in ms if m.safety_trip)
        print(f"  QP fallback rate : {fb_rate:.2f}%")
        print(f"  TVLQR active     : {tvlqr_rate:.1f}%")
        print(f"  Safety trips     : {safe_trips}")
        biases = np.array([m.bias_est for m in ms])
        print(f"  Bias est (final) : d1={biases[-1,0]:+.4f}  d2={biases[-1,1]:+.4f} Nm")

    def close(self):
        if self._fh:
            self._fh.close()

    # ── internal ──────────────────────────────────────────────────────────────

    def _row(self, m: StepMetrics) -> str:
        return (
            f"{m.step:>6}  {m.t_read_ms:>6.2f}  {m.t_ekf_ms:>6.2f}  "
            f"{m.t_model_ms:>6.2f}  {m.t_qp_ms:>6.2f}  {m.t_tvlqr_ms:>7.2f}  "
            f"{m.t_write_ms:>6.2f}  {m.t_total_ms:>6.2f}  {m.jitter_ms:>6.2f}  "
            f"{'Y' if m.tvlqr_active else 'N':>6}  {'Y' if m.qp_fallback else 'N':>3}  "
            f"{'Y' if m.safety_trip else 'N':>4}  "
            f"{m.u_applied[0]:>7.4f}  {m.u_applied[1]:>7.4f}  "
            f"{m.bias_est[0]:>7.4f}  {m.bias_est[1]:>7.4f}  "
            f"{m.x_est[0]:>7.4f}  {m.x_est[1]:>7.4f}  "
            f"{m.x_est[2]:>7.4f}  {m.x_est[3]:>7.4f}  {m.err_norm:>6.3f}"
        )

    def _print_summary(self, step: int):
        if not self.buf:
            return
        ms = list(self.buf)
        totals  = np.array([m.t_total_ms for m in ms])
        qps     = np.array([m.t_qp_ms    for m in ms])
        jitters = np.abs([m.jitter_ms    for m in ms])
        fb_rate = sum(1 for m in ms if m.qp_fallback) / len(ms) * 100
        biases  = np.array([m.bias_est for m in ms])
        errs    = np.array([m.err_norm for m in ms])
        print(
            f"\n  [step={step:>5}]"
            f"  loop {totals.mean():.1f}/{np.percentile(totals,99):.1f}ms (mean/p99)"
            f"  QP {qps.mean():.1f}/{np.percentile(qps,99):.1f}ms"
            f"  jitter {jitters.mean():.1f}ms"
            f"  fb={fb_rate:.0f}%"
            f"  err_mean={errs.mean():.3f}"
            f"  bias=[{biases[-1,0]:+.4f},{biases[-1,1]:+.4f}]",
            flush=True,
        )

    def _timing_table(self, ms: List[StepMetrics]):
        totals  = np.array([m.t_total_ms  for m in ms])
        qps     = np.array([m.t_qp_ms     for m in ms])
        ekfs    = np.array([m.t_ekf_ms    for m in ms])
        models  = np.array([m.t_model_ms  for m in ms])
        jitters = np.abs([m.jitter_ms     for m in ms])

        def row(name, a):
            a = np.array(a)
            print(f"  {name:<22} mean={a.mean():>6.2f}ms  p50={np.percentile(a,50):>6.2f}ms"
                  f"  p95={np.percentile(a,95):>6.2f}ms  p99={np.percentile(a,99):>6.2f}ms"
                  f"  max={a.max():>6.2f}ms")

        row("Loop total",      totals)
        row("QP solve",        qps)
        row("EKF update",      ekfs)
        row("Model inference", models)
        row("Timing jitter",   jitters)

    def _banner(self, title: str):
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def _write(self, line: str):
        print(line, flush=True)
        if self._fh:
            self._fh.write(line + "\n")


# ── TVLQR hybrid controller ────────────────────────────────────────────────────

class TVLQRController:
    """KNN-based TVLQR using a pre-computed trajectory policy.

    Policy .npz must contain:
        states      (N, 4)  — reference trajectory in hardware ordering
        times       (N,)    — time stamps (seconds)
        gains       (N, 2, 4) — LQR gains K at each waypoint
        feedforward (N, 2)  — feedforward d at each waypoint
        u_nom       (N, 2)  — nominal control along trajectory

    All arrays use hardware joint ordering: [q1, q2, q1_dot, q2_dot].
    Control law:  u = u_nom(t) - K_knn(x) @ (x - x_nom(t)) - d(t)
    """

    def __init__(
        self,
        policy_path:  str,
        n_neighbors:  int   = 10,
        activate_err: float = 0.5,   # switch from MPC to TVLQR below this error norm
        u_lim:        float = 0.15,
    ):
        data = np.load(policy_path)
        self.states      = data["states"]       # (N, 4) hw ordering
        self.times       = data["times"]        # (N,)
        self.gains       = data["gains"]        # (N, 2, 4)
        self.feedforward = data["feedforward"]  # (N, 2)
        self.u_nom       = data["u_nom"]        # (N, 2)
        self.n_neighbors = n_neighbors
        self.activate_err = activate_err
        self.u_lim        = u_lim
        self._t0: Optional[float] = None

    def reset(self):
        self._t0 = None

    def get_control(
        self,
        x_hw:     np.ndarray,    # hardware ordering [q1, q2, q1_dot, q2_dot]
        t_elapsed: float,        # seconds since episode start
    ) -> np.ndarray:
        """Return u in hardware ordering [tau1, tau2]."""
        if self._t0 is None:
            self._t0 = t_elapsed

        # Clamp time to trajectory range
        t = np.clip(t_elapsed, self.times[0], self.times[-1])

        # ── Interpolate u_nom, x_nom, d at time t ──
        idx = np.searchsorted(self.times, t)
        idx = np.clip(idx, 1, len(self.times) - 1)
        t0, t1 = self.times[idx-1], self.times[idx]
        alpha  = (t - t0) / (t1 - t0 + 1e-12)

        x_nom = (1-alpha) * self.states[idx-1]      + alpha * self.states[idx]
        u_n   = (1-alpha) * self.u_nom[idx-1]       + alpha * self.u_nom[idx]
        d     = (1-alpha) * self.feedforward[idx-1] + alpha * self.feedforward[idx]

        # ── KNN gain lookup on state space ──
        diffs = self.states - x_hw[np.newaxis, :]         # (N, 4)
        dists = np.sum(diffs**2, axis=1)
        nn    = np.argpartition(dists, min(self.n_neighbors, len(dists)-1))[:self.n_neighbors]
        w     = 1.0 / (dists[nn] + 1e-12)
        w    /= w.sum()
        K     = np.einsum("i,ijk->jk", w, self.gains[nn])   # (2, 4) weighted K

        # ── Control law ──
        u = u_n - K @ (x_hw - x_nom) - d
        return np.clip(u, -self.u_lim, self.u_lim)

    def should_activate(self, x_hw: np.ndarray, x_goal_hw: np.ndarray) -> bool:
        """True when close enough to goal for TVLQR to be reliable."""
        dq1 = math.atan2(math.sin(x_hw[0] - x_goal_hw[0]),
                         math.cos(x_hw[0] - x_goal_hw[0]))
        dq2 = math.atan2(math.sin(x_hw[1] - x_goal_hw[1]),
                         math.cos(x_hw[1] - x_goal_hw[1]))
        err  = math.sqrt(dq1**2 + x_hw[2]**2 + dq2**2 + x_hw[3]**2)
        return err < self.activate_err


# ── Hardware interfaces ────────────────────────────────────────────────────────

class HardwareInterface(ABC):
    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def read_state(self) -> Tuple[np.ndarray, float]:
        """Return (x_hw [q1,q2,q1d,q2d], read_latency_ms)."""

    @abstractmethod
    def write_torque(self, tau: np.ndarray) -> float:
        """Send [tau1, tau2] Nm. Return write_latency_ms."""

    @abstractmethod
    def disconnect(self) -> None: ...


class SimulationInterface(HardwareInterface):
    """Software-in-the-loop simulation using true RK4 dynamics."""

    def __init__(
        self,
        mpc,
        x0:          torch.Tensor,
        dt:          float          = 0.05,
        obs_sigma:   float          = 0.0,
        ctrl_bias:   Optional[np.ndarray] = None,
    ):
        self.mpc      = mpc
        self.x        = x0.clone().double()
        self.dt_t     = torch.tensor(dt, dtype=torch.float64)
        self.obs_sigma = obs_sigma
        self.ctrl_bias = torch.tensor(ctrl_bias, dtype=torch.float64) \
                         if ctrl_bias is not None else None
        self._u_last  = torch.zeros(2, dtype=torch.float64)

    def connect(self) -> bool:
        return True

    def read_state(self) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        self.x = self.mpc.true_RK4_disc(self.x, self._u_last, self.dt_t).detach()
        x_np = self.x.numpy()[OURS_TO_HW]    # → hardware ordering
        if self.obs_sigma > 0:
            x_np = x_np + np.random.randn(4) * self.obs_sigma
        return x_np, (time.perf_counter() - t0) * 1e3

    def write_torque(self, tau: np.ndarray) -> float:
        t0 = time.perf_counter()
        u = torch.tensor(tau, dtype=torch.float64)
        if self.ctrl_bias is not None:
            u = u + self.ctrl_bias
        u_lim = float(self.mpc.MPC_dynamics.u_max[0])
        self._u_last = u.clamp(-u_lim, u_lim)
        return (time.perf_counter() - t0) * 1e3

    def disconnect(self) -> None:
        self._u_last = torch.zeros(2, dtype=torch.float64)

    @property
    def true_state_ours(self) -> np.ndarray:
        return self.x.numpy()


class MABInterface(HardwareInterface):
    """pyCandle MAB Robotics interface.

    Uses IMPEDANCE mode with zero stiffness/damping = pure torque control.
    Motors: SHOULDER_ID=402 (joint 1), ELBOW_ID=382 (joint 2).
    Hardware state ordering: [q1, q2, q1_dot, q2_dot]
    """

    def __init__(self, u_lim: float = 0.15):
        self.u_lim    = u_lim
        self._candle  = None
        self._shoulder = None
        self._elbow    = None

    def connect(self) -> bool:
        try:
            import pyCandle
        except ImportError:
            raise ImportError(
                "pyCandle not found. Install the MAB Robotics pyCandle driver:\n"
                "  pip install pyCandle  (or build from MAB source)"
            )

        print("  [HW] Connecting to MAB hardware via pyCandle CAN ...", flush=True)
        candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True, pyCandle.USB)
        ids = candle.ping(pyCandle.CAN_BAUD_1M)

        if not ids:
            print("  [HW] ERROR: No CAN devices found.", flush=True)
            return False

        print(f"  [HW] Found CAN IDs: {ids}", flush=True)

        for dev_id in ids:
            candle.addMd80(dev_id)

        shoulder = elbow = None
        for md in candle.md80s:
            if md.getId() == SHOULDER_ID:
                shoulder = md
            elif md.getId() == ELBOW_ID:
                elbow = md

        if shoulder is None or elbow is None:
            found = [md.getId() for md in candle.md80s]
            print(f"  [HW] ERROR: Expected IDs {SHOULDER_ID},{ELBOW_ID}, found {found}")
            return False

        for md in candle.md80s:
            candle.controlMd80Mode(md.getId(), pyCandle.IMPEDANCE)
            md.setImpedanceControllerParams(0, 0)   # zero stiffness/damping → pure torque
            md.setTargetTorque(0)
            candle.controlMd80Enable(md.getId(), True)

        candle.begin()
        print(f"  [HW] Motors enabled. Shoulder={SHOULDER_ID}, Elbow={ELBOW_ID}")

        self._candle   = candle
        self._shoulder = shoulder
        self._elbow    = elbow
        return True

    def read_state(self) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        q1    = self._shoulder.getPosition()
        q2    = self._elbow.getPosition()
        q1d   = self._shoulder.getVelocity()
        q2d   = self._elbow.getVelocity()
        x_hw  = np.array([q1, q2, q1d, q2d], dtype=np.float64)
        return x_hw, (time.perf_counter() - t0) * 1e3

    def write_torque(self, tau: np.ndarray) -> float:
        t0 = time.perf_counter()
        tau = np.clip(tau, -self.u_lim, self.u_lim)
        self._shoulder.setTargetTorque(float(tau[0]))
        self._elbow.setTargetTorque(float(tau[1]))
        return (time.perf_counter() - t0) * 1e3

    def disconnect(self) -> None:
        if self._shoulder is not None:
            try:
                self._shoulder.setTargetTorque(0.0)
                self._elbow.setTargetTorque(0.0)
                time.sleep(0.05)
                self._candle.end()
            except Exception as e:
                print(f"  [HW] Warning during disconnect: {e}")
        print("  [HW] Disconnected.")


# ── Architecture auto-detection ────────────────────────────────────────────────

_ARCH_MAP = {
    "separated": "SeparatedLinearizationNetwork",
    "base":      "LinearizationNetwork",
    "sc":        "LinearizationNetworkSC",
}


def detect_architecture(state_dict: dict, metadata: Optional[dict] = None) -> str:
    """Infer architecture class name from checkpoint content."""
    # 1. Explicit architecture tag in metadata
    if metadata:
        arch = metadata.get("architecture", "")
        if "separated_fnet_qnet" in arch:
            return "SeparatedLinearizationNetwork"
        if "sc" in arch.lower() or "sincos" in arch.lower():
            return "LinearizationNetworkSC"
        if "linear" in arch.lower():
            return "LinearizationNetwork"

    # 2. Key-name heuristic: SeparatedLinearizationNetwork has f_net.* + q_net.*
    keys = list(state_dict.keys())
    if any(k.startswith("f_net.") for k in keys):
        return "SeparatedLinearizationNetwork"

    # 3. Encoder input-size heuristic: SC uses 30-dim input (5×6), base uses 20-dim (5×4)
    enc_key = next((k for k in keys if "encoder" in k and "weight" in k and "0" in k), None)
    if enc_key:
        in_dim = state_dict[enc_key].shape[1]
        if in_dim == 30:
            return "LinearizationNetworkSC"

    return "LinearizationNetwork"


def infer_model_kwargs(state_dict: dict, metadata: Optional[dict] = None) -> dict:
    """Extract constructor kwargs from checkpoint."""
    if metadata:
        kw = {
            "state_dim":       metadata.get("state_dim",       4),
            "control_dim":     metadata.get("control_dim",     2),
            "horizon":         metadata.get("horizon",         10),
            "hidden_dim":      metadata.get("hidden_dim",      128),
            "gate_range_q":    metadata.get("gate_range_q",    0.99),
            "gate_range_r":    metadata.get("gate_range_r",    0.20),
            "f_extra_bound":   metadata.get("f_extra_bound",   1.5),
            "f_kickstart_amp": metadata.get("f_kickstart_amp", 0.01),
        }
        return kw

    # Fallback: infer hidden_dim from weight shapes
    hidden_dim = 128
    for k, v in state_dict.items():
        if "trunk" in k and "weight" in k and "0" in k:
            hidden_dim = v.shape[0]
            break

    return dict(
        state_dim=4, control_dim=2, horizon=10, hidden_dim=hidden_dim,
        gate_range_q=0.99, gate_range_r=0.20,
        f_extra_bound=1.5, f_kickstart_amp=0.01,
    )


def resolve_model_path(spec: str) -> str:
    """Resolve model spec string to an actual .pth path."""
    # Direct path
    if os.path.isfile(spec):
        return spec

    # Shorthands
    patterns: Dict[str, str] = {
        "latest_v1":    "saved_models/hw_v1*/*.pth",
        "latest_v2":    "saved_models/hw_v2_nr_*FINAL*/*.pth",
        "latest_v2diag":"saved_models/hw_v2_nr_diag*/*.pth",
        "latest_v3":    "saved_models/hw_v3_u010*/*.pth",
        "latest_v4":    "saved_models/hw_v4_u007*/*.pth",
        "latest_v5":    "saved_models/hw_v5_sa015*/*.pth",
        "latest_v6":    "saved_models/hw_v6_sa010*/*.pth",
    }
    pat = patterns.get(spec)
    if pat:
        paths = glob.glob(pat)
        if paths:
            return max(paths, key=os.path.getmtime)
        # fallback to diag
        if spec == "latest_v2":
            paths = glob.glob("saved_models/hw_v2_nr_diag*/*.pth")
            if paths:
                return max(paths, key=os.path.getmtime)

    # Glob pattern
    paths = glob.glob(spec)
    if paths:
        return max(paths, key=os.path.getmtime)

    raise FileNotFoundError(f"Cannot find model: {spec!r}")


def load_model(
    ckpt_path: str,
    arch_override: Optional[str] = None,
) -> Tuple[object, dict, float]:
    """Load checkpoint → (model, metadata, u_lim).

    Returns model in eval mode on CPU.
    """
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        state_dict = data.get("model_state_dict", data)
        metadata   = data.get("metadata", None)
        # training_params may be at top level OR nested inside metadata
        tp = data.get("training_params") or \
             (metadata.get("training_params") if isinstance(metadata, dict) else None) or {}
        u_lim = tp.get("u_lim", 0.15) if tp else 0.15
    else:
        state_dict = data
        metadata   = None
        u_lim      = 0.15

    # Architecture resolution
    if arch_override:
        arch_name = _ARCH_MAP.get(arch_override.lower(), arch_override)
    else:
        arch_name = detect_architecture(state_dict, metadata)

    model_kwargs = infer_model_kwargs(state_dict, metadata)

    arch_cls = getattr(network_module, arch_name, None)
    if arch_cls is None:
        raise ValueError(f"Unknown architecture: {arch_name!r}")

    model = arch_cls(**model_kwargs).double()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected}")

    model.eval()
    return model, {"arch": arch_name, "u_lim": u_lim, **model_kwargs}, u_lim


# ── Controller ─────────────────────────────────────────────────────────────────

class DeployController:
    """
    Minimal-latency MPC + optional EKF + optional TVLQR control loop.

    Pipeline per step
    -----------------
    1. x_hw (hardware ordering) → permute → x_ours
    2. EKF  predict + update  → x_est, bias_est (ours ordering)
    3. State history → lin_net → gates_Q, gates_R, f_extra
    4. MPC QP → u_mpc (ours ordering)
    5. [Optional] TVLQR → u_tvlqr (hw ordering) if near goal
    6. Bias cancellation → u_final
    7. [Single-actuated] override tau2 with PD stiffness
    8. Apply torque limit + send
    """

    def __init__(
        self,
        model,
        mpc,
        x_goal:          torch.Tensor,
        dt:              float                  = 0.05,
        ekf_mode:        str                    = "ekf6",   # none/ekf4/ekf6
        ekf_Q_state:     Optional[torch.Tensor] = None,
        ekf_Q_bias:      Optional[torch.Tensor] = None,
        ekf_R:           Optional[torch.Tensor] = None,
        cancel_bias:     bool                   = True,
        bias_warmup:     int                    = 20,
        tvlqr:           Optional[TVLQRController] = None,
        tvlqr_blend:     bool                   = False,  # True=blend, False=hard switch
        tvlqr_threshold: float                  = 0.4,    # err norm to activate TVLQR
        single_actuated: bool                   = False,
        sa_kp:           float                  = SA_KP,
        sa_kd:           float                  = SA_KD,
        monitor_level:   int                    = MONITOR_STANDARD,
        log_file:        Optional[str]          = None,
        print_every:     int                    = 50,
    ):
        self.model           = model
        self.mpc             = mpc
        self.x_goal          = x_goal.double()
        self.x_goal_hw       = x_goal.numpy()[OURS_TO_HW]
        self.dt              = dt
        self.dt_t            = torch.tensor(dt, dtype=torch.float64)
        self.cancel_bias     = cancel_bias
        self.bias_warmup     = bias_warmup
        self.tvlqr           = tvlqr
        self.tvlqr_blend     = tvlqr_blend
        self.tvlqr_threshold = tvlqr_threshold
        self.single_actuated = single_actuated
        self.sa_kp           = sa_kp
        self.sa_kd           = sa_kd

        n_u = mpc.MPC_dynamics.u_min.shape[0]
        self.n_u = n_u
        self.u_lim_t = mpc.MPC_dynamics.u_max   # (n_u,)

        self.state_history: deque = deque(
            [torch.zeros(4, dtype=torch.float64)] * 5, maxlen=5
        )
        self.u_seq_guess = torch.zeros((mpc.N, n_u), dtype=torch.float64)
        self.u_prev      = torch.zeros(n_u, dtype=torch.float64)

        # EKF
        Q_s = ekf_Q_state if ekf_Q_state is not None else DEFAULT_Q_STATE
        Q_b = ekf_Q_bias  if ekf_Q_bias  is not None else DEFAULT_Q_BIAS
        R   = ekf_R       if ekf_R       is not None else DEFAULT_R_OBS

        if ekf_mode == "ekf6":
            self.ekf: Optional[object] = EKF6(mpc, Q_s, Q_b, R)
        elif ekf_mode == "ekf4":
            self.ekf = EKF4(mpc, Q_s, R)
        else:
            self.ekf = None
        self.ekf_mode = ekf_mode

        self.diag = DiagnosticsLogger(
            level=monitor_level,
            print_every=print_every,
            log_file=log_file,
        )
        self.step_count = 0
        self._episode_t0: float = 0.0

    # ── episode control ───────────────────────────────────────────────────────

    def reset(self, x0_hw: np.ndarray):
        x0 = torch.tensor(x0_hw[HW_TO_OURS], dtype=torch.float64)
        self.state_history = deque([x0.clone() for _ in range(5)], maxlen=5)
        self.u_seq_guess   = torch.zeros((self.mpc.N, self.n_u), dtype=torch.float64)
        self.u_prev        = torch.zeros(self.n_u, dtype=torch.float64)
        if self.ekf is not None:
            self.ekf.reset(x0)
        if self.tvlqr is not None:
            self.tvlqr.reset()
        self.step_count  = 0
        self._episode_t0 = time.perf_counter()

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self, x_hw: np.ndarray, t_read_ms: float = 0.0) -> Tuple[np.ndarray, StepMetrics]:
        """
        One control step.

        Args:
            x_hw      : hardware state [q1, q2, q1_dot, q2_dot]
            t_read_ms : read latency already measured externally

        Returns:
            tau_hw    : [tau1, tau2] Nm in hardware ordering
            metrics   : StepMetrics
        """
        t0 = time.perf_counter()
        t_elapsed = t0 - self._episode_t0

        # Permute to our ordering
        y = torch.tensor(x_hw[HW_TO_OURS], dtype=torch.float64)

        # ── EKF ──────────────────────────────────────────────────────────────
        t_ekf0 = time.perf_counter()
        if self.ekf is not None and self.step_count > 0:
            if self.ekf_mode == "ekf6":
                x_est, bias_est = self.ekf.step(y, self.u_prev)
            else:
                x_est, bias_est = self.ekf.step(y, self.u_prev)
        else:
            x_est    = y.clone()
            bias_est = torch.zeros(self.n_u, dtype=torch.float64)
            if self.ekf is not None:
                self.ekf.reset(x_est)
        t_ekf_ms = (time.perf_counter() - t_ekf0) * 1e3

        ekf_innov = (y - x_est).numpy()
        self.state_history.append(x_est.clone())

        # ── Model inference ───────────────────────────────────────────────────
        t_mod0 = time.perf_counter()
        hist_t = torch.stack(list(self.state_history), dim=0)
        with torch.no_grad():
            gates_Q, gates_R, f_extra, _, _, gates_Qf = self.model(
                hist_t,
                q_base_diag=self.mpc.q_base_diag,
                r_base_diag=self.mpc.r_base_diag,
            )
        t_model_ms = (time.perf_counter() - t_mod0) * 1e3

        # ── MPC QP ───────────────────────────────────────────────────────────
        t_qp0 = time.perf_counter()
        u_seq_c = self.u_seq_guess.clamp(
            min=-self.u_lim_t.unsqueeze(0),
            max= self.u_lim_t.unsqueeze(0),
        )
        u_mpc, U_full = self.mpc.control(
            x_est,
            x_est.unsqueeze(0).expand(self.mpc.N, -1).clone(),
            u_seq_c,
            self.x_goal,
            diag_corrections_Q=gates_Q,
            diag_corrections_R=gates_R,
            extra_linear_control=f_extra.reshape(-1),
            diag_corrections_Qf=gates_Qf,
        )
        t_qp_ms = (time.perf_counter() - t_qp0) * 1e3
        qp_fallback = (self.mpc.qp_fallback_count > 0)
        self.mpc.qp_fallback_count = 0

        # Warm-start next QP
        U_r = U_full.detach().view(self.mpc.N, self.n_u)
        self.u_seq_guess[:-1] = U_r[1:].clone()
        self.u_seq_guess[-1]  = U_r[-1].clone()

        # ── TVLQR ────────────────────────────────────────────────────────────
        t_tv0 = time.perf_counter()
        tvlqr_active = False
        u_tvlqr_hw = np.zeros(2)

        if self.tvlqr is not None:
            near_goal = self.tvlqr.should_activate(x_hw, self.x_goal_hw) \
                        if self.tvlqr_threshold > 0 else True
            if near_goal:
                u_tvlqr_hw   = self.tvlqr.get_control(x_hw, t_elapsed)
                tvlqr_active = True
        t_tvlqr_ms = (time.perf_counter() - t_tv0) * 1e3

        # ── Bias cancellation + combine ───────────────────────────────────────
        use_bias = (self.cancel_bias and self.ekf is not None
                    and self.step_count >= self.bias_warmup
                    and self.ekf_mode == "ekf6")
        if use_bias:
            u_cmd = torch.clamp(
                u_mpc.detach() - bias_est.detach(),
                min=-self.u_lim_t, max=self.u_lim_t,
            )
        else:
            u_cmd = u_mpc.detach().clone()

        # Combine MPC and TVLQR
        # Torque is always [tau_shoulder, tau_elbow] in both orderings — no permutation needed.
        u_cmd_np = u_cmd.numpy()
        if tvlqr_active:
            if self.tvlqr_blend:
                # Blend: use both, weight TVLQR more when closer to goal
                dq1  = math.atan2(math.sin(x_hw[0] - self.x_goal_hw[0]),
                                  math.cos(x_hw[0] - self.x_goal_hw[0]))
                err  = math.sqrt(dq1**2 + x_hw[2]**2)
                alpha = max(0.0, 1.0 - err / self.tvlqr_threshold)
                u_final = (1.0 - alpha) * u_cmd_np + alpha * u_tvlqr_hw
            else:
                u_final = u_tvlqr_hw.copy()   # hard switch — TVLQR takes over
        else:
            u_final = u_cmd_np

        # Apply torque limit
        u_lim_np = self.u_lim_t.numpy()
        u_final  = np.clip(u_final, -u_lim_np, u_lim_np)

        # ── Single-actuated override ──────────────────────────────────────────
        if self.single_actuated:
            q2  = x_hw[1]    # elbow position (hw ordering)
            q2d = x_hw[3]    # elbow velocity
            tau2_pd = -self.sa_kp * q2 - self.sa_kd * q2d
            tau2_pd = float(np.clip(tau2_pd, -u_lim_np[1], u_lim_np[1]))
            u_final[1] = tau2_pd

        self.u_prev = torch.tensor(u_final, dtype=torch.float64)

        t_total_ms = (time.perf_counter() - t0) * 1e3

        # Error norm (hardware-angle-wrap-aware)
        dq1  = math.atan2(math.sin(x_hw[0] - self.x_goal_hw[0]),
                          math.cos(x_hw[0] - self.x_goal_hw[0]))
        dq2  = math.atan2(math.sin(x_hw[1] - self.x_goal_hw[1]),
                          math.cos(x_hw[1] - self.x_goal_hw[1]))
        err_norm = math.sqrt(dq1**2 + x_hw[2]**2 + dq2**2 + x_hw[3]**2)

        m = StepMetrics(
            step          = self.step_count,
            t_read_ms     = t_read_ms,
            t_ekf_ms      = t_ekf_ms,
            t_model_ms    = t_model_ms,
            t_qp_ms       = t_qp_ms,
            t_tvlqr_ms    = t_tvlqr_ms,
            t_total_ms    = t_total_ms,
            qp_fallback   = qp_fallback,
            tvlqr_active  = tvlqr_active,
            u_mpc         = u_cmd_np.copy(),
            u_tvlqr       = u_tvlqr_hw.copy(),
            u_applied     = u_final.copy(),
            bias_est      = bias_est.numpy().copy(),
            ekf_innov     = ekf_innov.copy(),
            x_raw         = x_hw.copy(),
            x_est         = x_est.numpy().copy(),
            err_norm      = err_norm,
        )
        self._last_metrics = m
        self.step_count += 1
        return u_final, m


# ── Safety monitor ─────────────────────────────────────────────────────────────

def safety_check(x_hw: np.ndarray) -> bool:
    """Return True if state is within safe bounds (no E-stop needed)."""
    # Angle components only for position safety; velocities separately
    if abs(x_hw[2]) > SAFETY_VEL_LIMIT or abs(x_hw[3]) > SAFETY_VEL_LIMIT:
        return False
    return True


# ── Main control loop ──────────────────────────────────────────────────────────

def run_loop(
    controller: DeployController,
    interface:  HardwareInterface,
    x0_hw:     np.ndarray,
    dt:        float = 0.05,
    n_steps:   int   = 2000,
) -> List[StepMetrics]:
    controller.reset(x0_hw)

    dt_target = dt
    all_metrics: List[StepMetrics] = []
    t_tick = time.perf_counter()

    print(f"\n  Control loop: {n_steps} steps @ {1/dt:.0f} Hz  (dt={dt*1e3:.1f}ms)")
    print(f"  Press Ctrl+C to stop safely.\n")

    try:
        for step in range(n_steps):
            t_loop_start = time.perf_counter()

            # ── Read ──────────────────────────────────────────────────────────
            x_hw, t_read_ms = interface.read_state()

            # ── Safety ────────────────────────────────────────────────────────
            if not safety_check(x_hw):
                print(f"\n  [SAFETY] Trip at step {step}: vel={x_hw[2]:.2f},{x_hw[3]:.2f} rad/s")
                interface.write_torque(np.zeros(2))
                time.sleep(RESET_DELAY)
                controller.reset(x_hw)
                t_tick = time.perf_counter()
                continue

            # ── Control step ──────────────────────────────────────────────────
            tau, m = controller.step(x_hw, t_read_ms=t_read_ms)

            # ── Write ─────────────────────────────────────────────────────────
            t_write_start = time.perf_counter()
            interface.write_torque(tau)
            m.t_write_ms = (time.perf_counter() - t_write_start) * 1e3

            # ── Timing ────────────────────────────────────────────────────────
            t_next = t_tick + (step + 1) * dt_target
            sleep_s = t_next - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

            t_elapsed_step = time.perf_counter() - (t_tick + step * dt_target)
            m.jitter_ms  = (t_elapsed_step - dt_target) * 1e3
            m.t_sleep_ms = max(0.0, sleep_s * 1e3)

            controller.diag.log(m)
            all_metrics.append(m)

    except KeyboardInterrupt:
        print("\n  [Ctrl+C] Stopping safely.")

    finally:
        interface.write_torque(np.zeros(2))
        interface.disconnect()
        controller.diag.final_summary()
        controller.diag.close()

    return all_metrics


# ── Benchmark mode ─────────────────────────────────────────────────────────────

def run_benchmark(
    model,
    mpc,
    x_goal:    torch.Tensor,
    ekf_mode:  str = "ekf6",
    tvlqr:     Optional[TVLQRController] = None,
    n_warmup:  int = 100,
    n_bench:   int = 500,
):
    print("\n" + "="*65)
    print("  LATENCY BENCHMARK (no hardware I/O)")
    print("="*65)

    ctrl = DeployController(
        model=model, mpc=mpc, x_goal=x_goal,
        ekf_mode=ekf_mode, tvlqr=tvlqr,
        monitor_level=MONITOR_NONE,
    )
    x0_hw = np.zeros(4)
    ctrl.reset(x0_hw)

    for _ in range(n_warmup):
        ctrl.step(x0_hw)

    qp_t, mod_t, ekf_t, tv_t, tot_t = [], [], [], [], []
    for _ in range(n_bench):
        _, m = ctrl.step(x0_hw)
        qp_t.append(m.t_qp_ms)
        mod_t.append(m.t_model_ms)
        ekf_t.append(m.t_ekf_ms)
        tv_t.append(m.t_tvlqr_ms)
        tot_t.append(m.t_total_ms)

    q = np.percentile
    print(f"\n  {n_bench} steps after {n_warmup} warmup:")
    print(f"  {'Metric':<22}  {'mean':>7}  {'p50':>7}  {'p95':>7}  {'p99':>7}  {'max':>7}")
    print(f"  {'-'*58}")
    for name, arr in [
        ("QP solve (ms)",   qp_t),
        ("Model infer (ms)", mod_t),
        ("EKF update (ms)",  ekf_t),
        ("TVLQR (ms)",       tv_t),
        ("Loop total (ms)",  tot_t),
    ]:
        a = np.array(arr)
        print(f"  {name:<22}  {a.mean():>7.2f}  {q(a,50):>7.2f}  "
              f"{q(a,95):>7.2f}  {q(a,99):>7.2f}  {a.max():>7.2f}")

    dt_budget = 1000.0 / 20.0
    remaining = dt_budget - np.mean(tot_t)
    print(f"\n  Target period @ 20 Hz: {dt_budget:.1f} ms")
    print(f"  I/O budget remaining:  {remaining:.1f} ms")
    print("="*65 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MAB double pendulum deployment pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "--mode", choices=["sim", "hw", "bench"], default="sim",
        help="sim  = simulation  |  hw = real hardware  |  bench = latency benchmark",
    )
    p.add_argument(
        "--model", type=str, default="latest_v1",
        help=(
            "Model spec. Options:\n"
            "  latest_v1 / latest_v2 / latest_v2diag / latest_v3 / latest_v4 / latest_v5 / latest_v6\n"
            "  Path to .pth file\n"
            "  Glob pattern  (e.g. 'saved_models/hw_v1*/*.pth')"
        ),
    )
    p.add_argument(
        "--arch", choices=["auto", "separated", "base", "sc"], default="auto",
        help="Network architecture override (default: auto-detect from checkpoint)",
    )
    p.add_argument(
        "--actuate", choices=["double", "single"], default="double",
        help="double = both joints from MPC  |  single = shoulder only, elbow PD",
    )
    p.add_argument(
        "--ekf", choices=["none", "ekf4", "ekf6"], default="ekf6",
        help="Filter: none=raw observations  ekf4=state filter  ekf6=state+bias filter",
    )
    p.add_argument(
        "--stabilizer", choices=["none", "tvlqr"], default="none",
        help="Stabilizer: none=MPC only  tvlqr=switch to TVLQR when near goal",
    )
    p.add_argument(
        "--tvlqr_policy", type=str, default="tvlqr_policy_1.npz",
        help="Path to TVLQR policy .npz file",
    )
    p.add_argument(
        "--tvlqr_threshold", type=float, default=0.4,
        help="State-error norm to activate TVLQR stabiliser (default 0.4 rad)",
    )
    p.add_argument(
        "--tvlqr_blend", action="store_true",
        help="Blend MPC + TVLQR near threshold instead of hard switching",
    )
    p.add_argument(
        "--monitor", choices=list(MONITOR_LEVELS.keys()), default="standard",
        help=(
            "Verbosity:\n"
            "  none       — silent\n"
            "  minimal    — step counter + control every 100 steps\n"
            "  standard   — 50-step summaries with timing and error\n"
            "  verbose    — per-step table (all metrics)\n"
            "  diagnostic — verbose + file logging + final summary"
        ),
    )
    p.add_argument(
        "--u_lim", type=float, default=None,
        help="Torque limit override (Nm). Default: read from checkpoint or 0.15",
    )
    p.add_argument(
        "--freq", type=float, default=20.0,
        help="Control frequency in Hz (default 20 Hz)",
    )
    p.add_argument(
        "--steps", type=int, default=2000,
        help="Number of control steps (default 2000 = 100 s @ 20 Hz)",
    )
    p.add_argument(
        "--obs_sigma", type=float, default=0.0,
        help="Observation noise std for sim mode (rad / rad·s⁻¹)",
    )
    p.add_argument(
        "--bias", type=float, default=0.0,
        help="Constant torque bias injected in sim mode (Nm, both joints)",
    )
    p.add_argument(
        "--cancel_bias", action="store_true", default=True,
        help="Apply EKF6 bias cancellation (default on when ekf=ekf6)",
    )
    p.add_argument(
        "--no_cancel_bias", dest="cancel_bias", action="store_false",
    )
    p.add_argument(
        "--log", type=str, default=None,
        help="Write full per-step table to this file (auto-set for diagnostic mode)",
    )
    p.add_argument(
        "--print_every", type=int, default=50,
        help="Console summary interval in steps (default 50)",
    )
    p.add_argument(
        "--sa_kp", type=float, default=SA_KP,
        help=f"Single-actuated joint-2 PD stiffness kp (default {SA_KP} Nm/rad)",
    )
    p.add_argument(
        "--sa_kd", type=float, default=SA_KD,
        help=f"Single-actuated joint-2 PD damping kd (default {SA_KD} Nm·s/rad)",
    )
    return p


def main():
    args = build_parser().parse_args()

    # ── Resolve model ──────────────────────────────────────────────────────────
    try:
        ckpt_path = resolve_model_path(args.model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    arch_override = None if args.arch == "auto" else args.arch
    model, ckpt_meta, u_lim_ckpt = load_model(ckpt_path, arch_override)

    u_lim = args.u_lim if args.u_lim is not None else u_lim_ckpt

    print(f"\n{'='*70}")
    print(f"  MAB Double Pendulum Deployment")
    print(f"{'='*70}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Architecture: {ckpt_meta['arch']}")
    print(f"  hidden_dim  : {ckpt_meta.get('hidden_dim', '?')}")
    print(f"  Torque limit: {u_lim} Nm")
    print(f"  Mode        : {args.mode}")
    print(f"  Actuated    : {args.actuate}")
    print(f"  EKF         : {args.ekf}")
    print(f"  Stabilizer  : {args.stabilizer}")
    print(f"  Monitor     : {args.monitor}")
    print(f"  Freq        : {args.freq} Hz")
    print(f"{'='*70}\n")

    dt     = 1.0 / args.freq
    device = torch.device("cpu")
    x0     = torch.zeros(4, dtype=torch.float64)
    x_goal = torch.tensor([math.pi, 0.0, 0.0, 0.0], dtype=torch.float64)

    mpc = mpc_module.MPC_controller(
        x0=x0, x_goal=x_goal, N=10, device=device, u_lim=u_lim
    )
    mpc.dt = torch.tensor(dt, dtype=torch.float64)

    # ── TVLQR ─────────────────────────────────────────────────────────────────
    tvlqr = None
    if args.stabilizer == "tvlqr":
        if not os.path.isfile(args.tvlqr_policy):
            print(f"  WARNING: TVLQR policy not found: {args.tvlqr_policy!r}")
            print(f"  Falling back to MPC-only mode.")
        else:
            tvlqr = TVLQRController(
                policy_path=args.tvlqr_policy,
                activate_err=args.tvlqr_threshold,
                u_lim=u_lim,
            )
            print(f"  TVLQR policy loaded: {args.tvlqr_policy}")
            print(f"  Activation threshold: err < {args.tvlqr_threshold:.2f} rad")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if args.mode == "bench":
        run_benchmark(model, mpc, x_goal, ekf_mode=args.ekf, tvlqr=tvlqr)
        return

    # ── Log file ──────────────────────────────────────────────────────────────
    log_file = args.log
    if log_file is None and args.monitor == "diagnostic":
        log_file = "/tmp/hw_deploy_diagnostic.log"
        print(f"  Diagnostic log → {log_file}")

    monitor_level = MONITOR_LEVELS[args.monitor]

    # ── Interface ─────────────────────────────────────────────────────────────
    if args.mode == "sim":
        ctrl_bias = np.array([args.bias, args.bias]) if args.bias != 0.0 else None
        interface = SimulationInterface(
            mpc=mpc, x0=x0, dt=dt,
            obs_sigma=args.obs_sigma, ctrl_bias=ctrl_bias,
        )
    else:
        interface = MABInterface(u_lim=u_lim)

    if not interface.connect():
        print("  ERROR: Failed to connect. Exiting.")
        sys.exit(1)

    # ── Controller ────────────────────────────────────────────────────────────
    controller = DeployController(
        model            = model,
        mpc              = mpc,
        x_goal           = x_goal,
        dt               = dt,
        ekf_mode         = args.ekf,
        cancel_bias      = args.cancel_bias,
        tvlqr            = tvlqr,
        tvlqr_blend      = args.tvlqr_blend,
        tvlqr_threshold  = args.tvlqr_threshold,
        single_actuated  = (args.actuate == "single"),
        sa_kp            = args.sa_kp,
        sa_kd            = args.sa_kd,
        monitor_level    = monitor_level,
        log_file         = log_file,
        print_every      = args.print_every,
    )

    x0_hw = np.zeros(4)
    run_loop(controller, interface, x0_hw, dt=dt, n_steps=args.steps)


if __name__ == "__main__":
    main()
