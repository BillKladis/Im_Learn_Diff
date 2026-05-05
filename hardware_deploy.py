"""hardware_deploy.py — Plug-and-play deployment pipeline for MAB double pendulum.

Minimal-latency control loop with EKF6 bias estimation and extensive diagnostics.

Usage:
    # Simulation test (no hardware required):
    python hardware_deploy.py --mode sim --model saved_models/hw_v1_*/hw_v1_*.pth

    # Real hardware (fill in MABInterface.connect):
    python hardware_deploy.py --mode hw --model saved_models/hw_v1_*/hw_v1_*.pth

    # Quick latency benchmark:
    python hardware_deploy.py --mode bench --model saved_models/hw_v1_*/hw_v1_*.pth

State ordering:
    Ours:     [q1, q1_dot, q2, q2_dot]
    Hardware: [q1, q2,     q1_dot, q2_dot]
    Permutation: x_ours = x_hw[[0, 2, 1, 3]]
"""

import argparse
import math
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, "/home/user/Im_Learn_Diff")
os.chdir("/home/user/Im_Learn_Diff")

import lin_net as network_module
import mpc_controller as mpc_module
from ekf import EKF6

# ── Hardware permutation ───────────────────────────────────────────────────
# x_ours = x_hw[[0, 2, 1, 3]]   (swap q2 and q1_dot positions)
HW_TO_OURS = [0, 2, 1, 3]
OURS_TO_HW = [0, 2, 1, 3]   # same permutation is its own inverse


# ── Timing / monitoring ────────────────────────────────────────────────────

@dataclass
class StepMetrics:
    step:         int
    t_read_ms:    float   # hardware read latency (ms)
    t_ekf_ms:     float   # EKF predict+update (ms)
    t_model_ms:   float   # lin_net forward pass (ms)
    t_qp_ms:      float   # QP solve (ms)
    t_write_ms:   float   # hardware write latency (ms)
    t_total_ms:   float   # full loop time (ms)
    t_sleep_ms:   float   # idle time remaining before next tick (ms)
    jitter_ms:    float   # deviation from target period (ms)
    qp_fallback:  bool    # True if QP solver fell back to zero
    ekf_innov:    np.ndarray = field(default_factory=lambda: np.zeros(4))
    bias_est:     np.ndarray = field(default_factory=lambda: np.zeros(2))
    u_applied:    np.ndarray = field(default_factory=lambda: np.zeros(2))
    x_est:        np.ndarray = field(default_factory=lambda: np.zeros(4))


class DiagnosticsLogger:
    """Ring-buffer metric logger with periodic console summary."""

    def __init__(self, window: int = 200, log_file: Optional[str] = None,
                 print_every: int = 50):
        self.window = window
        self.print_every = print_every
        self.buf: deque = deque(maxlen=window)
        self.log_file = open(log_file, "w", buffering=1) if log_file else None

        hdr = (f"{'step':>6}  {'t_rd':>7}  {'t_ekf':>7}  {'t_mod':>7}  "
               f"{'t_qp':>7}  {'t_wr':>7}  {'t_tot':>7}  {'jitter':>8}  "
               f"{'fb':>3}  {'u1':>7}  {'u2':>7}  {'bias1':>7}  {'bias2':>7}  "
               f"{'q1':>7}  {'q1d':>7}  {'q2':>7}  {'q2d':>7}")
        self._write(hdr)
        self._write("-" * len(hdr))

    def log(self, m: StepMetrics):
        self.buf.append(m)
        row = (f"{m.step:>6}  {m.t_read_ms:>7.2f}  {m.t_ekf_ms:>7.2f}  "
               f"{m.t_model_ms:>7.2f}  {m.t_qp_ms:>7.2f}  {m.t_write_ms:>7.2f}  "
               f"{m.t_total_ms:>7.2f}  {m.jitter_ms:>8.2f}  "
               f"{'Y' if m.qp_fallback else 'N':>3}  "
               f"{m.u_applied[0]:>7.4f}  {m.u_applied[1]:>7.4f}  "
               f"{m.bias_est[0]:>7.4f}  {m.bias_est[1]:>7.4f}  "
               f"{m.x_est[0]:>7.4f}  {m.x_est[1]:>7.4f}  "
               f"{m.x_est[2]:>7.4f}  {m.x_est[3]:>7.4f}")
        self._write(row)

        if m.step > 0 and m.step % self.print_every == 0:
            self._print_summary()

    def _print_summary(self):
        if not self.buf:
            return
        ms = list(self.buf)
        totals  = np.array([m.t_total_ms for m in ms])
        qps     = np.array([m.t_qp_ms    for m in ms])
        jitters = np.array([abs(m.jitter_ms) for m in ms])
        fb_rate = sum(1 for m in ms if m.qp_fallback) / len(ms) * 100
        print(f"\n  [DIAG step={ms[-1].step}]"
              f"  loop: mean={totals.mean():.1f}ms  p99={np.percentile(totals,99):.1f}ms  max={totals.max():.1f}ms"
              f"  |  QP: mean={qps.mean():.1f}ms  p99={np.percentile(qps,99):.1f}ms"
              f"  |  jitter_abs: mean={jitters.mean():.1f}ms  max={jitters.max():.1f}ms"
              f"  |  QP fallbacks: {fb_rate:.1f}%", flush=True)

    def _write(self, line: str):
        print(line, flush=True)
        if self.log_file:
            self.log_file.write(line + "\n")

    def close(self):
        if self.log_file:
            self.log_file.close()

    def final_summary(self):
        if not self.buf:
            return
        ms = list(self.buf)
        totals  = np.array([m.t_total_ms for m in ms])
        qps     = np.array([m.t_qp_ms    for m in ms])
        ekfs    = np.array([m.t_ekf_ms   for m in ms])
        models  = np.array([m.t_model_ms for m in ms])
        jitters = np.array([abs(m.jitter_ms) for m in ms])
        fb_rate = sum(1 for m in ms if m.qp_fallback) / len(ms) * 100

        print("\n" + "="*70)
        print("  FINAL DIAGNOSTICS SUMMARY")
        print("="*70)
        print(f"  Loop total:  mean={totals.mean():.2f}ms  p50={np.percentile(totals,50):.2f}ms"
              f"  p95={np.percentile(totals,95):.2f}ms  p99={np.percentile(totals,99):.2f}ms"
              f"  max={totals.max():.2f}ms")
        print(f"  QP solve:    mean={qps.mean():.2f}ms  p95={np.percentile(qps,95):.2f}ms"
              f"  max={qps.max():.2f}ms")
        print(f"  EKF update:  mean={ekfs.mean():.2f}ms  max={ekfs.max():.2f}ms")
        print(f"  Model infer: mean={models.mean():.2f}ms  max={models.max():.2f}ms")
        print(f"  Jitter |abs|:mean={jitters.mean():.2f}ms  p99={np.percentile(jitters,99):.2f}ms"
              f"  max={jitters.max():.2f}ms")
        print(f"  QP fallback rate: {fb_rate:.2f}%")


# ── Hardware interface ─────────────────────────────────────────────────────

class HardwareInterface(ABC):
    """Abstract base — implement read_state/write_torque for your driver."""

    @abstractmethod
    def connect(self) -> bool:
        """Open connection to hardware. Returns True on success."""

    @abstractmethod
    def read_state(self) -> np.ndarray:
        """Return hardware state [q1, q2, q1_dot, q2_dot] (hardware ordering)."""

    @abstractmethod
    def write_torque(self, tau: np.ndarray) -> None:
        """Send torque [tau1, tau2] in Nm (hardware joint ordering)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Safely close connection (zero torque before disconnecting)."""


class SimulationInterface(HardwareInterface):
    """Software-in-the-loop: true dynamics, optional disturbances."""

    def __init__(self, mpc, x0: torch.Tensor, dt: float = 0.05,
                 obs_sigma: float = 0.0, ctrl_bias: Optional[np.ndarray] = None):
        self.mpc = mpc
        self.x   = x0.clone().double()
        self.dt  = torch.tensor(dt, dtype=torch.float64)
        self.obs_sigma  = obs_sigma
        self.ctrl_bias  = torch.tensor(ctrl_bias, dtype=torch.float64) \
                          if ctrl_bias is not None else None
        self._u_last = torch.zeros(2, dtype=torch.float64)

    def connect(self) -> bool:
        return True

    def read_state(self) -> np.ndarray:
        # Step physics with last torque
        self.x = self.mpc.true_RK4_disc(self.x, self._u_last, self.dt).detach()
        # Return in hardware ordering [q1, q2, q1_dot, q2_dot]
        x_np = self.x.numpy()[[0, 2, 1, 3]]
        if self.obs_sigma > 0:
            x_np = x_np + np.random.randn(4) * self.obs_sigma
        return x_np

    def write_torque(self, tau: np.ndarray) -> None:
        u = torch.tensor(tau, dtype=torch.float64)
        if self.ctrl_bias is not None:
            u = u + self.ctrl_bias
        u_lim = float(self.mpc.MPC_dynamics.u_max[0])
        self._u_last = u.clamp(-u_lim, u_lim)

    def disconnect(self) -> None:
        self._u_last = torch.zeros(2, dtype=torch.float64)

    @property
    def true_state(self) -> np.ndarray:
        return self.x.numpy()


class MABInterface(HardwareInterface):
    """
    Stub for MAB Robotics hardware.

    Fill in connect / read_state / write_torque to match your driver.

    Typical MAB interface options:
      A) ZMQ socket (common for MAB real-time bridge):
            import zmq
            ctx = zmq.Context()
            self.sock = ctx.socket(zmq.REQ)
            self.sock.connect("tcp://localhost:5555")

      B) Serial / CAN (direct motor controller):
            import serial
            self.ser = serial.Serial('/dev/ttyUSB0', 1000000)

      C) ROS topic:
            import rospy
            from sensor_msgs.msg import JointState
            rospy.Subscriber('/joint_states', JointState, self._state_cb)
            self.pub = rospy.Publisher('/joint_torques', ...)

    State returned by read_state must be in hardware ordering:
        [q1, q2, q1_dot, q2_dot]
    Angles in radians, velocities in rad/s, torque in Nm.
    """

    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host = host
        self.port = port
        self._conn = None

    def connect(self) -> bool:
        # TODO: implement hardware connection
        raise NotImplementedError(
            "Fill in MABInterface.connect() with your driver code.\n"
            "Use --mode sim for simulation testing."
        )

    def read_state(self) -> np.ndarray:
        # TODO: read [q1, q2, q1_dot, q2_dot] from hardware
        raise NotImplementedError

    def write_torque(self, tau: np.ndarray) -> None:
        # TODO: send [tau1, tau2] Nm to hardware
        raise NotImplementedError

    def disconnect(self) -> None:
        # TODO: zero torque and close connection
        raise NotImplementedError


# ── Controller ────────────────────────────────────────────────────────────

class DeployController:
    """
    Minimal-latency MPC + EKF control loop.

    Architecture:
      1. read_state  → hardware [q1, q2, q1_dot, q2_dot]
      2. permute     → our ordering [q1, q1_dot, q2, q2_dot]
      3. ekf.step    → filtered state + bias estimate
      4. lin_net     → Q/R gate corrections + f_extra
      5. mpc.control → QP solve → u_opt
      6. bias cancel → u_apply = clamp(u_opt - bias_est)
      7. write       → hardware
    """

    def __init__(
        self,
        model:      "SeparatedLinearizationNetwork",
        mpc:        "MPC_controller",
        x_goal:     torch.Tensor,
        dt:         float        = 0.05,
        use_ekf:    bool         = True,
        ekf_Q_state: Optional[torch.Tensor] = None,
        ekf_Q_bias:  Optional[torch.Tensor] = None,
        ekf_R:       Optional[torch.Tensor] = None,
        cancel_bias: bool        = True,
        bias_cancel_warmup: int  = 20,
        log_file:   Optional[str] = None,
        print_every: int         = 50,
    ):
        self.model   = model
        self.mpc     = mpc
        self.x_goal  = x_goal.double()
        self.dt      = dt
        self.dt_t    = torch.tensor(dt, dtype=torch.float64)
        self.cancel_bias = cancel_bias
        self.bias_cancel_warmup = bias_cancel_warmup

        n_u = mpc.MPC_dynamics.u_min.shape[0]
        self.n_u = n_u
        self.u_seq_guess = torch.zeros((mpc.N, n_u), dtype=torch.float64)
        self.state_history: deque = deque(
            [torch.zeros(4, dtype=torch.float64)] * 5, maxlen=5
        )
        self.u_prev = torch.zeros(n_u, dtype=torch.float64)

        # EKF setup
        if use_ekf:
            Q_s = ekf_Q_state if ekf_Q_state is not None else \
                  torch.diag(torch.tensor([1e-6, 1e-4, 1e-6, 1e-4], dtype=torch.float64))
            Q_b = ekf_Q_bias if ekf_Q_bias is not None else \
                  torch.eye(2, dtype=torch.float64) * 1e-3
            R   = ekf_R if ekf_R is not None else \
                  torch.diag(torch.tensor([4e-6, 1e-3, 4e-6, 1e-3], dtype=torch.float64))
            self.ekf: Optional[EKF6] = EKF6(mpc, Q_s, Q_b, R)
        else:
            self.ekf = None

        self.diag = DiagnosticsLogger(log_file=log_file, print_every=print_every)
        self.step_count = 0

    def reset(self, x0: torch.Tensor):
        """Call before starting a new episode."""
        x0 = x0.double()
        self.state_history = deque(
            [x0.clone() for _ in range(5)], maxlen=5
        )
        self.u_seq_guess = torch.zeros((self.mpc.N, self.n_u), dtype=torch.float64)
        self.u_prev = torch.zeros(self.n_u, dtype=torch.float64)
        if self.ekf is not None:
            self.ekf.reset(x0)
        self.step_count = 0

    def step(self, x_hw: np.ndarray) -> np.ndarray:
        """
        Run one control step given hardware state reading.

        Args:
            x_hw: [q1, q2, q1_dot, q2_dot] — hardware ordering

        Returns:
            tau: [tau1, tau2] Nm to send to hardware
        """
        t_step_start = time.perf_counter()

        # Permute to our ordering
        y = torch.tensor(x_hw[HW_TO_OURS], dtype=torch.float64)

        # ── EKF ────────────────────────────────────────────────────────
        t_ekf_start = time.perf_counter()
        if self.ekf is not None and self.step_count > 0:
            x_est, bias_est = self.ekf.step(y, self.u_prev)
        else:
            x_est = y.clone()
            bias_est = torch.zeros(self.n_u, dtype=torch.float64)
            if self.ekf is not None:
                self.ekf.reset(x_est)
        t_ekf_ms = (time.perf_counter() - t_ekf_start) * 1e3

        ekf_innov = (y - x_est).numpy()

        # Update state history with filtered state
        self.state_history.append(x_est.clone())

        # ── Model inference ─────────────────────────────────────────────
        t_model_start = time.perf_counter()
        hist_tensor = torch.stack(list(self.state_history), dim=0)
        with torch.no_grad():
            gates_Q, gates_R, f_extra, _, _, gates_Qf = self.model(
                hist_tensor,
                q_base_diag=self.mpc.q_base_diag,
                r_base_diag=self.mpc.r_base_diag,
            )
        t_model_ms = (time.perf_counter() - t_model_start) * 1e3

        # ── MPC QP ──────────────────────────────────────────────────────
        t_qp_start = time.perf_counter()
        u_lim_t = self.mpc.MPC_dynamics.u_max
        u_lin_seq = self.u_seq_guess.clamp(
            min=-u_lim_t.unsqueeze(0),
            max= u_lim_t.unsqueeze(0),
        )
        u_opt, U_opt_full = self.mpc.control(
            x_est,
            x_est.unsqueeze(0).expand(self.mpc.N, -1).clone(),
            u_lin_seq,
            self.x_goal,
            diag_corrections_Q=gates_Q,
            diag_corrections_R=gates_R,
            extra_linear_control=f_extra.reshape(-1),
            diag_corrections_Qf=gates_Qf,
        )
        t_qp_ms = (time.perf_counter() - t_qp_start) * 1e3

        qp_fallback = (self.mpc.qp_fallback_count > 0)
        self.mpc.qp_fallback_count = 0   # reset per-step counter

        # ── Bias cancellation ────────────────────────────────────────────
        if self.cancel_bias and self.ekf is not None and \
                self.step_count >= self.bias_cancel_warmup:
            u_cmd = torch.clamp(u_opt.detach() - bias_est.detach(),
                                min=-u_lim_t, max=u_lim_t)
        else:
            u_cmd = u_opt.detach().clone()

        self.u_prev = u_cmd.clone()

        # Warm-start next QP
        U_reshaped = U_opt_full.detach().view(self.mpc.N, self.n_u)
        self.u_seq_guess[:-1] = U_reshaped[1:].clone()
        self.u_seq_guess[-1]  = U_reshaped[-1].clone()

        t_total_ms = (time.perf_counter() - t_step_start) * 1e3

        # Placeholder read/write times — actual values filled by run_loop
        m = StepMetrics(
            step=self.step_count,
            t_read_ms=0.0,
            t_ekf_ms=t_ekf_ms,
            t_model_ms=t_model_ms,
            t_qp_ms=t_qp_ms,
            t_write_ms=0.0,
            t_total_ms=t_total_ms,
            t_sleep_ms=0.0,
            jitter_ms=0.0,
            qp_fallback=qp_fallback,
            ekf_innov=ekf_innov,
            bias_est=bias_est.numpy(),
            u_applied=u_cmd.numpy(),
            x_est=x_est.numpy(),
        )
        self._last_metrics = m
        self.step_count += 1

        return u_cmd.numpy()

    @property
    def last_metrics(self) -> StepMetrics:
        return self._last_metrics


def run_loop(
    controller: DeployController,
    interface:  HardwareInterface,
    x0_hw:     np.ndarray,
    dt:        float = 0.05,
    n_steps:   int   = 2000,
    log_file:  Optional[str] = None,
) -> list:
    """
    Main real-time control loop.

    Reads hardware state, runs one controller step, writes torque,
    then sleeps until the next control tick. All timing is measured
    and logged.

    Returns list of StepMetrics for offline analysis.
    """
    x0_ours = torch.tensor(x0_hw[HW_TO_OURS], dtype=torch.float64)
    controller.reset(x0_ours)

    dt_target_ns = int(dt * 1e9)
    all_metrics = []
    t_tick = time.perf_counter()

    print(f"\n  Starting control loop: {n_steps} steps @ {1/dt:.0f} Hz  (dt={dt*1e3:.1f}ms)")
    print(f"  Press Ctrl+C to stop safely.\n")

    try:
        for step in range(n_steps):
            t_loop_start = time.perf_counter()

            # ── Read ──────────────────────────────────────────────────
            t_read_start = time.perf_counter()
            x_hw = interface.read_state()
            t_read_ms = (time.perf_counter() - t_read_start) * 1e3

            # ── Control step ──────────────────────────────────────────
            tau = controller.step(x_hw)

            # ── Write ─────────────────────────────────────────────────
            t_write_start = time.perf_counter()
            interface.write_torque(tau)
            t_write_ms = (time.perf_counter() - t_write_start) * 1e3

            # ── Timing ────────────────────────────────────────────────
            t_loop_ms = (time.perf_counter() - t_loop_start) * 1e3
            t_next_tick = t_tick + (step + 1) * dt
            sleep_s = t_next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_actual_ms = (time.perf_counter() - t_tick - step * dt) * 1e3
            jitter_ms = t_actual_ms - dt * 1e3

            # Fill in loop-level timing
            m = controller.last_metrics
            m.t_read_ms  = t_read_ms
            m.t_write_ms = t_write_ms
            m.t_total_ms = t_loop_ms
            m.t_sleep_ms = max(0.0, sleep_s * 1e3)
            m.jitter_ms  = jitter_ms

            controller.diag.log(m)
            all_metrics.append(m)

    except KeyboardInterrupt:
        print("\n  [Ctrl+C] stopping loop safely.")
    finally:
        interface.write_torque(np.zeros(2))
        interface.disconnect()
        controller.diag.final_summary()

    return all_metrics


# ── Benchmark mode ─────────────────────────────────────────────────────────

def run_benchmark(model, mpc, x_goal, n_warmup=50, n_bench=500):
    """Measure model inference + QP solve time without hardware I/O overhead."""
    print("\n" + "="*60)
    print("  LATENCY BENCHMARK (no hardware I/O)")
    print("="*60)

    ctrl = DeployController(model=model, mpc=mpc, x_goal=x_goal,
                            use_ekf=True, log_file=None, print_every=10000)
    x0 = torch.zeros(4, dtype=torch.float64)
    ctrl.reset(x0)

    x_dummy_hw = np.zeros(4)

    # Warmup
    for _ in range(n_warmup):
        ctrl.step(x_dummy_hw)

    # Benchmark
    qp_times, model_times, ekf_times, total_times = [], [], [], []
    for _ in range(n_bench):
        ctrl.step(x_dummy_hw)
        m = ctrl.last_metrics
        qp_times.append(m.t_qp_ms)
        model_times.append(m.t_model_ms)
        ekf_times.append(m.t_ekf_ms)
        total_times.append(m.t_total_ms)

    q  = lambda a, p: np.percentile(a, p)
    print(f"\n  Over {n_bench} steps (after {n_warmup} warmup):")
    print(f"  {'Metric':<20}  {'mean':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'max':>8}")
    print(f"  {'-'*68}")
    for name, arr in [("QP solve (ms)", qp_times), ("Model infer (ms)", model_times),
                       ("EKF update (ms)", ekf_times), ("Loop total (ms)", total_times)]:
        a = np.array(arr)
        print(f"  {name:<20}  {a.mean():>8.2f}  {q(a,50):>8.2f}  {q(a,95):>8.2f}"
              f"  {q(a,99):>8.2f}  {a.max():>8.2f}")
    print(f"\n  Target loop: {1000/50:.1f} ms @ 20 Hz")
    budget_remaining = 50.0 - np.mean(total_times)
    print(f"  Remaining budget for I/O: {budget_remaining:.1f} ms")
    print("="*60 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str) -> "SeparatedLinearizationNetwork":
    MODEL_KWARGS = dict(
        state_dim=4, control_dim=2, horizon=10, hidden_dim=128,
        gate_range_q=0.99, gate_range_r=0.20,
        f_extra_bound=1.5, f_kickstart_amp=0.01,
    )
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = data.get("model_state_dict", data)

    # Infer u_lim from checkpoint metadata if available
    u_lim = 0.15
    if isinstance(data, dict) and "training_params" in data:
        u_lim = data["training_params"].get("u_lim", 0.15)

    model = network_module.SeparatedLinearizationNetwork(**MODEL_KWARGS).double()
    model.load_state_dict(state_dict)
    model.eval()
    return model, u_lim


def main():
    parser = argparse.ArgumentParser(description="MAB double pendulum deployment")
    parser.add_argument("--mode",  choices=["sim", "hw", "bench"], default="sim")
    parser.add_argument("--model", type=str, required=False,
                        help="Path to .pth checkpoint (default: latest hw_v1)")
    parser.add_argument("--u_lim", type=float, default=None,
                        help="Torque limit override (Nm). Default: from checkpoint or 0.15")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--obs_sigma", type=float, default=0.0,
                        help="Observation noise (sim mode only)")
    parser.add_argument("--bias", type=float, default=0.0,
                        help="Constant torque bias (sim mode only, Nm)")
    parser.add_argument("--no_ekf", action="store_true")
    parser.add_argument("--log", type=str, default="/tmp/hw_deploy.log")
    args = parser.parse_args()

    # Find checkpoint
    if args.model:
        ckpt = args.model
    else:
        import glob
        paths = glob.glob("saved_models/hw_v1*/*.pth")
        if not paths:
            print("No hw_v1 checkpoint found. Specify --model path.")
            return
        ckpt = max(paths, key=os.path.getmtime)
    print(f"  Loading model: {ckpt}")

    model, u_lim_ckpt = load_model(ckpt)
    u_lim = args.u_lim if args.u_lim is not None else u_lim_ckpt
    print(f"  Torque limit: {u_lim} Nm")

    x0     = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    x_goal = torch.tensor([math.pi, 0.0, 0.0, 0.0], dtype=torch.float64)
    mpc    = mpc_module.MPC_controller(x0=x0, x_goal=x_goal, N=10,
                                        device=torch.device("cpu"), u_lim=u_lim)

    if args.mode == "bench":
        run_benchmark(model, mpc, x_goal)
        return

    ctrl_bias = np.array([args.bias, args.bias]) if args.bias != 0 else None

    if args.mode == "sim":
        interface = SimulationInterface(
            mpc=mpc, x0=x0,
            obs_sigma=args.obs_sigma,
            ctrl_bias=ctrl_bias,
        )
    else:
        interface = MABInterface()

    ok = interface.connect()
    if not ok:
        print("Failed to connect to hardware.")
        return

    controller = DeployController(
        model=model, mpc=mpc, x_goal=x_goal,
        use_ekf=not args.no_ekf,
        cancel_bias=True,
        log_file=args.log,
        print_every=50,
    )

    x0_hw = np.zeros(4)  # start from rest in hardware ordering
    run_loop(controller, interface, x0_hw, n_steps=args.steps, log_file=args.log)


if __name__ == "__main__":
    main()
