"""
lin_net.py — Stage D LinearizationNetwork with kickstart bias on f_head.

WHAT IS NEW IN THIS REVISION:
  - The final-layer bias of f_head is initialised to a half-sinusoid pattern
    in the τ1 components, zero in the τ2 components.  This puts the network
    output at initialisation roughly at:
        f_extra[k, 0] ≈ kickstart_amp × sin(π (k + 0.5) / N)   for k=0..N-1
        f_extra[k, 1] ≈ 0
    so the very first rollout produces non-zero alternating-magnitude τ1
    rather than near-zero output.
  - The bias is LEARNABLE — the network will modify or discard it during
    training.  We are only providing a starting point that breaks the
    symmetry of the constant-direction failure mode.

The kickstart amplitude is configurable via f_kickstart_amp (default 1.0,
which after tanh×bound gives ~32% of full f_extra range — strong enough
to create real motion, weak enough to keep the QP solution in the interior
rather than saturating immediately).

Architecture otherwise unchanged from the previous Stage D version:
    state_history → state_encoder → trunk → {q_head, r_head, f_head}
"""

import math
import os
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class LinearizationNetwork(nn.Module):
    def __init__(
        self,
        state_dim:        int,
        control_dim:      int,
        horizon:          int,
        hidden_dim:       int   = 128,
        gate_range_q:     float = 0.95,
        gate_range_r:     float = 0.20,
        f_extra_bound:    float = 3.0,
        f_kickstart_amp:  float = 1.0,
    ):
        super().__init__()
        self.state_dim   = state_dim
        self.control_dim = control_dim
        self.hidden_dim  = hidden_dim
        self.horizon     = horizon

        for name, value in (
            ("gate_range_q", gate_range_q),
            ("gate_range_r", gate_range_r),
        ):
            if not (0.0 < value < 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {value}")
        if f_extra_bound <= 0:
            raise ValueError(f"f_extra_bound must be positive, got {f_extra_bound}")
        if f_kickstart_amp < 0:
            raise ValueError(f"f_kickstart_amp must be ≥ 0, got {f_kickstart_amp}")

        self.gate_range_q   = gate_range_q
        self.gate_range_r   = gate_range_r
        self.f_extra_bound  = f_extra_bound
        self.f_kickstart_amp = f_kickstart_amp

        # ── Input branch ────────────────────────────────────────────────────
        state_input_dim = 5 * state_dim
        enc_dim         = hidden_dim

        state_scale = torch.tensor(
            [math.pi, 8.0, math.pi, 8.0], dtype=torch.float64
        ).repeat(5)
        self.register_buffer("state_scale", state_scale)

        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_dim, enc_dim),
            nn.Tanh(),
        )

        # ── Trunk ───────────────────────────────────────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )

        # ── Heads ───────────────────────────────────────────────────────────
        q_out_dim = (horizon - 1) * state_dim
        r_out_dim = horizon * control_dim
        f_out_dim = horizon * control_dim

        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, q_out_dim),
        )
        self.r_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, r_out_dim),
        )
        self.f_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, f_out_dim),
        )

        self._initialize_weights()

        self.metadata = {
            "state_dim":       state_dim,
            "control_dim":     control_dim,
            "hidden_dim":      hidden_dim,
            "horizon":         horizon,
            "gate_range_q":    gate_range_q,
            "gate_range_r":    gate_range_r,
            "f_extra_bound":   f_extra_bound,
            "f_kickstart_amp": f_kickstart_amp,
            "architecture":    "stage_d_three_head_qrf_kickstart",
            "created_date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ──────────────────────────────────────────────────────────────────────
    def _initialize_weights(self):
        def _init_block(seq: nn.Sequential, final_std: float = 0.01):
            layers = [m for m in seq.modules() if isinstance(m, nn.Linear)]
            for layer in layers[:-1]:
                nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                nn.init.zeros_(layer.bias)
            final = layers[-1]
            nn.init.normal_(final.weight, mean=0.0, std=final_std)
            nn.init.zeros_(final.bias)

        _init_block(self.state_encoder, final_std=0.1)
        _init_block(self.trunk,         final_std=0.1)
        _init_block(self.q_head,        final_std=0.01)
        _init_block(self.r_head,        final_std=0.01)
        # f_head: small final std on weights so output ≈ bias at initialisation,
        # then the bias kickstart below dictates the starting f_extra.
        _init_block(self.f_head,        final_std=0.001)

        # ── Kickstart bias on f_head's final layer ──────────────────────
        # Half-sinusoid across the horizon in τ1; zero in τ2.
        # After tanh × bound, the output magnitude at step k is:
        #    bound × tanh(amp × sin(π(k+0.5)/N))
        # which for amp=1.0, bound=3.0 gives a smoothly-alternating
        # peak ≈ ±0.95 across the horizon.
        if self.f_kickstart_amp > 0.0:
            f_final = list(self.f_head.modules())[-1]   # last Linear
            n_u = self.control_dim
            init_bias = torch.zeros(self.horizon * n_u, dtype=f_final.bias.dtype)
            for k in range(self.horizon):
                phase = math.pi * (k + 0.5) / self.horizon  # in (0, π)
                init_bias[k * n_u + 0] = self.f_kickstart_amp * math.sin(phase)
                # τ2 stays at zero
            with torch.no_grad():
                f_final.bias.copy_(init_bias)

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x_sequence:  torch.Tensor,
        q_base_diag: Optional[torch.Tensor] = None,
        r_base_diag: Optional[torch.Tensor] = None,
    ):
        x_flat   = x_sequence.reshape(-1)
        x_normed = x_flat / self.state_scale
        state_emb = self.state_encoder(x_normed)
        features  = self.trunk(state_emb)

        raw_Q   = self.q_head(features).reshape(self.horizon - 1, self.state_dim)
        gates_Q = 1.0 + self.gate_range_q * torch.tanh(raw_Q)

        raw_R   = self.r_head(features).reshape(self.horizon, self.control_dim)
        gates_R = 1.0 + self.gate_range_r * torch.tanh(raw_R)

        raw_F   = self.f_head(features).reshape(self.horizon, self.control_dim)
        f_extra = self.f_extra_bound * torch.tanh(raw_F)

        q_diags = (q_base_diag.unsqueeze(0) * gates_Q) if q_base_diag is not None else None
        r_diags = (r_base_diag.unsqueeze(0) * gates_R) if r_base_diag is not None else None

        return gates_Q, gates_R, f_extra, q_diags, r_diags

    # ──────────────────────────────────────────────────────────────────────
    def save(self, filepath: str, metadata: Optional[Dict] = None):
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if metadata:
            self.metadata.update(metadata)
        torch.save(
            {"model_state_dict": self.state_dict(), "metadata": self.metadata},
            filepath,
        )

    @classmethod
    def load(cls, filepath: str, device: str = "cpu"):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        metadata   = checkpoint.get("metadata", {})
        model = cls(
            state_dim       = metadata.get("state_dim",       4),
            control_dim     = metadata.get("control_dim",     2),
            hidden_dim      = metadata.get("hidden_dim",      128),
            horizon         = metadata.get("horizon",         10),
            gate_range_q    = metadata.get("gate_range_q",    0.95),
            gate_range_r    = metadata.get("gate_range_r",    0.20),
            f_extra_bound   = metadata.get("f_extra_bound",   3.0),
            f_kickstart_amp = metadata.get("f_kickstart_amp", 0.0),  # 0 for back-compat
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).double()
        model.metadata = metadata
        return model


# ──────────────────────────────────────────────────────────────────────────
# Recorder & ModelManager (unchanged from previous Stage D version)
# ──────────────────────────────────────────────────────────────────────────
class NetworkOutputRecorder:
    def __init__(self):
        self.epochs: List[Dict] = []
        self._current_epoch_steps: List[Dict] = []

    def start_epoch(self):
        self._current_epoch_steps = []

    def record_step(
        self,
        gates_Q:   torch.Tensor,
        gates_R:   torch.Tensor,
        f_extra:   torch.Tensor,
        q_diags:   Optional[torch.Tensor],
        r_diags:   Optional[torch.Tensor],
        u_mpc:     torch.Tensor,
        state_err: torch.Tensor,
    ):
        with torch.no_grad():
            self._current_epoch_steps.append({
                "gates_Q":   gates_Q.detach().cpu().tolist(),
                "gates_R":   gates_R.detach().cpu().tolist(),
                "f_extra":   f_extra.detach().cpu().tolist(),
                "q_diags":   q_diags.detach().cpu().tolist() if q_diags is not None else None,
                "r_diags":   r_diags.detach().cpu().tolist() if r_diags is not None else None,
                "u_mpc":     u_mpc.detach().cpu().tolist(),
                "state_err": state_err.item() if isinstance(state_err, torch.Tensor) else state_err,
            })

    def end_epoch(self, epoch_loss: float):
        self.epochs.append({
            "steps":      self._current_epoch_steps,
            "epoch_loss": epoch_loss,
        })
        self._current_epoch_steps = []

    def epoch_summary(self, epoch_idx: int) -> Dict:
        epoch = self.epochs[epoch_idx]
        steps = epoch["steps"]
        if not steps:
            return {"epoch_loss": epoch["epoch_loss"], "num_steps": 0}

        errs    = [s["state_err"] for s in steps]
        u_norms = [float(torch.tensor(s["u_mpc"]).norm()) for s in steps]
        q_devs  = [float((torch.tensor(s["gates_Q"]) - 1.0).abs().mean()) for s in steps]
        r_devs  = [float((torch.tensor(s["gates_R"]) - 1.0).abs().mean()) for s in steps]
        f_norms = [float(torch.tensor(s["f_extra"]).norm()) for s in steps]
        f_tau1_first = [float(torch.tensor(s["f_extra"])[0, 0]) for s in steps]

        return {
            "epoch_loss":         epoch["epoch_loss"],
            "num_steps":          len(steps),
            "mean_state_err":     float(sum(errs) / len(errs)),
            "mean_u_norm":        float(sum(u_norms) / len(u_norms)),
            "mean_Q_gate_dev":    float(sum(q_devs) / len(q_devs)),
            "mean_R_gate_dev":    float(sum(r_devs) / len(r_devs)),
            "mean_f_extra_norm":  float(sum(f_norms) / len(f_norms)),
            "mean_f_tau1_first":  float(sum(f_tau1_first) / len(f_tau1_first)),
        }

    def save_pt(self, filepath: str):
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save({"recorder": self.epochs}, filepath)


class ModelManager:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "saved_models"
        )
        os.makedirs(self.base_dir, exist_ok=True)

    def save_training_session(
        self, model, loss_history, training_params,
        session_name=None, recorder=None,
    ):
        session_name = session_name or f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = os.path.join(self.base_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        model.save(
            os.path.join(session_dir, f"{session_name}.pth"),
            {"training_loss_history": loss_history, "training_params": training_params},
        )
        if recorder:
            recorder.save_pt(os.path.join(session_dir, f"{session_name}_network_outputs.pt"))
        return session_dir