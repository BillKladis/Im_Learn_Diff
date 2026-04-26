import torch
import torch.nn as nn
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import math

class LinearizationNetwork(nn.Module):
    def __init__(
        self,
        state_dim:    int,
        control_dim:  int,
        horizon:      int,
        hidden_dim:   int   = 128,
        gate_range:   float = 0.5,
        gate_range_q: Optional[float] = None,
        gate_range_r: Optional[float] = None,
        gate_range_e: Optional[float] = None,
        n_res:        int   = 5,
    ):
        super().__init__()
        self.state_dim   = state_dim
        self.control_dim = control_dim
        self.hidden_dim  = hidden_dim
        self.horizon     = horizon
        self.gate_range  = gate_range
        self.gate_range_q = gate_range if gate_range_q is None else gate_range_q
        self.gate_range_r = gate_range if gate_range_r is None else gate_range_r
        self.gate_range_e = gate_range if gate_range_e is None else gate_range_e
        self.n_res       = n_res

        for name, value in (
            ("gate_range_q", self.gate_range_q),
            ("gate_range_r", self.gate_range_r),
            ("gate_range_e", self.gate_range_e),
        ):
            if value >= 1.0:
                raise ValueError(f"{name} must be < 1.0")

        state_input_dim = 5 * state_dim
        res_input_dim   = n_res * state_dim
        enc_dim         = hidden_dim // 2
        trunk_input_dim = enc_dim * 2

        q_out_dim = (horizon - 1) * state_dim
        r_out_dim = horizon       * control_dim
        e_out_dim = horizon       # 1 weight per step for Energy
        
        # A 4x4 lower triangular matrix has 10 non-zero elements
        qf_out_dim = (state_dim * (state_dim + 1)) // 2

        state_scale = torch.tensor([math.pi, 8.0, math.pi, 8.0], dtype=torch.float64).repeat(5)
        self.register_buffer('state_scale', state_scale)
        self.state_encoder = nn.Sequential(nn.Linear(state_input_dim, enc_dim), nn.Tanh())

        self.register_buffer('res_scale', torch.tensor([1.0, 5.0, 1.0, 5.0], dtype=torch.float64).repeat(n_res))
        self.res_encoder = nn.Sequential(nn.Linear(res_input_dim, enc_dim), nn.Tanh())

        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )

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

        # ── Dense Qf Head ──────────────────────────────────────────────
        self.qf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, qf_out_dim),
        )

        # ── Linearization Control Offset Head ──────────────────────────
        self.u_lin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, r_out_dim),
        )
        self.register_buffer('u_lin_bound', torch.tensor(1.0, dtype=torch.float64))

        # ── NEW: Tunable Energy Shaping Head ───────────────────────────
        self.e_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, e_out_dim),
        )

        # ── Expert Seeded Baseline ─────────────────────────────────────
        # Qf diagonal: [q1=20, q1_dot=60, q2=30, q2_dot=30]
        # Higher q1_dot terminal weight prevents overshoot during swing-up.
        # q2/q2_dot weights match the Q matrix's anti-fold priorities.
        base_L_matrix = torch.zeros((state_dim, state_dim), dtype=torch.float64)
        base_L_matrix[0, 0] = math.sqrt(20.0)   # Qf_q1     ≈ 20
        base_L_matrix[1, 1] = math.sqrt(60.0)   # Qf_q1dot  ≈ 60 (prevents overshoot)
        base_L_matrix[2, 2] = math.sqrt(30.0)   # Qf_q2     ≈ 30
        base_L_matrix[3, 3] = math.sqrt(30.0)   # Qf_q2dot  ≈ 30
        base_L_matrix[1, 0] = 1.0
        base_L_matrix[3, 2] = 1.0

        tril_idx = torch.tril_indices(row=state_dim, col=state_dim, offset=0)
        base_L_flat = base_L_matrix[tril_idx[0], tril_idx[1]]
        
        self.register_buffer('L_base', base_L_flat.double())
        self.register_buffer('qf_bound', torch.tensor(5.0, dtype=torch.float64))

        self._initialize_weights()
        
        self.metadata = {
            'state_dim':       state_dim,
            'control_dim':     control_dim,
            'hidden_dim':      hidden_dim,
            'horizon':         horizon,
            'gate_range':      gate_range,
            'gate_range_q':    self.gate_range_q,
            'gate_range_r':    self.gate_range_r,
            'gate_range_e':    self.gate_range_e,
            'n_res':           n_res,
            'architecture':    'five_branch_dense_qf_u_lin_energy',
            'created_date':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

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
        _init_block(self.res_encoder,   final_std=0.1)
        _init_block(self.trunk, final_std=0.1)
        _init_block(self.q_head, final_std=0.01)
        _init_block(self.r_head, final_std=0.01)
        _init_block(self.qf_head, final_std=0.001)
        _init_block(self.u_lin_head, final_std=0.001)
        _init_block(self.e_head, final_std=0.01) # Initialize energy gates near 1.0

    def forward(
        self,
        x_sequence:       torch.Tensor,
        residual_history: torch.Tensor,
        q_base_diag:      Optional[torch.Tensor] = None,
        r_base_diag:      Optional[torch.Tensor] = None,
    ):
        x_flat   = x_sequence.reshape(-1)
        x_normed = x_flat / self.state_scale
        state_emb   = self.state_encoder(x_normed)

        res_flat   = residual_history.reshape(-1)
        res_normed = res_flat / self.res_scale
        res_emb     = self.res_encoder(res_normed)

        combined    = torch.cat([state_emb, res_emb], dim=-1)
        features    = self.trunk(combined)

        raw_Q   = self.q_head(features).reshape(self.horizon - 1, self.state_dim)
        gates_Q = 1.0 + self.gate_range_q * torch.tanh(raw_Q)

        raw_R   = self.r_head(features).reshape(self.horizon, self.control_dim)
        gates_R = 1.0 + self.gate_range_r * torch.tanh(raw_R)

        # ── Dense Qf Matrix ───────────────────────────────────────────
        raw_qf_offset = self.qf_head(features)
        bounded_offset = torch.tanh(raw_qf_offset) * self.qf_bound
        L_flat = self.L_base + bounded_offset
        L = torch.zeros((self.state_dim, self.state_dim), device=features.device, dtype=torch.float64)
        tril_indices = torch.tril_indices(row=self.state_dim, col=self.state_dim, offset=0)
        L[tril_indices[0], tril_indices[1]] = L_flat
        Qf_dense = L @ L.T + 1e-3 * torch.eye(self.state_dim, device=features.device, dtype=torch.float64)

        # ── Control Linearization Offset ──────────────────────────────
        raw_u_lin = self.u_lin_head(features).reshape(self.horizon, self.control_dim)
        u_lin_delta = self.u_lin_bound * torch.tanh(raw_u_lin)

        # ── Energy Shaping Gates ──────────────────────────────────────
        raw_E   = self.e_head(features).reshape(self.horizon)
        gates_E = 1.0 + self.gate_range_e * torch.tanh(raw_E)

        q_diags = (q_base_diag.unsqueeze(0) * gates_Q) if q_base_diag is not None else None
        r_diags = (r_base_diag.unsqueeze(0) * gates_R) if r_base_diag is not None else None

        return gates_Q, gates_R, gates_E, Qf_dense, q_diags, r_diags, u_lin_delta

    # (save and load remain unchanged, omit for brevity)
    def save(self, filepath: str, metadata: Dict = None):
        parent = os.path.dirname(filepath)
        if parent: os.makedirs(parent, exist_ok=True)
        if metadata: self.metadata.update(metadata)
        torch.save({'model_state_dict': self.state_dict(), 'metadata': self.metadata}, filepath)

    @classmethod
    def load(cls, filepath: str, device: str = 'cpu'):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        metadata   = checkpoint.get('metadata', {})
        legacy_gate_range = metadata.get('gate_range', 0.5)
        model = cls(
            state_dim   = metadata.get('state_dim',   4),
            control_dim = metadata.get('control_dim', 2),
            hidden_dim  = metadata.get('hidden_dim',  128),
            horizon     = metadata.get('horizon',     10),
            gate_range  = legacy_gate_range,
            gate_range_q= metadata.get('gate_range_q', legacy_gate_range),
            gate_range_r= metadata.get('gate_range_r', legacy_gate_range),
            gate_range_e= metadata.get('gate_range_e', legacy_gate_range),
            n_res       = metadata.get('n_res',       2),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).double()
        model.metadata = metadata
        return model

class NetworkOutputRecorder:
    def __init__(self):
        self.epochs: List[Dict] = []
        self._current_epoch_steps: List[Dict] = []

    def start_epoch(self):
        self._current_epoch_steps = []

    def record_step(
        self, gates_Q: torch.Tensor, gates_R: torch.Tensor, gates_E: torch.Tensor,
        q_diags: Optional[torch.Tensor], r_diags: Optional[torch.Tensor],
        u_mpc: torch.Tensor, state_err: torch.Tensor, residual_norm: float = 0.0,
        Qf_dense: Optional[torch.Tensor] = None, u_lin_delta: Optional[torch.Tensor] = None,
    ):
        with torch.no_grad():
            self._current_epoch_steps.append({
                'gates_Q':       gates_Q.detach().cpu().tolist(),
                'gates_R':       gates_R.detach().cpu().tolist(),
                'gates_E':       gates_E.detach().cpu().tolist(),
                'Qf_dense':      Qf_dense.detach().cpu().tolist() if Qf_dense is not None else None, 
                'q_diags':       q_diags.detach().cpu().tolist() if q_diags is not None else None,
                'r_diags':       r_diags.detach().cpu().tolist() if r_diags is not None else None,
                'u_lin_delta':   u_lin_delta.detach().cpu().tolist() if u_lin_delta is not None else None,
                'u_mpc':         u_mpc.detach().cpu().tolist(),
                'state_err':     state_err.item() if isinstance(state_err, torch.Tensor) else state_err,
                'residual_norm': residual_norm,
            })

    def end_epoch(self, epoch_loss: float):
        self.epochs.append({'steps': self._current_epoch_steps, 'epoch_loss': epoch_loss})
        self._current_epoch_steps = []

    def epoch_summary(self, epoch_idx: int) -> Dict:
        epoch = self.epochs[epoch_idx]
        steps = epoch['steps']
        if not steps: return {'epoch_loss': epoch['epoch_loss'], 'num_steps': 0}

        errs      = [s['state_err'] for s in steps]
        u_norms   = [float(torch.tensor(s['u_mpc']).norm()) for s in steps]
        res_norms = [s.get('residual_norm', 0.0) for s in steps]
        q_devs    = [float((torch.tensor(s['gates_Q']) - 1.0).abs().mean()) for s in steps]
        r_devs    = [float((torch.tensor(s['gates_R']) - 1.0).abs().mean()) for s in steps]
        e_devs    = [float((torch.tensor(s['gates_E']) - 1.0).abs().mean()) for s in steps]

        # NEW — these used to be looked up but never computed → caused nans
        u_lin_norms = [float(torch.tensor(s['u_lin_delta']).norm())
                    for s in steps if s.get('u_lin_delta') is not None]
        qf_norms    = [float(torch.tensor(s['Qf_dense']).norm())
                    for s in steps if s.get('Qf_dense') is not None]

        return {
            'epoch_loss':         epoch['epoch_loss'],
            'num_steps':          len(steps),
            'mean_state_err':     float(sum(errs) / len(errs)),
            'mean_u_norm':        float(sum(u_norms) / len(u_norms)),
            'mean_Q_gate_dev':    float(sum(q_devs) / len(q_devs)),
            'mean_R_gate_dev':    float(sum(r_devs) / len(r_devs)),
            'mean_E_gate_dev':    float(sum(e_devs) / len(e_devs)),
            'mean_residual_norm': float(sum(res_norms) / len(res_norms)),
            'mean_u_lin_norm':    float(sum(u_lin_norms)/len(u_lin_norms)) if u_lin_norms else float('nan'),
            'mean_qf_norm':       float(sum(qf_norms)/len(qf_norms))       if qf_norms    else float('nan'),
        }

    def save_json(self, filepath: str, summaries_only: bool = False):
        parent = os.path.dirname(filepath)
        if parent: os.makedirs(parent, exist_ok=True)
        data = {'num_epochs': len(self.epochs), 'summaries': [self.epoch_summary(i) for i in range(len(self.epochs))]} if summaries_only else {'num_epochs': len(self.epochs), 'epochs': self.epochs}
        with open(filepath, 'w') as f: json.dump(data, f, indent=2)

    def save_pt(self, filepath: str):
        parent = os.path.dirname(filepath)
        if parent: os.makedirs(parent, exist_ok=True)
        torch.save({'recorder': self.epochs}, filepath)

class ModelManager:
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        os.makedirs(self.base_dir, exist_ok=True)
    def save_training_session(self, model, loss_history, training_params, session_name=None, recorder=None):
        session_name = session_name or f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = os.path.join(self.base_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        model.save(os.path.join(session_dir, f"{session_name}.pth"), {'training_loss_history': loss_history, 'training_params': training_params})
        if recorder: recorder.save_pt(os.path.join(session_dir, f"{session_name}_network_outputs.pt"))
        return session_dir
