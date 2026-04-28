# Double Pendulum Swing-up: Results Summary

## Best result: **goal_dist = 0.0612** at default `q_base_diag = [12, 5, 50, 40]`
**WITHOUT** any reference trajectory.

Reproduce: `python exp_no_demo.py`
Best model: `saved_models/stageD_nodemo_20260428_123448/`

## Demo CSVs
- `demo_csv/rollout_epoch0_untrained.csv` (untrained, goal_dist=2.84)
- `demo_csv/rollout_final_trained.csv` (trained, goal_dist=0.06)

## Key insights

### 1. The "core challenge" was solved by suppressing BOTH q1 and q1d
The working state-phase Q-profile target uses `pump=[0.01, 0.01, 1, 1]`,
not `[0.01, 1, 1, 1]`.  Suppressing q1 alone leaves q1d_cost=5 in the QP
which damps q1_dot and kills energy pumping.

### 2. The reference trajectory is not needed
A synthetic cosine-eased energy ramp from -14.7 to +14.7 J as the
target gives TIGHTER swing-up than the demo (0.06 vs 0.14).  The demo's
own residual error at the end was hurting performance.

### 3. The network is naturally robust to noise
Without any Kalman filter, the model achieves 100% success up to:
- σ_q = 0.10 rad (≈ 5.7°)
- σ_qd = 0.50 rad/s

The encoder's 5-frame state history provides implicit smoothing.
Breaks down around σ_q = 0.20.  Noise FT gives marginal improvement.

### 4. Generalisation across initial states is partial
Expansion training (alternating x0=zero with random perturbations) gives
3/7 success across symmetric perturbations.  Failures are symmetric:
both positive AND negative q1 starting positions fail.  The trained
network has a baked-in swing-up direction.

## Loss design that worked

Component | Role
---|---
energy tracking (synthetic ramp -14.7→+14.7) | Smooth pumping signal
state-phase Q-profile target (W=100) | Suppress q1+q1d cost during pump
end-phase Q-gate increase (W=80, last 20 steps) | Stabilise at goal
q1+q1d kickstart bias (-3) | Initial gates near 0.05 (pump prior)

## Loss components contributing to final loss
- Track loss (energy difference, scaled by E_range²)
- Profile penalty (||gates_Q - state_phased_target||²)
- End-phase penalty (||1 - gates_Q[:,0:2]||² for last 20 steps)
- Terminal anchor (disabled — caused f-saturation)

## Files
- `exp_no_demo.py` — best swing-up (synthetic energy ramp)
- `exp_end_q_high.py` — end-phase Q-gate-up version (with demo)
- `exp_noise_test.py` — noise robustness rollout
- `exp_robust_finetune.py` — noise-augmented fine-tuning
- `exp_expansion.py` — random x0 fine-tuning (forgets clean)
- `exp_expansion2.py` — alternating x0 (preserves clean)
- `dump_success_csv.py` — generate before/after CSVs

## Training progression (the path)
1. Initial Q-modulation experiments at default q_base_diag: FAIL (saturation)
2. Curriculum learning (q1=0→3→6→12): works to q1=6, fails at 9
3. State-phase profile target with q1 only: 1.91 (close)
4. State-phase profile with q1+q1d: 0.25 (BREAKTHROUGH)
5. End-phase Q-gate increase: 0.20
6. Softer kickstart bias=-3: 0.14
7. **No reference trajectory + synthetic energy ramp: 0.06** ← best

## Robustness summary

| Noise level | σ_q | σ_qd | mean | success |
|---|---|---|---|---|
| Clean | 0 | 0 | 0.06 | 100% |
| Low | 0.01 | 0.05 | 0.04 | 100% |
| Medium | 0.05 | 0.20 | 0.05 | 100% |
| High | 0.10 | 0.50 | 0.18 | 100% |
| xhigh | 0.20 | 1.00 | 0.98 | 80% |
| Brutal | 0.30 | 1.50 | 2.41 | 60% |
| Extreme | 0.50 | 2.50 | 4.16 | 20% |
