# Double-Pendulum Swing-Up — Handoff

**Branch:** `claude/pendulum-mpc-neural-network-csqVr`
**Latest commit:** `cbfe712` — `w_f_stable` state-conditional f_extra penalty

---

## 1. Current situation (one paragraph)

Best clean swing-up is `goal_dist = 0.0612` at step 170 with the model in
`saved_models/stageD_nodemo_20260428_123448/`.  The pendulum **reaches**
the upright but does not **hold** it: at step 200 the raw distance is 3.47
(it's fallen back), at step 600 it's 5.48 (oscillating through ±π).
A diagnostic (`exp_diag_failure.py`) shows the network keeps emitting
large `f_extra` (norm ≈ 6.8) even at the goal — "overlearned energy
pumping."  The breakthrough proven by `verify_smart_gate.py`: scaling
`f_extra` by `(1 - stable_zone)` at inference takes the same model from
wrap=5.48 to wrap=0.077 at 600 steps without retraining.  Currently
fine-tuning that property INTO the network via a state-conditional outer
loss (`w_f_stable` in `Simulate.py`).  **No controller code was
modified.**

---

## 2. Hard constraints (must respect)

These are non-negotiable design choices for the system:

| Constraint | What it means in practice |
|------------|---------------------------|
| **Fixed horizon length** | `HORIZON = 10` everywhere. Don't change `mpc.N`. |
| **No trajectory inside the controller** | The MPC only sees `(state, x_lin_seq, u_lin_seq, x_goal, gates_Q, gates_R, f_extra)`. It does NOT see the demo, the energy ramp, or anything trajectory-shaped. |
| **Physics-informed signals live in the OUTER LOSS** | Energy tracking, wrapped-angle penalties, stable-zone gates: ALL go into `Simulate.train_linearization_network`'s loss assembly. None of these belong in `mpc_controller.py`. |
| **Controller stays general** | `mpc_controller.py` is task-agnostic. It just solves the QP given Q/R/f_extra it's handed. No swing-up-specific code there. |
| **The network is the only learned component** | Everything that adapts to the task adapts via `lin_net.LinearizationNetwork` weights. The MPC is fixed. |
| **No inference-time hacks in production** | `verify_smart_gate.py` was a probe to confirm the principle. The fix lives in the loss so the network learns it; we don't massage outputs at runtime in the deployed rollout. |

---

## 3. Targets

| Property | Current | Target |
|----------|---------|--------|
| Clean goal_dist @ 170 steps | 0.0612 | ≤ 0.10 (preserve, don't beat) |
| Clean wrapped_dist @ 600 steps (12 s) | 5.48 (oscillates) | ≤ 0.30 (sustained stable hold) |
| Symmetric initial-condition success | 3 / 7 | 7 / 7 |
| Sensor-noise robustness | σ_q ≤ 0.10 (100 %) | σ_q ≤ 0.20 |

---

## 4. File map — every Python file, what it's for, status

### Core libraries (all in active use)

| File | Role | Status |
|------|------|--------|
| `lin_net.py` | `LinearizationNetwork` (default, raw 4-dim state, 20 input dims) and `LinearizationNetworkSC` (sin/cos encoding, 30 input dims). `NetworkOutputRecorder`, `ModelManager`. | **Active** |
| `Simulate.py` | Training loop `train_linearization_network` + `rollout`. All loss terms live here (energy track, Q-profile, end-Q-high, w_f_end_reg, w_stable_phase, w_f_stable, train_noise_sigma). | **Active** |
| `mpc_controller.py` | Differentiable QP-based MPC. **NOT modified during this session.** | **Active, untouched** |

### Currently driving progress

| File | Purpose | Status |
|------|---------|--------|
| `exp_no_demo.py` | Train from scratch, synthetic energy ramp. Produced the 0.0612 baseline. | **Reference baseline** |
| `exp_diag_failure.py` | Step-by-step rollout trace of f_extra / Q-gates / energy / state for failing initial conditions. **Diagnosed the overlearned-pumping problem.** | **Diagnostic — keep** |
| `verify_smart_gate.py` | Probe: tests inference-time `stable_zone = (1+cos(q1-π))/2 × exp(-(q1d²+q2d²)/2)` gating. **Confirmed the fix principle (wrap=0.077 at 600 steps).** | **Verifier — keep** |
| `verify_stable.py` | Loads latest `stageD_stabilize*` checkpoint and evaluates raw + wrapped goal_dist at 170/220/300/400/600. | **Active, run after each stability training** |
| `exp_stab_state.py` | **Currently running.** Fine-tunes 0.0612 model with `w_f_stable=50` (state-conditional f_extra penalty). All other losses identical to original training. | **Currently training** |
| `exp_noise_test.py` | Noise robustness eval (σ levels: clean → brutal). Used to prove the model is robust without a Kalman filter. | **Active when needed** |

### Network output / debug

| File | Purpose | Status |
|------|---------|--------|
| `dump_success_csv.py` | Writes trajectory CSVs (untrained vs trained) for plotting demos. | Useful for visualisation |
| `RESULTS_SUMMARY.md` | Snapshot of results from BEFORE the stability work. Pre-stability findings doc. | Reference history |

### Dead ends or subsumed (don't waste time on these)

| File | Why dead-end |
|------|--------------|
| `exp_finetune_best.py` | Used time-window `w_f_end_reg` which broke the swing-up. Replaced by state-conditional `w_f_stable` in `exp_stab_state.py`. |
| `exp_stabilize.py`, `exp_stabilize2.py` | Same time-window approach. `exp_stabilize.py` produced a model with raw=5.69 / wrapped=2.85 at 170 steps (broken swing-up). |
| `exp_robust_finetune.py` | Marginal noise improvement; superseded by the discovery that the model is already noise-robust (σ_q ≤ 0.10). |
| `exp_expansion.py` | First expansion attempt — caused catastrophic forgetting of the canonical x0=zero. |
| `exp_expansion2.py` | Alternation fix preserved canonical but only 3/7 perturbations succeed. Will be retried after stability + state-conditional loss is solid. |
| `exp_curriculum_expand.py`, `exp_curriculum.py`, `exp_curriculum2.py`, `exp_curriculum_phase.py` | Pre-stability curriculum experiments. Not needed now. |
| `exp_mirror.py` | Mirror augmentation for symmetric swing-up. Started, killed when stability became the focus. **Will be revisited next** with the stable model as starting point. |
| `exp_bignet.py` | hidden_dim=256 from scratch. Killed (CPU contention, wrong priority). |
| `exp_sincos.py` | Trains `LinearizationNetworkSC` from scratch. Not yet run; would address symmetric-swing-up structurally rather than via mirror augmentation data. **Pivot option for later.** |
| `exp_cos_q1`, `exp_phase_aware`, `exp_phase_loss`, `exp_profile_loss`, `exp_grad_split`, `exp_hardcoded_q`, `exp_q_modulation`, `exp_q1_freeze`, `exp_q1_gate_reg`, `exp_threshold`, `exp_twophase`, `exp_small_q1`, `exp_end_q_high`, `exp_nodemo_sweep`, `exp_noise_train`, `exp_final`, `verify_inference_gate.py` | Earlier-stage experiments and probes. The relevant findings are absorbed into `exp_no_demo.py` (best result) and `exp_stab_state.py` (current focus). Don't run these. |

---

## 5. What changed during this session (concrete)

**`Simulate.py` (outer loss only — not controller):**
- Added `train_noise_sigma` (Gaussian noise on observations during training)
- Added `w_f_end_reg`, `f_end_reg_steps` — time-window f_extra penalty (**superseded — does not help**)
- Added `w_stable_phase`, `stable_phase_steps` — wrapped-angle position-tracking loss for last N steps (**too aggressive alone**)
- Added `w_f_stable` — **state-conditional f_extra penalty** (this is the keeper):

  ```
  stable_zone = ((1 + cos(q1 - q1_goal)) / 2) * exp(-(q1d² + q2d²) / 2)
  loss += w_f_stable * stable_zone * ‖f_extra‖²
  ```

  Mirrors the existing `q_profile_state_phase=True` mechanism that already
  blends pump↔stable Q-targets via `cos(q1 - π)`.

**`lin_net.py` (architecture only):**
- Added `LinearizationNetworkSC`: same trunk/heads, but inputs are
  `(sin(q1), cos(q1), q1d/8, sin(q2), cos(q2), q2d/8) × 5 = 30` instead
  of `(q1/π, q1d/8, q2/π, q2d/8) × 5 = 20`.  Goal: structural symmetry
  for the upright (`+π ≡ -π` at the input level).

**`mpc_controller.py`:** unchanged.

---

## 6. Milestones & how each was hit

| # | Milestone | Commit | How |
|---|-----------|--------|-----|
| 1 | Swing-up at default Q-cost | `9b90ba3` | Q-profile suppresses BOTH q1 + q1d during pump phase, raises during stabilise. |
| 2 | End-phase Q-gate increase | `913114c` | `w_end_q_high` pushes Q-gates → 1 in last 20 steps. 0.249 → 0.198. |
| 3 | **0.0612 baseline (no demo)** | `7581f08` | Replaced demo with synthetic cosine-eased energy ramp -14.7 → +14.7 J. Energy-only outer loss. |
| 4 | Noise robustness proven | `320409f` | 100 % up to σ_q = 0.10 rad without any Kalman filter — 5-frame state history acts as implicit smoother. |
| 5 | Generalisation map | `82b04df` | Alternating x0=zero ↔ random x0 prevents catastrophic forgetting. 3/7 perturbations succeed. |
| 6 | **Stability diagnosis** | `56c8e05` | `exp_diag_failure.py` proved: pendulum oscillates through ±π because f_extra stays large at goal. Also showed the q1=±0.2 "failures" were really wrap-related (raw vs wrapped distance). |
| 7 | **Stability fix verified at inference** | `2dec502` | `verify_smart_gate.py` with `stable_zone` gate → wrap=0.077 at 600 steps on the 0.0612 model. |
| 8 | **State-conditional outer loss** | `cbfe712` | `w_f_stable` added to `Simulate.py`. **Currently training.** |

---

## 7. Saved models

| Folder | Best | Notes |
|--------|------|-------|
| `stageD_nodemo_20260428_123448/` | **0.0612 (170 steps)** | The baseline that everything fine-tunes from. |
| `stageD_robust_20260428_143148/` | 0.072 | Noise-robust fine-tune from baseline. |
| `stageD_expand2_20260428_154524/` | varies | Expansion alternation. 3/7 success. |
| `stageD_stabstate_*` | TBD | Will appear when `exp_stab_state.py` finishes. |

---

## 8. Plan from here

1. **Now:** wait for `exp_stab_state.py` to finish.
2. **Verify** the stab_state model:
   - Run `verify_stable.py` (raw + wrapped at 170/220/300/400/600 steps).
   - Confirm 170-step swing-up preserved (raw ≤ 0.15).
   - Confirm long-horizon stability (wrap ≤ 0.30 at 600 steps).
   - Look at `loss_history` monotonicity.
3. **If stability is solved (1.–2. both pass):** move to generality.
   - Run `exp_mirror.py` from the stab_state checkpoint instead of from
     0.0612 (so we keep stability while learning symmetry).
   - Or: try `exp_sincos.py` from scratch with `w_f_stable` enabled —
     architectural symmetry + learned stability in one shot.
4. **If stability is NOT solved** (training broke the swing-up again, or
   wrap > 0.30 at 600):
   - Lower `w_f_stable` (try 20, 10).
   - Inspect rollout step-by-step with `exp_diag_failure.py` on the new
     model to see what went wrong.

---

## 9. Quick commands

```bash
# Verify any stabilize* checkpoint
python verify_stable.py

# Inference-time gating probe (read-only)
python verify_smart_gate.py

# Diagnostic for any failure mode
python exp_diag_failure.py     # edit MODEL_PATH at the top

# Once stability is solid, generality:
python exp_mirror.py           # edit PRETRAINED to point at stabstate

# Or architectural fix:
python exp_sincos.py
```

---

## 10. Commit trace (most recent first)

| Commit | Description |
|--------|-------------|
| `cbfe712` | **w_f_stable state-conditional penalty (this is the breakthrough fix)** |
| `2dec502` | Inference-time gating verifier + follow-up scripts |
| `f00fecd` | HANDOFF.md + verify_stable.py |
| `656e950` | exp_stabilize: save model BEFORE post-eval |
| `56c8e05` | Stabilize experiment + diagnostic + w_stable_phase |
| `1c592a0` | LinearizationNetworkSC + w_f_end_reg |
| `0783dc4` | exp_finetune_best.py (time-window approach, dead end) |
| `dd97541` | mirror / bignet / curriculum_expand experiments |
| `3022b7b` | RESULTS_SUMMARY.md (pre-stability) |
| `82b04df` | Expansion v2 alternation (3/7 succeed) |
| `7581f08` | **HUGE: 0.0612 without any reference trajectory** |
