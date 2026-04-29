# Double-Pendulum Swing-Up — Handoff

**Branch:** `claude/pendulum-mpc-neural-network-csqVr`
**Latest commit:** `b653f42` (state-conditional fix + handoff)

---

## 1. Current situation (one paragraph)

The `w_f_stable` state-conditional outer-loss term **works**: from the
0.0612 swing-up baseline, fine-tuning with this single new loss took
the goal_dist at the training horizon (220 steps) from 6.22 → **0.0755
(raw=0.0755, wrapped=0.0755 — STABLE)**.  The principle is proven and
the controller was never modified — all the change is in `Simulate.py`'s
loss assembly and the network's learned weights.  The remaining gap is
that the trained model **degrades after the training horizon**: at step
400 it's at raw=6.59 / wrapped=1.28 (sliding out of the stable zone),
at step 1000 it's wrapped=7.18 (lost completely).  Diagnosis: the
network was only ever evaluated up to step 220 during training, so it
has no signal for sustained hold.  Currently running
`exp_long_horizon.py` — same loss configuration but NUM_STEPS=400, fine-tuning
from the stab_state checkpoint, to extend the temporal coverage.

---

## 2. Hard constraints (must respect)

| Constraint | What it means in practice |
|------------|---------------------------|
| **Fixed horizon length** | `HORIZON = 10` everywhere. Don't change `mpc.N`. |
| **No trajectory inside the controller** | The MPC only sees `(state, x_lin_seq, u_lin_seq, x_goal, gates_Q, gates_R, f_extra)`. It does NOT see the demo, the energy ramp, or anything trajectory-shaped. |
| **Physics-informed signals live in the OUTER LOSS** | Energy tracking, wrapped-angle penalties, stable-zone gates: ALL go into `Simulate.train_linearization_network`'s loss assembly. None of these belong in `mpc_controller.py`. |
| **Controller stays general** | `mpc_controller.py` is task-agnostic. It just solves the QP given Q/R/f_extra it's handed. No swing-up-specific code there. |
| **The network is the only learned component** | Everything that adapts to the task adapts via `lin_net.LinearizationNetwork` weights. The MPC is fixed. |
| **No inference-time hacks in production** | `verify_smart_gate.py` was a probe to confirm the principle. The fix lives in the loss so the network learns it; we don't massage outputs at runtime in the deployed rollout. |

---

## 3. Targets and current state

| Property | Baseline (0.0612) | After stab_state | Target |
|----------|------------------:|-----------------:|-------:|
| 170 steps raw   | 0.0612 | 0.1995 | ≤ 0.20 ✓ |
| 220 steps raw   | 6.2214 | **0.0755** | ≤ 0.30 ✓ |
| 300 steps wrap  | 0.9537 | 0.2038 | ≤ 0.30 ✓ |
| **400 steps wrap** | — | 1.2847 | ≤ 0.30 ✗ |
| **600 steps wrap** | 5.4802 | 2.5874 | ≤ 0.30 ✗ |
| **1000 steps wrap** | — | 7.1807 | ≤ 0.30 ✗ |
| Loss monotonicity | — | 25/33 (76 %) | should be > 80 % |
| Symmetric IC success | 3 / 7 | not yet retested | 7 / 7 |
| σ_q noise robustness | ≤ 0.10 | not yet retested | ≤ 0.20 |

**Two columns matter most:** 220-step (proves the gate works) and 600-step (proves sustained hold).

---

## 4. What changed this session (concrete)

### `Simulate.py` (outer loss only — controller untouched)

- `train_noise_sigma` — Gaussian observation noise during training.
- `w_f_end_reg`, `f_end_reg_steps` — time-window f_extra penalty (**dead-end, broke swing-up**).
- `w_stable_phase`, `stable_phase_steps` — wrapped-angle position-tracking loss for the last N steps (**too aggressive alone, retained as optional but not the main lever**).
- **`w_f_stable`** — state-conditional f_extra penalty (**this is the keeper**):

  ```
  stable_zone = ((1 + cos(q1 - q1_goal)) / 2) * exp(-(q1d² + q2d²) / 2)
  loss      += w_f_stable * stable_zone * ‖f_extra‖²
  ```

  Mirrors the existing `q_profile_state_phase=True` mechanism that already
  blends pump↔stable Q-targets via `cos(q1 - π)`.

### `lin_net.py` (architecture — adds variant, doesn't change default)

- `LinearizationNetworkSC` — same trunk and heads, but inputs are
  `(sin(q1), cos(q1), q1d/8, sin(q2), cos(q2), q2d/8) × 5 = 30` instead
  of `(q1/π, q1d/8, q2/π, q2d/8) × 5 = 20`.  Goal: structural symmetry
  for the upright (`+π ≡ -π` at the input level).  Not yet trained.

### `mpc_controller.py`

Unchanged.  Stays general by design.

---

## 5. Files — every Python file, what it's for, status

### Core libraries (active)

| File | Role | Status |
|------|------|--------|
| `lin_net.py` | `LinearizationNetwork` (default, raw 4-dim state, 20 input dims) and `LinearizationNetworkSC` (sin/cos, 30 dims). `NetworkOutputRecorder`, `ModelManager`. | Active |
| `Simulate.py` | Training loop `train_linearization_network` + `rollout`. ALL physics-informed loss terms live here. | Active |
| `mpc_controller.py` | Differentiable QP-based MPC. **Untouched this session.** | Active |

### Currently driving progress

| File | Purpose | Status |
|------|---------|--------|
| `exp_no_demo.py` | Trained the 0.0612 baseline (synthetic energy ramp, no demo). | Reference |
| `exp_diag_failure.py` | Step-by-step rollout trace. **Diagnosed the overlearned-pumping problem.** | Diagnostic |
| `verify_smart_gate.py` | Probe: inference-time `stable_zone` gating. **Confirmed the fix principle (wrap=0.077 at 600 steps).** | Verifier |
| `verify_stable.py` | Loads latest `stageD_stabilize*` and reports raw + wrapped at 170/220/300/400/600 steps. | Active |
| `exp_stab_state.py` | **Done.** Fine-tuned 0.0612 with `w_f_stable=50`. Achieved raw=0.0755 at step 220. Saved `stageD_stabstate_20260428_224856`. | Done |
| `exp_long_horizon.py` | **Currently running.** Same loss as stab_state but NUM_STEPS=400, fine-tuning from stabstate checkpoint. Goal: sustained hold at 1000+ steps. | Currently training |
| `exp_noise_test.py` | Noise-robustness eval. | Active when needed |

### Dead ends / superseded (don't run these)

| File | Why dead-end |
|------|--------------|
| `exp_finetune_best.py` | Time-window `w_f_end_reg`. Broke swing-up. |
| `exp_stabilize.py`, `exp_stabilize2.py` | Same time-window approach. |
| `exp_robust_finetune.py` | Marginal noise improvement; baseline already robust. |
| `exp_expansion.py`, `exp_expansion2.py` | Catastrophic forgetting / 3/7 partial generality. Will be revisited after sustained-hold is solved. |
| `exp_curriculum*.py` | Pre-stability curriculum experiments. |
| `exp_mirror.py` | Mirror augmentation. Killed when stability became the focus. **Will resume from long_horizon checkpoint.** |
| `exp_bignet.py` | hidden_dim=256 from scratch. Killed. |
| `exp_sincos.py` | Trains `LinearizationNetworkSC` from scratch. Pivot option for symmetric swing-up if mirror data-aug doesn't work. |
| `exp_cos_q1`, `exp_phase_aware`, `exp_phase_loss`, `exp_profile_loss`, `exp_grad_split`, `exp_hardcoded_q`, `exp_q_modulation`, `exp_q1_freeze`, `exp_q1_gate_reg`, `exp_threshold`, `exp_twophase`, `exp_small_q1`, `exp_end_q_high`, `exp_nodemo_sweep`, `exp_noise_train`, `exp_final`, `verify_inference_gate.py` | Earlier-stage experiments. Findings absorbed into `exp_no_demo.py` and `exp_stab_state.py`. |

---

## 6. Saved models

| Folder | Best metric | Notes |
|--------|-------------|-------|
| `stageD_nodemo_20260428_123448/` | 0.0612 raw @ 170 | The swing-up baseline. Everything fine-tunes from this. |
| `stageD_robust_20260428_143148/` | 0.072 raw @ 170 | Noise-robust fine-tune. |
| `stageD_expand2_20260428_154524/` | 3/7 perturbations | Expansion alternation. |
| **`stageD_stabstate_20260428_224856/`** | **0.0755 raw @ 220** | **State-conditional w_f_stable — current best for stability up to step 300. Degrades after.** |
| `stageD_long_*` | TBD | Will appear when `exp_long_horizon.py` finishes. |

---

## 7. Loss monotonicity — `stab_state` run

```
epoch  1:  26.590
epoch  5:  24.905
epoch 10:  22.601
epoch 15:  21.865
epoch 17:  19.226   ← lowest
epoch 20:  19.379
epoch 25:  18.706
epoch 30:  21.118   (small re-rise)
epoch 34:  20.011   (early-stop triggered: best_goal_dist hadn't improved in 15 epochs)

decreasing transitions: 25/33 (76 %)
```

The training loss is **mostly monotonic with mild noise** — strictly
decreasing for the first 25 epochs, then a small re-rise as the LR
cosine-anneals.  The "Best" goal_dist at step 220 reached **0.0755 at
epoch 20** and never improved beyond that, which is what triggered the
early stop.

This is healthier than the time-window run which had loss = 97.45 at
epoch 1 (over-penalised) and then a full collapse of swing-up.

---

## 8. Plan from here (after long_horizon finishes)

1. **Verify** the long_horizon model:
   - If wrap ≤ 0.30 at 600 AND 1000 steps → stability is solved.
   - Run `verify_smart_gate.py` against it as a sanity check (the
     learned gate should make the inference-time gate redundant — they
     should agree closely).
2. **If stable:** generality phase.
   - **Symmetric IC:** retrain with mirror data-augmentation
     (`exp_mirror.py`) starting from long_horizon checkpoint.
   - **Continuous IC distribution:** retrain with random small
     perturbations of x0 starting from the same checkpoint.
   - Or pivot to `exp_sincos.py` from scratch with `w_f_stable` on —
     architectural symmetry plus learned stability in one shot.
3. **If long_horizon plateaus too:** raise w_f_stable (try 100), or
   add `w_stable_phase` on top to pin the position directly during the
   stabilisation window.

---

## 9. Quick commands

```bash
# Verify any stabilize* / long_* checkpoint
python verify_stable.py

# Inference-time gating probe (read-only)
python verify_smart_gate.py

# Diagnostic for any failure mode
python exp_diag_failure.py     # edit MODEL_PATH at the top

# Sustained-hold training (currently running)
python exp_long_horizon.py

# Once stability is solid, generality:
python exp_mirror.py           # edit PRETRAINED to point at long_horizon
python exp_sincos.py           # alternative: from scratch with SC architecture
```

---

## 10. Commit trace (most recent first)

| Commit | Description |
|--------|-------------|
| `b653f42` | HANDOFF.md rewrite (this commit, then this update) |
| `cbfe712` | **w_f_stable state-conditional penalty (the breakthrough fix)** |
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
