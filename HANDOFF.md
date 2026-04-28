# Double-Pendulum Swing-Up — Handoff Document

**Last update:** 2026-04-28
**Branch:** `claude/pendulum-mpc-neural-network-csqVr`
**Latest commit:** `656e950` — exp_stabilize: save model BEFORE post-eval

---

## 1. Where we started

A neural-network-augmented MPC for double-pendulum swing-up. The architecture
is:

```
state_history (5 frames × 4 dims) → encoder → trunk → {Q_head, R_head, f_head}
                                                       ↓       ↓        ↓
                                                    gates_Q gates_R  f_extra
                                                       ↓       ↓        ↓
                                              MPC (10-step horizon QP)
                                                       ↓
                                                  applied torque
```

The Q/R heads modulate the MPC's diagonal cost weights per-step; `f_extra` is
a feedforward control offset.  Default cost matrix `q_base_diag = [12, 5, 50, 40]`.

**Initial state at session start:**
- Best swing-up: `goal_dist = 0.0612` at step 170 (8.5 s)
- Model: `saved_models/stageD_nodemo_20260428_123448/`
- Trained from scratch with **NO reference trajectory** (synthetic cosine
  energy ramp from -14.7 J → +14.7 J as the only target signal).
- Already proven robust to sensor noise: 100 % success up to σ_q = 0.10 rad,
  σ_qd = 0.50 rad/s.

---

## 2. Current implementation files

### Core libraries

| File | Role |
|------|------|
| `lin_net.py` | Network classes. `LinearizationNetwork` (raw inputs, 20 dims) and `LinearizationNetworkSC` (sin/cos encoded, 30 dims). |
| `Simulate.py` | Training loop and rollouts. Includes all loss terms. |
| `mpc_controller.py` | The differentiable QP-based MPC. |

### `Simulate.py` — loss components added during this session

| Param | Effect |
|-------|--------|
| `train_noise_sigma` | Inject Gaussian noise on state observations during training. |
| `w_f_end_reg`, `f_end_reg_steps` | Penalise `‖f_extra‖²` in the last N steps so the network stops pumping at the goal. |
| `w_stable_phase`, `stable_phase_steps` | Direct position tracking (wrapped q1 + normalised velocities) for the last N steps to teach holding position. |

The pre-existing `w_q_profile`, `q_profile_pump`, `q_profile_stable`,
`q_profile_state_phase`, `w_end_q_high`, `end_phase_steps` were already in place.

### Experiment files (all `exp_*.py` are entry points)

| File | Goal | Status |
|------|------|--------|
| `exp_no_demo.py` | Train from scratch, synthetic energy ramp | Best result: 0.0612 |
| `exp_noise_test.py` | Evaluate noise robustness across σ levels | Done; 100 % up to high noise |
| `exp_robust_finetune.py` | Fine-tune with noise injection | Marginal gain |
| `exp_expansion.py`, `exp_expansion2.py` | Random-x0 expansion training | 3/7 perturbations succeed |
| `exp_mirror.py` | Train on x0 AND mirror -x0 each iteration | Started; killed (priority shift to stability) |
| `exp_bignet.py` | hidden_dim=256 from scratch | Started; killed (CPU contention) |
| `exp_curriculum_expand.py` | Grow x0 perturbation gradually | Started; killed |
| `exp_finetune_best.py` | Push clean dist below 0.06 with f_extra suppression | Killed (priority shift) |
| `exp_sincos.py` | Train `LinearizationNetworkSC` from scratch | Not yet run |
| `exp_diag_failure.py` | Per-step rollout diagnostic with f_extra/Q-gate trace | **Run; revealed the stability bug** |
| `exp_stabilize.py` | Fine-tune for stability (this is the active focus) | **Currently running (stabilize3)** |
| `verify_stable.py` | Load latest stabilize model, eval at 170/220/300/400/600 steps | Ready to run after stabilize3 |
| `dump_success_csv.py` | Write trajectory CSVs (untrained vs trained) for plotting | Done previously |

### Demo CSVs (already pushed to repo)

- `demo_csv/rollout_epoch0_untrained.csv` — pre-training (goal_dist=2.84)
- `demo_csv/rollout_final_trained.csv` — best swing-up (goal_dist=0.06)

---

## 3. Analytical challenges and milestones

### Milestone 1 — Swing-up at default `q_base_diag` (commit `9b90ba3`)
**Challenge:** original Q-modulation with `q_base_diag=[12,5,50,40]` saturated
the controls and never reached the top.
**Solution:** state-phase Q-profile that pushes the q1 cost DOWN during the
pump phase (so the QP doesn't fight energy injection) and back UP near the
goal.  Crucial detail: **both q1 AND q1d** must be suppressed during pump
(`pump=[0.01, 0.01, 1, 1]`); suppressing q1 alone leaves the q1d cost
damping the angular velocity and killing energy buildup.

### Milestone 2 — End-phase Q-gate stabilisation (commit `913114c`)
**Challenge:** the swing-up reached the upright but didn't tighten enough at
the end.
**Solution:** add `w_end_q_high * (1 - gates_Q[:, 0:2])²` for the last 20
steps, forcing the Q-gates UP near the goal so the MPC starts caring about
position error in the final stretch. Improved 0.249 → 0.198.

### Milestone 3 — No reference trajectory (commit `7581f08`, **0.0612**)
**Challenge:** the demo trajectory's own residual error was hurting
performance — the network was tracking demo errors, not physics.
**Solution:** replace demo with a synthetic cosine-eased energy ramp from
E(q1=0)=-14.7 J to E(q1=π)=+14.7 J.  Energy alone gives the smooth pumping
gradient (∂E/∂q̇ = τ·q̇).  This was the breakthrough that produced the best
clean result.

### Milestone 4 — Noise robustness proven (commit `320409f`)
**Challenge:** is the model robust to sensor noise without a Kalman filter?
**Solution:** rollouts with Gaussian noise injected only into the
state_history (true dynamics clean).  100 % success up to σ_q = 0.10 rad
(≈ 5.7°), σ_qd = 0.50 rad/s.  Breakdown around σ_q = 0.20.  The 5-frame
state history acts as implicit smoothing.

### Milestone 5 — Generalisation map (commits `a486e0e`, `82b04df`)
**Challenge:** does the model generalise to different initial conditions?
**Findings:**
- Naive expansion training causes catastrophic forgetting (loses x0=zero)
- Alternation (1 epoch x0=zero + 1 epoch random) preserves clean
- Result: 3/7 symmetric perturbations succeed; failures are symmetric

### **Milestone 6 — STABILITY DIAGNOSIS (current work)**
**Challenge:** at NUM_STEPS=170 the goal_dist is 0.0612 but at NUM_STEPS=200
it's 3.47 and at 250 it's 6.20.  What's happening?
**Diagnostic** (`exp_diag_failure.py`):
- Step 169: q1=182.27°, fNorm=6.8 (still pumping at the goal!)
- Step 200: pendulum has fallen back to q1≈0
- Step 250: pendulum has swung back UP to q1≈-π (raw=6.2, **wrapped=0.6**)

**Diagnosis:** "overlearned energy pumping" — the network keeps applying
large feedforward torque even after reaching the goal, causing the pendulum
to overshoot, fall, and oscillate through ±π indefinitely.

**Also discovered:** the previous "x0=+0.2 fails" finding was partly an
artifact of the **raw** goal_dist metric.  q1=+0.2 actually reaches q1≈-π
(raw=6.04 because |+π − (-π)| = 2π ≈ 6.28; wrapped=small).  The pendulum
swings up successfully, just to the "other wrap" of the same upright pose.

### Milestone 6 fixes (in progress, commit `56c8e05`, `656e950`)
Three new loss terms ALL active in the last 50 steps:
1. `w_stable_phase = 20`: direct position tracking using **wrapped** q1
   error (so gradient drives toward EITHER +π or -π).
2. `w_f_end_reg = 80`: suppresses f_extra in the stabilisation window.
3. `w_end_q_high = 160`: doubles the Q-gate-up push.

**Current run progress (stabilize3):**
- Pre-train: 250 steps raw=6.20, wrapped=0.60 (oscillating)
- Epoch 1: GoalDist=6.22, fNorm=7.8
- Epoch 15: GoalDist=1.88
- **Epoch 18-20: best=0.4305** (from raw distance, with WRAPPED smaller)
- fNorm dropped 7.8 → 4.3 (f_extra successfully suppressed)
- Early stop expected around epoch 33

---

## 4. Git commit trace (most-recent first)

| Commit | Description |
|--------|-------------|
| `656e950` | exp_stabilize: save model BEFORE post-eval (bug fix) |
| `56c8e05` | Stabilize experiment + diagnostic + w_stable_phase loss |
| `1c592a0` | Sin/cos encoding network + f_extra end-phase regularization |
| `0783dc4` | exp_finetune_best.py to push clean dist below 0.06 |
| `dd97541` | Three new generalisation experiments (mirror, bignet, curriculum) |
| `3022b7b` | RESULTS_SUMMARY.md documenting all key findings |
| `82b04df` | Expansion v2 alternation (3/7 succeed, clean preserved) |
| `a486e0e` | Expansion v1 (catastrophic forgetting) |
| `e29fc47` | Robustness: noise injection finetune |
| `320409f` | Noise robustness proven |
| `be5dbbe` | Demo CSVs with best 0.0612 |
| `7581f08` | **HUGE: 0.0612 without any reference trajectory** |
| `9b90ba3` | First swing-up at default q_base_diag (0.2486) |

---

## 5. Saved models inventory

| Folder | Best metric | Notes |
|--------|-------------|-------|
| `stageD_nodemo_20260428_123448` | 0.0612 (170 steps) | **Best clean swing-up** — primary checkpoint used for fine-tuning |
| `stageD_robust_20260428_143148` | 0.072 (170 steps) | Noise-finetune from above |
| `stageD_expand2_20260428_154524` | varies | Expansion alternation |
| `stageD_stabilize_*` | TBD | Will appear when stabilize3 completes |

---

## 6. Next steps and targets

### Immediate (after stabilize3 finishes)

1. Run `verify_stable.py` to confirm the saved checkpoint truly holds
   the upright at 300/400/600 steps.
2. If wrap_dist < 0.3 at 600 steps → **stability problem solved**.
3. If wrap_dist still > 0.3, fine-tune again from the new checkpoint with:
   - smaller LR (5e-5)
   - higher `w_stable_phase` (50)
   - longer `stable_phase_steps` (80, with `NUM_STEPS=250`)

### Medium-term — generalisation (after stability)

4. Apply the stable model as a starting point for `exp_mirror.py`
   (mirror x0 augmentation).  Stability + mirror should give symmetric
   swing-up that also holds.
5. Try `exp_sincos.py` from scratch — sin/cos encoding gives structural
   symmetry without needing mirror training data.  q1=+π and q1=-π map to
   identical inputs by construction.

### Long-term — robustness in deployment

6. Combine stable + sincos + noise injection: final model that swings up
   from any initial condition under noise and stays at the top.
7. Test "perturbation while at goal" scenario: poke the pendulum after it
   has stabilised and check it recovers.

### Targets

| Property | Current | Target |
|----------|---------|--------|
| Clean goal_dist @ 170 steps | 0.0612 | ≤ 0.05 |
| Clean wrapped_dist @ 600 steps | ~0.6 (oscillating) | ≤ 0.3 |
| Symmetric initial condition success | 3/7 | 7/7 |
| Sensor noise robustness | σ_q ≤ 0.10 | σ_q ≤ 0.20 |

---

## 7. Key files for handoff

- `RESULTS_SUMMARY.md` — original results doc (pre-stability work)
- `HANDOFF.md` — this file
- `Simulate.py` — training loop with all loss terms
- `lin_net.py` — `LinearizationNetwork` (default) + `LinearizationNetworkSC` (sin/cos)
- `mpc_controller.py` — differentiable MPC

To continue work:

```bash
# Verify the latest stabilize checkpoint
python verify_stable.py

# If stable, kick off mirror training from the new checkpoint
# (edit PRETRAINED in exp_mirror.py to point at stageD_stabilize_*)
python exp_mirror.py

# Or try sin/cos from scratch
python exp_sincos.py
```
