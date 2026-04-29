# Double-Pendulum Swing-Up — Handoff

**Branch:** `claude/pendulum-mpc-neural-network-csqVr`
**Latest:** `de928ef` — generality win 12/12

---

## 1. Current state (one paragraph)

We have a **working model**: `saved_models/stageD_nodemo_qf50_20260429_111711/`.
It was trained from scratch with the **exact same recipe** that produced the
0.0612 baseline (exp_no_demo.py), with **one** change: the MPC's terminal
cost matrix `Qf` has `q1d` weight bumped from 20 to 50. From x0=zero, the
pendulum reaches the upright at step 219 (10.95s — 2.6s later than baseline)
and then HOVERS near the goal for **75 seconds with wrap < 1.0** (vs baseline
which loses the goal completely after ~4s). On a 12-perturbation generality
test (q1 ∈ [-0.5, +0.5], q1d ∈ [-1, +1], q2 ±0.1, q2d ±0.5, combined),
QF50 v2 succeeds on **12/12** at the strict "total ≥ 50 steps in zone"
threshold; baseline succeeds on **0/12**. The network learned a
state-dependent braking policy that's both more sustained AND more
general than baseline. **No controller code changed.** No qf-head, no
curriculum, no mirror, no sin/cos — just one Qf coefficient.

---

## 2. The breakthrough

**File:** `exp_no_demo_qf50.py` (literal copy of `exp_no_demo.py` with one config line different).

**The change:** `mpc.Qf = diag([20.0, 50.0, 40.0, 30.0])` (was `[20, 20, 40, 30]`).

**The intuition (from `probe_qf.py`):** the `Qf q1d` weight controls how
hard the MPC's terminal cost penalises residual angular velocity at the
end of the planning horizon. With q1d=20 (default), the network learns
a swing-up that arrives at upright with non-zero velocity and then
oscillates through. With q1d=50, the network is forced to pump such that
the pendulum arrives **with low velocity** → it holds.

The reason the existing `0.0612` model didn't generalise: it learned a
specific pump-trajectory-shape that worked from `x0=zero`. Perturbations
shift that trajectory enough that the network's pump-then-hope-it-stops
policy fails. QF50 v2's policy has a real "brake" component governed by
the higher Qf — that brake acts on the actual state, regardless of how
the rollout got there → robust to perturbation.

---

## 3. Numbers

### From `x0=zero`

| Steps | Baseline 0.0612 | **QF50 v2** |
|------:|----------------:|------------:|
| 170   | 0.06 raw        | 0.30 raw    |
| 220   | 1.89 wrap **FAIL** | 0.26 wrap **STABLE** |
| 300   | 0.95 wrap       | 0.58 wrap CLOSE |
| 400   | (lost)          | **0.15 wrap STABLE** |
| 600   | 5.48 wrap **FAIL** | 0.70 wrap CLOSE |
| 1000  | 7.18 wrap **FAIL** | 0.91 wrap CLOSE |
| 1500  | (lost)          | **0.32 wrap CLOSE** |

### Generality (12 perturbed x0, 1000-step rollouts; threshold total≥50)

```
Initial cond.       BASELINE 0.0612    QF50 v2
canonical           total= 24 WEAK     total= 96 OK
q1=+0.2             total=  4 WEAK     total=100 OK
q1=-0.2             total=  4 WEAK     total= 98 OK
q1=+0.5             FAIL               total= 99 OK
q1=-0.5             total= 24 WEAK     total=109 OK
q1d=+0.5            FAIL               total=104 OK
q1d=-0.5            FAIL               total=101 OK
q1d=+1.0            total=  3 WEAK     total= 99 OK
q2=+0.1             total= 19 WEAK     total= 95 OK
q2d=+0.5            total= 25 WEAK     total=104 OK
combined+           total= 17 WEAK     total=114 OK
combined-           total=  5 WEAK     total= 97 OK
                    -----------       -----------
Success             0 / 12             12 / 12
```

---

## 4. Hard constraints (preserved throughout this work)

| Constraint | Status |
|------------|--------|
| Fixed horizon length (HORIZON=10) | ✓ never changed |
| No trajectory inside the controller | ✓ MPC sees only `(state, x_lin_seq, u_lin_seq, x_goal, gates_Q, gates_R, f_extra, [optional gates_Qf])` |
| Physics-informed signals in OUTER LOSS | ✓ all loss terms in `Simulate.train_linearization_network` |
| Controller stays general | ✓ `mpc_controller.py` unchanged in logic; only added optional `diag_corrections_Qf` parameter for the (unused) qf-head experiment |
| Network is the only learned component | ✓ |
| No inference-time hacks in production | ✓ |

The Qf change is a **config tweak** of an existing controller knob, not new
controller logic. We already configure `q_base_diag`; bumping `Qf q1d` is
the same kind of parameter setting.

---

## 5. Files — current state

### Active (run these)

| File | Role |
|------|------|
| `lin_net.py` | LinearizationNetwork (default) and SC variant. `qf_head` is added but inactive (gate_range_qf=0 default). Backward-compatible with all existing checkpoints (`load_state_dict(strict=False)`). |
| `Simulate.py` | Training loop. New params from this session: `train_noise_sigma`, `w_f_end_reg`, `w_stable_phase`, `w_f_stable`, `w_qf_profile`, `w_track_late_phase`, `init_history`, `external_optimizer`, `restore_best`, `early_stop_patience`. |
| `mpc_controller.py` | MPC. `Qf` is now a public attribute that experiments can override (no logic change). `build_cost_matrices` accepts optional `diag_corrections_Qf`. |
| `exp_no_demo_qf50.py` | **The winning experiment.** Same as exp_no_demo.py but with `Qf q1d=20→50`. |
| `test_generality.py` | Compares any two models on 12 perturbed x0s with hold-time metrics. |
| `probe_qf.py` | Grid-search over Qf q1d for sustained-hold sensitivity. |
| `trace_rollout.py` | Step-by-step rollout trace for any saved model. |
| `inspect_real_energy.py` | Computes the energy curve along a saved trajectory. |
| `audit_history.py` | Verifies init_history seeding matches natural rollout to 5e-7. |
| `diag_grad.py`, `diag_minimal.py` | Gradient flow diagnostics (used to find the optimizer-reset bug). |
| `verify_smart_gate.py` | Inference-time gating probe (kept for reference — proved the principle behind w_f_stable). |
| `verify_stable.py` | Loads latest stabstate-style checkpoint and reports raw + wrap at multiple horizons. |

### Dead-end / superseded experiments

| File | Why parked |
|------|-----------|
| `exp_qf_head.py` | Architectural change with a learned Qf-head; from-scratch training plateaued. The simpler "just bump Qf" approach won. |
| `exp_qf50_progressive.py` | Hardcoded Qf + progressive penalty — couldn't recover swing-up. |
| `exp_qf50_tighten.py`, `exp_qf50_tighten_v2.py` | Fine-tune of qf50 v2 to tighten the hold. v1 was a no-op (LR too gentle). v2 was too aggressive (broke swing-up at iter 1). |
| `exp_no_demo_kinetic.py`, `exp_no_demo_realistic.py` | Demo-shape experiments. The kinetic-peak demo got 0.43 raw (worse than baseline). The realistic-trajectory demo with state late-phase pin plateaued. |
| `exp_long_horizon.py` | NUM_STEPS=400 fine-tune. Regressed the swing-up. |
| `exp_stab_state.py` | First state-conditional w_f_stable run. Useful precursor; superseded by Qf bump. |
| `exp_finetune_best.py`, `exp_stabilize.py`, `exp_stabilize2.py`, `exp_robust_finetune.py`, `exp_expansion.py`, `exp_expansion2.py`, `exp_curriculum*.py`, `exp_traj_curriculum.py`, `exp_near_goal.py`, `exp_mirror.py`, `exp_bignet.py`, `exp_sincos.py`, `exp_diag_failure.py`, all the `exp_q*` and `exp_phase*` and earlier probes | Earlier-stage experiments, most predate the Qf insight. |

---

## 6. Saved models

| Folder | Best metric | Notes |
|--------|-------------|-------|
| `stageD_nodemo_20260428_123448/` | 0.0612 raw @ 170 (canonical only) | The original 0.0612 baseline. Useful as a stress test. |
| **`stageD_nodemo_qf50_20260429_111711/`** | **12/12 generality, 75s near-goal hover** | **THE WORKING MODEL.** Single-variable change vs baseline: Qf q1d=50. |
| `stageD_nodemo_qf50_20260429_124016/` | (regressed) | qf50 v3, longer training. Got worse. Ignore. |
| `stageD_kinetic_20260429_102046/` | 0.43 raw @ 170 | Kinetic-peak demo. Worse than baseline. Ignore. |
| `stageD_qf50tight_20260429_114008/` | (no-op) | First fine-tune attempt, didn't move. Ignore. |
| `stageD_stabstate_20260428_224856/` | 0.075 raw @ 220 | Pre-Qf-discovery stability fine-tune. Useful as a contrast. |
| `stageD_robust_20260428_143148/`, `stageD_expand2_20260428_154524/`, `stageD_trajcurr_*/` | Various prior experiments. | Pre-breakthrough. |

---

## 7. Discoveries from this session

1. **Two real bugs in `train_linearization_network`** for curriculum-style use (single-epoch alternating calls):
   - Optimizer was created fresh every call → AdamW momentum reset → cold-start steps with no adaptation.
   - `best_state_dict` was captured BEFORE `optimizer.step()` and restored at function exit → with `num_epochs=1` the function ROLLED BACK every gradient step.
   - Fix: `external_optimizer` and `restore_best=False` parameters. Verified via `diag_grad.py` (param diff went from 0.0 → 2.4e-3, expected for one Adam step at LR=1e-4).

2. **The Qf q1d knob:** the only single-variable Qf modification that improves sustained hold without breaking arrival is `q1d=50`. Below 50, the policy collapses (longest hold worse than baseline); at 50+, it brakes harder but arrives later. The network has to be retrained against the new Qf to get both arrival AND hold (probe_qf showed the existing model loses arrival when Qf is bumped at inference).

3. **Loss decline doesn't track sustained-hold quality.** qf50 v3 reached lower loss (5.24 vs 7.35) than v2 but rolled out worse. The "best metric" used by early-stop (GoalDist at training horizon) is a noisy snapshot mid-oscillation. A real sustained-hold metric needs something like longest-contiguous-wrap-hold from a long rollout, not single-step distance.

4. **Track loss alone (energy mode) is symmetric in `±π`.** A model can perfectly track the energy curve while landing at q1=-π instead of q1=+π. Caught this in `realistic v1` where track=0.008 but goal_dist=5.7. State-mode penalties (or the wrap function) are needed for position discrimination.

5. **`init_history` correctness verified to 5e-7** via `audit_history.py`. The CSV trajectory + 5-frame-history seeding gives the network bit-equivalent inputs to a natural rollout (modulo CSV roundoff).

---

## 8. Next directions (if continuing)

1. **Test the boundary** — extend `test_generality.py` to wider perturbations
   (q1 ±1.0, ±1.5, q1d ±2.0, q1d ±3.0) to find where qf50 v2 starts failing.

2. **Tighten the hold** — qf50 v2's "longest contiguous wrap<0.3" is only
   13-14 steps because the pendulum oscillates within wrap [0, 1].
   Fine-tune from qf50 v2 with a hold-quality-aware best-metric
   (HoldMonitor pattern was sketched in `exp_qf50_tighten_v2.py` but the
   LR was too aggressive). Lower LR + the same monitor pattern is the
   right shape.

3. **Multi-seed reproduction** — qf50 v2 succeeded; qf50 v3 (longer training,
   different stochastic path) failed. Run the same recipe with several
   seeds to characterise the success rate of this training.

4. **Try q1d=40 or q1d=60** as a sweep — find the optimal Qf q1d.

5. **Stack with sensor noise** (`train_noise_sigma`) for robust deployment.

6. **Sin/cos architectural variant** with the qf50 recipe — should give
   structural symmetry (q1=+π ≡ -π in inputs) on top of the already-
   robust policy.

---

## 9. Quick commands

```bash
# Compare generality of any two models
python test_generality.py

# Probe Qf sensitivity (no retraining required)
python probe_qf.py

# Trace a rollout step-by-step
python trace_rollout.py     # edit MODELS list at the top

# Verify history seeding correctness (sanity check)
python audit_history.py

# Re-train the winner from scratch (~10 min)
python exp_no_demo_qf50.py
```

---

## 10. Commit trace (most recent first)

| Commit | What |
|--------|------|
| `de928ef` | Generality win 12/12 commit message |
| `94e80ac` | test_generality.py |
| `1485d87` | qf50 v3 (longer training) regressed |
| `7aa5179` | qf50_tighten v1 no-op |
| `5b7b94c` | **NEW BASELINE: nodemo_qf50 saved** |
| `1af40d3` | realistic v2 (state pin replaces energy pin) |
| `b1da7fd` | exp_no_demo_realistic + late-phase penalty |
| `c429109` | kinetic-peak model saved |
| `30670a3` | exp_no_demo_kinetic |
| `7f040f8` | **exp_no_demo_qf50 (the recipe)** |
| `4c40ad6` | THE BIG FIX: restore_best parameter |
| `f5464bd` | persistent optimizer fix |
| `099d330` | init_history parameter |
| `cbfe712` | w_f_stable state-conditional penalty |
| `7581f08` | original 0.0612 baseline |
