# Double-Pendulum Swing-Up — Handoff

**Branch:** `claude/pendulum-mpc-neural-network-csqVr`
**Status:** **Holding-at-top is unsolved.** Both qf50 v2 and stab_state were found to OSCILLATE through the upright (qf50 v2: 0% wrap<0.1; stab_state: 3.3%). Neither truly "holds". The previous "12/12 generality" and "35/35 boundary" results measured time-in-loose-zone (wrap<0.3), which the pendulum hits often when continuously swinging through.

---

## 1. CURRENT FOCUS: real holding at the top

The user's hard requirement: pendulum must STAY upright after swing-up, not oscillate through it. Nothing trained so far does this reliably.

**Currently running (parallel):**
- `exp_combined_hold.py` (PID 29759) — Qf q1d=50 + w_f_stable=50 + w_stable_phase=30 in last 130 steps
- `exp_combined_v2.py` (PID 31581) — same + w_stable_phase=80 (stronger pin)

Combines every ingredient that contributed to a partial holding success.

**Hold metric:** fraction of post-arrival time with `wrap < 0.1`. This was 0% for qf50 v2, 3.3% for stab_state.

---

## 2. Past partial successes (BUILD FROM THESE)

| File | Key ingredient | Effect on holding |
|------|----------------|-------------------|
| `exp_no_demo.py` (0.0612 baseline) | track + q_profile + end_q_high | clean swing-up; arrives but doesn't hold |
| `exp_stab_state.py` (`w_f_stable=50`) | state-conditional f_extra suppression near goal | longest contiguous=74 (3.7s) — best 'real' hold so far |
| `exp_no_demo_qf50.py` (`Qf q1d=50`) | terminal velocity brake | wider perturbation basin, 35/35 OK at total≥50 — but 0% true holding |

The right fix is COMBINING these (currently testing).

---

## 3. Confirmed bugs (all fixed)

1. **Optimizer-reset bug** — `train_linearization_network` created fresh AdamW every call, killing momentum across `num_epochs=1` curriculum loops. Fix: `external_optimizer` parameter.
2. **`restore_best=False` bug** — function was rolling back to PRE-step weights at exit. Fix: `restore_best=False`.
3. **Eval-cadence bug** in monitors — `epoch == 0` always True inside the train fn (each call has num_epochs=1). Fix: track outer-iter count externally.

`diag_grad.py` and `diag_minimal.py` verify the fixes.

---

## 4. Hard constraints (preserved throughout)

| Constraint | Status |
|------------|--------|
| Fixed horizon length (HORIZON=10) | ✓ |
| No trajectory in controller | ✓ |
| Physics-informed signals in OUTER LOSS | ✓ |
| Controller stays general | ✓ (Qf is config) |
| Network is the only learned component | ✓ |
| No inference-time hacks in production | ✓ |

---

## 5. Active files

| File | Role |
|------|------|
| `lin_net.py` | LinearizationNetwork + qf_head + LinearizationNetworkSC |
| `Simulate.py` | Training loop. Many params added this session. |
| `mpc_controller.py` | MPC. Optional `diag_corrections_Qf` param. |
| `exp_no_demo.py` | The 0.0612 baseline recipe |
| `exp_no_demo_qf50.py` | qf50 recipe (Qf q1d=50) |
| `exp_stab_state.py` | w_f_stable=50 recipe |
| `exp_combined_hold.py` | **All-in: qf50 + w_f_stable + w_stable_phase** |
| `exp_combined_v2.py` | Same + stronger w_stable_phase |
| `trace_qf50.py`, `trace_stabstate.py` | Detailed wrap-distance trace |
| `test_generality.py`, `test_boundary.py` | Perturbation tests |
| `probe_qf.py` | Qf grid-search, no retraining |
| `audit_history.py` | Verifies init_history seeding |
| `diag_grad.py`, `diag_minimal.py` | Found the fixed bugs |

---

## 6. Saved working models

| Folder | Real hold? | Notes |
|--------|------------|-------|
| `stageD_nodemo_20260428_123448/` | **No** (12 contiguous steps only) | Original 0.0612 baseline. Useful as Phase 1. |
| `stageD_stabstate_20260428_224856/` | **Partial** (3.3% wrap<0.1) | Best 'real' hold so far. |
| `stageD_nodemo_qf50_20260429_111711/` | **No** (0% wrap<0.1, 35/35 swing-through) | Wide basin but oscillates through goal. |

**No model truly holds yet.** That's the open task.

---

## 7. Next steps (in order of likely impact)

1. **Wait for combined / combined_v2 results.** If either gives wrap<0.1 fraction > 50%, holding is solved.
2. **If still oscillating:** the network's swing-up timing is the issue. Try training with a track_mode that includes velocity (or full state, not just energy).
3. **If still oscillating:** add an explicit "no swing-through" loss — penalty when q1 crosses past π with nonzero velocity in the late phase.
4. **Reproduction concerns:** the qf50 recipe is seed-fragile. Wait for hold to be solved first, then run multi-seed sweep.

---

## 8. User's noted ideas (for later)

- LR-smash on plateau: when finding a good minimum, drop LR very low to refine within the sharp minimum
- Curriculum / sequential loss combination

---

## 9. Commit trace

| Commit | What |
|--------|------|
| `211578b` | combined v2 (stable_phase=80) |
| `3d52b49` | exp_combined_hold + traces revealing oscillation |
| `95bb539` | real_hold + overnight seedsweep |
| `2885e15` | qf50 v2 boundary 35/35 (later found illusory) |
| `7f040f8` | exp_no_demo_qf50 — the qf50 recipe |
| `4c40ad6` | THE BIG FIX: restore_best parameter |
| `f5464bd` | persistent optimizer fix |
| `cbfe712` | w_f_stable state-conditional penalty |
| `7581f08` | original 0.0612 baseline |
