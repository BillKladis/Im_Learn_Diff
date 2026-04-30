# Double-Pendulum Swing-Up — HANDOFF

**Branch:** `claude/pendulum-mpc-neural-network-csqVr` (will be made main)
**Status: holding-at-top NOT yet solved.** Multi-day session of attempts. This document is a complete picture for whoever continues.

---

## TL;DR

The pipeline reliably swings the pendulum **near** upright but doesn't actually **stay** there. The pendulum continuously **oscillates** or even **swings around** through the upright position. We have one model (`stageD_nodemo_qf50_20260429_111711`, "qf50 v2") that achieves 12/12 on a perturbation grid AND 35/35 on a wider grid — but those grids only measure "did it touch the loose zone wrap<0.3" not "did it hold". A detailed trace showed qf50 v2 spends **0%** of time with `wrap < 0.10` after arrival. Same for stab_state (best previous "real" hold: only 3.3% wrap<0.1). Multiple attempts to add hold-quality losses (w_stable_phase, w_f_stable, w_f_pos_only, w_hold_reward, distillation) either break the swing-up or get stuck in pre-pump phase.

---

## 1. THE GOAL — what success looks like

The user's hard requirement:

> "Right now the most important thing is to make sure that we are not oscillating at the top and we are staying there."

A successful model:
- Reaches `q1 ≈ π, q1d ≈ 0, q2 ≈ 0, q2d ≈ 0` from `x0 = [0, 0, 0, 0]` once (not endlessly cycling)
- **Stays** there: `wrap_dist < 0.1` for tens of seconds (i.e. > 200 contiguous steps at dt=0.05)
- Generalizes: holds across `q1 ± 1.0+`, `q1d ± 2.0+`, perturbed q2/q2d
- Is **reproducible**: training the recipe on a fresh seed produces a similar model (currently it does NOT)

**Hold metric** to track during training and eval:
- `fraction wrap<0.10` over the post-arrival 1000-step window (THIS is the real hold quality; > 50% = good)
- NOT `total time in zone (wrap<0.3)` — that misleads, see below
- NOT `goal_dist@training_horizon` — that's a single-step snapshot, easily fooled by passing-through

---

## 2. CRITICAL DON'T — no Q-only swing-up

**We do not want the network to do the swing-up via `gates_Q` only with `f_extra ≈ 0`.**

What we keep observing: under the new losses with state-pin / hold-reward, the network finds a degenerate "local minimum" where `f_extra ≈ 0.001` (kickstart bias frozen) and the entire swing-up is driven by `gates_Q + Qf` shaping the QP's optimal control. This:
- Was observed in `combinedv3`, `simple_pin`, `pin_warmup` runs (all stuck at `fnorm=0.001` for tens of epochs)
- Produces a "swing-up" that the QP can't sustain past the training horizon (rollout doesn't reach goal in 600-step eval despite low track loss in 300-step training)
- Doesn't generalize — the gates-only policy is fragile because the QP only plans 10 steps ahead

The original 0.0612 baseline (`exp_no_demo.py`) had `fnorm` growing to ~5-8 — the network learned a real `f_extra` pumping pattern. **That's the path we want to preserve.** New losses must not "starve" the gradient to f_head.

Workaround for next attempts:
- Keep loss weights MUCH smaller initially (warmup) so the swing-up develops first
- OR explicitly initialize `f_kickstart_amp=1.0` (sinusoidal pumping bias) so f_extra starts non-zero
- OR add an explicit `w_f_active` reward for `||f_extra||` magnitude in the early phase

---

## 3. WHAT WORKED (partial successes — build on these)

| File | Recipe | What it gave us |
|------|--------|-----------------|
| `exp_no_demo.py` | track-energy + q_profile + end_q_high (170 steps, Qf default) | The 0.0612 baseline. Reaches goal at step 167. Doesn't hold. **Real f_extra-based swing-up.** |
| `exp_stab_state.py` | 0.0612 + `w_f_stable=50` (state-conditional f_extra suppression near goal, with VELOCITY gating) | Held for 74 contiguous steps (3.7s) — best 'real hold' so far. Still oscillates eventually. Fine-tune from 0.0612. |
| `exp_no_demo_qf50.py` | 0.0612 + `Qf q1d=20→50` | qf50 v2: 12/12 on q1±0.5 / q1d±1, 35/35 on wider. But oscillates THROUGH the upright (full revolutions per the trace). Wide basin, no real hold. |

**Best holding-quality model:** `saved_models/stageD_stabstate_20260428_224856/` (3.3% wrap<0.1 — small but the only nonzero we've seen).

---

## 4. WHAT DIDN'T WORK (don't waste cycles re-trying these)

| Attempt | Result | Reason |
|---------|--------|--------|
| Phased fine-tune (Phase 1 = 0.0612, Phase 2 = +hold reward) | Phase 1's swing-up policy is in a SHARP local minimum. ANY gradient step in Phase 2 (any new loss component) destroys arrival on iter 1. Tried with multiple hold-reward magnitudes, multiple LRs, with/without warmup. | Loaded policy too sharp; can't fine-tune. |
| Joint training with hold_reward from epoch 1 | Network never starts pumping (fnorm=0.001 forever). | Hold reward gradient is absent (pendulum never reaches goal) but other terms hold f_extra near zero. |
| `Qf q1d` sweep (20, 40, 50, 60, 80) | Only `Qf q1d=20` (default) and `q1d=50` work. Other values give weak or non-existent holds. | Sharp optimum in Qf space, plus our seed-fragility. |
| Sin/cos input encoding (`LinearizationNetworkSC`) seed 10 | best=2.008 (poor swing-up, didn't converge in 60 epochs) | Different input dim (30 vs 20) shifts loss landscape; same seed-fragility. |
| Wider net (hidden=256) | Stuck at fnorm=0.001 for 20+ epochs. | Same as 128-net stuck. |
| High weight decay (1e-2) | Stuck at kickstart for 50 epochs. | Decay > gradient, network can't move. |
| SAM-approximation (weight noise injection) | Slow, no advantage observed before kill. | |
| State-conditional distillation (`w_distill_goal`) | Suppressed f_extra everywhere → no swing-up. | Distillation regularizer too aggressive. |
| `w_stable_phase=200` (max state-pin) | Loss landscape dominated by stable_phase, network stuck. | Single-loss tyranny. |
| Multi-seed sweep (seeds 0, 1, 10, 20, 30, 40, 50) | All produced 0/6 generality. qf50 v2 was a lucky default-RNG seed. | Recipe is fundamentally seed-fragile. |
| Trajectory curriculum (fine-tune from successful trajectory states) | Bug-fixed but still didn't transfer well. | After bug fixes, gradient too small. |

---

## 5. THE TWO BUGS WE FIXED (and they're real)

Both in `train_linearization_network` (Simulate.py):

**Bug 1: Optimizer reset.** `train_linearization_network` created a fresh `AdamW` every call. With curriculum-style code that calls it many times with `num_epochs=1`, AdamW's momentum (m, v) reset every call → cold-start steps with no adaptation → essentially zero cumulative learning despite gradients being computed.
- Fix: `external_optimizer` parameter. If provided, the function uses your optimizer instead of creating one.

**Bug 2: `restore_best` rollback.** At end of function, `lin_net.load_state_dict(best_state_dict)` was called UNCONDITIONALLY. `best_state_dict` was captured **before** `optimizer.step()` each epoch. With `num_epochs=1`, the function:
1. Computes loss → backward
2. Saves PRE-step weights as 'best'
3. Steps optimizer (weights change)
4. Reverts to PRE-step weights at exit
- Result: every gradient step in curriculum-style use was a NO-OP for the network parameters. Optimizer state advanced (Adam's `step=1, exp_avg=0.015`) but params unchanged.
- Fix: `restore_best=False` parameter. Set to False for curriculum-style use; default True for one-shot training.

Verified by `diag_grad.py` (sum-of-max-param-diffs went from `0.0` → `2.4e-3`, matching expected Adam step magnitude).

**Also a print/eval bug** in monitor classes: each `train_linearization_network` call has internal `epoch` start at 0, so `epoch == 0` always fires. Use external counter for outer-iteration tracking.

---

## 6. KEY INSIGHTS FROM TRACING

**`trace_qf50.py` and `trace_stabstate.py`** dump the full 2000-step rollout's wrap-distance time series. They revealed:

- **qf50 v2** (the "winner"): pendulum goes through FULL revolutions. Sample wrap values at successive snapshots (50-step intervals): `0.27, 0.29, 6.70, 6.86, 0.39, 0.66, 0.92, 1.54, 1.18, 0.35, 2.37, 3.33, 0.75, ...`. Energy at snapshots oscillates from ~9 (bottom) to ~15 (upright). `q1` values like `9.42, -3.93` indicate multi-revolution wrapping. **0% time with wrap<0.10**, 11.6% with wrap<0.30.

- **stab_state**: similar oscillation pattern but smaller amplitude. **3.3% time with wrap<0.10**, 8.6% with wrap<0.30.

- The "12/12 generality" and "35/35 boundary" used `total time in zone` (wrap<0.3 ≥ 50 steps over 1000-step rollout) which is satisfied by repeated PASSES through the goal. Misleading.

---

## 7. THE STATE-CONDITIONAL `w_f_stable` TRAP (asymmetric in velocity)

`w_f_stable` is currently:
```
stable_zone = ((1 + cos(q1 - π)) / 2) * exp(-(q1d² + q2d²) / 2)
loss += w_f_stable * stable_zone * ||f_extra||²
```

The `exp(-v²/2)` kills the penalty when velocity is high. So during a swing-through (high velocity at upright position), the penalty is ~13% of full at v=2 rad/s. **The network can keep pumping during pass-through with little penalty.** That's the oscillation mechanism.

Already added `w_f_pos_only` (position-only gate, no velocity factor) in Simulate.py but didn't yet validate.

---

## 8. CURRENTLY RUNNING (none — system rebooted at 02:44)

System rebooted overnight. ALL training processes died. Last log states:

- `simplepin` (NUM_STEPS=300, w_stable_phase=30 only) — was at ep 45 with `track=0.368` after slow steady drop from 0.670. fnorm still 0.001 (gates-Q-only path — the BAD path). Did not save.
- `pinwarmup` (0.0612 phase 0-29, then add stable_phase=30) — was at ep 35 with `track=0.093` (great energy fit) after pin engaged. fnorm=0.001 (also gates-Q-only). Did not save.
- `combinedv3` killed earlier; `bigger`, `sincos`, `seedsweep`, etc. all parked.

---

## 9. ARCHITECTURE / CODE STATE

### Modified files (preserve these — they have real fixes)

- `Simulate.py` — many new params: `train_noise_sigma`, `w_f_end_reg`, `w_stable_phase`, `w_f_stable`, `w_f_pos_only`, `w_qf_profile`, `w_track_late_phase`, `w_distill_goal`, `w_hold_reward`, `init_history`, `external_optimizer`, `restore_best`, `early_stop_patience`. ALL are backward-compatible (zero defaults).
- `lin_net.py` — added `qf_head` (default disabled via `gate_range_qf=0`), added `LinearizationNetworkSC` (sin/cos encoding). All loaders use `strict=False` so legacy checkpoints still work.
- `mpc_controller.py` — added optional `diag_corrections_Qf` parameter. None default → exact legacy behavior.

### Hard constraints (preserved through all this)

- Fixed horizon length (HORIZON=10)
- No trajectory in controller
- All physics-informed signals in OUTER LOSS
- Controller stays general (Qf is a config knob like `q_base_diag` is)
- Network is the only learned component
- No inference-time hacks

---

## 10. SAVED MODELS WORTH LOADING

| Folder | Status |
|--------|--------|
| `stageD_nodemo_20260428_123448/` | The 0.0612 baseline. Real f_extra swing-up. Doesn't hold. |
| `stageD_nodemo_qf50_20260429_111711/` | qf50 v2. Wide perturbation basin. Oscillates through goal (full revolutions). |
| `stageD_stabstate_20260428_224856/` | Best partial hold (3.3% wrap<0.1). Fine-tuned from 0.0612 with `w_f_stable=50`. |

---

## 11. NEXT DIRECTIONS (in priority order)

1. **First priority: fix HOLDING.** Need a model that has > 50% time wrap<0.10 in a 1000-step rollout from x0=zero. Once that exists, generality and reproducibility can follow.

2. **Avoid the gates-Q-only trap.** Whatever loss is added must keep f_extra growing. Suggested: explicit reward `+w * mean(||f_extra||)` during the swing-up phase (steps 0-170) so the network has positive incentive to use f_extra.

3. **Re-think the demo.** Current demo: q1 ramps 0→π, then holds. Maybe demo should INCLUDE a short braking phase (q1d non-zero approaching π, then ramping to 0) so the network has a smooth hand-off from pumping to braking in the energy curve.

4. **Periodic checkpointing.** Add a `save_every_N_epochs` to training loops so a system reboot doesn't lose hours of work.

5. **Lower the bar gradually.** Train for "hold for 50 steps" first (easier), then "hold for 200 steps", then "hold for 1000 steps".

6. **Reproducibility (after holding works).** Multi-seed sweep with the working recipe. SAM or weight averaging if seeds vary.

7. **Boundary characterization** of any holding model.

---

## 12. RUNNING / VERIFICATION COMMANDS

```bash
# Verify any model's actual hold quality (not just touch-passes)
# (modify PRETRAINED in the file)
python trace_qf50.py
python trace_stabstate.py

# Test perturbation generality (wider grid)
python test_boundary.py
python test_generality.py

# Train the 0.0612 baseline (proven swing-up; ~10 min)
python exp_no_demo.py
```

---

## 13. THE ONE THING TO REMEMBER

**Track loss (energy) and gates-Q can both be satisfied without the pendulum actually holding still at upright.** Energy is symmetric (kinetic vs potential trade-off — the pendulum can have any energy at any q1 given matching q1d). Gates are state-shaping but the QP only plans 10 steps ahead.

Without a `f_extra`-driven swing-up (which is open-loop, time-varying torque pattern that the network commits to), the QP alone cannot reliably plan a multi-second swing-up trajectory.

**Future losses must not starve the f_extra pathway.** That's the real constraint.
