# Double-Pendulum Swing-Up — HANDOFF

**Branch:** `claude/continue-handoff-work-RhI5l`
**Status (2026-05-01 session 11): RECORD 87.3% stands. Gate grid remaining in progress. v7 training. v9/v10 input bug FIXED and relaunched.**

**RECORD**: thresh=0.850 natural diagonal → **87.3%** (f01), arr=242, post=99.3%
**CONFIRMED BY**: threshold sweep, gate grid, AND scale=5.0 eval — all peak at 87.3%

---

## SESSION 6 LATEST (continued past context window)

### UNIVERSAL FAILURE PATTERN — CONFIRMED ACROSS 5 EXPERIMENTS

| Experiment | Why it failed | Result |
|---|---|---|
| dual-thresh(0.60) | wider zone → earlier boost | arr=322 (baseline 242) → 83% |
| scale=4× (global) | Q[q1]=76 everywhere | 0% (known) |
| Q-max warmup (v4,v5) | Q[q1]=23.8 everywhere after warmup | 0% both |
| posgate (smooth) | α=2.27 at q1=90° (60° from top) | 0% |

**IRON LAW**: Any Q[q1] boost that activates at q1=90° or earlier destroys swing-up. The swing-up energy-pumping trajectory REQUIRES low Q[q1] outside |q1-π| < ~53° (en < 0.8).

The 87.2% model's hardcoded threshold (error_norm < 0.80) is near-optimal. The learned gate must MATCH or SHARPEN this threshold.

### POSGATE FINAL AUTOPSY (exp_posgate.py, PID 12563 — KILLED)

Supervised pre-train produced SMOOTH activation profile:
- q1=60°: Q_adj[q1]=1.078 (activating during swing-up!)
- q1=90°: Q_adj[q1]=2.271 (half the full boost at exactly the wrong time)
- Only matches wrapper at q1=150°+

Root cause: pre-train had only TOP (q1≈π) and BOTTOM (q1≈0) examples. The MLP interpolated smoothly between them, activating at all intermediate angles. No negative examples in the 60°-127° zone.

Initial eval: 0.0% (catastrophic). Training loop started but killed — training from 0.0% is very slow and the root cause is pre-training design.

### v4/v5 FINAL AUTOPSY (exp_stageE_alternating.py — BOTH KILLED)

Both got 0.0% because Q-max warmup made gates_Q[q1]=1.9851 GLOBALLY (all states). The stability test (lr=1e-4) showed warmup is TOO STABLE — bottom states can't recover their original low Q[q1].

Shared q_head weights = global effect. No state-conditional behavior possible.

### NEW APPROACH: SCALEGATE (exp_scalegate.py)

User said: "Maybe the network can learn to scale a single scalar. That thing that created the switch instead of hard coding it in."

Design:
- Single scalar α ∈ [0,1] scales the FIXED dQ direction (Q_adj = α × dQ_ref)
- lin_net fully frozen
- Only 57 parameters (5→8→1 with Sigmoid)
- Supervised pre-train with BALANCED sampling:
  - Active examples: error_norm < 0.80 (near-top, q1≈π±25°, small velocities)
  - Inactive examples: bottom states + INTERMEDIATE states (60°-127° from top)
  - Binary cross-entropy loss (clean, binary targets)

**v1 gate profile after 1000-step pre-train (CORRECT):**
```
q1=  0°  cos=-1.000  en=3.142  wrapper=off  α=0.0000  ✓
q1= 60°  cos=-0.500  en=2.094  wrapper=off  α=0.0000  ✓ (posgate was 1.078!)
q1= 90°  cos=+0.000  en=1.571  wrapper=off  α=0.0000  ✓ (posgate was 2.271!)
q1=120°  cos=+0.500  en=1.047  wrapper=off  α=0.0542  ✓ (nearly off)
q1=127°  cos=+0.602  en=0.925  wrapper=off  α=0.3657  ~ (transition zone)
q1=150°  cos=+0.866  en=0.524  wrapper=ON   α=0.9944  ✓
q1=180°  cos=+1.000  en=0.000  wrapper=ON   α=0.9994  ✓
```

**v2 design** (exp_scalegate_v2.py): error_norm as DIRECT INPUT → 2-input gate [error_norm, cos(q1-π)].
More trivially learnable (threshold is literally on one feature). Sharper profile:
- q1=127° α=0.146 (vs 0.366 for v1)
- q1=150°+ α=0.999

### SESSION 7: SCALEGATE v3/v4 DESIGN + RUNNING

**Scalegate v1 autopsy (arr=321):** α=0.366 at q1=127° — sigmoid too soft, disrupts swing-up identically to dual-thresh(0.60).

**Scalegate v2 autopsy (arr=237, post=93.9%):** Better (α=0.063 at 127°), but gate saturated (α≈0 or ≈1 at all training states) → zero gradient → training stuck at 82.8%. Wrong ramp shape vs wrapper (sigmoid too steep in wrong region).

**KEY INSIGHT (why 87.2% wrapper works):**
The wrapper uses a LINEAR RAMP (not binary step): `gate = ((near_pi - 0.80) / 0.20).clamp(0,1)`
- At q1=127° (near_pi=0.801): gate=0.005 ≈ 0 (just barely on)
- At q1=140° (near_pi=0.883): gate=0.415 (partial)
- At q1=150° (near_pi=0.933): gate=0.665 (partial)
- At q1=180° (near_pi=1.000): gate=1.000 (full)

This gradual ramp PREVENTS saturation and allows gradient flow throughout the ramp region.

**Scalegate v4 design (exp_scalegate_v4.py) — RUNNING PID 23979:**
- α = (w × near_pi + b).clamp(0,1), 2 learnable params
- Init: w=5.0, b=-4.0 → EXACTLY the 87.2% wrapper formula
- Uses full (9,4) dQ_ref + (10,2) dR_ref from SCALE4_CKPT (both buffers)
- lin_net frozen, 200 epochs, LR=1e-2

CONFIRMED: Initial eval = 87.2% (arr=242, post=99.1%) — exact replication of wrapper.

v4 ep=10 result:
- thresh moved from 0.800 → 0.7753 (training pushes threshold LOWER)
- arr=240 (improved), post=98.8% (slightly worse), f01=87.0% (below 87.2%)
- Training gradient: both top-start and bottom-start say "make α larger → lower threshold"
- This is a fundamental mismatch: training loss ≠ f01 metric

Training dynamics: gradient always pushes toward wider activation (lower thresh) because:
1. Top-start: "hold better → α larger → lower thresh"
2. Bottom-start approach zone: "reach π → α larger → lower thresh"
Neither provides countergradient for maintaining arr quality.

v4-high (KILLED before first result): init thresh=0.85 — killed to free CPU for static evals.
v4 (KILLED after ep=10): threshold drifted 0.800→0.7753, f01=87.0% (degraded from 87.2%).

**Threshold sweep results (exp_thresh_upper_sweep.py — COMPLETE):**
| thresh | zone  | f01     | arr | post   | notes |
|--------|-------|---------|-----|--------|-------|
| 0.750  | 60.0° | 82.3%   | 239 | 93.4%  | wide → faster arr, terrible hold |
| 0.800  | 53.1° | 87.2%   | 242 | 99.1%  | prior baseline |
| 0.825  | 49.5° | 87.2%   | 242 | 99.2%  | marginal improvement |
| 0.850  | 45.6° | **87.3%★** | 242 | 99.3%  | PEAK — natural diagonal optimum |
| 0.875  | 41.4° | 87.2%   | 241 | 99.1%  | drops off |
| 0.900  | 36.9° | 87.2%   | 242 | 99.1%  | plateau |
| 0.925  | 31.8° | 86.8%   | 238 | 98.5%  | cliff — earlier arr but WORSE post |
| 0.950  | 25.8° | 83.0%   | 237 | 94.1%  | severe degradation |

KEY FINDING: Clear peak at thresh=0.850. Above that, high-threshold gates concentrate activation 
very close to goal — this gives slightly EARLIER arrival (arr 242→237) but drastically worse hold 
(post 99.3%→94.1%). The narrow activation zone can't sustain the hold.

At thresh=0.925/0.950: the gate activates like a sharp impulse at near_pi=0.93+ which is too 
late for gradual Q-cost buildup. Post drops because the hold isn't established gradually.

**Gate grid (exp_gate_grid_eval.py, PID 10134)**: Fixed torch.no_grad() bug → restarted. Warmup 
confirmed 87.2% ✓. Now running 14 (w,b) off-diagonal configs. First result (baseline BASELINE 
w=5,b=-4): 87.2% ✓. Results still coming in (~12 min per config).

**dQ/dR shape correction (critical fix):**
SCALE4_CKPT stores:
- best_delta_Q: shape (9,4) — per-step values, q1d dim varies [4.35, 3.56, ..., 4.36]
- best_delta_R: shape (10,2) — per-step values, ≈±0.41

Prior scalegate v1/v2/v3/v4 (before fix) were using mean(dQ) → (4,) which loses per-step information.
Fixed in v4 and threshold sweep to use full tensors.

### MATHEMATICAL CEILING ANALYSIS

**Key insight**: With current setup (fixed lin_net + dQ_ref from SCALE4_CKPT), 87.2% is NEAR the theoretical max:

Best achievable f01 = (2000 - min_arr) × max_post / 2000

From experiments:
- min_arr observed = 239 (thresh=0.75, wide gate helps approach)
- max_post observed = 99.2% (thresh=0.825)
- These occur at DIFFERENT thresholds → can't combine easily

**Upper bound**: (2000-239) × 0.992 / 2000 = 1761 × 0.992 / 2000 = 87.3%

This means **87.3% is approximately the theoretical maximum** for the current lin_net/dQ_ref combo.
The current 87.2% is essentially at the ceiling — improvements are at most 0.1%.

For meaningful improvement beyond 88%:
1. **Reduce arr from 242 → ≤200**: Requires faster swing-up controller (new lin_net training)
2. **Different dQ direction**: Retrain dQ_ref for a different threshold (expensive)
3. **Hybrid gates**: Maybe v6/v7 can decouple arr improvement from post degradation

**Theorem**: With arr=242 and ANY post (even 100%), f01 ≤ 87.9%. To reach 90%+, need arr≤200.

### RUNNING EXPERIMENTS (session 11)

| PID | Script | Status | Notes |
|-----|--------|--------|-------|
| 2470 | eval_grid_remaining.py | RUNNING — 2/5 done | /tmp/gate_grid_remaining.log |
| 2471 | exp_scalegate_v7.py | RUNNING — training started | velocity-aware 3 params |
| 18638 | exp_scalegate_v9.py | RUNNING — imitation pretraining | FIXED input bug |
| 19001 | exp_scalegate_v10.py | RUNNING — imitation pretraining | FIXED input bug |

Gate grid remaining results so far (2/5):
  w=8.000 b=-6.400: 87.2%  arr=243  post=99.2%  [steeper at thresh=0.800 — no improvement]
  w=10.000 b=-8.000: 87.1%  arr=243  post=99.1%  [steepest at thresh=0.800 — worse]
  w=8.000 b=-6.240: TBD  [thresh=0.780 wide+steep]
  w=8.000 b=-6.000: TBD  [thresh=0.750 widest steep]
  w=5.069 b=-3.930: TBD  [v4 ep=10 approx]

COMPLETED/KILLED in session 9 (before reboot):
- PID 24704 (thresh_sweep): COMPLETE
- PID 10134 (gate_grid): 9/14 done at reboot
- PID 11634 (v6 decoupled): KILLED ep=20 — stagnant 87.1%, drifting wrong
- PID 20698 (v7 first run): KILLED — k went negative, fixed with softplus
- PID 30320 (v7 second run): KILLED by reboot at ep=10 (87.2%, improving)
- PID 1934 (scale5 eval): COMPLETE — peak 87.3%

KEY BUG FIXED (session 9): v7's gate_k was unconstrained → went to k=-0.063 → 0.0% f01.
Fix: gate_k = softplus(gate_k_raw), constrains k≥0. Verified working.

### V9/V10 INPUT BUG — FOUND AND FIXED (session 10-11)

ROOT CAUSE: v9 and v10 fed raw q1 to the MLP as the first input feature. During actual
swing-up rollouts, q1 accumulates beyond [0,2π] (e.g., 3π, -π). Network trained on
q1∈[0,2π] gets out-of-distribution inputs → random/high alpha during swing-up → 
fe suppressed → catastrophic 0.0% f01 on initial eval despite perfect imitation pretraining.

FIX: Replace raw q1 with near_pi = (1+cos(q1-π))/2 ∈ [0,1] as first input feature.
near_pi is angle-wrapped and always in-distribution regardless of q1 accumulation.

Changes made to both v9 and v10:
1. get_alpha(): compute near_pi from x_last[0], pass [near_pi, q1d, q2, q2d] to network
2. imitation_pretrain(): use near_pi as first column of state batch (training matches inference)
3. verification check: use np_val (near_pi) not q1_val as network input

v10 also got a _features() helper method to centralize the feature extraction.

v9 (PID 18638) and v10 (PID 19001) relaunched — expect initial eval ≈87.3% (matching imitation target).

Logs: /tmp/v9_fixed.log, /tmp/v10_fixed.log

### THRESHOLD SWEEP RESULTS (COMPLETE — session 9)

| thresh | zone  | f01     | arr | post   | notes |
|--------|-------|---------|-----|--------|-------|
| 0.750  | 60.0° | 82.3%   | 239 | 93.4%  | wider → better arr, MUCH worse hold |
| 0.800  | 53.1° | 87.2%   | 242 | 99.1%  | prior best |
| 0.825  | 49.5° | 87.2%   | 242 | 99.2%  | same f01, better post |
| 0.850  | 45.6° | **87.3%★** | 242 | 99.3%  | PEAK — natural diagonal optimum |
| 0.875  | 41.4° | 87.2%   | 241 | 99.1%  | drops off |
| 0.900  | 36.9° | 87.2%   | 242 | 99.1%  | plateau |
| 0.925  | 31.8° | 86.8%   | 238 | 98.5%  | steep drop (arr earlier, post much worse) |
| 0.950  | 25.8° | 83.0%   | 237 | 94.1%  | cliff — near-total collapse of hold |

SHAPE ANALYSIS:
- 0.750→0.800: +4.9pp (post 93.4%→99.1% — early activation hurts hold)
- 0.825→0.850: +0.1pp, post +0.1%  ← PEAK
- 0.850→0.875: -0.1pp (begins drop)
- 0.850→0.925: arr improves 242→238 BUT post collapses 99.3%→98.5% → net loss
- 0.925→0.950: catastrophic — narrow zone can't maintain hold

CEILING: f01_max(arr=242, post=100%) = 1758/2000 = 87.9%  [current gap: 0.6%]
Path to 88%+: need arr≤230, NOT achievable via gate tuning alone (requires faster swing-up)

**TRAINING DYNAMICS (v4 ep=10 analysis):**
- Gradient training from thresh=0.80 ALWAYS pushes threshold LOWER (toward 0.775)
- This is WRONG direction: threshold sweep shows 0.85 > 0.80
- Training loss ≠ f01 metric: loss rewards "more alpha = better hold" everywhere
- Fundamental mismatch: gradient pushes toward wider gate, wider gate is actually WORSE

**IMPLICATION FOR v5-v8**: Init from thresh=0.85 (w=5.0, b=-4.25) not 0.80.
Gradient will push thresh lower, but if we start at 0.85, we might stabilize near 0.82-0.85 range.

**v6 DECOUPLED GATE — KEY RESULTS:**

First run (wrong Q gate: w=5, b=-4.25, max_alpha_Q=0.75 at π):
  f01=86.8%, arr=238, post=98.5%  [suboptimal Q hurt post, but arr=238 CONFIRMED fe helps!]

SCIENTIFIC FINDING: Early fe suppression at thresh=0.75 INDEPENDENTLY reduces arr from 242 → 238.
This is 4 steps faster arrival without the Q boost being widened.

Root cause of low post: Q gate formula (w=5, b=-4.25) gives max alpha_Q=0.75 at π (not 1.0!).
Correct natural diagonal for thresh=0.85: w_Q=6.667, b_Q=-5.667 → full activation at π.

**v6 proper (RUNNING PID 11634): fe=(4.0, -3.0, thresh=0.75), Q=(6.667, -5.667, thresh=0.85)**
Initial eval result: f01=87.1%, arr=240, post=98.9% — BELOW 87.3% record. Training started.
The decoupled gate (early fe + natural Q) does NOT beat coupled 0.850 at initialization.
Possible reason: early fe suppression (0.75-0.85 range) without Q boost creates guidance vacuum.
Training may improve params. Watching for training epochs > 87.3%.

**CRITICAL BUG FIXED**: gate grid had `torch.no_grad()` in eval2k (breaks cvxpylayers).
HANDOFF section 15.6 explicitly says NEVER wrap rollout with no_grad. Fixed in commit 46676ef.
Gate grid restarted (PID 10134) without the bug — confirmed "Solved/Inaccurate" is GONE.

---

## SESSION 9-10 FINDINGS (2026-05-01)

### GATE GRID RESULTS (9/14 configs completed, 5 steeper-slope configs restarted)

| w | b | thresh | full | f01 | arr | post | notes |
|---|---|--------|------|-----|-----|------|-------|
| 5.000 | -4.000 | 0.800 | 1.000 | 87.2% | 242 | 99.1% | baseline |
| 6.667 | -5.667 | 0.850 | 1.000 | **87.3%★** | 242 | 99.3% | NATURAL DIAGONAL PEAK |
| 8.000 | -7.200 | 0.900 | 1.025 | 87.1% | 239 | 98.9% | natural diag drops |
| 10.000 | -9.500 | 0.950 | 1.050 | 83.2% | 236 | 94.3% | cliff |
| 5.000 | -3.900 | 0.780 | 0.980 | 87.0% | 240 | 98.9% | lower thresh hurts |
| 5.000 | -3.750 | 0.750 | 0.950 | 82.6% | 239 | 93.8% | wide gate, bad hold |
| 5.000 | -4.125 | 0.825 | 1.025 | 87.2% | 241 | 99.1% | max_alpha=0.875 at goal |
| 5.000 | -4.250 | 0.850 | 1.050 | 87.1% | 240 | 99.0% | max_alpha=0.750 (v4-high bug) |
| 5.000 | -4.375 | 0.875 | 1.075 | 86.9% | 238 | 98.6% | max_alpha=0.625 — bad |
| 8.000 | -6.400 | 0.800 | 0.925 | TBD | — | — | RESTARTED |
| 10.000 | -8.000 | 0.800 | 0.900 | TBD | — | — | RESTARTED |
| 8.000 | -6.240 | 0.780 | 0.905 | TBD | — | — | RESTARTED |
| 8.000 | -6.000 | 0.750 | 0.875 | TBD | — | — | RESTARTED |
| 5.069 | -3.930 | 0.775 | 0.973 | TBD | — | — | RESTARTED |

KEY FINDINGS from gate grid:
1. Natural diagonal (full=1.000 at goal) is STRICTLY BEST for any given threshold
2. full < 1.0 at goal: gate never opens fully → worse hold (87.1% or below)
3. full > 1.0 at goal (but clamped to 1.0): similar or slightly worse than natural diagonal
4. The optimum is confirmed at thresh=0.850, natural diagonal: w=6.667, b=-5.667

### SCALE=5.0 EVAL WITH OPTIMAL THRESHOLD (COMPLETE)

| thresh | f01 | arr | post | notes |
|--------|-----|-----|------|-------|
| 0.800 | 87.2% | 243 | 99.3% | arr slightly later than scale4 |
| 0.825 | **87.3%** | 242 | 99.3% | ties record |
| 0.850 | **87.3%** | 242 | 99.3% | ties record (same as scale4 at 0.850) |

CONCLUSION: Scale=5.0 does NOT improve over scale=4.0. Both peak at 87.3%.
The dQ_ref magnitude is optimized by scale training; adding more at inference doesn't help.

### V6 DECOUPLED GATE AUTOPSY (KILLED at ep=20)

Decoupled fe (thresh=0.75) + Q (thresh=0.85, natural diagonal):
- Initial eval: 87.1%, arr=240, post=98.9% — BELOW record even at initialization
- ep=10: 87.1%, th_fe=0.739, th_Q=0.826 — both drifting lower (wrong direction)
- ep=20: 87.1%, th_fe=0.739, th_Q=0.810 — still drifting lower, no improvement

CONCLUSION: Decoupled fe+Q gates don't help. The coupled approach (same gate for both)
at thresh=0.850 is superior. Early fe suppression without Q boost creates a "guidance
vacuum" that slightly hurts overall performance.

### V7 VELOCITY-AWARE GATE (RESTARTED)

Gate formula: proximity = near_pi - k_eff × q1d²; alpha = (w×proximity + b).clamp(0,1)
k_eff = softplus(k_raw) ≥ 0 always (CRITICAL: unconstrained k went negative → 0.0% f01!)

Run 1 (unconstrained k): k → -0.063 at ep=10 → 0.0% f01 (gate fires MORE at high velocity,
disrupts swing-up). FIXED by softplus constraint.

Run 2 (softplus, init k_eff=0.05, thresh=0.850, KILLED BY REBOOT at ep=10):
- Initial eval: 87.0% (velocity suppression at k=0.05 slightly hurts arrival)
- ep=10: **87.2%** ★ (improving! thresh=0.830, k=0.047 — k stayed positive, converging)

Run 3 (CURRENTLY RUNNING, PID 2471): same init, restarted after reboot.
Expected: improve through ep=10→20, potentially reach or exceed 87.3%.

---

## SESSION 6 QUICK-READ (earlier portion, 2026-04-30)

**STABILITY TEST RESULTS (qmax_stability.py, no CVXPY needed):**
After 500-step Q-max warmup (gates_Q[q1]: 0.013→1.985):

| Test | Setup | After 200 training steps |
|---|---|---|
| A | 70% top (Q-max aux) + 30% bottom (param-regularize to orig) | 1.9851 → **1.9851** (ZERO CHANGE) |
| B | 100% bottom only (worst case, no Q-max at all) | 1.9851 → **1.9849** (negligible) |
| C | 70% top (no Q-max aux) + 30% bottom | [in progress] |

**Key conclusion**: At lr=1e-4, warmup gains are EXTREMELY STABLE. Even 100% bottom training barely moves them. This means:
1. Top Q gains persist through alternating training ✓
2. BUT: bottom states also stay at 1.985, which likely HURTS swing-up (confirmed below)

**DUAL-THRESH SWEEP NEW RESULT:**
- thresh_dQ=0.600 → f01=83.2%, **arr=322** (vs baseline 242!)
- Wider boost zone delays swing-up by 80 steps — confirms high Q at non-top states is harmful
- thresh_dQ=0.800 → 87.2% (optimal is current setting)
- Lower thresh_dQ not viable: disrupts the swing-up trajectory

**CRITICAL DISCOVERY: Q-restore pre-phase creates NO differentiation.**
After 100 Q-restore steps at lr=1e-3:
- top: 1.985 → 0.123 (dropped 94%!)
- bottom: 1.985 → 0.113 (dropped 94%!)

Both states converge to SIMILAR values. The Q-restore gradient affects GLOBAL q_head weights
equally — it does NOT selectively lower bottom Q while keeping top high. After restore, both
are near the original values (top≈0.013 orig, bottom≈0.111 orig). Warmup has zero net benefit.

**Root cause of Q-restore failure**: The q_head maps FEATURES → Q. After warmup, ALL state
features map to high Q (globally). Q-restore at bottom features creates gradients that flow
through the SHARED q_head weights, affecting ALL states (top included). Even though the gradient
at top features is smaller (cos_sim=0.175), with 100 steps at lr=1e-3, the accumulated effect
reverts top almost as much as bottom.

**Dead experiments:**
- v2 (restore_steps=200, lr=1e-3): killed — pre-restore reverts both top AND bottom equally
- v3 (restore_steps=300, lr=1e-3): killed — same issue, confirmed by data
- v4 (warmup+70%top): 0.0% initial eval — global Q-max disrupts swing-up
- v5 (warmup+100%bottom): 0.0% initial eval — same root cause
- posgate (437 params): 0.0% initial eval — smooth pre-train activates at q1=60°

---

## SESSION 5 QUICK-READ (2026-04-30, continued)

**KEY DISCOVERY: The tanh saturation CAN be fixed by the atanh-inverted Q-max loss.**

The critical problem was: `raw_Q[q1] ≈ -3.47` at the top → `gates_Q[q1] = 1+0.99*tanh(-3.47) ≈ 0.013` (saturated).
MSE loss on `gates_Q` has gradient: `0.99*(1-tanh²(-3.47)) ≈ 0.004` (near zero! useless).

**Fix (atanh inversion):** Recover `raw_Q = atanh((gates_Q - 1) / 0.99)`, then MSE on `raw_Q → +3.0`.
- Gradient: atanh amplifies 250× near saturation, canceling the tanh suppression → net gradient ≈ 1
- Probe result (qmax_probe.py, 2000 steps, lr=1e-3):
  - Step 0: gates_Q[q1] = 0.014, raw_Q = -3.052, loss = 38.97
  - Step 200: gates_Q[q1] = **1.985**, raw_Q = +3.000, loss ≈ 0 (CONVERGED!)
  - Stable thereafter (step 400, 600: same 1.985)

**Effective Q[q1] after fix:**
- From lin_net alone: 12 × 1.985 = **23.8** (vs 0.156 before → 153× improvement)
- With scale=4× wrapper: 12 × (1.985 + 4.354) = **76.1**

**Critical bug fixed: lin_net expects 5-step state history, NOT HORIZON=10.**
- `lin_net.py:80`: `state_input_dim = 5 * state_dim = 20` (hardcoded)
- Both Stage E scripts were using `expand(HORIZON, -1)` → shape (10,4) → crash
- Fixed to `expand(5, -1)` (STATE_HIST constant)

**CRITICAL DISCOVERY: Q-max effect is GLOBAL, not state-specific.**

Probe (qmax_probe.py) ran 2000 steps of atanh Q-max at lr=1e-3:
- gates_Q[q1] at TOP: 0.013 → **1.985** (target achieved, converges in 200 steps ✓)
- gates_Q[q1] at BOTTOM (q1=0): 0.111 → **1.976** (ALSO increased! global change!)
- ALL states (q1=0-π): gates_Q[q1] ≈ 1.98 everywhere

Why? The q_head.4 output layer is shared for all inputs. Gradient from top states also
increases raw_Q[q1] for bottom-state trunk representations (they're correlated, not orthogonal).

**Original gates_Q[q1] profile (posonly_ft):**
| q1 | gates_Q[q1] | eff Q[q1] |
|---|---|---|
| 0.0 (bottom) | 0.111 | 1.33 |
| 1.0 | 0.049 | 0.59 |
| 2.0 | 0.021 | 0.25 |
| π (top) | 0.013 | 0.16 |

Pattern: monotonically decreasing! Network learned to care LESS about q1 as it approaches π.
After Q-max: uniform 1.98 everywhere → breaks this learned pattern.

**Risk: Global Q[q1]=23.8 during swing-up might disrupt f_extra-based energy pumping.**
Threshold sweep confirmed: scale=4× (Q[q1]=76 during swing-up) → catastrophic (0%).
Scale=1× (Q[q1]=23.8) during swing-up: unknown, might be OK or harmful.

**Fix: Q-restore contrastive loss for bottom states.**
Added `q_restore_aux_step()`: pushes raw_Q at bottom states → ORIGINAL raw_Q (atanh of orig gates_Q).
This creates true state-conditional behavior without relying on trunk discrimination.

**Running experiments (as of ~20:30):**

| PID | Script | Status |
|-----|--------|--------|
| 25595 | exp_dual_thresh.py --sweep | thresh=0.0-0.5 all 0%, confirming threshold is all-or-nothing |
| 25964 | exp_boost_v2.py (from scale=4× ckpt) | ep=10: 87.2% (gradient doesn't improve from 4× init) |
| 11208 | exp_stageE_qhead_ft.py --with_wrapper | Running, compiling CVXPY (tracking loss only, no Q-max) |
| 28128 | exp_stageE_alternating.py --with_wrapper | Q-max warmup + Q-restore, compiling CVXPY |

**Stage E design (exp_stageE_alternating.py v3):**
1. Q-max warmup: 500 steps at lr=1e-3 → gates_Q[q1]≈1.985 everywhere (fast convergence)
2. Q-restore warmup: bottom epochs push raw_Q back to original over time
3. Alternating training: 70% top (tracking + Q-max aux), 30% bottom (tracking + Q-restore)
4. w_q_bonus=2.0, w_restore=2.0 — balanced competition between top gain and bottom anchor

**Dual-thresh sweep (COMPLETE prediction): no improvement possible via lower threshold.**
thresh_dQ=0.0-0.5: all 0% (catastrophic). thresh=0.6-0.8 will recover to ~87.2%. thresh=0.85-0.90: slightly lower.
This confirms scale=4× delta_Q cannot be activated smoothly — must snap on at threshold=0.80.

**Theoretical ceiling with current wrapper approach:**
- arr≈242 (fixed by swing-up dynamics), post≈99.1-99.3%
- Max f01 = (2000-242)/2000 × 1.0 = 87.9%
- To exceed 88%: need arr < 242 (faster swing-up) via bottom training
- Stage E alternating's bottom epochs MIGHT achieve this by improving swing-up Q profile

---

## SESSION 4 QUICK-READ (2026-04-30)

**ROBUSTNESS CONFIRMED: avg 83.2% across 8 diverse starting conditions.**
The 82.9% boost model (learned delta_Q) was tested from 8 different x0 values. Average boost=83.2%, average zero=26.1%. The only outlier is x0=[+0.1,0,0,0] (64.4%) which has arr=701 — a swing-up dynamics quirk where small positive q1 causes long roundabout trajectory. All others: 82-89%.

| x0 | boost f<0.10 | zero f<0.10 | arr | Δ |
|---|---|---|---|---|
| [0,0,0,0] (standard) | 82.9% | 26.2% | 236 | +56.6% |
| [0.1,0,0,0] | 64.4% | 20.2% | 701 | +44.2% |
| [-0.1,0,0,0] | 85.9% | 25.3% | 248 | +60.5% |
| [0.5,0,0,0] | 86.7% | 28.4% | 256 | +58.3% |
| [-0.5,0,0,0] | **88.5%** | 29.5% | **214** | +59.0% |
| [0,0.5,0,0] | 86.7% | 26.6% | 238 | +60.0% |
| [0,-0.5,0,0] | 85.7% | 27.7% | 273 | +58.0% |
| [0,0,0.3,0] | 85.3% | 24.4% | 281 | +60.8% |

**Failure mode analysis (from [0,0,0,0]):**
- Swing-up time: 236 steps → 11.8% failure contribution (timing only)
- Hold-phase failures: 6.1% of 1764 hold steps = 108 steps → 5.4% contribution  
- Path to 92%+: reduce hold-phase fall-off (6.1% → ~0%) OR reduce swing time (arr→0)

**SESSION 4 SCALE SWEEP BREAKTHROUGH:**

Simply scaling the trained delta_Q by a constant factor reveals non-monotonic behavior:
```
 scale    f<0.10    arr    post      eff Q[q1]
  0.00    26.2%    326    31.3%        0.16
  0.25    23.9%    327    28.6%        3.42
  0.50    32.8%    236    37.2%        6.69
  0.75    83.1%    236    94.2%        9.95
  1.00    82.9%    236    93.9%       13.22  ← trained
  1.25    82.7%    237    93.8%       16.48
  1.50    82.5%    237    93.5%       19.75
  2.00    81.8%    238    92.8%       26.28
  3.00    87.0%    239    98.8%       39.34  ← NEW BEST!
```

Key findings:
- **87.0% at scale=3.0** — 4.1pp better than trained 82.9%, no retraining needed!
- **post_arr=98.8%** at scale=3.0 vs 93.9% at scale=1.0 — dramatically better hold quality
- **arr stays similar** (236-239) for scales 0.5-3.0 — swing-up not hurt by higher delta_Q
- Non-monotonic: scale 1.0→2.0 degrades, then scale=3.0 jumps to new high
- Why scale=3.0 works: high dQ[q1,q1d] gives strong restoring force, NEGATIVE dQ[q2]=-0.315
  reduces Q[q2] weight (50→34), enabling natural swing-up approach
- Extended sweeps COMPLETE: peak is **87.2% at scale=4.0-5.0**. See full table:

```
 scale    f01    arr    post      eff Q[q1]
  0.00   26.2%   326   31.3%        0.16
  0.60   64.8%   236   73.5%        7.99
  0.65   83.1%   236   94.2%        8.65  ← threshold jump
  0.70   83.0%   236   94.1%
  1.00   82.9%   236   93.9%       13.22  ← gradient-trained attractor
  1.50   82.5%   237   93.5%
  2.00   81.8%   238   92.8%
  2.50   86.8%   238   98.5%       32.81  ← second regime starts
  3.00   87.0%   239   98.8%       39.34
  4.00   87.2%   242   99.1%       52.41  ← CURRENT BEST (tied)
  5.00   87.2%   243   99.3%       65.47  ← CURRENT BEST (tied)
  7.00   87.1%   243   99.1%       91.60
 10.00    0.0%  None    N/A       130.79  ← catastrophic failure
```

- Checkpoints saved: `stageD_scale4.0x_dQ_20260430_192447`, `stageD_scale5.0x_dQ_20260430_192447`
- Two distinct performance regimes: 83% (scale 0.65-2.0), 87% (scale 2.5-7.0)
- Gradient training converges to the 83% regime; scale=4-5 is NOT reachable by near-top gradient
- Why scale=4-5 works: Q[q1]=52.4 (4× base), Q[q2]=29.1 (42% reduction) → strong q1 hold + relaxed q2

**Gradient training from 82.9% always degrades (confirmed by boost_v2, boost_continue):**
- boost_continue: 82.9% → 82.4% at ep=40 (degrading)
- boost_v2 (fresh x0): 82.9% → 82.7% at ep=60 (degrading slower but still degrading)
- optinit_2.0 (larger init): converges to 82.6% — same basin as 0.987 init

**Key insight: The 82.9% is a stable attractor of near-top training loss, but the loss
landscape ≠ 2000-step eval metric. Scale=3.0× gives 87.0% without gradient training.**

**New experiments written:**
- `exp_dq_scale_sweep.py`: Forward-only sweep of delta_Q scale factors (ran: 87% at scale=3.0!)
- `exp_thresh_sweep.py`: Forward-only sweep of gate thresholds
- `exp_dual_thresh.py`: Separate gate thresholds for delta_Q activation vs f_extra zeroing
- `exp_boost_v3.py`: Mix near-top + swing-up training (robustness-oriented)
- `exp_boost_v4.py`: Adds delta_Qf (terminal cost learning, new axis)

**SESSION 3 QUICK-READ (2026-04-30)**

**BEST RESULT: 82.9% frac<0.10** (2000-step from [0,0,0,0]), post_arr=93.9%, arr=236 steps. Learned delta_Q=[1.089,1.023,-0.105,0.077] correction via 20 epochs of gradient training from near-top x0. ZeroFNet baseline was 26.2%. Target (>50%) EXCEEDED.

**Root cause discovered (session 3):** At top state [π,0,0,0], the network outputs `gates_Q[q1]≈0.013` and `gates_Q[q1d]≈0.013` (near-zero!), giving effective Q[q1]=0.156 and Q[q1d]=0.064 (vs. base 12.0 and 5.0 — 98.7% suppressed!). The MPC is essentially ignoring q1/q1d deviations near the top, causing the pendulum to oscillate rather than hold. Fix: add `delta_Q[:,0]≈+0.987` and `delta_Q[:,1]≈+0.987` to restore full Q base weights.

**Scalar Q boost confirmed harmful:** q_boost=0.05 → 24.5% (WORSE than 26.2%). Uniform boost also increases q2/q2d which already have full gates (≈1.0), causing interference.

**Experiments this session:**
| Experiment | Result | Status |
|---|---|---|
| ZeroFNet (thresh=0.80, no training) | **26.2%** | baseline |
| holdboost_ft v1 (hold_reward, ep=15) | **26.5%** ↑ | Done |
| exp_qboost_targeted (scalar sweep) | q=0.05→24.5% | Done (scalar hurts) |
| exp_holdboost_nearstart (near-top starts) | — | Killed (backprop too slow, no ckpt in 110min) |
| exp_q1restore_test dq0=0.00 | **26.2%** | Done |
| exp_q1restore_test dq0=0.25 | **36.8%** (+10.6pp) | Done |
| **exp_optinit_holdboost 0.987 0.987** | **82.9%** POST_ARR=93.9% | DONE ★★★ |
| exp_boost_continue (from 82.9% init) | ep20=82.7%, ep40=82.4% | Killed (degrading) |
| exp_optinit trial 2 (0.987,0.987) | **82.9%** — SAME! | Done — REPRODUCIBLE ★★★ |
| **exp_robust_eval (8 x0 starts)** | avg=83.2%, all +44-61% over zero | Done ★ |

**REPRODUCIBILITY CONFIRMED:** Trial 2 with different random seed gives IDENTICAL 82.9% result.
- Trial 1: dQ_mean=[1.089, 1.023, -0.105, 0.077]
- Trial 2: dQ_mean=[1.087, 1.001, -0.103, 0.076]
Very consistent! The optimal delta_Q is a stable attractor for the 20-epoch training from (0.987, 0.987).

**KEY FINDING: More training hurts.** Starting from 82.9% and continuing gradient training degrades to 82.4% at ep=40. The 20-epoch sweet spot is robust.

**Saved checkpoint (BEST):** `saved_models/stageD_optinit_holdboost_dq0.99x0.99_20260430_165519/`
- best_frac01_2000step: 0.8286 (82.9%)
- best_delta_Q mean: [1.089, 1.023, -0.105, 0.077]
- best_delta_R mean: [0.081, -0.083]

**Robust eval complete.** Avg 83.2% boost vs 26.1% zero across 8 starts. See Session 4 quick-read above.

**Next queued:** `exp_boost_v2.py` (fresh x0 per epoch, cosine LR) — most promising next step.

---

## TL;DR (original)

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

## 13. HOW I WORKED (self-debrief)

This is a Claude (Anthropic) session log. Some honest notes on the process so the next agent (or human) doesn't repeat the same patterns:

**Approach that worked:**
- Aggressive parallel experiments (2-3 trainings at once on a 4-core CPU). Found qf50 v2 fast, found the optimizer-reset bug fast, found the trace-revealed oscillation fast.
- Diagnostic scripts (`trace_qf50.py`, `diag_grad.py`, `audit_history.py`, `probe_qf.py`) caught real bugs and revealed misleading metrics. **The "12/12 generality" was an artifact** until `trace_qf50.py` showed actual oscillation.
- Hard constraints upheld throughout (no controller logic changes, all signals in outer loss).
- Frequent commits + detailed messages — restoring context after reboots was painless.

**Mistakes made:**
- **Trusted aggregate metrics too long.** "12/12 OK" and "35/35 boundary" were claimed wins for hours before tracing revealed the underlying oscillation. The eval threshold (`total time in zone ≥ 50` over 1000 steps) was too lenient — passing-through goal repeatedly satisfies it. **Lesson: validate the evaluation metric BEFORE trusting it.**
- **Over-engineered "robust" attempts.** SAM-approximation, distillation, sin/cos encoding, weight decay sweeps — most were one-off attempts that didn't pan out. Should have done a careful single-variable ablation before stacking 3+ losses.
- **Killed promising runs prematurely.** qf50 v2's first run was killed at epoch 40 with `GoalDist=0.80` thinking it would diverge; it was actually converging. Re-ran it and got the 12/12 result.
- **Didn't periodically save checkpoints.** Two system reboots killed multi-hour trainings. Should have added `save_every_N_epochs` early.
- **Stalled when waiting for monitors.** The monitor pattern (background process emitting events) timed out at 30 min repeatedly; I'd "wait" passively rather than poll the log file directly. Better: use `Bash` with `until grep -q` for blocking waits on specific signals.
- **Got pulled into the "gates-Q only" trap repeatedly.** Multiple attempts produced models with `fnorm ≈ 0` (no f_extra) without recognizing the problem until late. This is now flagged in section 2.

**Communication patterns:**
- Reported every milestone tersely (1-2 sentences). User noted some events were "stalls" — I was waiting for monitor events but should have been polling actively.
- Asked the user for direction at decision points; could have been more autonomous on small calls.
- Named all experiments with descriptive prefixes (`exp_combined_v3`, `exp_pin_warmup`, etc.) so they're self-documenting.

**Tools used:**
- `Monitor` for streaming log events (with persistent + non-persistent variants)
- `Bash run_in_background` + `until grep -q` for blocking waits on specific log lines
- `git commit && git push` after every meaningful change to survive reboots
- `TodoWrite` for tracking multi-step plans
- `Edit` for surgical file changes; `Write` only for new files
- The `Skill` tool (e.g. `simplify`, `claude-api`) was not needed for this work

**Total session footprint:**
- ~30 experiment scripts written (most parked as dead-ends)
- ~15 saved models (most don't truly hold)
- 80+ commits to the working branch
- 2 real bugs fixed (optimizer reset, restore_best rollback)
- 1 working-ish model (`qf50 v2` — wide basin but oscillates)

**For the next agent:** read sections 2 (no Q-only swing-up), 7 (velocity-asymmetry trap), and 11 (next directions) before writing any new training script. The HANDOFF is the truth; the experiments themselves are exploratory artifacts.

---

## 14. THE ONE THING TO REMEMBER

**Track loss (energy) and gates-Q can both be satisfied without the pendulum actually holding still at upright.** Energy is symmetric (kinetic vs potential trade-off — the pendulum can have any energy at any q1 given matching q1d). Gates are state-shaping but the QP only plans 10 steps ahead.

Without a `f_extra`-driven swing-up (which is open-loop, time-varying torque pattern that the network commits to), the QP alone cannot reliably plan a multi-second swing-up trajectory.

**Future losses must not starve the f_extra pathway.** That's the real constraint.

---

## 15. SESSIONS 2–3 KEY FINDINGS (2026-04-30)

### 15.1 ZeroFNet — 26.2% with no training

Discovery: applying an inference-time soft gate that zeros `f_extra` near the top (no gradient training) gives the best hold quality so far. No parameter changes needed.

```python
gate = ((near_pi - thresh) / (1 - thresh)).clamp(0, 1)
f_extra_effective = f_extra * (1 - gate)
```

Best threshold: 0.80 → **26.2% frac<0.10** over 2000 steps from `[0,0,0,0]`.
Pendulum arrives at step 326, post-arrival hold quality: 31.3% of steps < 0.10.

Models evaluated (all from saved_models/): the `posonly_ft_final` model from stageD_posonly_ft_20260430_083618 is the BEST swing-up model. Use this as the base.

### 15.2 The Trajectory Coupling Root Cause

ALL gradient-based training failed because:
1. Full model: shared trunk corrupted by competing gradients (swing-up vs hold)
2. Q/R-only training: weights shared across all states → changing for hold → changes swing-up
3. Near-top initial state training: Q/R changes alter trajectory → alter f_extra

The only safe approach: **additive corrections that only activate near the top**.

### 15.3 HoldBoostWrapper — 56 trainable parameters

```python
q_shape = (H-1, state_dim)  # (9, 4)
r_shape = (H, control_dim)  # (10, 2)
delta_Q = nn.Parameter(zeros(q_shape))  # init=0 → identical to ZeroFNet
delta_R = nn.Parameter(zeros(r_shape))
```

Forward: `gates_Q += gate * delta_Q`, `gates_R += gate * delta_R`, `f_extra *= (1-gate.detach())`

DECOUPLING GUARANTEE: `∂loss_swing/∂delta_Q ≈ 0` because gate≈0 during swing-up.

Result at ep=15 (hold_reward, LR=1e-3): delta_Q.mean=0.0066 → **26.5%** (first positive training result).

### 15.4 Root cause: gates_Q[q1]≈0.013 at top state

```python
# Diagnostic run at top state [pi, 0, 0, 0]:
# gates_Q mean per dim: [0.0130, 0.0129, 1.0012, 0.9986]
# Effective Q:          [0.156,  0.064,  50.06,  39.94]
# Base Q:               [12.0,   5.0,    50.0,   40.0 ]
#                        ^98.7% suppressed!  ^fine
```

The baseline network outputs near-zero gates for q1 and q1d at the top, so the MPC barely penalizes q1/q1d deviations. This is why the pendulum oscillates rather than holding. Required fix:

```python
delta_Q[:, 0] = +0.987   # restores gates_Q[q1]  from 0.013 → 1.000
delta_Q[:, 1] = +0.987   # restores gates_Q[q1d] from 0.013 → 1.000
delta_Q[:, 2] = 0.0      # q2 already at 1.001, leave alone
delta_Q[:, 3] = 0.0      # q2d already at 0.999, leave alone
```

Why scalar boost fails: q_boost=0.05 gives +5% to q2/q2d (already at full base), causing interference that outweighs the +384% benefit to q1.

### 15.5 Key results (2026-04-30 ~17:00 UTC) — MAJOR BREAKTHROUGH

**exp_optinit_holdboost.py** with init (0.987, 0.987):
- Initial eval (fixed, no training): 26.0% (dq1=0.987 alone is harmful!)
- After 20 epochs gradient training (near-top x0, 200-step rollout, LR=5e-3):
  **82.9% frac<0.10**, frac<0.30=83.7%, arr=236, post_arr=93.9%

Training converged in 3.6 minutes (218s) and triggered early stop at EXCELLENT_HOLD=50%.

**Why it works:** The gradient training learned the optimal near-top Q matrix:
- Q[q1] = 12.0 × (0.013 + 1.089) = **13.2** (full restoration + slight boost)
- Q[q1d] = 5.0 × (0.013 + 1.023) = **5.2** (full restoration — contrary to naive expectation)
- Q[q2] = 50.0 × (1.001 - 0.105) = **44.8** (slight reduction — reduces over-control)
- Q[q2d] = 40.0 × (0.987 + 0.077) = **42.6** (slight increase — more damping)
Combined with f_extra=0 (no oscillation driver), the MPC now properly tracks q1 at all horizon steps.

**Why naive fixed dq1=0.987 gave only 26.0%:** The initial fixed (0.987, 0.987) was suboptimal. Gradient training found the right balance within 20 epochs.

**exp_holdboost_nearstart** killed at 110min: backprop through QP is ~0.9s/call, BUT the effective training was much faster when only one process ran (CPU freed = 3.6min/chunk instead of 60min). Key insight: avoid running competing heavy processes.

**exp_q1restore_test.py results (partial, stopped after 2 evals):**
```
  config    f<0.10    frac<0.30    arr    post
baseline     26.2%     42.3%      326    31.3%
 q1=0.25     36.8%     51.3%      327    44.0%  ← partial fix already +10.6pp
```

### 15.6 cvxpylayers gotcha

**Never use `torch.no_grad()` in eval loops.** The QP solver (cvxpylayers) needs autograd active even for forward-only evaluation — wrapping rollout with `no_grad()` produces wrong/zero control outputs (pendulum never arrives). The `train_module.rollout()` function already uses `no_grad()` internally for the lin_net forward pass only, which is correct.

**Compilation cost:** Each fresh Python process takes ~25 min for first QP solve. Keep experiments in long-lived processes. Multiple processes compete for CPU — avoid running more than 2 heavy experiments simultaneously.

### 15.7 Next steps in priority order

1. **Check trial 2 result** (PID 23020): Does a second 20-epoch run from (0.987,0.987) reproduce or exceed 82.9%? Log: `/tmp/optinit_trial2.log`. Expected ~17:42 UTC.
2. **If trial 2 ≥ 82.9%**: Result is reproducible. Accept as final. Update HANDOFF, push.
3. **If trial 2 < 82.9%**: Run `exp_robust_eval.py` to test 82.9% model across multiple x0 starts. Then try `exp_boost_v2.py` (fresh x0 every epoch, different LR).
4. **Robustness testing**: `python exp_robust_eval.py` (8 different starting conditions). Requires CVXPY compilation.
5. **For future sessions**: Full retraining of lin_net (Stage E) that doesn't need the wrapper. This would be a multi-hour training run but would produce a "natively" better model.

### 15.7b What to do if we want to go beyond 82.9%

Options in increasing complexity:
1. **Different random seeds** for the 20-epoch training run (trial 2 tests this)
2. **Different init**: Try (1.5, 0.0) or (0.987, 0.0) — different starting point for gradient
3. **exp_boost_v2**: Fresh x0 every epoch, cosine LR — might find better optimum
4. **State-dependent delta_Q**: Train a small linear model W×x+b for delta_Q instead of fixed. More expressive (180 params vs 36).
5. **Direct network weight editing**: Modify q_head weights so raw_Q[q1] at top → 0. More fundamental fix.
6. **Full Stage E retraining**: Retrain lin_net with hold-quality loss + ZeroFNet gate + stage-aware training. Would take hours but could reach 90%+.

### 15.8 How to reproduce 82.9% result

```python
import torch, math
import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module

POSONLY = "saved_models/stageD_posonly_ft_20260430_083618/stageD_posonly_ft_20260430_083618.pth"
BEST_CKPT = "saved_models/stageD_optinit_holdboost_dq0.99x0.99_20260430_165519/stageD_optinit_holdboost_dq0.99x0.99_20260430_165519.pth"

# Load best delta_Q from checkpoint
ckpt = torch.load(BEST_CKPT, map_location='cpu', weights_only=False)
tp = ckpt['metadata']['training_params']
best_dQ = torch.tensor(tp['best_delta_Q'], dtype=torch.float64)  # shape (9,4)
best_dR = torch.tensor(tp['best_delta_R'], dtype=torch.float64)  # shape (10,2)

# Load lin_net (frozen) + apply FixedHoldBoost wrapper
lin_net = network_module.LinearizationNetwork.load(POSONLY, device='cpu').double()
# Then use HoldBoostWrapper(lin_net, thresh=0.8, dQ_init=best_dQ, dR_init=best_dR)
# → 82.9% frac<0.10 on 2000-step eval from [0,0,0,0]
```
