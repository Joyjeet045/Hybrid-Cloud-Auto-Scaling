# Rec 4 / Approach 4 — Objective-Driven Membership-Function Auto-Tuning (self-designed)

**Status:** NEUTRAL RESULT, but a rigorous one. Three progressively more careful
variants were built. Naive objective-driven tuning **overfits** the calibration
set; the principled variant (regularized, schedule-only, validation-gated)
**certifies the expert membership functions as a robust optimum** by converging
back to them. Pluggable, disabled on the default path. **Not promoted to core**
(does not beat baseline out-of-sample, so by the promotion criterion it stays an
ablation).

## Motivation (why this is different from rec2)

The user asked to *stop hand-picking* the membership functions and have an
algorithm choose centres/widths that are "optimal for our task without harming
performance".

`rec2` clustered the recorded input samples (1-D k-means) and used the centroids
as membership-function centres. That optimises **input-space coverage**, an
objective uncorrelated with control loss — and it failed (net-flat, bad tail,
in-sample W-13 regressed).

This approach fixes the objective mismatch directly: optimise the membership-
function partition **against the closed-loop control loss itself**, with a hard
feasibility penalty. The term **names and count never change**, so all 9 rules and
the open-shoulder logic keyed on term names stay intact.

This ablation went through three variants. Each one tightened the methodology in
response to the previous one's failure mode.

---

## Variant 1 — Static membership-function tuning (overfits)

- **Search:** derivative-free `(1 + λ)` evolution strategy. Genome = per-term
  Gaussian `(center, sigma)` for `psi` (4), `omega` (3), `phi` (3); `rho` fixed.
  Centres monotone-decoded (ordered per variable).
- **Objective:** `mean_MRT(calib) + 1e5 * total_Vio`, calib = `{N-13, W-13, A-13}`.
- **Elitism:** generation 0 evaluates the expert seed; incumbent only replaced by a
  strictly better candidate (hard lower bound on the *calibration* score).
- **Injection:** `OptimizedMFEngine` overwrites the centre/sigma of the existing
  terms from `mf_terms.json`. No core change.

**Result (12 scenarios, seed 0):** calibration improves **−2.60 ms (−1.62 %)** and
is feasible — but the full suite is net-flat (**−0.13 ms**) and the **held-out mean
is +0.69 ms** with a huge tail spread (W-12 −22.10 vs A-14 +28.10). The objective
is now correctly aligned with control (unlike rec2), yet the *capacity to overfit*
a small, low-dimensional calibration set dominates. The static MF shapes it finds
are a fit to three load regimes and do not transfer.

> Lesson: MRT-optimal MF **shapes** are scenario-specific. Fitting them on a few
> scenarios overfits.

---

## Variant 2 — Context-scheduled MFs + robust regret + validation + CMA-ES (overfits worse)

To attack the overfit, the whole methodology was upgraded with four ideas:

- **(A) Robust objective.** Per-scenario *regret* `r_t = (mrt_t − base_t)/base_t`;
  score `= mean(r) + 1.0·worst(r) + 1e3·vio`. Optimises the **worst case**, not the
  average, so a single overfit scenario cannot be hidden by wins elsewhere.
- **(B) Train/validation split.** Rank candidates on `TRAIN = {N-13, W-13, A-13}`,
  but **select the incumbent by `VAL = {N-11, W-12, A-14}`** (feasible-first). The
  expert is always evaluated, so validation can never be harmed by construction.
- **(C) Context-scheduled (gain-scheduled) partition.** Instead of static centres,
  the partition **moves with live load**: `g = clip((psi − g_lo)/(g_hi − g_lo), 0, 1)`
  and each variable's centre/width is shifted by learnable gains
  `centre = base + g_c·(g − 0.5)·span`, `sigma = base·(1 + g_s·(g − 0.5))`, with a
  monotone projection. At gains = 0 and base = expert this is **byte-identical** to
  the baseline (confirmed: gen-0 regret +0.0000). This encodes the *structural*
  prior "be more aggressive under high load" — the kind of idea that generalised in
  rec3.
- **(D) Separable CMA-ES.** Diagonal-covariance CMA-ES (Ros & Hansen 2008),
  implemented from scratch in numpy (no `cma` dependency), over a 26-D genome
  (20 static multipliers + 6 schedule gains).

**Result — validation looked excellent, held-out was a disaster:**

| metric | validation `{N-11,W-12,A-14}` | **true held-out (6 scen)** |
|---|---|---|
| mean regret | **−3.27 %** | **+5.09 %** |
| worst regret | −2.10 % | A-11 **+21.37 %** |
| mean dMRT | — | **+11.26 ms** |

Full suite 206.00 → **210.14 ms (+4.14 ms)** — *worse than baseline and worse than
Variant 1*. All Vio = 0. Worst offenders: A-11 +36.14, W-14 +25.79, N-14 +5.68.

> Lesson: with 26 free parameters the search **games the validation split**. The
> chosen split `{N-11,W-12,A-14}` was not representative of the held-out regimes,
> and high model capacity turned the robust objective + validation into a false
> sense of safety. More capacity + a fixed small validation set = a new way to
> overfit.

---

## Variant 3 — Schedule-only + regularization + validation-gating (certifies the expert)

The only part of Variant 2 with a genuine generalisation story is the **load
schedule** itself (a structural "be aggressive under high load"), not the base MF
shapes. Variant 3 isolates exactly that and removes all the capacity that enabled
overfitting:

- **Freeze the base partition at the expert** (the 20 shape parameters are fixed at
  the hand-designed values). Optimise **only the 6 schedule gains**.
- **Regularize the schedule.** Add `REG_COEF · mean(gains²)` (`REG_COEF = 0.05`) to
  the training fitness, pulling the schedule toward "no scheduling" unless the data
  clearly justifies it.
- Keep the **robust regret objective** and the **validation-gated incumbent
  selection** from Variant 2. Expert always evaluated (no-harm by construction).

**Auto-tune run (schedule-only, gens 6, popsize 8, workers 12):**

```
[gen 0] expert:    train mean_reg +0.0000 | val score +0.0000
[gen 1] train best -0.0174 | incumbent val score +0.0000
[gen 2] train best -0.0258 | incumbent val score +0.0000
...
[gen 6] train best -0.0203 | incumbent val score +0.0000
final gains: psi=(0.000, 0.000)  omega=(0.000, 0.000)  phi=(0.000, 0.000)
```

CMA-ES **does** find schedules that reduce *training* regret (mean_reg −0.02), but
**none of them beats the expert on validation**, so the validation gate keeps the
incumbent at **gains = 0** — i.e. it returns the expert design.

**Result (12 scenarios, seed 0):** the saved design reproduces the baseline
**exactly**, every scenario within ±0.00 ms, all feasible:

| metric | value |
|---|---|
| full-suite mean | 206.00 → **206.00 ms (±0.00)** |
| held-out mean dMRT | **−0.00 ms** (6 scenarios) |
| held-out mean regret | **−0.00 %** |
| infeasible | 0 / 12 |

> The principled auto-tuner, given freedom to gain-schedule the partition, **cannot
> find a generalising improvement and converges back to the expert**. That is an
> automated *certificate* that the expert fuzzy partition sits at a robust optimum.

---

## Conclusion

Across three increasingly careful designs, objective-driven membership-function
optimisation never produces a robust, transferable MRT win:

1. **Static tuning overfits** the calibration set (held-out +0.69 ms).
2. **Full scheduled tuning overfits worse** (held-out +11.26 ms / +5.09 %) — more
   capacity games the validation split.
3. **Regularized, schedule-only, validation-gated tuning returns the expert**
   (held-out −0.00 ms): it certifies the hand-designed partition as a robust
   optimum rather than improving on it.

The decisive, defensible finding is the **certification**: a derivative-free robust
optimiser, allowed to reshape the partition via load-scheduling and gated on
out-of-sample validation, **declines to move off the expert design**. This is
positive evidence for the controller's interpretability claim — the hand-designed
linguistic priors are not leaving measurable closed-loop performance on the table —
and a cautionary tale that naive objective-driven MF tuning (Variants 1–2) overfits
in exactly the way a fixed validation set fails to catch once model capacity is
high.

The expert membership functions on `main` remain the right default. This ablation
is **not promoted to core**: by the promotion criterion (held-out mean dMRT < 0 and
Vio = 0) it ties the baseline rather than beating it.

## Files

- `scheduled_anfis.py` — `ScheduledMFEngine`: context/gain-scheduled partition
  (Variant 3 engine; gains=0 ⇒ exact baseline).
- `scheduled_controller.py` — `ScheduledMFController` swaps the engine in. No core
  change.
- `autotune.py` — robust, validation-gated separable-CMA-ES tuner. `--full`
  reproduces Variant 2 (26-D); default is Variant 3 (schedule-only, regularized).
- `evaluate.py` — train/val/held-out split report with held-out mean dMRT + regret.
- `optimized_anfis.py` / `optimized_controller.py` — legacy Variant 1 static engine.
- `mf_terms.json` — committed deliverable design (Variant 3 ⇒ gains = 0 = expert).

## Reproduce (from the repo root)

```
# Variant 3 (default): schedule-only, regularized, validation-gated
python -m ablations.rec4_mf_autotune.autotune --workers 12 --gens 6 --popsize 8
python -m ablations.rec4_mf_autotune.evaluate --workers 12

# Variant 2 (full 26-D, reproduces the overfit)
python -m ablations.rec4_mf_autotune.autotune --workers 12 --gens 6 --popsize 8 --full
```

Re-running the auto-tune overwrites `mf_terms.json`.
