# Rec 4 / Approach 4 — Objective-Driven Membership-Function Auto-Tuning (self-designed)

**Status:** NEUTRAL-TO-NEGATIVE RESULT. Improves the calibration set but does not
generalise. Pluggable, disabled on the default path.

## Motivation (why this is different from rec2)

The user asked to *stop hand-picking* the membership functions and have an
algorithm choose centres/widths that are "optimal for our task without harming
performance".

`rec2` clustered the recorded input samples (1-D k-means) and used the centroids
as membership-function centres. That optimises **input-space coverage**, an
objective uncorrelated with control loss — and it failed (net-flat, bad tail,
in-sample W-13 regressed).

This approach fixes the objective mismatch directly: optimise the membership-
function centres/widths **against the closed-loop control loss itself** (mean MRT
on a calibration set), with a hard feasibility penalty. To guarantee "no harm" it
is **seeded at the expert membership functions and kept elitist**, so the result
can never be worse than the hand-tuned baseline *on the calibration set* by
construction.

## Method

- **Search:** derivative-free `(1 + λ)` evolution strategy. Genome = per-term
  Gaussian `(center, sigma)` for `psi` (4 terms), `omega` (3), `phi` (3); `rho`
  is fixed. Centres are **monotone-decoded** (ordered per variable) so the
  linguistic ordering stays valid; term **names and count are unchanged**, so all
  9 rules and the open-shoulder logic keyed on term names stay intact.
- **Objective:** `mean_MRT(calib) + 1e5 * total_Vio`. Calib set =
  `{N-13, W-13, A-13}` (workload-diverse), seed 0, evaluated in parallel.
- **Elitism:** generation 0 evaluates the expert seed; the incumbent is only
  replaced by a strictly better candidate. The expert is therefore a hard lower
  bound on the calibration score.
- **Injection:** an `OptimizedMFEngine(ANFISEngine)` overwrites the centre/sigma
  of the *existing* terms from a tuned `mf_terms.json`; an
  `OptimizedMFController` swaps it in. **No core change.**

## Auto-tune run (gens 5, popsize 8, workers 12)

```
[gen 0] expert seed:      mean_mrt = 160.371  vio = 0.000
[gen 1] -> 159.357  <- new incumbent
[gen 2] -> 158.101  <- new incumbent
[gen 5] -> 157.768  <- new incumbent
calibration mean MRT: 157.768  (total vio 0.000)
```

The elitist guarantee holds: the tuned membership functions are **-2.60 ms
(-1.62%)** on the calibration set versus the expert seed, all feasible. So far,
so good — "no harm" is satisfied where we optimised.

## Results (12 scenarios, seed 0, `dMRT = tuned - base`)

| scen | base | tuned | dMRT | Vio | split |
|---|---|---|---|---|---|
| N-11 | 152.91 | 158.40 | +5.49 | 0.0 | held-out |
| N-12 | 157.39 | 156.95 | -0.44 | 0.0 | held-out |
| N-13 | 140.52 | 140.52 | +0.00 | 0.0 | calib |
| W-11 | 229.34 | 229.69 | +0.35 | 0.0 | held-out |
| W-12 | 261.53 | 239.43 | **-22.10** | 0.0 | held-out |
| W-13 | 210.52 | 210.80 | +0.28 | 0.0 | calib |
| A-11 | 169.09 | 168.08 | -1.01 | 0.0 | held-out |
| A-12 | 162.74 | 165.74 | +3.00 | 0.0 | held-out |
| A-13 | 130.07 | 121.98 | **-8.09** | 0.0 | calib |
| N-14 | 249.87 | 250.21 | +0.34 | 0.0 | held-out |
| W-14 | 383.11 | 375.62 | -7.49 | 0.0 | held-out |
| A-14 | 224.89 | 252.99 | **+28.10** | 0.0 | held-out |
| **mean** | **206.00** | **205.87** | **-0.13** | infeasible=0 | |

**Held-out mean dMRT: +0.69 ms (9 scenarios).** Calibration set improves -2.60 ms;
the full suite is net-flat (-0.13 ms, noise level).

## Why it does not generalise

The optimiser does exactly what it was told: it drives down MRT **on the three
calibration scenarios** (A-13 -8.09 in-sample). But the membership functions it
finds are a **fit to those three load regimes**, and the gain does not transfer:
the held-out set is slightly *worse* on average (+0.69 ms) with very high
variance (W-12 -22.10 vs A-14 +28.10). The objective is correctly aligned with
control (unlike rec2), yet the *capacity to overfit* a small, low-dimensional
calibration set dominates.

The decisive observation: with the objective now *correct*, the expert membership
functions are already **at or near the objective-optimal operating point** — the
best a task-driven search can do on held-out data is roughly tie them. That is a
positive finding for the method's interpretability claim: the hand-designed
linguistic priors are not leaving measurable performance on the table.

## Conclusion

Automated, objective-driven membership-function selection **matches** expert tuning
but does not beat it out of sample: a -2.60 ms in-sample gain that washes out to
-0.13 ms (net-flat) across all 12 scenarios, with a high-variance tail. "Without
harming performance" is satisfied on the calibration set and roughly on aggregate,
but there is no robust, transferable win. The expert membership functions on `main`
remain the right default; this confirms they are already well-placed.

## Reproduce (from the repo root)

```
python -m ablations.rec4_mf_autotune.autotune --workers 12 --gens 5 --popsize 8
python -m ablations.rec4_mf_autotune.evaluate --workers 12
```

The tuned `mf_terms.json` (the deliverable design) is committed; re-running the
auto-tune overwrites it.
