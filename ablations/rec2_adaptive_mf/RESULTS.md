# Rec 2 / Approach 2 — Data-Driven (Clustered) Membership Functions

**Status:** NEUTRAL-TO-NEGATIVE RESULT. Pluggable, disabled on the default path.

## Idea

Replace the hand-tuned Gaussian membership-function centres/sigmas for the four
ANFIS inputs (`psi`, `omega`, `phi`, `rho`) with values learned from data. One or
more baseline episodes are run with input recording on; per variable the recorded
samples are clustered with deterministic 1-D k-means (quantile init, no RNG) and
the cluster centres become the new membership-function centres (sigmas = 0.6x
nearest-centre gap, floored). Term names and order are preserved so all 9 rules
stay intact.

## Method

- Baseline (`main`): all-12 mean MRT **206.00 ms**, all `Vio = 0`, 9/9 STAR wins.
- Evaluation: `evaluate.py`, 12 scenarios, 12 workers, seed 0. `dMRT = branch - base`.

## Results

### Variant A — single-scenario calibration (calibrate on A-12)

| metric | base | branch | delta |
|---|---|---|---|
| mean MRT | 206.00 | 205.73 | **-0.27 ms (flat)** |
| feasibility (Vio) | 0 | 0 | ok |
| worst case (W-14) | 383.11 | 474.51 | **+91.40 ms** |
| W-13 | 210.52 | 233.41 | +22.89 ms |
| best case (A-11) | 169.09 | 129.47 | -39.62 ms |

Net change is noise-level (-0.13%) but with large, scenario-dependent swings:
solid wins (A-11, A-12, W-12) are cancelled by a catastrophic W-14 regression. The
A-12-calibrated membership functions do not transfer to the heavy
Wikipedia / 14-service regime.

### Variant B — workload-diverse pooled calibration (pool N-13 + W-13 + A-13)

| metric | base | branch | delta |
|---|---|---|---|
| mean MRT | 206.00 | 213.04 | **+7.05 ms (worse)** |
| feasibility (Vio) | 0 | 0 | ok |
| worst case (W-14) | 383.11 | 510.09 | **+126.98 ms** |
| W-13 (in-sample!) | 210.52 | 223.11 | **+12.59 ms** |

Pooling across workloads made it strictly worse. The decisive observation:
**W-13 regressed even though it was in the calibration set.**

## Why it fails (confirmed empirically)

1-D clustering optimises input-space separation / coverage, an objective that is
uncorrelated with the closed-loop control loss (MRT). Moving membership-function
centres to cluster centroids changes which rules fire and how strongly, but in a
direction that has no reason to reduce latency -- and empirically does not.
Pooling across workloads washes out per-workload structure, degrading every regime
and blowing up the heaviest scenario (W-14) the most. That an in-sample scenario
(W-13) still regresses is direct evidence the calibration objective is misaligned
with control.

## Conclusion

Adaptive membership functions do not improve NF-DiagScale. Best variant is
net-flat with a bad worst case; the more principled pooled variant is strictly
worse. The hand-tuned (expert) membership functions on `main` remain the better,
more robust choice.

## Reproduce (from the repo root)

```
python -m ablations.rec2_adaptive_mf.evaluate --workers 12
python -m ablations.rec2_adaptive_mf.evaluate --workers 12 --pooled
```
