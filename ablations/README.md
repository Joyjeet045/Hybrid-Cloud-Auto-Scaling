# Ablation Studies

Self-contained, pluggable negative-result experiments for NF-DiagScale. Each one
plugs into the **unmodified** core controller/engine via subclassing, so the
baseline architecture is preserved and the reporting pipeline reproduces the same
figures. Nothing here runs on the default path.

| Folder | PDF recommendation | Idea | Result |
|---|---|---|---|
| `rec1_gnn_selection/` | Rec 2 / Approach 1 | Replace the analytic critical-path selector with a distilled 2-layer GCN | **Strong negative** (+267% to +337% MRT) |
| `rec2_adaptive_mf/` | Rec 2 / Approach 2 | Replace the expert membership functions with data-driven (clustered) ones | **Neutral-to-negative** (flat mean, worse tail) |

Baseline (`main`, seed 0): all-12 mean MRT **206.00 ms**, all `Vio = 0`, 9/9 STAR
wins. The self-designed GNN load-forecast residual corrector (formerly the
`rec3_gnn_forecast` ablation) **beat this** out of sample (-1.94 ms overall,
-2.34 ms held-out, `Vio = 0`) and has been **promoted into the core controller**
as the default; it is no longer an ablation. A second self-designed study
(objective-driven membership-function auto-tuning) was **neutral** — a regularized,
validation-gated, load-scheduled CMA-ES tuner certified the expert partition as a
robust optimum (it returned the expert design unchanged) — so it was dropped rather
than kept.

## How they stay pluggable without touching core

- `rec1` adds a `GnnSelectionController` that overrides the no-op
  `_select_bottleneck` extension point on the core controller.
- `rec2` adds an `AdaptiveMFEngine` (subclass of `ANFISEngine`) and an
  `AdaptiveMFController` that swaps it in.

Each folder has its own `evaluate.py` (parallel 12-scenario A/B vs baseline) and a
`RESULTS.md` write-up. See each `RESULTS.md` for exact reproduce commands.
