# Ablation Studies

Self-contained, pluggable negative-result experiments for NF-DiagScale. Each one
plugs into the **unmodified** core controller/engine via subclassing, so the
baseline architecture is preserved and the reporting pipeline reproduces the same
figures. Nothing here runs on the default path.

| Folder | PDF recommendation | Idea | Result |
|---|---|---|---|
| `rec1_gnn_selection/` | Rec 2 / Approach 1 | Replace the analytic critical-path selector with a distilled 2-layer GCN | **Strong negative** (+267% to +337% MRT) |
| `rec2_adaptive_mf/` | Rec 2 / Approach 2 | Replace the expert membership functions with data-driven (clustered) ones | **Neutral-to-negative** (flat mean, worse tail) |
| `rec3_gnn_forecast/` | Self-designed (Approach 3) | Add a GCN **residual corrector on the per-type load forecast** (not the selector); free labels = realised next load | **Positive** (-1.94 ms overall, -2.34 ms held-out, `Vio = 0`) |
| `rec4_mf_autotune/` | Self-designed (Approach 4) | Auto-tune the membership functions against the **control objective** — from static ES to a regularized, validation-gated, load-**scheduled** separable-CMA-ES tuner | **Neutral (rigorous)**: naive tuning overfits (held-out +0.69 → +11.26 ms); the principled schedule-only variant **certifies the expert partition** (returns it, held-out ±0.00 ms) |

Baseline to beat (`main`, seed 0): all-12 mean MRT **206.00 ms**, all `Vio = 0`,
9/9 STAR wins. `rec3` is the first ablation that beats it; see its `RESULTS.md`
for the torch-free-inference promotion path (the gain is a tiny `4x16` / `16x1`
GCN that can be re-expressed in pure NumPy).

## How they stay pluggable without touching core

- `rec1` adds a `GnnSelectionController` that overrides the no-op
  `_select_bottleneck` extension point on the core controller.
- `rec2` adds an `AdaptiveMFEngine` (subclass of `ANFISEngine`) and an
  `AdaptiveMFController` that swaps it in.
- `rec3` adds a `GnnForecastController` that overrides the no-op
  `_refine_forecast` extension point (additive, fail-safe residual on the
  load forecast).
- `rec4` adds a `ScheduledMFEngine` (subclass of `ANFISEngine`, context/gain-
  scheduled partition) and a `ScheduledMFController` that swaps it in; the tuned
  `mf_terms.json` design at `gains = 0` is byte-identical to the expert baseline.

Each folder has its own `evaluate.py` (parallel 12-scenario A/B vs baseline) and a
`RESULTS.md` write-up. See each `RESULTS.md` for exact reproduce commands.
