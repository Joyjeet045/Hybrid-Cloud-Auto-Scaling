# Rec 2 / Approach 1 — GNN-Distilled Bottleneck Selection

**Status:** STRONG NEGATIVE RESULT. Pluggable, disabled on the default path.

## Idea

Replace NF-DiagScale's interpretable analytic critical-path score (the
criticality-blended `score` that selects *which* microservice to scale each
interval) with a 2-layer Graph Convolutional Network (Kipf & Welling 2017),
trained by **distillation** to reproduce that analytic score from the service DAG
plus local per-node signals. PyTorch (CPU), hand-written dense GCN, no PyTorch
Geometric.

Node features (local; **no** analytic rank in the honest variant):
`psi, lat_risk, pressure, et/deadline, log1p(lam)/10, vcpu_norm`.
Target: the analytic per-graph argmax (which service the baseline would scale).

## Method

- Baseline (`main`): all-12 mean MRT **206.00 ms**, all `Vio = 0`, 9/9 STAR wins.
- Distillation data: 3 baseline episodes (N-13, W-13, A-13), 1440 decision-step
  graphs, seed 0.
- Objective: softmax cross-entropy over a graph's nodes with the analytic argmax
  as the target class (MSE on the score level was tried first and scrambles the
  ranking — see below).
- Evaluation: `evaluate.py`, 12 scenarios, 12 workers, seed 0.

## Results

### Training (in-sample selection agreement = GCN argmax == analytic argmax)

| training setup | in-sample agreement |
|---|---|
| MSE on score level (300 ep) | **9.1%** (~random) |
| cross-entropy, local features (400 ep) | **41.9%** |
| cross-entropy, **+ analytic rank fed in**, hidden 32, 2000 ep | **89.5%** |

### Closed-loop evaluation (12 scenarios; mean MRT, ms; lower is better)

| variant | mean MRT | vs base | feasibility |
|---|---|---|---|
| baseline (`main`, analytic score) | 206.00 | — | Vio = 0 |
| GNN, local features (42% agreement) | 755.41 | **+549.41 (+267%)** | Vio = 0 |
| GNN, +rank ceiling (89.5% agreement) | 900.52 | **+694.52 (+337%)** | Vio = 0 |

Per-scenario the GNN inflates MRT 3-9x (e.g. W-11 229 -> 1808, W-14 383 -> 1883).
`Vio = 0` only because the budget guard still blocks overspend; the controller is
simply scaling the wrong services, starving the true bottleneck.

## Why it fails (confirmed empirically)

1. **No upside by construction.** Distillation trains the GNN to imitate the
   analytic score; its best behaviour is to match a signal already computed
   exactly, cheaply and interpretably.
2. **Selection is an argmax, not a regression.** MSE distillation (9% agreement)
   fits the level but destroys the ranking that drives control.
3. **GCN smoothing fights per-node selection.** Neighbour averaging blurs the
   local pressure/rank spike that identifies the bottleneck, capping in-sample
   agreement at ~90% even when the rank is provided.
4. **Compounding errors (behavioural cloning).** Agreement is measured on the
   baseline's trajectory. Once the GNN drives decisions, a single wrong pick
   shifts the system off-distribution and errors snowball (Ross & Bagnell 2011) --
   which is why the 89.5%-agreement model is *worse* closed-loop (+694 ms) than
   the 42% one (+549 ms).

## Conclusion

A GNN-distilled selector does not improve NF-DiagScale; it is catastrophic and,
by construction, cannot exceed the analytic critical-path score it imitates. The
exact, cheap, interpretable score on `main` is strictly superior.

## Reproduce (from the repo root)

```
python -m ablations.rec1_gnn_selection.train_gnn --epochs 400 --lr 0.02
python -m ablations.rec1_gnn_selection.train_gnn --include-rank --epochs 2000 --lr 0.03 --hidden 32
python -m ablations.rec1_gnn_selection.evaluate --workers 12
```
