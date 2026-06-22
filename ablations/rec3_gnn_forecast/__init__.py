"""GNN residual load forecaster (self-designed ablation).

A Graph Convolutional Network adds a topology-aware residual to the per-series
Kalman+Holt forecast by propagating upstream observed load along the service DAG.
Fail-safe by construction (a zero residual reduces to the baseline) and trained on
free labels (the realised next-interval load), with no counterfactual simulation.
"""
