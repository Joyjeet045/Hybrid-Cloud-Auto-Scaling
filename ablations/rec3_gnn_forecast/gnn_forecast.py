"""GNN residual load forecaster (self-designed ablation).

Motivation. The baseline forecasts each microservice's load with an independent
per-series Kalman+Holt filter, which is blind to the call graph: in a microservice
DAG a service's near-future load is largely driven by the *current* load of its
upstream callers. This module adds a Graph Convolutional Network that predicts a
*residual* correction to the per-series forecast by aggregating upstream observed
load along the DAG. Because the correction is additive on top of the existing
forecast, a GCN that learns ~zero residual reduces exactly to the baseline.

Labels are free and require no counterfactual simulation: the target residual for
interval ``k`` is simply ``realised_load[k+1] - kalman_forecast[k]`` per service,
read straight from the next interval of any rollout.

The graph convolution is the dense Kipf & Welling (2017) propagation with a
*directed, upstream* adjacency (a node aggregates from its predecessors), so the
message passing carries load forward along the invocation edges. No PyTorch
Geometric dependency; torch is imported lazily so the recorder stays torch-free.
"""
from __future__ import annotations

import numpy as np

FORECAST_FEATURES = ("obs_n", "et_n", "vcpu_n", "rank")
FEATURE_DIM = len(FORECAST_FEATURES)
_VCPU_NORM = 16.0 * 15.0
_LOAD_NORM = 10.0


def build_forecast_inputs(types, obs, et, vcpu, rank, succ, deadline):
    """Build ``(order, X, A_hat)`` for one decision step.

    ``obs/et/vcpu/rank`` are per-type dicts. The adjacency is directed upstream:
    for a DAG edge ``u -> s`` node ``s`` aggregates from ``u`` (load flows
    downstream), row-normalised with self-loops so each node averages itself with
    its upstream callers.
    """
    order = sorted(types)
    n = len(order)
    idx = {t: i for i, t in enumerate(order)}

    X = np.zeros((n, FEATURE_DIM), dtype=np.float64)
    for t, i in idx.items():
        X[i, 0] = np.log1p(max(0.0, float(obs[t]))) / _LOAD_NORM
        X[i, 1] = float(et[t]) / max(deadline, 1e-6)
        X[i, 2] = float(vcpu[t]) / _VCPU_NORM
        X[i, 3] = float(rank[t])

    A = np.zeros((n, n), dtype=np.float64)
    for t, i in idx.items():
        for s in succ.get(t, []) or []:
            j = idx.get(s)
            if j is not None:
                A[j, i] = 1.0

    A_hat = _row_norm_selfloop(A)
    return order, X, A_hat


def _row_norm_selfloop(A):
    n = A.shape[0]
    a_tilde = A + np.eye(n)
    deg = a_tilde.sum(axis=1, keepdims=True)
    deg[deg == 0.0] = 1.0
    return a_tilde / deg


class ForecastGCN:
    """Two-layer GCN regressor predicting a standardised per-node load residual.

    Importable without torch; torch is imported lazily inside :meth:`_build`. The
    output is de-standardised with the stored ``(y_mean, y_std)`` so callers get a
    raw additive residual in load units.
    """

    def __init__(self, in_dim=FEATURE_DIM, hidden=16, seed=0,
                 y_mean=0.0, y_std=1.0):
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std) if abs(float(y_std)) > 1e-9 else 1.0
        self._torch = None
        self._module = None
        self._build()

    def _ensure_torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def _build(self):
        torch = self._ensure_torch()
        nn = torch.nn
        torch.manual_seed(self.seed)

        class _Net(nn.Module):
            def __init__(self, in_dim, hidden):
                super().__init__()
                self.w1 = nn.Linear(in_dim, hidden, bias=True)
                self.w2 = nn.Linear(hidden, 1, bias=True)
                self.act = nn.ReLU()

            def forward(self, X, A_hat):
                h = self.act(A_hat @ self.w1(X))
                out = A_hat @ self.w2(h)
                return out.squeeze(-1)

        self._module = _Net(self.in_dim, self.hidden).double()

    @property
    def module(self):
        return self._module

    def forward_std(self, X, A_hat):
        """Standardised forward pass (used during training)."""
        torch = self._ensure_torch()
        xt = torch.from_numpy(np.ascontiguousarray(X))
        at = torch.from_numpy(np.ascontiguousarray(A_hat))
        return self._module(xt, at)

    def predict(self, X, A_hat):
        """Raw (de-standardised) residual vector for one graph."""
        torch = self._ensure_torch()
        self._module.eval()
        with torch.no_grad():
            out = self.forward_std(X, A_hat)
        return out.cpu().numpy() * self.y_std + self.y_mean

    def save(self, path):
        torch = self._ensure_torch()
        torch.save({
            "in_dim": self.in_dim, "hidden": self.hidden, "seed": self.seed,
            "y_mean": self.y_mean, "y_std": self.y_std,
            "state_dict": self._module.state_dict(),
        }, path)

    @classmethod
    def load(cls, path):
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"], seed=ckpt["seed"],
                  y_mean=ckpt.get("y_mean", 0.0), y_std=ckpt.get("y_std", 1.0))
        obj._module.load_state_dict(ckpt["state_dict"])
        obj._module.double()
        return obj
