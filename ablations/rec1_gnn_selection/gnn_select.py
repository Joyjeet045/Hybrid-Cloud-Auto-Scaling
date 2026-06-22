"""Hand-written Graph Convolutional Network for bottleneck selection (PDF Rec 2,
Approach 1).

This is the distillation variant: a 2-layer GCN is trained offline to reproduce
the controller's analytic selection priority (the criticality-blended
critical-path ``score``) from local per-microservice signals plus the service
DAG. The analytic upward-rank centrality is deliberately withheld from the node
features, so the GCN must recover the spatial/structural component through message
passing -- exactly the capability a GNN is claimed to add over the interpretable
score.

No PyTorch Geometric dependency: the graph convolution is the standard Kipf &
Welling (2017) propagation ``A_hat = D^-1/2 (A + I) D^-1/2`` implemented with
dense tensors (service graphs here have <= ~15 nodes, so dense is trivial).

``build_graph_inputs`` is pure NumPy so the training recorder stays torch-free;
tensors are only created inside the model forward.

Node features (NO analytic rank / score, so there is no target leakage):
psi, lat_risk, pressure, et/deadline, log1p(forecast load)/10, vCPU fraction.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("psi", "lat_risk", "pressure", "et_n", "lam_n", "vcpu_n")
FEATURE_DIM = len(FEATURE_NAMES)
_VCPU_NORM = 16.0 * 15.0


def feature_dim(include_rank: bool = False) -> int:
    """Input dimensionality given the optional analytic-rank feature."""
    return FEATURE_DIM + (1 if include_rank else 0)


def build_graph_inputs(feats: dict, succ: dict, deadline: float,
                       include_rank: bool = False):
    """Build ``(types, X, A_hat)`` for a single decision step.

    ``feats`` is the controller's per-type feature dict; ``succ`` maps a type to
    its downstream DAG successor types. Returns the sorted type list, the
    ``[N, feature_dim]`` node-feature matrix and the symmetric-normalised
    adjacency ``A_hat`` (with self-loops) as NumPy arrays. When ``include_rank``
    is set the analytic HEFT upward rank is appended as an extra feature (the
    ceiling variant: the GCN is handed the structural signal directly and can
    reproduce the analytic selector almost exactly).
    """
    types = sorted(feats.keys())
    n = len(types)
    idx = {t: i for i, t in enumerate(types)}

    dim = feature_dim(include_rank)
    X = np.zeros((n, dim), dtype=np.float64)
    for t, i in idx.items():
        f = feats[t]
        X[i, 0] = f["psi"]
        X[i, 1] = f["lat_risk"]
        X[i, 2] = f["pressure"]
        X[i, 3] = f["et"] / max(deadline, 1e-6)
        X[i, 4] = np.log1p(max(0.0, f["lam"])) / 10.0
        X[i, 5] = f["type_total_vcpu"] / _VCPU_NORM
        if include_rank:
            X[i, 6] = f["rank"]

    A = np.zeros((n, n), dtype=np.float64)
    for t, i in idx.items():
        for s in succ.get(t, []) or []:
            j = idx.get(s)
            if j is not None:
                A[i, j] = 1.0
                A[j, i] = 1.0

    A_hat = _normalize_adj(A)
    return types, X, A_hat


def _normalize_adj(A: np.ndarray) -> np.ndarray:
    """Symmetric normalisation ``D^-1/2 (A + I) D^-1/2`` (Kipf & Welling 2017)."""
    n = A.shape[0]
    A_tilde = A + np.eye(n)
    deg = A_tilde.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0.0, deg ** -0.5, 0.0)
    return (A_tilde * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]


class GCNScorer:
    """Two-layer GCN regressor. Importable without torch at module load; torch is
    imported lazily inside :meth:`_ensure_torch`."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 16, seed: int = 0,
                 include_rank: bool = False):
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.include_rank = bool(include_rank)
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

    def predict(self, X: np.ndarray, A_hat: np.ndarray) -> np.ndarray:
        """Forward pass for a single graph; returns a length-N score vector."""
        torch = self._ensure_torch()
        self._module.eval()
        with torch.no_grad():
            xt = torch.from_numpy(np.ascontiguousarray(X))
            at = torch.from_numpy(np.ascontiguousarray(A_hat))
            out = self._module(xt, at)
        return out.cpu().numpy()

    def save(self, path: str) -> None:
        torch = self._ensure_torch()
        torch.save({
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "seed": self.seed,
            "include_rank": self.include_rank,
            "state_dict": self._module.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str) -> "GCNScorer":
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"], seed=ckpt["seed"],
                  include_rank=ckpt.get("include_rank", False))
        obj._module.load_state_dict(ckpt["state_dict"])
        obj._module.double()
        return obj
