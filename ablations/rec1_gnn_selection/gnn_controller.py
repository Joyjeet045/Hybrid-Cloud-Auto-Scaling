"""Pluggable GNN-selection controller (PDF Rec 2, Approach 1).

Subclass of the unmodified core :class:`NFGDiagScaleController` that overrides the
no-op ``_select_bottleneck`` extension point to (a) record distillation samples
and/or (b) replace the analytic critical-path argmax with the distilled GCN's
argmax. The core control loop is untouched, so with no weights and no recording
this behaves exactly like the baseline.
"""
from __future__ import annotations

import os

import numpy as np

from nfg_diagscale.hgraph_policy.controller import NFGDiagScaleController
from ablations.rec1_gnn_selection import gnn_select


class GnnSelectionController(NFGDiagScaleController):
    def __init__(self, config, *, weights=None, include_rank=False, **kwargs):
        super().__init__(config, **kwargs)
        self._gnn_model = None
        self._gnn_include_rank = bool(include_rank)
        self._gnn_record = None
        if weights and os.path.exists(weights):
            self._gnn_model = gnn_select.GCNScorer.load(weights)
            self._gnn_include_rank = bool(self._gnn_model.include_rank)

    def record_on(self):
        """Start buffering ``(X, A_hat, y)`` distillation samples in act()."""
        self._gnn_record = []

    def pop_records(self):
        """Return and clear the buffered distillation samples."""
        rec = self._gnn_record or []
        self._gnn_record = None
        return rec

    def _select_bottleneck(self, state, feats, default_type):
        if self._gnn_record is None and self._gnn_model is None:
            return default_type
        succ = getattr(state, "succ", {}) or {}
        types, X, A_hat = gnn_select.build_graph_inputs(
            feats, succ, self.deadline, include_rank=self._gnn_include_rank)
        if self._gnn_record is not None:
            y = np.array([feats[t]["score"] for t in types], dtype=np.float64)
            self._gnn_record.append((X, A_hat, y))
        if self._gnn_model is not None and len(types) > 1:
            pred = self._gnn_model.predict(X, A_hat)
            return types[int(np.argmax(pred))]
        return default_type
