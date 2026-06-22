"""Pluggable controller with the GNN residual load forecaster.

Subclass of the unmodified core :class:`NFGDiagScaleController` that overrides the
no-op ``_refine_forecast`` extension point to add a graph-aware residual to the
per-series Kalman+Holt forecast. With no weights and no recording it returns the
baseline forecast unchanged, so it reproduces the baseline exactly.

Two modes:
  * recording (``record_on``): buffers per-interval ``(graph, base_pred, observed)``
    so :mod:`train_forecast` can build free residual labels from a baseline rollout;
  * serving (``weights=...``): adds the trained GCN's residual to each forecast.
"""
from __future__ import annotations

import os

import numpy as np

from nfg_diagscale.hgraph_policy.controller import NFGDiagScaleController
from ablations.rec3_gnn_forecast import gnn_forecast


class GnnForecastController(NFGDiagScaleController):
    def __init__(self, config, *, weights=None, **kwargs):
        super().__init__(config, **kwargs)
        self._fc_model = None
        self._fc_record = None
        self._fc_cache = {}
        if weights and os.path.exists(weights):
            self._fc_model = gnn_forecast.ForecastGCN.load(weights)

    def record_on(self):
        """Start buffering per-interval forecast records for offline labelling."""
        self._fc_record = []

    def pop_records(self):
        """Return and clear the buffered forecast records."""
        rec = self._fc_record if self._fc_record is not None else []
        self._fc_record = None
        return rec

    def _build_interval_graph(self, state, by_type):
        types = list(by_type.keys())
        obs = {s: self._type_last_load(by_type[s]) for s in types}
        et = {s: float(state.proc_time.get(s, 0.0)) for s in types}
        vcpu = {s: float(sum(c.vcpu for c in by_type[s])) for s in types}
        rank = {s: float(state.rank.get(s, 0.0)) for s in types}
        return gnn_forecast.build_forecast_inputs(
            types, obs, et, vcpu, rank, state.succ, self.deadline)

    def _refine_forecast(self, con_type, predicted_lam, observed, state, by_type):
        if self._fc_model is None and self._fc_record is None:
            return predicted_lam

        slot = int(state.slot_index)
        cache = self._fc_cache
        if cache.get("slot") != slot:
            order, X, A_hat = self._build_interval_graph(state, by_type)
            cache.clear()
            cache["slot"] = slot
            cache["order"] = order
            cache["idx"] = {t: i for i, t in enumerate(order)}
            cache["X"] = X
            cache["A"] = A_hat
            if self._fc_model is not None and len(order) > 0:
                cache["resid"] = self._fc_model.predict(X, A_hat)
            if self._fc_record is not None:
                rec = {"order": order, "X": X, "A": A_hat, "base": {}, "obs": {}}
                self._fc_record.append(rec)
                cache["rec"] = rec

        if self._fc_record is not None and "rec" in cache:
            cache["rec"]["base"][int(con_type)] = float(predicted_lam)
            cache["rec"]["obs"][int(con_type)] = float(observed)

        if self._fc_model is not None:
            i = cache["idx"].get(con_type)
            if i is not None:
                predicted_lam = max(0.0, float(predicted_lam) + float(cache["resid"][i]))
        return predicted_lam
