"""Pluggable controller that uses the data-driven ANFIS engine (PDF Rec 2,
Approach 2).

Subclass of the unmodified core :class:`NFGDiagScaleController` that swaps the
baseline ANFIS engine for :class:`AdaptiveMFEngine` after construction. The core
control loop is untouched; with calibration disabled in the config this behaves
exactly like the baseline.
"""
from __future__ import annotations

from nfg_diagscale.hgraph_policy.controller import NFGDiagScaleController
from ablations.rec2_adaptive_mf.adaptive_anfis import AdaptiveMFEngine


class AdaptiveMFController(NFGDiagScaleController):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.anfis = AdaptiveMFEngine(config)
