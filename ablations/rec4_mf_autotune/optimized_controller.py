"""Pluggable controller that swaps in the objective-tuned ANFIS engine.

Subclass of the unmodified core :class:`NFGDiagScaleController`. The only change is
to replace the decision engine with :class:`OptimizedMFEngine`, which reads the
tuned membership functions from ``config["fuzzify"]``. With tuning off the engine
is identical to the baseline, so this controller reproduces the baseline exactly.
"""
from __future__ import annotations

from nfg_diagscale.hgraph_policy.controller import NFGDiagScaleController
from ablations.rec4_mf_autotune.optimized_anfis import OptimizedMFEngine


class OptimizedMFController(NFGDiagScaleController):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.anfis = OptimizedMFEngine(config)
