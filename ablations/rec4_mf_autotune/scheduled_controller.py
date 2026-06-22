"""Pluggable controller that swaps in the context-scheduled ANFIS engine.

Subclass of the unmodified core :class:`NFGDiagScaleController`. The only change is
to replace the decision engine with :class:`ScheduledMFEngine`, which reads the
load-scheduled membership-function design from ``config["fuzzify"]``. With
scheduling off the engine is identical to the baseline, so this controller
reproduces the baseline exactly.
"""
from __future__ import annotations

from nfg_diagscale.hgraph_policy.controller import NFGDiagScaleController
from ablations.rec4_mf_autotune.scheduled_anfis import ScheduledMFEngine


class ScheduledMFController(NFGDiagScaleController):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.anfis = ScheduledMFEngine(config)
