"""Pluggable ANFIS engine with data-driven membership functions (PDF Rec 2,
Approach 2).

Subclass of the unmodified core :class:`ANFISEngine`. After the baseline engine is
built (with the hand-set ``LINGUISTIC_TERMS``), the premise membership functions
are optionally replaced by ones calibrated from data (see :mod:`adaptive_mf`),
preserving each term's name and order so the rule base stays valid. It also adds
an input recorder used only during offline calibration. With calibration off this
is identical to the baseline engine.
"""
from __future__ import annotations

from nfg_diagscale.decision.anfis import ANFISEngine


class AdaptiveMFEngine(ANFISEngine):
    def __init__(self, config):
        super().__init__(config)
        self._record = None
        from ablations.rec2_adaptive_mf import adaptive_mf as _amf
        terms = _amf.adaptive_terms(config)
        if terms is not None:
            self._rebuild_mf(terms)

    def _rebuild_mf(self, terms):
        self.mf_params = {}
        for var_name, var_terms in terms.items():
            self.mf_params[var_name] = {}
            for term_name, (center, sigma) in var_terms.items():
                self.mf_params[var_name][term_name] = {
                    "center": float(center),
                    "sigma": float(sigma),
                }

    def record_inputs(self, flag=True):
        """Start/stop recording the (psi, omega, phi, rho) inputs seen by decide()."""
        self._record = [] if flag else None

    def pop_recorded_inputs(self):
        """Return the recorded input tuples and clear the recorder."""
        rec = self._record if self._record is not None else []
        self._record = None
        return rec

    def decide(self, psi, omega, phi, rho, **kwargs):
        if self._record is not None:
            self._record.append((psi, omega, phi, rho))
        return super().decide(psi, omega, phi, rho, **kwargs)
