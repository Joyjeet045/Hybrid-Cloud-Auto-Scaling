"""Pluggable ANFIS engine with objective-tuned membership functions.

Subclass of the unmodified core :class:`ANFISEngine`. After the baseline engine is
built with the hand-set ``LINGUISTIC_TERMS``, the Gaussian centres/widths are
optionally overwritten by an optimiser-tuned set (see :mod:`autotune`), preserving
every term's name and order so the rule base stays valid. With tuning off this is
identical to the baseline engine.
"""
from __future__ import annotations

from nfg_diagscale.decision.anfis import ANFISEngine


class OptimizedMFEngine(ANFISEngine):
    def __init__(self, config):
        super().__init__(config)
        fz = config.get("fuzzify", {})
        terms = fz.get("mf_terms")
        if fz.get("optimized_mf", False) and terms:
            self._rebuild_mf(terms)

    def _rebuild_mf(self, terms):
        """Overwrite centre/sigma of existing terms only (names/order preserved)."""
        for var_name, var_terms in terms.items():
            if var_name not in self.mf_params:
                continue
            for term_name, center_sigma in var_terms.items():
                if term_name not in self.mf_params[var_name]:
                    continue
                self.mf_params[var_name][term_name]["center"] = float(center_sigma[0])
                self.mf_params[var_name][term_name]["sigma"] = float(center_sigma[1])
