"""Context-scheduled ANFIS engine: a load-adaptive (gain-scheduled) fuzzy partition.

This is the novel contribution over the static rec4 engine. Instead of one frozen
Gaussian partition, the membership-function centres/widths become a smooth function
of a live, observable load signal ``g in [0, 1]`` derived from the load factor
``psi``:

    centre_term(g) = clip(base_centre + gc_var * (g - 0.5) * span_var, lo, hi)
    sigma_term(g)  = max(0.05, base_sigma * (1 + gs_var * (g - 0.5)))

so the fuzzy partition *self-schedules* with the operating point -- one design
adapts its aggressiveness to low- vs high-load regimes instead of compromising on a
single static set (the failure mode of the static rec4). Term names, order and
count are preserved, so the rule base and the open-shoulder membership logic are
unchanged; at ``gains = 0`` with ``base = expert`` it reproduces the baseline
engine exactly.

Enabled via ``config["fuzzify"] = {"scheduled_mf": True, "design": <design>}`` where
``design`` is produced by :mod:`autotune`. With the flag off it is identical to the
baseline :class:`ANFISEngine`.
"""
from __future__ import annotations

from nfg_diagscale.decision.anfis import ANFISEngine

VARS = ("psi", "omega", "phi")
DOMAIN = {"psi": (0.3, 3.0), "omega": (0.08, 1.0), "phi": (0.03, 1.0)}
_MIN_GAP_FRAC = 0.02


class ScheduledMFEngine(ANFISEngine):
    def __init__(self, config):
        super().__init__(config)
        fz = config.get("fuzzify", {})
        self._scheduled = bool(fz.get("scheduled_mf", False))
        if not self._scheduled:
            return
        design = fz["design"]
        self._base = design["base_terms"]            # {var: {term: [centre, sigma]}}
        self._gains = design["gains"]                # {var: [gc, gs]}
        self._g_lo = float(design.get("g_lo", 0.4))
        self._g_hi = float(design.get("g_hi", 1.6))
        self._order = {v: list(self._base[v].keys()) for v in VARS}

    def _context(self, psi):
        """Map the load factor to a bounded schedule signal g in [0, 1]."""
        g = (float(psi) - self._g_lo) / (self._g_hi - self._g_lo + 1e-9)
        return min(1.0, max(0.0, g))

    def _apply_schedule(self, psi):
        g = self._context(psi)
        for var in VARS:
            lo, hi = DOMAIN[var]
            span = hi - lo
            gap = span * _MIN_GAP_FRAC
            gc, gs = self._gains[var]
            prev = lo
            for term in self._order[var]:
                bc, bs = self._base[var][term]
                centre = bc + gc * (g - 0.5) * span
                sigma = max(0.05, bs * (1.0 + gs * (g - 0.5)))
                centre = min(max(centre, prev + gap), hi)
                centre = max(centre, lo)
                self.mf_params[var][term]["center"] = float(centre)
                self.mf_params[var][term]["sigma"] = float(sigma)
                prev = centre

    def decide(self, psi, omega, phi, rho, n_current, cores_current,
               corrective=None):
        if self._scheduled:
            self._apply_schedule(psi)
        return super().decide(psi, omega, phi, rho, n_current, cores_current,
                              corrective)
