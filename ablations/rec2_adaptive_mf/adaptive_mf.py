"""Data-driven (clustered) ANFIS membership-function calibration (PDF Rec 2,
Approach 2).

Instead of the hand-set ``LINGUISTIC_TERMS`` centres/sigmas, calibrate the premise
membership functions once, offline, from the empirical distribution of the ANFIS
inputs (``psi, omega, phi, rho``). One or more calibration episodes are run with
the baseline membership functions, the observed signal values are collected, and a
deterministic 1-D k-means (quantile-initialised, no RNG) places one fuzzy term per
cluster. Term names and ordering are preserved, so the open-shoulder ``gaussian_mf``
semantics and the rule base remain valid; only each term's centre/sigma moves to
where the data concentrates.

The result is cached, so calibration runs only once per process. A re-entrancy
guard makes the calibration episode itself fall back to the baseline membership
functions (otherwise building its controller would recurse).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np

_BUDGET = 200.0
_DEADLINE = 500.0
_INTERVALS = 480

_CACHE: dict = {}
_CALIBRATING = False

_VAR_ORDER = ("psi", "omega", "phi", "rho")


def _kmeans_1d(x, k, iters=100):
    """Deterministic 1-D k-means (quantile init, Lloyd updates); sorted centres."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.linspace(0.0, 1.0, k)
    uniq = np.unique(x)
    if uniq.size <= k:
        if uniq.size == k:
            return np.sort(uniq)
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            return np.full(k, lo)
        return np.linspace(lo, hi, k)

    qs = np.linspace(0.0, 1.0, k + 2)[1:-1]
    centres = np.quantile(x, qs)
    for _ in range(iters):
        d = np.abs(x[:, None] - centres[None, :])
        lab = d.argmin(axis=1)
        new = np.array([x[lab == j].mean() if np.any(lab == j) else centres[j]
                        for j in range(k)])
        new.sort()
        if np.allclose(new, centres):
            break
        centres = new
    return centres


def _sigmas_from_centres(centres, floor=0.05):
    """Sigma per term = 0.6x the gap to the nearest neighbouring centre, floored."""
    k = len(centres)
    if k == 1:
        return np.array([max(floor, 0.15)])
    sig = np.empty(k)
    for i in range(k):
        if i == 0:
            gap = centres[1] - centres[0]
        elif i == k - 1:
            gap = centres[k - 1] - centres[k - 2]
        else:
            gap = min(centres[i] - centres[i - 1], centres[i + 1] - centres[i])
        sig[i] = max(floor, 0.6 * abs(gap))
    return sig


def _collect_samples(config, scen_list, seed):
    """Run one baseline episode per (app, workload) and pool the recorded ANFIS
    input tuples across all of them."""
    from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
    from ablations.rec2_adaptive_mf.adaptive_controller import AdaptiveMFController

    samples = []
    for app, workload in scen_list:
        env = HGraphScaleEnv(app=app, workload=workload, seed=seed, budget=_BUDGET)
        state = env.reset(test=True)
        ctrl = AdaptiveMFController(config, deadline=_DEADLINE,
                                    total_intervals=_INTERVALS)
        ctrl.reset(budget_T=_BUDGET, total_intervals=_INTERVALS)
        ctrl.anfis.record_inputs(True)

        done = False
        with contextlib.redirect_stdout(io.StringIO()):
            while not done:
                action = ctrl.act(state)
                state, _r, done, _info = env.step(action)
        samples.extend(ctrl.anfis.pop_recorded_inputs())
    return samples


def _build_terms(samples, base_terms):
    """Cluster the recorded samples into new (centre, sigma) per term, preserving
    the term names/order of ``base_terms``."""
    if not samples:
        return None
    arr = np.asarray(samples, dtype=float)
    new_terms = {}
    for var, terms in base_terms.items():
        if var not in _VAR_ORDER:
            new_terms[var] = dict(terms)
            continue
        col = arr[:, _VAR_ORDER.index(var)]
        names = list(terms.keys())
        centres = _kmeans_1d(col, len(names))
        sigmas = _sigmas_from_centres(centres)
        new_terms[var] = {nm: (float(centres[i]), float(sigmas[i]))
                          for i, nm in enumerate(names)}
    return new_terms


def adaptive_terms(config):
    """Return data-driven LINGUISTIC_TERMS, or None to signal "use the baseline".

    Cached per (calibration scenarios, seed). Returns None while a calibration
    episode is in flight (re-entrancy guard) so that episode runs on the baseline
    membership functions.
    """
    global _CALIBRATING
    if _CALIBRATING:
        return None

    fz = config.get("fuzzify", {})
    if not fz.get("adaptive_mf", False):
        return None

    seed = int(fz.get("calib_seed", 0))
    calib_tags = fz.get("calib_tags", None)

    from run_star_comparison import SCENARIOS
    if calib_tags:
        scen_list = [SCENARIOS[t] for t in calib_tags]
        key = (tuple(calib_tags), seed)
    else:
        app = fz.get("calib_app", "A12")
        workload = fz.get("calib_workload", "alibaba")
        scen_list = [(app, workload)]
        key = ((app, workload), seed)
    if key in _CACHE:
        return _CACHE[key]

    from nfg_diagscale.decision.fuzzy_rules import LINGUISTIC_TERMS

    _CALIBRATING = True
    try:
        samples = _collect_samples(config, scen_list, seed)
        terms = _build_terms(samples, LINGUISTIC_TERMS)
    finally:
        _CALIBRATING = False

    _CACHE[key] = terms
    return terms
