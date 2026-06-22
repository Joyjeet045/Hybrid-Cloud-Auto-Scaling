"""Derivative-free auto-tuning of the ANFIS Gaussian membership functions.

A (1+lambda) evolution strategy searches the centre/width of every fuzzy term in
``psi``, ``omega`` and ``phi`` (``rho`` is a fixed binary anchor) to minimise the
mean closed-loop MRT over a small, workload-diverse calibration set, subject to a
hard zero-violation constraint (cost overruns are penalised heavily). The search
is:

  * seeded at the expert ``LINGUISTIC_TERMS`` (the incumbent at generation 0),
  * elitist -- the incumbent is re-evaluated every generation and only replaced by
    a strictly better candidate, so the returned design can never be worse than the
    hand-tuned baseline on the calibration scenarios, and
  * monotone-decoded -- term centres stay ordered (low < moderate < ...) so the
    linguistic partition, and therefore the rule base, stays valid.

The tuned design is written to ``mf_terms.json`` next to this file; evaluate.py
loads it for the held-out 12-scenario comparison.

Run from the repo root:
    python -m ablations.rec4_mf_autotune.autotune --workers 12
    python -m ablations.rec4_mf_autotune.autotune --workers 12 --quick
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from nfg_diagscale.config import load_config
from nfg_diagscale.decision.fuzzy_rules import LINGUISTIC_TERMS
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from run_star_comparison import BUDGET, DEADLINE, SCENARIOS
from ablations.rec4_mf_autotune.optimized_controller import OptimizedMFController

HERE = os.path.dirname(os.path.abspath(__file__))
TERMS_PATH = os.path.join(HERE, "mf_terms.json")
INTERVALS = 480
VARS = ("psi", "omega", "phi")
DOMAIN = {"psi": (0.3, 3.0), "omega": (0.08, 1.0), "phi": (0.03, 1.0)}
CALIB_TAGS = ("N-13", "W-13", "A-13")
VIO_PENALTY = 1.0e5


def build_params():
    """Flat list of optimisable terms in canonical order with expert seeds."""
    params = []
    for var in VARS:
        for term, (center, sigma) in LINGUISTIC_TERMS[var].items():
            params.append({"var": var, "term": term,
                           "c0": float(center), "s0": float(sigma)})
    return params


def decode(theta, params):
    """Map a normalised parameter vector to a valid, monotone terms dict."""
    by_var = OrderedDict()
    for k, p in enumerate(params):
        by_var.setdefault(p["var"], []).append((k, p))

    terms = {}
    for var, items in by_var.items():
        lo, hi = DOMAIN[var]
        gap = (hi - lo) * 0.04
        n = len(items)
        prev = lo
        out = {}
        for j, (k, p) in enumerate(items):
            tc = float(np.clip(theta[2 * k], 0.5, 1.8))
            ts = float(np.clip(theta[2 * k + 1], 0.5, 2.0))
            center = p["c0"] * tc
            upper = hi - gap * (n - 1 - j)
            center = min(max(center, prev + gap), upper)
            center = float(max(center, lo))
            sigma = float(max(0.05, p["s0"] * ts))
            out[p["term"]] = [center, sigma]
            prev = center
        terms[var] = out
    return terms


def expert_terms():
    return {var: {term: [float(c), float(s)]
                  for term, (c, s) in LINGUISTIC_TERMS[var].items()}
            for var in VARS}


def _eval_one(task):
    cand_id, tag, terms, seed = task
    app, workload = SCENARIOS[tag]
    cfg = load_config()
    cfg["fuzzify"] = {"optimized_mf": True, "mf_terms": terms}
    env = HGraphScaleEnv(app=app, workload=workload, seed=seed, budget=BUDGET)
    state = env.reset(test=True)
    ctrl = OptimizedMFController(cfg, deadline=DEADLINE, total_intervals=INTERVALS)
    ctrl.reset(budget_T=BUDGET, total_intervals=INTERVALS)
    done = False
    info = {}
    with contextlib.redirect_stdout(io.StringIO()):
        while not done:
            action = ctrl.act(state)
            state, _r, done, info = env.step(action)
    mrt = float(info.get("average_resptime", float("nan")))
    cost = float(info.get("VM_cost", 0.0))
    return cand_id, tag, mrt, max(0.0, cost - BUDGET)


def evaluate_population(terms_list, calib_tags, seed, workers):
    """Return (score, mean_mrt, total_vio) per candidate via a parallel pool."""
    tasks = []
    for ci, terms in enumerate(terms_list):
        for tag in calib_tags:
            tasks.append((ci, tag, terms, seed))
    agg = {ci: {"mrt": [], "vio": 0.0} for ci in range(len(terms_list))}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for ci, tag, mrt, vio in ex.map(_eval_one, tasks):
            agg[ci]["mrt"].append(mrt)
            agg[ci]["vio"] += vio
    out = []
    for ci in range(len(terms_list)):
        mean_mrt = float(np.mean(agg[ci]["mrt"]))
        total_vio = float(agg[ci]["vio"])
        score = mean_mrt + VIO_PENALTY * total_vio
        out.append((score, mean_mrt, total_vio))
    return out


def autotune(workers, seed, gens, popsize, sigma0, calib_tags):
    params = build_params()
    dim = 2 * len(params)
    rng = np.random.default_rng(seed)

    incumbent = np.ones(dim)
    incumbent_terms = decode(incumbent, params)
    s0, m0, v0 = evaluate_population([incumbent_terms], calib_tags, seed, workers)[0]
    best_score, best_mrt, best_vio = s0, m0, v0
    print(f"[gen 0] expert seed: mean_mrt={m0:.3f} vio={v0:.3f} score={s0:.3f}")

    sigma = sigma0
    for g in range(1, gens + 1):
        cands = [incumbent.copy()]
        for _ in range(popsize - 1):
            cands.append(incumbent + sigma * rng.standard_normal(dim))
        terms_list = [decode(c, params) for c in cands]
        scored = evaluate_population(terms_list, calib_tags, seed, workers)
        order = sorted(range(len(cands)), key=lambda i: scored[i][0])
        bi = order[0]
        if scored[bi][0] < best_score - 1e-9:
            best_score, best_mrt, best_vio = scored[bi]
            incumbent = cands[bi].copy()
            incumbent_terms = terms_list[bi]
            tag = "  <- new incumbent"
        else:
            tag = ""
        print(f"[gen {g}] best_cand mean_mrt={scored[bi][1]:.3f} "
              f"vio={scored[bi][2]:.3f} | incumbent mean_mrt={best_mrt:.3f}"
              f" sigma={sigma:.3f}{tag}")
        sigma *= 0.82

    return incumbent_terms, best_mrt, best_vio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--sigma0", type=float, default=0.18)
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget for a plumbing smoke test")
    args = ap.parse_args()

    gens, popsize = args.gens, args.popsize
    if args.quick:
        gens, popsize = 2, 4

    print(f"Auto-tuning MF on {list(CALIB_TAGS)}  gens={gens} popsize={popsize} "
          f"workers={args.workers}")
    terms, mrt, vio = autotune(args.workers, args.seed, gens, popsize,
                               args.sigma0, CALIB_TAGS)
    payload = {
        "calib_tags": list(CALIB_TAGS),
        "seed": args.seed,
        "calib_mean_mrt": mrt,
        "calib_total_vio": vio,
        "mf_terms": terms,
    }
    with open(TERMS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved tuned MFs -> {TERMS_PATH}")
    print(f"calibration mean MRT: {mrt:.3f}  (total vio {vio:.3f})")


if __name__ == "__main__":
    main()
