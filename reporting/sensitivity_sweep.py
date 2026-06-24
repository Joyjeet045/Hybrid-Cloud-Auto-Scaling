"""One-at-a-time (OAT) hyperparameter sensitivity sweep for NF-DiagScale.

Varies each of the three tunable knobs around its default, holding the other two
fixed, and reports the all-12-scenario mean MRT, the worst-case scenario MRT, and
the number of budget-infeasible scenarios. A flat MRT curve with Vio==0 across the
range demonstrates the controller is robust to the exact setting.

  zeta  = criticality.weight        (criticality blend)     default 0.4
  kappa = adaptive.corrective_weight (deterministic anchor)  default 0.45
  eta   = adaptive.eta              (consequent adapt rate)  default 0.10

Usage:  python reporting/sensitivity_sweep.py --workers 12
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nfg_diagscale.config import load_config
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from nfg_diagscale.hgraph_policy.controller import NFGDiagScaleController

BUDGET, DEADLINE, TOTAL = 200.0, 500.0, 480
SCENARIOS = {
    "N-11": ("A11", "nasa"), "N-12": ("A12", "nasa"), "N-13": ("A13", "nasa"),
    "W-11": ("A11", "wiki"), "W-12": ("A12", "wiki"), "W-13": ("A13", "wiki"),
    "A-11": ("A11", "alibaba"), "A-12": ("A12", "alibaba"), "A-13": ("A13", "alibaba"),
    "N-14": ("A14", "nasa"), "W-14": ("A14", "wiki"), "A-14": ("A14", "alibaba"),
}

DEFAULT = (0.4, 0.45, 0.10)
ZETA_VALS = [0.0, 0.2, 0.4, 0.6, 0.8]
KAPPA_VALS = [0.25, 0.45, 0.65]
ETA_VALS = [0.05, 0.10, 0.20]


def _unique_configs():
    cfgs = {DEFAULT}
    for z in ZETA_VALS:
        cfgs.add((z, DEFAULT[1], DEFAULT[2]))
    for k in KAPPA_VALS:
        cfgs.add((DEFAULT[0], k, DEFAULT[2]))
    for e in ETA_VALS:
        cfgs.add((DEFAULT[0], DEFAULT[1], e))
    return sorted(cfgs)


def run_one(args):
    (zeta, kappa, eta), tag = args
    app, workload = SCENARIOS[tag]
    cfg = load_config()
    cfg.setdefault("criticality", {})["weight"] = zeta
    cfg.setdefault("adaptive", {})["corrective_weight"] = kappa
    cfg.setdefault("adaptive", {})["eta"] = eta
    env = HGraphScaleEnv(app=app, workload=workload, seed=0, budget=BUDGET)
    state = env.reset(test=True)
    ctrl = NFGDiagScaleController(cfg, deadline=DEADLINE, total_intervals=TOTAL)
    ctrl.reset(budget_T=BUDGET, total_intervals=TOTAL)
    done, info = False, {}
    with contextlib.redirect_stdout(io.StringIO()):
        while not done:
            state, _r, done, info = env.step(ctrl.act(state))
    mrt = float(info.get("average_resptime", float("nan")))
    cost = float(info.get("VM_cost", float("nan")))
    return {"cfg": [zeta, kappa, eta], "tag": tag, "mrt": mrt,
            "vio": max(0.0, cost - BUDGET)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    configs = _unique_configs()
    jobs = [(c, t) for c in configs for t in SCENARIOS]
    print(f"Running {len(jobs)} episodes ({len(configs)} configs x 12 scenarios) "
          f"over {args.workers} workers...")
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run_one, jobs):
            results.setdefault(tuple(r["cfg"]), []).append(r)

    def summarise(cfg):
        rs = results[cfg]
        mrts = np.array([x["mrt"] for x in rs])
        infeasible = sum(1 for x in rs if x["vio"] > 1e-9)
        return mrts.mean(), mrts.max(), infeasible

    out = {"default": list(DEFAULT), "rows": []}
    print("\n=== Sensitivity: criticality blend zeta (kappa=0.45, eta=0.10) ===")
    print("  zeta   meanMRT   maxMRT   infeasible")
    for z in ZETA_VALS:
        m, mx, inf = summarise((z, DEFAULT[1], DEFAULT[2]))
        tag = "  <- default" if z == DEFAULT[0] else ""
        print(f"  {z:<5}  {m:7.2f}  {mx:7.2f}   {inf}{tag}")
        out["rows"].append({"param": "zeta", "value": z, "mean": m, "max": mx, "infeasible": inf})

    print("\n=== Sensitivity: anchor blend kappa (zeta=0.4, eta=0.10) ===")
    print("  kappa  meanMRT   maxMRT   infeasible")
    for k in KAPPA_VALS:
        m, mx, inf = summarise((DEFAULT[0], k, DEFAULT[2]))
        tag = "  <- default" if k == DEFAULT[1] else ""
        print(f"  {k:<5}  {m:7.2f}  {mx:7.2f}   {inf}{tag}")
        out["rows"].append({"param": "kappa", "value": k, "mean": m, "max": mx, "infeasible": inf})

    print("\n=== Sensitivity: adapt rate eta (zeta=0.4, kappa=0.45) ===")
    print("  eta    meanMRT   maxMRT   infeasible")
    for e in ETA_VALS:
        m, mx, inf = summarise((DEFAULT[0], DEFAULT[1], e))
        tag = "  <- default" if e == DEFAULT[2] else ""
        print(f"  {e:<5}  {m:7.2f}  {mx:7.2f}   {inf}{tag}")
        out["rows"].append({"param": "eta", "value": e, "mean": m, "max": mx, "infeasible": inf})

    with open(os.path.join(_REPO_ROOT, "reporting", "sensitivity_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote reporting/sensitivity_results.json")


if __name__ == "__main__":
    main()
