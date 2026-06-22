"""Parallel 12-scenario A/B evaluation of the auto-tuned membership functions vs
the expert-tuned baseline.

Run from the repo root (after autotune.py has produced mf_terms.json):
    python -m ablations.rec4_mf_autotune.evaluate --workers 12
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor

from nfg_diagscale.config import load_config
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from run_star_comparison import BUDGET, DEADLINE, SCENARIOS
from ablations.rec4_mf_autotune.optimized_controller import OptimizedMFController

HERE = os.path.dirname(os.path.abspath(__file__))
TERMS_PATH = os.path.join(HERE, "mf_terms.json")
INTERVALS = 480

BASELINE = {
    "N-11": 152.91, "N-12": 157.39, "N-13": 140.52,
    "W-11": 229.34, "W-12": 261.53, "W-13": 210.52,
    "A-11": 169.09, "A-12": 162.74, "A-13": 130.07,
    "N-14": 249.87, "W-14": 383.11, "A-14": 224.89,
}
TAGS = list(BASELINE)
CALIB_TAGS = {"N-13", "W-13", "A-13"}


def _run_one(task):
    tag, seed, terms = task
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
    return tag, mrt, max(0.0, cost - BUDGET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(TERMS_PATH):
        raise SystemExit(f"Tuned MFs not found at {TERMS_PATH}; run autotune.py first.")
    with open(TERMS_PATH, "r", encoding="utf-8") as fh:
        terms = json.load(fh)["mf_terms"]

    tasks = [(tag, args.seed, terms) for tag in TAGS]
    out = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for tag, mrt, vio in ex.map(_run_one, tasks):
            out[tag] = (mrt, vio)

    print(f"{'scen':<6}{'base':>9}{'tuned':>9}{'dMRT':>9}{'Vio':>8}  split")
    print("-" * 50)
    dsum = bsum = 0.0
    ninf = 0
    held_delta = []
    for tag in TAGS:
        mrt, vio = out[tag]
        base = BASELINE[tag]
        dsum += mrt - base
        bsum += mrt
        ninf += int(vio > 1e-6)
        split = "calib" if tag in CALIB_TAGS else "held-out"
        if tag not in CALIB_TAGS:
            held_delta.append(mrt - base)
        print(f"{tag:<6}{base:>9.2f}{mrt:>9.2f}{mrt - base:>+9.2f}{vio:>8.1f}  {split}")
    n = len(TAGS)
    print("-" * 50)
    print(f"{'mean':<6}{sum(BASELINE.values()) / n:>9.2f}{bsum / n:>9.2f}"
          f"{dsum / n:>+9.2f}   infeasible={ninf}")
    if held_delta:
        print(f"held-out mean dMRT: {sum(held_delta) / len(held_delta):+.2f}  "
              f"({len(held_delta)} scenarios)")


if __name__ == "__main__":
    main()
