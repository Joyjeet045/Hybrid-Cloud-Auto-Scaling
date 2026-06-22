"""Parallel 12-scenario A/B evaluation of the data-driven membership functions vs
the tuned baseline (PDF Rec 2, Approach 2).

Run from the repo root:
    python -m ablations.rec2_adaptive_mf.evaluate --workers 12
    python -m ablations.rec2_adaptive_mf.evaluate --workers 12 --pooled
"""
from __future__ import annotations

import argparse
import contextlib
import io
from concurrent.futures import ProcessPoolExecutor

from nfg_diagscale.config import load_config
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from run_star_comparison import BUDGET, DEADLINE, SCENARIOS
from ablations.rec2_adaptive_mf.adaptive_controller import AdaptiveMFController

INTERVALS = 480

BASELINE = {
    "N-11": 152.91, "N-12": 157.39, "N-13": 140.52,
    "W-11": 229.34, "W-12": 261.53, "W-13": 210.52,
    "A-11": 169.09, "A-12": 162.74, "A-13": 130.07,
    "N-14": 249.87, "W-14": 383.11, "A-14": 224.89,
}
TAGS = list(BASELINE)


def _run_one(task):
    tag, seed, fuzzify = task
    app, workload = SCENARIOS[tag]
    cfg = load_config()
    cfg["fuzzify"] = fuzzify
    env = HGraphScaleEnv(app=app, workload=workload, seed=seed, budget=BUDGET)
    state = env.reset(test=True)
    ctrl = AdaptiveMFController(cfg, deadline=DEADLINE, total_intervals=INTERVALS)
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
    ap.add_argument("--pooled", action="store_true",
                    help="workload-diverse calibration (pool N-13, W-13, A-13)")
    args = ap.parse_args()

    if args.pooled:
        fuzzify = {"adaptive_mf": True, "calib_tags": ["N-13", "W-13", "A-13"],
                   "calib_seed": args.seed}
    else:
        fuzzify = {"adaptive_mf": True, "calib_app": "A12",
                   "calib_workload": "alibaba", "calib_seed": args.seed}

    tasks = [(tag, args.seed, fuzzify) for tag in TAGS]
    out = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for tag, mrt, vio in ex.map(_run_one, tasks):
            out[tag] = (mrt, vio)

    print(f"{'scen':<6}{'base':>9}{'branch':>9}{'dMRT':>9}{'Vio':>8}")
    print("-" * 41)
    dsum = bsum = 0.0
    ninf = 0
    for tag in TAGS:
        mrt, vio = out[tag]
        base = BASELINE[tag]
        dsum += mrt - base
        bsum += mrt
        ninf += int(vio > 1e-6)
        print(f"{tag:<6}{base:>9.2f}{mrt:>9.2f}{mrt - base:>+9.2f}{vio:>8.1f}")
    n = len(TAGS)
    print("-" * 41)
    print(f"{'mean':<6}{sum(BASELINE.values()) / n:>9.2f}{bsum / n:>9.2f}"
          f"{dsum / n:>+9.2f}   infeasible={ninf}")


if __name__ == "__main__":
    main()
