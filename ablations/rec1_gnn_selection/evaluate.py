"""Parallel 12-scenario A/B evaluation of the GNN-distilled selector vs the tuned
baseline (PDF Rec 2, Approach 1).

Run from the repo root after training weights:
    python -m ablations.rec1_gnn_selection.train_gnn
    python -m ablations.rec1_gnn_selection.evaluate --workers 12
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
from concurrent.futures import ProcessPoolExecutor

from nfg_diagscale.config import load_config
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from run_star_comparison import BUDGET, DEADLINE, SCENARIOS
from ablations.rec1_gnn_selection.gnn_controller import GnnSelectionController

INTERVALS = 480
HERE = os.path.dirname(__file__)
WEIGHTS = os.path.join(HERE, "gnn_weights.pt")

BASELINE = {
    "N-11": 169.66, "N-12": 165.80, "N-13": 138.74,
    "W-11": 230.53, "W-12": 241.24, "W-13": 197.11,
    "A-11": 149.40, "A-12": 153.25, "A-13": 136.35,
    "N-14": 255.74, "W-14": 369.63, "A-14": 249.47,
}
TAGS = list(BASELINE)


def _run_one(task):
    tag, seed, weights = task
    app, workload = SCENARIOS[tag]
    cfg = load_config()
    env = HGraphScaleEnv(app=app, workload=workload, seed=seed, budget=BUDGET)
    state = env.reset(test=True)
    ctrl = GnnSelectionController(cfg, weights=weights, deadline=DEADLINE,
                                  total_intervals=INTERVALS)
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
    ap.add_argument("--weights", type=str, default=WEIGHTS)
    args = ap.parse_args()

    tasks = [(tag, args.seed, args.weights) for tag in TAGS]
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
