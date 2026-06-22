"""Parallel 12-scenario A/B evaluation of the context-scheduled, robustly-tuned
membership functions vs the expert-tuned baseline.

Reports the TRAIN / VALIDATION / HELD-OUT split so generalisation is explicit: the
held-out scenarios were never seen by the optimiser. ``dMRT = tuned - base`` and
relative regret are both shown; the headline number is the held-out mean.

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
from ablations.rec4_mf_autotune.scheduled_controller import ScheduledMFController

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


def _run_one(task):
    tag, seed, design = task
    app, workload = SCENARIOS[tag]
    cfg = load_config()
    cfg["fuzzify"] = {"scheduled_mf": True, "design": design}
    env = HGraphScaleEnv(app=app, workload=workload, seed=seed, budget=BUDGET)
    state = env.reset(test=True)
    ctrl = ScheduledMFController(cfg, deadline=DEADLINE, total_intervals=INTERVALS)
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
        raise SystemExit(f"Design not found at {TERMS_PATH}; run autotune.py first.")
    with open(TERMS_PATH, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    design = {"base_terms": d["base_terms"], "gains": d["gains"],
              "g_lo": d["g_lo"], "g_hi": d["g_hi"]}
    train_tags = set(d.get("train_tags", []))
    val_tags = set(d.get("val_tags", []))

    def split_of(tag):
        if tag in train_tags:
            return "train"
        if tag in val_tags:
            return "val"
        return "held-out"

    tasks = [(tag, args.seed, design) for tag in TAGS]
    out = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for tag, mrt, vio in ex.map(_run_one, tasks):
            out[tag] = (mrt, vio)

    print(f"{'scen':<6}{'base':>9}{'tuned':>9}{'dMRT':>9}{'reg%':>8}{'Vio':>7}  split")
    print("-" * 56)
    dsum = bsum = 0.0
    ninf = 0
    held_delta, held_reg = [], []
    for tag in TAGS:
        mrt, vio = out[tag]
        base = BASELINE[tag]
        reg = (mrt - base) / base * 100.0
        dsum += mrt - base
        bsum += mrt
        ninf += int(vio > 1e-6)
        split = split_of(tag)
        if split == "held-out":
            held_delta.append(mrt - base)
            held_reg.append(reg)
        print(f"{tag:<6}{base:>9.2f}{mrt:>9.2f}{mrt - base:>+9.2f}"
              f"{reg:>+8.2f}{vio:>7.1f}  {split}")
    n = len(TAGS)
    print("-" * 56)
    print(f"{'mean':<6}{sum(BASELINE.values()) / n:>9.2f}{bsum / n:>9.2f}"
          f"{dsum / n:>+9.2f}{'':>8}  infeasible={ninf}")
    if held_delta:
        print(f"held-out mean dMRT: {sum(held_delta) / len(held_delta):+.2f} ms  "
              f"({len(held_delta)} scenarios)")
        print(f"held-out mean regret: {sum(held_reg) / len(held_reg):+.2f}%")


if __name__ == "__main__":
    main()
