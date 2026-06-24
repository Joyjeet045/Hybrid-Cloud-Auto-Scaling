"""Clean (single-process) measurement of NF-DiagScale's per-interval decision time.

Run SEQUENTIALLY so the timings are not polluted by CPU contention. Reports the
decision-time distribution for the deployed full controller (GCN residual on) and,
for reference, with the GCN residual off, on representative scenarios spanning
11--14 microservices. Also records per-interval (#containers, time) so the paper
can state how cost scales with deployment size.

Usage:  python reporting/measure_overhead.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nfg_diagscale.config import load_config
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from nfg_diagscale.hgraph_policy.controller import NFGDiagScaleController

BUDGET, DEADLINE, TOTAL = 200.0, 500.0, 480
SCEN = {"A-11": ("A11", "alibaba"), "A-13": ("A13", "alibaba"), "W-14": ("A14", "wiki")}


def measure(tag, gcn_on):
    app, workload = SCEN[tag]
    cfg = load_config()
    if not gcn_on:
        cfg.setdefault("forecast", {})["gnn_residual"] = False
    env = HGraphScaleEnv(app=app, workload=workload, seed=0, budget=BUDGET)
    state = env.reset(test=True)
    ctrl = NFGDiagScaleController(cfg, deadline=DEADLINE, total_intervals=TOTAL)
    ctrl.reset(budget_T=BUDGET, total_intervals=TOTAL)
    times, ncons = [], []
    done = False
    with contextlib.redirect_stdout(io.StringIO()):
        while not done:
            t0 = time.perf_counter()
            action = ctrl.act(state)
            times.append((time.perf_counter() - t0) * 1000.0)
            ncons.append(len(state.containers))
            state, _r, done, _info = env.step(action)
    warm = np.array(times[2:])  # drop lazy-init warmup
    return {
        "tag": tag, "gcn_on": gcn_on, "intervals": len(times),
        "mean_ms": float(warm.mean()), "median_ms": float(np.median(warm)),
        "p95_ms": float(np.percentile(warm, 95)), "max_ms": float(warm.max()),
        "max_containers": int(max(ncons)),
    }


def main():
    rows = []
    for tag in SCEN:
        for gcn_on in (True, False):
            r = measure(tag, gcn_on)
            rows.append(r)
            tagstr = f"{tag} GCN-{'on' if gcn_on else 'off'}"
            print(f"  {tagstr:<14} mean={r['mean_ms']:6.2f}  median={r['median_ms']:6.2f}  "
                  f"p95={r['p95_ms']:6.2f}  max={r['max_ms']:7.2f} ms  "
                  f"(<= {r['max_containers']} containers)")
    on = [r for r in rows if r["gcn_on"]]
    off = [r for r in rows if not r["gcn_on"]]
    print("\n=== Decision time (full controller, GCN on) ===")
    print(f"  mean   {np.mean([r['mean_ms'] for r in on]):.2f} ms/interval")
    print(f"  median {np.mean([r['median_ms'] for r in on]):.2f} ms")
    print(f"  p95    {np.mean([r['p95_ms'] for r in on]):.2f} ms")
    print(f"  max    {max(r['max_ms'] for r in on):.2f} ms")
    print(f"  GCN marginal: mean {np.mean([r['mean_ms'] for r in on]) - np.mean([r['mean_ms'] for r in off]):+.2f} ms")
    print(f"  control interval = 180000 ms (3 min); decision is "
          f"{100*np.mean([r['mean_ms'] for r in on])/180000:.4f}% of it")
    with open(os.path.join(_REPO_ROOT, "reporting", "overhead_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\nWrote reporting/overhead_results.json")


if __name__ == "__main__":
    main()
