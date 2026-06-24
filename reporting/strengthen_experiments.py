"""Extra experiments for the NF-DiagScale paper (publication strengthening).

Produces, on the 12 benchmark scenarios (seed 0):
  * diagonal vs vertical-only vs horizontal-only scaling   -> diagonal ablation
  * GCN residual on/off, criticality blend on/off          -> positive component ablation
  * per-interval controller decision time (ms)             -> computational overhead

All runs reuse the vendored HGraphScale simulator through the same controller the
paper deploys; the single-axis and component variants are obtained by wrapping the
controller at runtime (no change to the core code path). Results are written to
``reporting/strengthen_results.json``.

Usage:  python reporting/strengthen_experiments.py --workers 12
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nfg_diagscale.config import load_config
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from nfg_diagscale.hgraph_policy.controller import NFGDiagScaleController

BUDGET = 200.0
DEADLINE = 500.0
TOTAL_INTERVALS = 480

SCENARIOS = {
    "N-11": ("A11", "nasa"), "N-12": ("A12", "nasa"), "N-13": ("A13", "nasa"),
    "W-11": ("A11", "wiki"), "W-12": ("A12", "wiki"), "W-13": ("A13", "wiki"),
    "A-11": ("A11", "alibaba"), "A-12": ("A12", "alibaba"), "A-13": ("A13", "alibaba"),
    "N-14": ("A14", "nasa"), "W-14": ("A14", "wiki"), "A-14": ("A14", "alibaba"),
}
TAGS = list(SCENARIOS)


def _make_cfg(experiment):
    cfg = load_config()
    if experiment == "gcn_off":
        cfg.setdefault("forecast", {})["gnn_residual"] = False
    elif experiment == "crit_off":
        cfg.setdefault("criticality", {})["weight"] = 0.0
    return cfg


def _wrap_controller(ctrl, experiment):
    """Restrict the controller to a single scaling axis by wrapping it at runtime."""
    if experiment == "vertical":
        # Vertical-only: never request replica changes and never overflow into a
        # new replica, so the replica count is frozen at the initial deployment.
        orig_decide = ctrl.anfis.decide

        def v_decide(*a, **k):
            d = dict(orig_decide(*a, **k))
            d["delta_n"] = 0
            return d

        ctrl.anfis.decide = v_decide

        def v_scale_up(replicas, delta_total, replica_vcpu, budget_room, hours_remaining):
            target = max(replicas, key=lambda c: (c.aver_resptime, c.qlen))
            headroom = int(target.max_scal_vcpu)
            delta_total = min(int(delta_total), headroom)  # cap at headroom: no new VM
            if delta_total <= 0:
                return None
            return (target.con_id, int(delta_total))

        ctrl._scale_up = v_scale_up

        def v_scale_down(replicas, delta_total):
            target = min(replicas, key=lambda c: (c.aver_resptime, c.qlen))
            room_down = -(int(target.vcpu) - ctrl.min_cores)  # never remove a replica
            delta_total = max(int(delta_total), room_down)
            if delta_total >= 0:
                return None
            return (target.con_id, int(delta_total))

        ctrl._scale_down = v_scale_down

    elif experiment == "horizontal":
        # Horizontal-biased: never request per-replica vCPU changes, so capacity
        # grows/shrinks by replica count (the platform is vertical-first, so a
        # scale-up still consumes a replica's headroom before spawning a replica).
        orig_decide = ctrl.anfis.decide

        def h_decide(*a, **k):
            d = dict(orig_decide(*a, **k))
            d["delta_c"] = 0
            return d

        ctrl.anfis.decide = h_decide


def run_one(args):
    experiment, tag = args
    app, workload = SCENARIOS[tag]
    cfg = _make_cfg(experiment)
    env = HGraphScaleEnv(app=app, workload=workload, seed=0, budget=BUDGET)
    state = env.reset(test=True)
    ctrl = NFGDiagScaleController(cfg, deadline=DEADLINE, total_intervals=TOTAL_INTERVALS)
    ctrl.reset(budget_T=BUDGET, total_intervals=TOTAL_INTERVALS)
    _wrap_controller(ctrl, experiment)

    times = []
    done, info = False, {}
    with contextlib.redirect_stdout(io.StringIO()):
        while not done:
            t0 = time.perf_counter()
            action = ctrl.act(state)
            times.append((time.perf_counter() - t0) * 1000.0)
            state, _r, done, info = env.step(action)

    mrt = float(info.get("average_resptime", float("nan")))
    cost = float(info.get("VM_cost", float("nan")))
    vio = max(0.0, cost - BUDGET)
    warm = times[2:] if len(times) > 2 else times  # drop lazy-init warmup
    return {
        "experiment": experiment, "tag": tag,
        "mrt": mrt, "cost": cost, "vio": vio,
        "decide_mean_ms": float(np.mean(warm)),
        "decide_median_ms": float(np.median(warm)),
        "decide_p95_ms": float(np.percentile(warm, 95)),
        "decide_max_ms": float(np.max(warm)),
        "intervals": len(times),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=str, default="reporting/strengthen_results.json")
    args = ap.parse_args()

    experiments = ["baseline", "vertical", "horizontal", "gcn_off", "crit_off"]
    jobs = [(e, t) for e in experiments for t in TAGS]
    print(f"Running {len(jobs)} episodes over {args.workers} workers...")

    results = {e: {} for e in experiments}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run_one, jobs):
            results[r["experiment"]][r["tag"]] = r
            print(f"  {r['experiment']:<11} {r['tag']:<5} "
                  f"MRT={r['mrt']:7.2f}  Vio={r['vio']:.1f}  "
                  f"decide={r['decide_mean_ms']:.3f} ms")

    # ---- summaries ----
    def mean_mrt(e):
        return float(np.mean([results[e][t]["mrt"] for t in TAGS]))

    base = mean_mrt("baseline")
    print("\n=== Diagonal ablation (all-12 mean MRT, ms) ===")
    print(f"  diagonal (full)   {base:7.2f}")
    print(f"  vertical-only     {mean_mrt('vertical'):7.2f}  "
          f"(+{mean_mrt('vertical')-base:.2f})")
    print(f"  horizontal-only   {mean_mrt('horizontal'):7.2f}  "
          f"(+{mean_mrt('horizontal')-base:.2f})")
    print("\n=== Component ablation (all-12 mean MRT, ms) ===")
    print(f"  full (GCN+crit)   {base:7.2f}")
    print(f"  GCN residual off  {mean_mrt('gcn_off'):7.2f}  "
          f"(delta {base-mean_mrt('gcn_off'):+.2f} from removing GCN)")
    print(f"  criticality off   {mean_mrt('crit_off'):7.2f}  "
          f"(delta {base-mean_mrt('crit_off'):+.2f} from removing criticality)")
    all_decide = [results["baseline"][t]["decide_mean_ms"] for t in TAGS]
    all_p95 = [results["baseline"][t]["decide_p95_ms"] for t in TAGS]
    all_max = [results["baseline"][t]["decide_max_ms"] for t in TAGS]
    print("\n=== Decision-time overhead (deployed full controller) ===")
    print(f"  mean   {np.mean(all_decide):.3f} ms/interval")
    print(f"  p95    {np.mean(all_p95):.3f} ms")
    print(f"  max    {np.max(all_max):.3f} ms")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
