"""STAR Table 3 comparison harness for NFG-DiagScale on the HGraphScale env.

Runs the NFG-DiagScale controller on the 9 scenarios reported in the STAR paper
(Fang et al., 2026, "STAR: Spatial-Temporal Autoscaling ...", ESWA) — the
NASA / Wikipedia / Alibaba workloads crossed with the 11-, 12- and 13-microservice
applications — and tabulates the achieved mean response time (MRT, ms) and cost
violation (Vio = max(0, daily VM cost - $200 budget); STAR Eq. 7) next to the
numbers reported in STAR Table 3.

We report only what we actually measured: every NFG-DiagScale cell is produced by
running the vendored HGraphScale simulator. The STAR / baseline columns are quoted
verbatim from the STAR paper for reference; we do not recompute them.

Usage:
    python run_star_comparison.py                # all 9 scenarios, seed 0
    python run_star_comparison.py --seeds 0 1 2  # mean over seeds
    python run_star_comparison.py --scenario N-11
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics

from nfg_diagscale.config import load_config
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from nfg_diagscale.hgraph_policy.controller import NFGDiagScaleController

# Per-day cost budget from the STAR/HGraphScale protocol ($200/day). Vio (STAR
# Eq. 7) is the dollar amount by which the realised cost exceeds this budget.
BUDGET = 200.0
DEADLINE = 500.0

# STAR Table 3 (MRT_ms, Vio). Quoted verbatim from the STAR paper for reference.
STAR_TABLE = {
    "N-11": {"AWS-Scale": (410.42, 0.0), "ProScale": (305.57, 52.03), "DeepScale": (306.60, 0.0), "DRPC": (289.92, 67.10), "STAR": (187.15, 0.0)},
    "N-12": {"AWS-Scale": (688.52, 0.0), "ProScale": (387.72, 22.88), "DeepScale": (532.65, 0.0), "DRPC": (433.23, 10.40), "STAR": (288.85, 0.0)},
    "N-13": {"AWS-Scale": (899.04, 0.0), "ProScale": (406.82, 0.81), "DeepScale": (493.62, 44.04), "DRPC": (532.34, 0.0), "STAR": (195.76, 0.0)},
    "W-11": {"AWS-Scale": (489.73, 0.0), "ProScale": (532.46, 0.0), "DeepScale": (318.29, 34.75), "DRPC": (415.48, 28.18), "STAR": (350.16, 0.0)},
    "W-12": {"AWS-Scale": (864.65, 0.0), "ProScale": (687.00, 0.0), "DeepScale": (549.98, 0.0), "DRPC": (512.40, 12.52), "STAR": (413.26, 0.0)},
    "W-13": {"AWS-Scale": (1080.44, 0.0), "ProScale": (482.13, 13.18), "DeepScale": (675.37, 0.0), "DRPC": (491.68, 56.17), "STAR": (304.75, 0.0)},
    "A-11": {"AWS-Scale": (702.46, 0.0), "ProScale": (680.03, 8.65), "DeepScale": (447.48, 0.0), "DRPC": (339.05, 0.0), "STAR": (274.29, 0.0)},
    "A-12": {"AWS-Scale": (1195.81, 0.0), "ProScale": (721.31, 0.0), "DeepScale": (399.66, 4.64), "DRPC": (410.70, 0.0), "STAR": (348.62, 0.0)},
    "A-13": {"AWS-Scale": (944.85, 0.0), "ProScale": (393.54, 23.05), "DeepScale": (337.48, 0.0), "DRPC": (227.29, 48.78), "STAR": (220.32, 0.0)},
}

# scenario tag -> (app id, workload name)
SCENARIOS = {
    "N-11": ("A11", "nasa"), "N-12": ("A12", "nasa"), "N-13": ("A13", "nasa"),
    "W-11": ("A11", "wiki"), "W-12": ("A12", "wiki"), "W-13": ("A13", "wiki"),
    "A-11": ("A11", "alibaba"), "A-12": ("A12", "alibaba"), "A-13": ("A13", "alibaba"),
    # App-14 is beyond STAR Table 3 (which stops at 13); baselined separately.
    "N-14": ("A14", "nasa"), "W-14": ("A14", "wiki"), "A-14": ("A14", "alibaba"),
}


def run_scenario(tag, seed, cfg):
    """Run one full episode; return (MRT_ms, VM_cost, Vio)."""
    app, workload = SCENARIOS[tag]
    env = HGraphScaleEnv(app=app, workload=workload, seed=seed, budget=BUDGET)
    # The episode length (test horizon) is fixed by the trace; pass it to the
    # controller so its cost proration matches the simulator.
    state = env.reset(test=True)
    # Determine horizon once: STAR day-2 test split = 480 intervals (= exactly one
    # day, so the $200/day budget maps exactly onto the test episode).
    total_intervals = 480
    ctrl = NFGDiagScaleController(cfg, deadline=DEADLINE, total_intervals=total_intervals)
    ctrl.reset(budget_T=BUDGET, total_intervals=total_intervals)

    done = False
    info = {}
    # Silence the simulator's per-episode diagnostic prints.
    with contextlib.redirect_stdout(io.StringIO()):
        while not done:
            action = ctrl.act(state)
            state, _reward, done, info = env.step(action)

    mrt = float(info.get("average_resptime", float("nan")))
    cost = float(info.get("VM_cost", float("nan")))
    vio = max(0.0, cost - BUDGET)
    return mrt, cost, vio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--scenario", type=str, nargs="+", default=None,
                    help="Run only these scenario tags (e.g. N-14 W-14 A-14).")
    ap.add_argument("--out", type=str, default="star_comparison_results.json")
    args = ap.parse_args()

    cfg = load_config()
    tags = args.scenario if args.scenario else list(SCENARIOS)

    results = {}
    print(f"\nNFG-DiagScale vs STAR Table 3  (seeds={args.seeds}, budget=${BUDGET:.0f}/day)")
    print("=" * 92)
    print(f"{'Scenario':<9}{'AWS-Scale':>12}{'ProScale':>13}{'DeepScale':>13}"
          f"{'DRPC':>13}{'STAR':>12}{'NFG-DiagScale':>16}")
    print("-" * 92)

    wins = 0
    total = 0
    for tag in tags:
        runs = [run_scenario(tag, s, cfg) for s in args.seeds]
        mrt = statistics.mean(r[0] for r in runs)
        cost = statistics.mean(r[1] for r in runs)
        vio = statistics.mean(r[2] for r in runs)
        results[tag] = {"MRT": mrt, "VM_cost": cost, "Vio": vio,
                        "per_seed": [{"seed": s, "MRT": r[0], "cost": r[1], "Vio": r[2]}
                                     for s, r in zip(args.seeds, runs)]}

        if tag in STAR_TABLE:
            star_mrt, star_vio = STAR_TABLE[tag]["STAR"]
            beat = mrt < star_mrt and vio <= 1e-6
            wins += int(beat)
            total += 1
            mark = "  WIN" if beat else ""

            def cell(name):
                m, v = STAR_TABLE[tag][name]
                return f"{m:.1f}/{v:.0f}"

            print(f"{tag:<9}{cell('AWS-Scale'):>12}{cell('ProScale'):>13}"
                  f"{cell('DeepScale'):>13}{cell('DRPC'):>13}{cell('STAR'):>12}"
                  f"{mrt:>10.2f}/{vio:.0f}{mark}")
        else:
            print(f"{tag:<9}{'n/a':>12}{'n/a':>13}{'n/a':>13}{'n/a':>13}{'n/a':>12}"
                  f"{mrt:>10.2f}/{vio:.0f}   (no STAR baseline)")

    print("-" * 92)
    if total:
        print(f"NFG-DiagScale beats STAR on {wins}/{total} STAR scenarios (lower MRT, Vio=0).")

    out_path = os.path.abspath(args.out)
    payload = {"budget": BUDGET, "seeds": args.seeds, "star_table": STAR_TABLE,
               "nfg_diagscale": results}
    # Merge with any existing results so running a subset (e.g. the A14 tags)
    # extends the file instead of dropping the scenarios already measured.
    if os.path.exists(out_path):
        try:
            with open(out_path) as fh:
                prev = json.load(fh)
            merged = dict(prev.get("nfg_diagscale", {}))
            merged.update(results)
            payload["nfg_diagscale"] = merged
        except (json.JSONDecodeError, OSError):
            pass
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved detailed results -> {out_path}\n")


if __name__ == "__main__":
    main()
