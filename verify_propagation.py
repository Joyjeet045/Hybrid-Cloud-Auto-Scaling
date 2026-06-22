"""Focused verification: can dependency propagation (Rec 4) ever help?

Fixes criticality at the tuned default (weight=0.4, balanced preset) and sweeps
ONLY the propagation weight over a fine grid that includes very small values,
evaluating on all 12 scenarios. If no weight > 0 beats weight == 0 on mean MRT
(with Vio == 0), propagation is not useful and stays disabled.

Usage:
    python verify_propagation.py --workers 14
"""
from __future__ import annotations

import argparse
import json
import os

from tune_hyperparams import TRAIN, HELDOUT, evaluate, score, _run_one  # noqa: F401

CRIT = {"weight": 0.4, "alpha": 0.4, "beta": 0.4, "gamma": 0.2}
PROP_WEIGHTS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
PROP_HOPS = [1, 2]
ALL12 = TRAIN + HELDOUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 4))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hop_weights = {1: [w for w in PROP_WEIGHTS if w > 0.0],
                   2: [0.01, 0.02, 0.05, 0.1]}
    grid = [{"criticality": dict(CRIT), "propagation": {"weight": 0.0, "hops": 1}}]
    combos = [(0.0, 1)]
    for h in PROP_HOPS:
        for w in hop_weights[h]:
            grid.append({"criticality": dict(CRIT),
                         "propagation": {"weight": w, "hops": h}})
            combos.append((w, h))

    print(f"Propagation verification: criticality fixed at weight=0.4 balanced.")
    print(f"sweeping weight x hops -> {len(grid)} configs x {len(ALL12)} scenarios = "
          f"{len(grid) * len(ALL12)} episodes on {args.workers} workers")
    print("=" * 78)

    raw = evaluate(grid, ALL12, args.workers, args.seed)

    base_mean = None
    rows = []
    for ci, (w, h) in enumerate(combos):
        _, mean_mrt, ninf = score(raw[ci])
        if w == 0.0:
            base_mean = mean_mrt
        rows.append((w, h, mean_mrt, ninf))

    print(f"\nMean MRT over all 12 scenarios (prop disabled reference = {base_mean:.3f} ms):")
    print("-" * 78)
    best_improving = None
    for w, h, mean_mrt, ninf in rows:
        delta = mean_mrt - base_mean
        flag = "  INFEASIBLE(Vio>0)" if ninf else ""
        if w == 0.0:
            tag = "  <- reference"
        else:
            tag = "  IMPROVES" if delta < -1e-6 and not ninf else ""
        if w > 0.0 and delta < -1e-6 and ninf == 0:
            if best_improving is None or mean_mrt < best_improving[2]:
                best_improving = (w, h, mean_mrt, delta)
        label = "disabled" if w == 0.0 else f"w={w:<5} hops={h}"
        print(f"  prop {label:<16} {mean_mrt:8.3f} ms  (dMRT {delta:+8.3f}){flag}{tag}")

    per_scen_help = {}
    base_per = raw[0]
    for ci, (w, h) in enumerate(combos):
        if w == 0.0:
            continue
        for tag, (mrt, _c, vio) in raw[ci].items():
            base_mrt = base_per[tag][0]
            if vio <= 1e-6 and mrt < base_mrt - 1e-6:
                prev = per_scen_help.get(tag)
                if prev is None or mrt < prev[1]:
                    per_scen_help[tag] = (w, h, mrt, base_mrt)

    print("\n" + "=" * 78)
    if best_improving is None:
        print("VERDICT: no propagation (weight>0, any hops) improves mean MRT with Vio=0.")
        print("         -> propagation is NOT useful; keep propagation.weight = 0.")
    else:
        w, h, m, d = best_improving
        print(f"VERDICT: propagation w={w} hops={h} improves mean MRT to {m:.3f} ms (dMRT {d:+.3f}).")
        print("         -> propagation CAN help; consider enabling.")

    if per_scen_help:
        print("\nPer-scenario: propagation beat the reference on these (best config):")
        for tag, (w, h, mrt, base_mrt) in sorted(per_scen_help.items()):
            print(f"   {tag}: {base_mrt:.2f} -> {mrt:.2f} ms  at prop w={w} hops={h}")
    else:
        print("\nPer-scenario: propagation did not help on ANY single scenario.")

    out = {
        "criticality_fixed": CRIT,
        "reference_mean_mrt": base_mean,
        "sweep": [
            {"weight": w, "hops": h, "mean_mrt": mm, "n_infeasible": ninf}
            for (w, h, mm, ninf) in rows
        ],
        "verdict": ("not_useful" if best_improving is None else "useful"),
        "per_scenario_help": {
            tag: {"weight": w, "hops": h, "mrt": mrt, "base_mrt": base_mrt}
            for tag, (w, h, mrt, base_mrt) in per_scen_help.items()
        },
    }
    with open("propagation_verification.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n[saved] propagation_verification.json")


if __name__ == "__main__":
    main()
