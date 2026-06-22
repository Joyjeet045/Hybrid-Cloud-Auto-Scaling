"""Deterministic hyperparameter search for the criticality (Rec 3) and
dependency-propagation (Rec 4) knobs of NF-DiagScale.

Both knobs only change *which* microservice is selected for scaling each
interval; at ``criticality.weight == 0`` and ``propagation.weight == 0`` the
controller reproduces the published baseline exactly. This script sweeps a
small, fixed grid over those knobs, runs the HGraphScale scenarios in parallel,
and reports the configuration that minimises mean response time (MRT) subject to
a hard zero-cost-violation feasibility constraint.

The search is split train/validation to guard against overfitting:
  * TRAIN     -> used to rank configurations.
  * HELDOUT   -> used only to validate the chosen configuration generalises.

Usage:
    python tune_hyperparams.py                 # full grid, seed 0
    python tune_hyperparams.py --workers 12
    python tune_hyperparams.py --top 3 --out tuning_results.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor

from nfg_diagscale.config import load_config
from run_star_comparison import run_scenario

# Larger apps across all three workloads (more headroom to differ from baseline).
TRAIN = ["N-12", "N-13", "W-12", "W-13", "A-12", "A-13"]
# Untouched during the search; used only to check the winner generalises.
HELDOUT = ["N-11", "W-11", "A-11", "N-14", "W-14", "A-14"]

INFEASIBLE_PENALTY = 1.0e6  # any cost violation makes a config inadmissible


def make_grid():
    """Return the list of {section: overrides} dicts to evaluate."""
    crit_settings = [
        {"weight": 0.0, "alpha": 0.4, "beta": 0.4, "gamma": 0.2},  # criticality off (baseline)
        {"weight": 0.4, "alpha": 0.4, "beta": 0.4, "gamma": 0.2},  # balanced
        {"weight": 0.4, "alpha": 0.2, "beta": 0.6, "gamma": 0.2},  # centrality-heavy
        {"weight": 0.7, "alpha": 0.2, "beta": 0.6, "gamma": 0.2},  # centrality-heavy, strong
        {"weight": 0.7, "alpha": 0.5, "beta": 0.3, "gamma": 0.2},  # load-heavy, strong
    ]
    prop_weights = [0.0, 0.35, 0.7]
    grid = []
    for crit in crit_settings:
        for pw in prop_weights:
            grid.append({
                "criticality": dict(crit),
                "propagation": {"weight": pw, "hops": 1},
            })
    return grid


def cfg_with(overrides):
    cfg = load_config()
    for section, vals in overrides.items():
        cfg.setdefault(section, {})
        cfg[section].update(vals)
    return cfg


def _run_one(task):
    """Worker: run a single (config, scenario) episode. Picklable, top-level."""
    ci, tag, seed, overrides = task
    cfg = cfg_with(overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        mrt, cost, vio = run_scenario(tag, seed, cfg)
    return ci, tag, float(mrt), float(cost), float(vio)


def score(results):
    """Lower is better. Mean MRT with a hard penalty for any cost violation."""
    mrts = [r[0] for r in results.values()]
    vios = [r[2] for r in results.values()]
    n_infeasible = sum(1 for v in vios if v > 1e-6)
    mean_mrt = sum(mrts) / max(len(mrts), 1)
    return (mean_mrt + INFEASIBLE_PENALTY * n_infeasible + 100.0 * sum(vios),
            mean_mrt, n_infeasible)


def evaluate(grid, tags, workers, seed):
    """Run every (config, tag) pair in one pool; return {ci: {tag: (mrt,cost,vio)}}."""
    tasks = [(ci, tag, seed, ov) for ci, ov in enumerate(grid) for tag in tags]
    raw = {ci: {} for ci in range(len(grid))}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for ci, tag, mrt, cost, vio in ex.map(_run_one, tasks):
            raw[ci][tag] = (mrt, cost, vio)
    return raw


def fmt_cfg(ov):
    c, p = ov["criticality"], ov["propagation"]
    if c["weight"] == 0.0 and p["weight"] == 0.0:
        return "BASELINE (crit=0, prop=0)"
    cpart = (f"crit w={c['weight']} (a={c['alpha']},b={c['beta']},g={c['gamma']})"
             if c["weight"] > 0 else "crit=0")
    ppart = f"prop w={p['weight']}" if p["weight"] > 0 else "prop=0"
    return f"{cpart} | {ppart}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 4))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--out", type=str, default="tuning_results.json")
    args = ap.parse_args()

    grid = make_grid()
    print(f"Hyperparameter search: {len(grid)} configs x {len(TRAIN)} train scenarios "
          f"= {len(grid) * len(TRAIN)} episodes on {args.workers} workers")
    print("=" * 88)

    # --- TRAIN: rank configurations -------------------------------------------
    train_raw = evaluate(grid, TRAIN, args.workers, args.seed)
    ranked = []
    for ci, ov in enumerate(grid):
        s, mean_mrt, ninf = score(train_raw[ci])
        ranked.append((s, mean_mrt, ninf, ci, ov))
    ranked.sort(key=lambda r: r[0])

    base_ci = next(i for i, ov in enumerate(grid)
                   if ov["criticality"]["weight"] == 0.0 and ov["propagation"]["weight"] == 0.0)
    base_mean = score(train_raw[base_ci])[1]

    print(f"\nTRAIN ranking (mean MRT over {len(TRAIN)} scenarios, baseline={base_mean:.3f} ms):")
    print("-" * 88)
    for rank, (s, mean_mrt, ninf, ci, ov) in enumerate(ranked, 1):
        delta = mean_mrt - base_mean
        flag = "  INFEASIBLE" if ninf else ""
        print(f"{rank:>2}. {mean_mrt:8.3f} ms  (dMRT {delta:+7.3f}){flag}   {fmt_cfg(ov)}")

    # --- VALIDATION: best K + baseline on the held-out scenarios ---------------
    top_cis = [r[3] for r in ranked[:args.top] if r[2] == 0]
    val_cis = list(dict.fromkeys([base_ci] + top_cis))
    val_grid = [grid[ci] for ci in val_cis]
    val_raw = evaluate(val_grid, HELDOUT, args.workers, args.seed)

    print(f"\nVALIDATION on held-out {HELDOUT}:")
    print("-" * 88)
    base_val = None
    val_scores = {}
    for local_i, ci in enumerate(val_cis):
        s, mean_mrt, ninf = score(val_raw[local_i])
        val_scores[ci] = (mean_mrt, ninf)
        if ci == base_ci:
            base_val = mean_mrt
    for local_i, ci in enumerate(val_cis):
        mean_mrt, ninf = val_scores[ci]
        delta = mean_mrt - base_val
        flag = "  INFEASIBLE" if ninf else ""
        tag = "  <- baseline" if ci == base_ci else ""
        print(f"   {mean_mrt:8.3f} ms  (dMRT {delta:+7.3f}){flag}   {fmt_cfg(grid[ci])}{tag}")

    best = ranked[0]
    payload = {
        "seed": args.seed,
        "train_scenarios": TRAIN,
        "heldout_scenarios": HELDOUT,
        "baseline_train_mrt": base_mean,
        "ranking": [
            {"rank": r + 1, "score": s, "mean_mrt": m, "n_infeasible": n,
             "config": ov, "per_scenario": train_raw[ci]}
            for r, (s, m, n, ci, ov) in enumerate(ranked)
        ],
        "best": {"config": best[4], "train_mrt": best[1], "n_infeasible": best[2]},
    }
    with open(os.path.abspath(args.out), "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 88)
    print(f"BEST: {fmt_cfg(best[4])}")
    print(f"      train MRT {best[1]:.3f} ms  vs baseline {base_mean:.3f} ms "
          f"(dMRT {best[1] - base_mean:+.3f})")
    print(f"Saved -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
