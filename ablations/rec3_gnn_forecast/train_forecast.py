"""Train the GNN residual load forecaster from baseline rollouts.

Pipeline:
  1. Run the controller in record mode (no GCN) on a workload-diverse calibration
     set, buffering per-interval ``(graph, kalman_forecast, observed)``.
  2. Build free residual labels: for each service the target at interval ``k`` is
     ``observed[k+1] - kalman_forecast[k]`` (the realisation of that forecast).
  3. Fit the two-layer GCN to the standardised residuals (MSE) and report the
     in-sample forecast-error reduction over the per-series baseline.

Run from the repo root:
    python -m ablations.rec3_gnn_forecast.train_forecast [--epochs 300 --hidden 16]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from nfg_diagscale.config import load_config
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from run_star_comparison import BUDGET, DEADLINE, SCENARIOS
from ablations.rec3_gnn_forecast.forecast_controller import GnnForecastController
from ablations.rec3_gnn_forecast import gnn_forecast

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "forecast_weights.pt")
INTERVALS = 480
CALIB_TAGS = ("N-13", "W-13", "A-13")


def _collect_one(task):
    tag, seed = task
    app, workload = SCENARIOS[tag]
    cfg = load_config()
    env = HGraphScaleEnv(app=app, workload=workload, seed=seed, budget=BUDGET)
    state = env.reset(test=True)
    ctrl = GnnForecastController(cfg, deadline=DEADLINE, total_intervals=INTERVALS)
    ctrl.reset(budget_T=BUDGET, total_intervals=INTERVALS)
    ctrl.record_on()
    done = False
    with contextlib.redirect_stdout(io.StringIO()):
        while not done:
            action = ctrl.act(state)
            state, _r, done, _info = env.step(action)
    return ctrl.pop_records()


def collect(calib_tags, seed, workers):
    tasks = [(tag, seed) for tag in calib_tags]
    per_scenario = []
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
        for records in ex.map(_collect_one, tasks):
            per_scenario.append(records)
    return per_scenario


def build_groups(per_scenario):
    groups = []
    all_targets = []
    base_abs = []
    for records in per_scenario:
        for k in range(len(records) - 1):
            r, r2 = records[k], records[k + 1]
            idx_map = {t: i for i, t in enumerate(r["order"])}
            idxs, targets = [], []
            for t, base in r["base"].items():
                if t in r2["obs"]:
                    node_i = idx_map.get(t)
                    if node_i is None:
                        continue
                    resid = float(r2["obs"][t]) - float(base)
                    idxs.append(node_i)
                    targets.append(resid)
                    all_targets.append(resid)
                    base_abs.append(abs(resid))
            if idxs:
                groups.append({
                    "X": r["X"], "A": r["A"],
                    "idx": np.asarray(idxs, dtype=np.int64),
                    "y": np.asarray(targets, dtype=np.float64),
                })
    return groups, np.asarray(all_targets), np.asarray(base_abs)


def train(groups, y_mean, y_std, hidden, epochs, lr, seed):
    import torch

    model = gnn_forecast.ForecastGCN(hidden=hidden, seed=seed,
                                     y_mean=y_mean, y_std=y_std)
    module = model.module
    module.train()
    opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)

    order = list(range(len(groups)))
    last = 0.0
    for ep in range(epochs):
        rng.shuffle(order)
        total = 0.0
        for gi in order:
            g = groups[gi]
            xt = torch.from_numpy(np.ascontiguousarray(g["X"]))
            at = torch.from_numpy(np.ascontiguousarray(g["A"]))
            out = module(xt, at)
            pred = out[torch.from_numpy(g["idx"])]
            tgt = torch.from_numpy((g["y"] - y_mean) / y_std)
            loss = loss_fn(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(g["idx"])
        last = total / max(1, sum(len(groups[i]["idx"]) for i in order))
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:>4}  train MSE(std) {last:.4f}")
    module.eval()
    return model


def insample_error(groups, model):
    """Compare baseline |residual| vs |residual - gcn_pred| on the training graphs."""
    base_mae, corr_mae, n = 0.0, 0.0, 0
    for g in groups:
        pred = model.predict(g["X"], g["A"])[g["idx"]]
        base_mae += float(np.sum(np.abs(g["y"])))
        corr_mae += float(np.sum(np.abs(g["y"] - pred)))
        n += len(g["idx"])
    return base_mae / max(1, n), corr_mae / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-3)
    args = ap.parse_args()

    print(f"Collecting baseline rollouts on {list(CALIB_TAGS)} ...")
    per_scenario = collect(CALIB_TAGS, args.seed, args.workers)
    groups, all_targets, _ = build_groups(per_scenario)
    print(f"  {len(groups)} interval-graphs, {all_targets.size} residual labels")

    y_mean = float(np.mean(all_targets))
    y_std = float(np.std(all_targets))
    print(f"  residual mean={y_mean:.3f} std={y_std:.3f}")

    model = train(groups, y_mean, y_std, args.hidden, args.epochs, args.lr, args.seed)
    base_mae, corr_mae = insample_error(groups, model)
    print(f"\nin-sample forecast |residual| MAE: baseline {base_mae:.3f}  "
          f"-> corrected {corr_mae:.3f}  ({100.0 * (base_mae - corr_mae) / max(base_mae, 1e-9):+.1f}%)")

    model.save(WEIGHTS)
    print(f"Saved GCN forecaster -> {WEIGHTS}")


if __name__ == "__main__":
    main()
