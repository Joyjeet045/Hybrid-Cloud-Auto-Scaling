"""Offline distillation trainer for the GCN bottleneck selector (PDF Rec 2,
Approach 1).

Runs baseline NF-DiagScale episodes (via :class:`GnnSelectionController` in record
mode) on a few calibration scenarios, collecting ``(node_features, DAG,
analytic_score)`` triples, then trains the 2-layer GCN to reproduce the analytic
selection argmax. Selection is an argmax, so the objective is a softmax
cross-entropy over a graph's nodes with the analytic argmax as the target class
(a plain MSE fit of the score level scrambles the ranking). Weights are saved next
to this file as ``gnn_weights.pt``.

Run from the repo root:
    python -m ablations.rec1_gnn_selection.train_gnn                      # local features
    python -m ablations.rec1_gnn_selection.train_gnn --include-rank --epochs 2000 --hidden 32
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle

import numpy as np

from nfg_diagscale.config import load_config
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from run_star_comparison import BUDGET, DEADLINE, SCENARIOS
from ablations.rec1_gnn_selection import gnn_select
from ablations.rec1_gnn_selection.gnn_controller import GnnSelectionController

INTERVALS = 480
DEFAULT_TAGS = ["N-13", "W-13", "A-13"]
HERE = os.path.dirname(__file__)
WEIGHTS = os.path.join(HERE, "gnn_weights.pt")


def collect(cfg, tags, seed):
    """Run baseline episodes with the recorder on; return distillation samples."""
    records = []
    for tag in tags:
        app, workload = SCENARIOS[tag]
        env = HGraphScaleEnv(app=app, workload=workload, seed=seed, budget=BUDGET)
        state = env.reset(test=True)
        ctrl = GnnSelectionController(cfg, deadline=DEADLINE, total_intervals=INTERVALS)
        ctrl.reset(budget_T=BUDGET, total_intervals=INTERVALS)
        ctrl.record_on()
        done = False
        with contextlib.redirect_stdout(io.StringIO()):
            while not done:
                action = ctrl.act(state)
                state, _r, done, _info = env.step(action)
        recs = ctrl.pop_records()
        records.extend(recs)
        print(f"  {tag}: {len(recs)} decision-step graphs")
    return records


def train(records, epochs, lr, hidden, seed, include_rank):
    """Full-batch softmax cross-entropy GCN training over per-graph argmax."""
    import torch

    scorer = gnn_select.GCNScorer(in_dim=gnn_select.feature_dim(include_rank),
                                  hidden=hidden, seed=seed, include_rank=include_rank)
    model = scorer.module
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()

    data = []
    for X, A, y in records:
        if X.shape[0] <= 1:
            continue
        data.append((
            torch.from_numpy(np.ascontiguousarray(X)),
            torch.from_numpy(np.ascontiguousarray(A)),
            torch.tensor([int(np.argmax(y))], dtype=torch.long),
        ))

    for ep in range(epochs):
        opt.zero_grad()
        losses = [ce(model(Xt, At).unsqueeze(0), tgt) for Xt, At, tgt in data]
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        if ep == 0 or (ep + 1) % 50 == 0:
            print(f"  epoch {ep + 1:4d}  mean CE {float(loss.detach()):.6e}")
    return scorer


def selection_agreement(scorer, records):
    """Fraction of multi-node steps where GCN argmax == analytic argmax."""
    agree = n = 0
    for X, A, y in records:
        if X.shape[0] <= 1:
            continue
        pred = scorer.predict(X, A)
        agree += int(np.argmax(pred) == np.argmax(y))
        n += 1
    return (agree / n if n else float("nan")), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--include-rank", action="store_true")
    args = ap.parse_args()

    cfg = load_config()

    suffix = "-rank" if args.include_rank else ""
    cache = os.path.join(HERE, f"_records_{'-'.join(args.tags)}_s{args.seed}{suffix}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            records = pickle.load(fh)
        print(f"loaded {len(records)} cached graph samples from {os.path.basename(cache)}")
    else:
        print(f"collecting distillation data on {args.tags} (seed {args.seed}) ...")
        records = collect(cfg, args.tags, args.seed)
        with open(cache, "wb") as fh:
            pickle.dump(records, fh)
        print(f"  total {len(records)} graph samples (cached)")

    print("training GCN ...")
    scorer = train(records, args.epochs, args.lr, args.hidden, args.seed,
                   bool(args.include_rank))

    agr, n = selection_agreement(scorer, records)
    print(f"in-sample selection agreement: {agr * 100:.1f}%  over {n} multi-node steps")

    scorer.save(WEIGHTS)
    print(f"saved weights -> {WEIGHTS}")


if __name__ == "__main__":
    main()
