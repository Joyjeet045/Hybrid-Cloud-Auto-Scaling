"""Online-adaptation convergence figure for NF-DiagScale.

Runs one deterministic episode, captures the controller's ``learn_trace`` (recorded
at every Stage-6 consequent update), and plots two views of convergence:
  (top)    the mean fuzzy consequents (vCPU and replica action magnitudes)
           settling to stable values as the bounded online rule adapts them;
  (bottom) the rolling-mean magnitude of the adaptation signal decaying into the
           deadzone band, i.e. the loop reaching a chatter-free steady state.

Usage:  python reporting/make_convergence_fig.py [--scenario A-12]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nfg_diagscale.config import load_config
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from nfg_diagscale.hgraph_policy.controller import NFGDiagScaleController

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
BUDGET, DEADLINE, TOTAL = 200.0, 500.0, 480
SCEN = {"A-12": ("A12", "alibaba"), "W-13": ("A13", "wiki"), "W-14": ("A14", "wiki")}


def _rolling(x, w=15):
    if len(x) < 2:
        return x
    w = max(1, min(w, len(x)))
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def collect(scenario):
    app, workload = SCEN[scenario]
    cfg = load_config()
    eps = float(cfg.get("adaptive", {}).get("deadzone_eps", 0.05))
    env = HGraphScaleEnv(app=app, workload=workload, seed=0, budget=BUDGET)
    state = env.reset(test=True)
    ctrl = NFGDiagScaleController(cfg, deadline=DEADLINE, total_intervals=TOTAL)
    ctrl.reset(budget_T=BUDGET, total_intervals=TOTAL)
    done = False
    with contextlib.redirect_stdout(io.StringIO()):
        while not done:
            state, _r, done, _info = env.step(ctrl.act(state))
    return ctrl.learn_trace, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="A-12", choices=list(SCEN))
    args = ap.parse_args()

    trace, eps = collect(args.scenario)
    if not trace:
        print("no adaptation steps recorded; try a heavier scenario")
        return
    slot = np.array([t["slot"] for t in trace], dtype=float)
    s_dc = np.array([t["mean_s_dc"] for t in trace])
    s_dn = np.array([t["mean_s_dn"] for t in trace])
    signal = np.array([t["signal"] for t in trace])

    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 160, "font.size": 11,
        "axes.titlesize": 11, "axes.labelsize": 10, "axes.grid": True,
        "grid.alpha": 0.3, "axes.axisbelow": True, "legend.fontsize": 9,
        "legend.frameon": False,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 5.4), sharex=True)

    ax1.plot(slot, s_dc, color="#1565c0", lw=1.6, label=r"mean vCPU consequent $\bar{s}_{\Delta c}$")
    ax1.plot(slot, s_dn, color="#2e7d32", lw=1.6, label=r"mean replica consequent $\bar{s}_{\Delta n}$")
    ax1.set_ylabel("consequent value")
    ax1.legend(loc="best", ncol=1)
    ax1.set_title(f"Online adaptation on {args.scenario}: bounded consequents settle to a "
                  "stable operating point", loc="left", fontsize=10.5)

    ax2.axhspan(-eps, eps, color="0.85", label=f"deadzone $\\pm\\epsilon$ ($\\epsilon={eps:g}$)")
    ax2.plot(slot, signal, color="0.6", lw=0.7, alpha=0.7, label="adaptation signal")
    ax2.plot(slot, _rolling(signal), color="#d32f2f", lw=1.8, label="rolling mean (15)")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("control interval")
    ax2.set_ylabel("adaptation signal")
    ax2.legend(loc="best", ncol=2)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "fig_convergence.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {os.path.relpath(path, _REPO_ROOT)}  ({len(trace)} adaptation steps)")
    print(f"  final consequents: s_dc={s_dc[-1]:.3f}  s_dn={s_dn[-1]:.3f}")
    print(f"  |signal| mean first-quarter {np.mean(np.abs(signal[:len(signal)//4])):.3f} "
          f"-> last-quarter {np.mean(np.abs(signal[-len(signal)//4:])):.3f}")


if __name__ == "__main__":
    main()
