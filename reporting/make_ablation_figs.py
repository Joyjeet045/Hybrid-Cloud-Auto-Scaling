"""Ablation-study figures for NF-DiagScale (per-scenario degradation).

Renders the two negative-result ablations into publication figures from the
measured per-scenario closed-loop results (12 scenarios, seed 0). Reproduce the
underlying numbers with:
  python -m ablations.rec1_gnn_selection.train_gnn --epochs 400 --lr 0.02
  python -m ablations.rec1_gnn_selection.evaluate  --workers 12
  python -m ablations.rec2_adaptive_mf.evaluate    --workers 12            # single-scenario calib
  python -m ablations.rec2_adaptive_mf.evaluate    --workers 12 --pooled   # pooled calib

This script only visualises the measured numbers; it does not re-run the sweeps.

Usage:
  python reporting/make_ablation_figs.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Measured per-scenario closed-loop MRT (ms), seed 0, all variants Vio = 0.
# ---------------------------------------------------------------------------
SCEN = ["N-11", "N-12", "N-13", "W-11", "W-12", "W-13",
        "A-11", "A-12", "A-13", "N-14", "W-14", "A-14"]

# Default (main) NF-DiagScale controller -- the ablation baseline.
BASE = np.array([152.91, 157.39, 140.52, 229.34, 261.53, 210.52,
                 169.09, 162.74, 130.07, 249.87, 383.11, 224.89])

# Ablation A: analytic critical-path selector replaced by a distilled 2-layer GCN
# (local features, 71.9% in-sample selection agreement).
GNN = np.array([271.27, 271.70, 541.44, 373.76, 372.38, 443.59,
                232.72, 223.76, 442.59, 560.85, 448.13, 481.92])

# Ablation B: expert membership functions replaced by k-means-learned ones,
# under two calibration regimes.
MF_SINGLE = np.array([161.81, 150.44, 138.96, 223.86, 229.83, 228.51,   # calib on A-12
                      127.70, 140.83, 129.49, 230.84, 479.17, 216.10])
MF_POOLED = np.array([137.70, 151.16, 139.08, 251.91, 236.19, 218.30,   # calib on N/W/A-13
                      121.77, 162.48, 129.55, 240.88, 389.05, 237.89])

BASE_MEAN = float(BASE.mean())  # 206.00

# ---------------------------------------------------------------------------
# Ablation C / single-axis (deployed full controller = GCN residual + criticality).
# Measured by reporting/strengthen_experiments.py (seed 0, all Vio = 0).
# ---------------------------------------------------------------------------
FULL = np.array([151.55, 161.94, 140.40, 223.07, 243.84, 208.55,
                 169.17, 162.05, 129.94, 250.70, 382.30, 225.15])
# Single-axis restrictions.
VERT = np.array([363.88, 340.20, 430.56, 680.93, 627.58, 803.87,
                 333.63, 279.40, 351.39, 777.22, 1463.23, 631.59])
HORIZ = np.array([158.34, 155.89, 145.90, 230.52, 253.75, 224.27,
                  144.83, 149.88, 144.00, 241.39, 414.24, 231.88])
# Component removals (GCN_OFF == BASE, the criticality-on/residual-off variant).
GCN_OFF = BASE
CRIT_OFF = np.array([159.56, 163.33, 138.86, 227.75, 255.87, 221.36,
                     207.69, 156.27, 130.13, 252.37, 392.42, 218.64])
FULL_MEAN = float(FULL.mean())  # 204.06



def _style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 160, "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
        "legend.fontsize": 9, "legend.frameon": False,
    })


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {os.path.relpath(path, _REPO_ROOT)}")
    return path


def fig_ablation_gnn():
    """Ablation A: per-scenario MRT, analytic selector vs distilled GCN."""
    x = np.arange(len(SCEN))
    bw = 0.40
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - bw / 2, BASE, bw, label="analytic selector (baseline)",
           color="#2e7d32", edgecolor="k", linewidth=0.5)
    b1 = ax.bar(x + bw / 2, GNN, bw, label="GNN selector (71.9% agr.)",
                color="#d32f2f", edgecolor="k", linewidth=0.5)
    for b in b1:
        ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(SCEN)
    ax.set_ylabel("response time (ms)")
    ax.set_ylim(0, GNN.max() * 1.18)
    ax.set_xlabel("scenario")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Ablation A: a GCN distilled from the analytic critical-path selector "
                 "regresses\nevery scenario ($+61$ to $+401$ ms); all-12 mean "
                 f"{BASE_MEAN:.0f}$\\to${GNN.mean():.0f} ms "
                 f"($+{100 * (GNN.mean() - BASE_MEAN) / BASE_MEAN:.0f}\\%$, Vio still 0)",
                 loc="left", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, "fig_abl_gnn.png")


def fig_ablation_mf():
    """Ablation B: per-scenario signed MRT change vs the expert baseline."""
    d_single = MF_SINGLE - BASE
    d_pooled = MF_POOLED - BASE
    x = np.arange(len(SCEN))
    bw = 0.40
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - bw / 2, d_single, bw, label="learned MFs, single-scenario calib.",
           color="#ff9800", edgecolor="k", linewidth=0.5)
    ax.bar(x + bw / 2, d_pooled, bw, label="learned MFs, pooled calib.",
           color="#1565c0", edgecolor="k", linewidth=0.5)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axhline(d_single.mean(), color="#ff9800", ls="--", lw=1.0,
               label=f"single mean $\\Delta={d_single.mean():+.1f}$ ms")
    ax.axhline(d_pooled.mean(), color="#1565c0", ls="--", lw=1.0,
               label=f"pooled mean $\\Delta={d_pooled.mean():+.1f}$ ms")
    ax.set_xticks(x)
    ax.set_xticklabels(SCEN)
    ax.set_xlabel("scenario")
    ax.set_ylabel(r"$\Delta$ MRT vs expert baseline (ms)" "\n(positive = worse)")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Ablation B: learned membership functions help some scenarios and "
                 "hurt others\n(W-14 $+96$, W-11 $+23$, in-sample W-13 $+8$/$+18$); "
                 "no robust, reliable gain",
                 loc="left", fontsize=10.5)
    pad = max(abs(d_single).max(), abs(d_pooled).max()) * 1.18
    ax.set_ylim(-pad, pad)
    fig.tight_layout()
    return _save(fig, "fig_abl_mf.png")


def fig_ablation_diag():
    """Single-axis ablation: diagonal vs vertical-only vs horizontal-only (log)."""
    x = np.arange(len(SCEN))
    bw = 0.27
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - bw, FULL, bw, label=f"diagonal (full), mean {FULL.mean():.0f} ms",
           color="#2e7d32", edgecolor="k", linewidth=0.5)
    ax.bar(x, VERT, bw, label=f"vertical-only, mean {VERT.mean():.0f} ms",
           color="#d32f2f", edgecolor="k", linewidth=0.5)
    ax.bar(x + bw, HORIZ, bw, label=f"horizontal-only, mean {HORIZ.mean():.0f} ms",
           color="#1565c0", edgecolor="k", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(SCEN)
    ax.set_xlabel("scenario")
    ax.set_ylabel("response time (ms, log scale)")
    ax.legend(loc="upper left", ncol=1)
    ax.set_title("Single-axis ablation: forbidding new replicas (vertical-only) nearly "
                 "triples\nmean MRT ($+189\\%$); horizontal-only is near-diagonal "
                 "(vertical-first platform), but diagonal wins",
                 loc="left", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, "fig_abl_diag.png")


def fig_ablation_comp():
    """Ablation C: signed per-scenario MRT change when a kept component is removed."""
    d_gcn = GCN_OFF - FULL
    d_crit = CRIT_OFF - FULL
    x = np.arange(len(SCEN))
    bw = 0.40
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - bw / 2, d_gcn, bw, label="$-$ GCN forecast residual",
           color="#1565c0", edgecolor="k", linewidth=0.5)
    ax.bar(x + bw / 2, d_crit, bw, label="$-$ criticality blend",
           color="#ff9800", edgecolor="k", linewidth=0.5)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axhline(d_gcn.mean(), color="#1565c0", ls="--", lw=1.0,
               label=f"$-$GCN mean $\\Delta={d_gcn.mean():+.2f}$ ms")
    ax.axhline(d_crit.mean(), color="#ff9800", ls="--", lw=1.0,
               label=f"$-$crit. mean $\\Delta={d_crit.mean():+.2f}$ ms")
    ax.set_xticks(x)
    ax.set_xticklabels(SCEN)
    ax.set_xlabel("scenario")
    ax.set_ylabel(r"$\Delta$ MRT when component removed (ms)" "\n(positive = component helps)")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Ablation C: both accepted components are net positive \u2014 the residual "
                 "helps bursty\nWikipedia traces, criticality helps the DAG-central A-11 "
                 "($+38$ ms); Vio still 0",
                 loc="left", fontsize=10.5)
    pad = max(abs(d_gcn).max(), abs(d_crit).max()) * 1.18
    ax.set_ylim(-pad, pad)
    fig.tight_layout()
    return _save(fig, "fig_abl_comp.png")


def main():
    _style()
    print("\n=== NF-DiagScale ablation figures (per-scenario) ===")
    fig_ablation_gnn()
    fig_ablation_mf()
    fig_ablation_diag()
    fig_ablation_comp()
    print(f"\nDone. Figures -> {os.path.relpath(FIG_DIR, _REPO_ROOT)}/\n")


if __name__ == "__main__":
    main()
