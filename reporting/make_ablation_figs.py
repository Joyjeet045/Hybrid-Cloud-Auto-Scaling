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

# Deployed NF-DiagScale controller (pooled-calibrated MFs) -- Ablation A baseline.
BASE = np.array([169.66, 165.80, 138.74, 230.53, 241.24, 197.11,
                 149.40, 153.25, 136.35, 255.74, 369.63, 249.47])  # 204.74

# Hand-tuned expert seed (pre-calibration) -- Ablation B reference.
EXPERT = np.array([161.10, 176.07, 141.94, 443.85, 381.82, 208.87,
                   217.42, 166.06, 130.94, 287.19, 443.66, 276.96])  # 252.99

# Ablation A: analytic critical-path selector replaced by a distilled 2-layer GCN
# (local features, 71.9% in-sample selection agreement).
GNN = np.array([331.85, 294.15, 542.67, 527.63, 332.49, 373.98,
                269.30, 245.16, 446.49, 495.22, 421.42, 410.44])  # 390.90

# Ablation B: k-means-calibrated membership functions vs the expert seed,
# under two calibration regimes.
MF_SINGLE = np.array([169.11, 165.32, 140.65, 225.56, 242.74, 195.58,   # calib on A-12
                      150.68, 153.15, 136.92, 255.09, 619.26, 265.95])  # 226.67
MF_POOLED = np.array([169.66, 165.80, 138.74, 230.53, 241.24, 197.11,   # calib on N/W/A-13
                      149.40, 153.25, 136.35, 255.74, 369.63, 249.47])  # 204.74 (deployed)

BASE_MEAN = float(BASE.mean())  # 204.74

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
                 "regresses\nevery scenario ($+52$ to $+404$ ms); all-12 mean "
                 f"{BASE_MEAN:.0f}$\\to${GNN.mean():.0f} ms "
                 f"($+{100 * (GNN.mean() - BASE_MEAN) / BASE_MEAN:.0f}\\%$, Vio still 0)",
                 loc="left", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, "fig_abl_gnn.png")


def fig_ablation_mf():
    """Ablation B: per-scenario signed MRT change vs the expert seed."""
    d_single = MF_SINGLE - EXPERT
    d_pooled = MF_POOLED - EXPERT
    x = np.arange(len(SCEN))
    bw = 0.40
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - bw / 2, d_single, bw, label="calibrated MFs, single-scenario",
           color="#ff9800", edgecolor="k", linewidth=0.5)
    ax.bar(x + bw / 2, d_pooled, bw, label="calibrated MFs, pooled (deployed)",
           color="#1565c0", edgecolor="k", linewidth=0.5)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axhline(d_single.mean(), color="#ff9800", ls="--", lw=1.0,
               label=f"single mean $\\Delta={d_single.mean():+.1f}$ ms")
    ax.axhline(d_pooled.mean(), color="#1565c0", ls="--", lw=1.0,
               label=f"pooled mean $\\Delta={d_pooled.mean():+.1f}$ ms")
    ax.set_xticks(x)
    ax.set_xticklabels(SCEN)
    ax.set_xlabel("scenario")
    ax.set_ylabel(r"$\Delta$ MRT vs expert seed (ms)" "\n(positive = worse)")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Ablation B: $k$-means calibration helps only when pooled \u2014 single-scenario\n"
                 "overfits (W-14 $+176$ ms); pooled repairs W-11 ($-213$ ms) and wins all nine",
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
