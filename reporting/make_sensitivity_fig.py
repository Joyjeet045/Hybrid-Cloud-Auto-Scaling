"""Hyperparameter-sensitivity figure for NF-DiagScale.

Reads ``reporting/sensitivity_results.json`` (produced by sensitivity_sweep.py)
and renders the all-12 mean MRT versus each of the three tunable knobs in a 1x3
panel, marking the default and the catastrophic anchor-under-weight point.

Usage:  python reporting/make_sensitivity_fig.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

PANELS = [
    ("zeta", r"criticality blend $\zeta$", 0.4),
    ("kappa", r"anchor blend $\kappa$", 0.45),
    ("eta", r"adaptation rate $\eta$", 0.10),
]


def main():
    with open(os.path.join(_REPO_ROOT, "reporting", "sensitivity_results.json")) as f:
        data = json.load(f)
    rows = data["rows"]

    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 160, "font.size": 11,
        "axes.titlesize": 11, "axes.labelsize": 10, "axes.grid": True,
        "grid.alpha": 0.3, "axes.axisbelow": True, "legend.fontsize": 8.5,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
    # Shared y-limits for the two flat knobs so their small swings read as flat;
    # kappa keeps its own range so the under-weight cliff stays visible.
    ylims = {"zeta": (190, 222), "eta": (190, 222), "kappa": (185, 335)}
    for ax, (param, label, default) in zip(axes, PANELS):
        pts = sorted([(r["value"], r["mean"]) for r in rows if r["param"] == param])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color="#1565c0", lw=1.8, ms=5, zorder=3)
        # mark the default point directly (no legend to avoid stray markers)
        dy = next(y for x, y in pts if abs(x - default) < 1e-9)
        ax.plot([default], [dy], marker="*", color="#d32f2f", ms=14,
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        ax.annotate(f"default\n({default:g})", (default, dy),
                    textcoords="offset points", xytext=(7, 8),
                    fontsize=8.5, color="#d32f2f", fontweight="bold")
        ax.set_xlabel(label)
        ax.set_title(label, fontsize=10.5)
        ax.set_ylim(*ylims[param])
        # annotate the catastrophic kappa point
        if param == "kappa":
            xbad, ybad = pts[0]
            ax.annotate(f"{ybad:.0f} ms\n(anchor\nunder-weighted)",
                        (xbad, ybad), textcoords="offset points", xytext=(16, -34),
                        fontsize=8, color="#d32f2f",
                        arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=0.8))
    axes[0].set_ylabel("all-12 mean MRT (ms)")
    fig.suptitle("Hyperparameter sensitivity (one-at-a-time, all 12 scenarios, seed 0; "
                 "Vio = 0 throughout)", fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "fig_sensitivity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {os.path.relpath(path, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
