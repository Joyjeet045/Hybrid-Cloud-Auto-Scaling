"""Statistical validation (Rec 7) for NF-DiagScale vs the STAR-family baselines.

Reads ``star_comparison_results.json`` and runs non-parametric significance
tests on the scenarios where *every* method has a result (the shared N/W/A x
11-13 cases -- the three ``-14`` scenarios only have NF-DiagScale numbers, so
they cannot enter a cross-method comparison). Emits paper-ready tables and
figures. It does NOT run the simulator or alter any existing result -- it only
post-processes the numbers already in the JSON.

Produces
--------
tables/table6_friedman_ranks.{csv,md}
        Friedman omnibus test + per-method mean rank (1 = best / lowest MRT).
tables/table7_pairwise_significance.{csv,md}
        NF-DiagScale vs each baseline: Wilcoxon signed-rank p (raw + Holm),
        Cliff's delta effect size and its magnitude.
figures/fig11_cd_diagram.png
        Nemenyi critical-difference diagram (Friedman post-hoc).
figures/fig12_effect_sizes.png
        Cliff's delta of NF-DiagScale's MRT advantage per baseline.

Usage
-----
    python reporting/stats_validation.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
FIG_DIR = os.path.join(_HERE, "figures")
TAB_DIR = os.path.join(_HERE, "tables")
RESULTS = os.path.join(_REPO_ROOT, "star_comparison_results.json")

_NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def load_mrt_matrix(path=RESULTS):
    """Return (scenarios, methods, mrt) with mrt[i, j] = MRT of method j on
    scenario i. ``methods`` lists the baselines first, NF-DiagScale last."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    star = data["star_table"]
    nfg = data["nfg_diagscale"]

    scenarios = [s for s in star if s in nfg]
    baselines = sorted({m for s in scenarios for m in star[s]})
    methods = baselines + ["NF-DiagScale"]

    mrt = np.empty((len(scenarios), len(methods)), dtype=float)
    for i, s in enumerate(scenarios):
        for j, m in enumerate(baselines):
            mrt[i, j] = float(star[s][m][0])
        mrt[i, -1] = float(nfg[s]["MRT"])
    return scenarios, methods, mrt


def mean_ranks(mrt):
    """Per-scenario ranks (1 = lowest MRT), averaged over scenarios."""
    ranks = np.vstack([stats.rankdata(row) for row in mrt])
    return ranks.mean(axis=0)


def cliffs_delta(a, b):
    """Cliff's delta of a vs b. Negative => a tends to be SMALLER (better MRT)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = int(np.sum(a[:, None] > b[None, :]))
    lt = int(np.sum(a[:, None] < b[None, :]))
    return (gt - lt) / (a.size * b.size)


def cliffs_magnitude(d):
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.330:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def _write_table(name, headers, rows, title=None):
    os.makedirs(TAB_DIR, exist_ok=True)
    csv_path = os.path.join(TAB_DIR, name + ".csv")
    md_path = os.path.join(TAB_DIR, name + ".md")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(f'"{h}"' for h in headers) + "\n")
        for r in rows:
            fh.write(",".join(f'"{c}"' for c in r) + "\n")
    with open(md_path, "w", encoding="utf-8") as fh:
        if title:
            fh.write(f"**{title}**\n\n")
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for r in rows:
            fh.write("| " + " | ".join(str(c) for c in r) + " |\n")
    print(f"  [tab] {os.path.relpath(md_path, _REPO_ROOT)}  (+ .csv)")


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"  [fig] {os.path.relpath(path, _REPO_ROOT)}")


def _cliques(ranks_sorted, cd):
    """Maximal groups of (sorted) methods whose rank span is below CD."""
    k = len(ranks_sorted)
    groups = []
    for i in range(k):
        j = k - 1
        while j > i and (ranks_sorted[j] - ranks_sorted[i]) >= cd:
            j -= 1
        if j > i:
            groups.append((i, j))
    maximal = []
    for g in groups:
        if not any(o != g and o[0] <= g[0] and g[1] <= o[1] for o in groups):
            maximal.append(g)
    return maximal


def fig_cd_diagram(methods, ranks, cd, name="fig11_cd_diagram.png"):
    k = len(methods)
    idx = np.argsort(ranks)
    names = [methods[i] for i in idx]
    rsorted = np.array([ranks[i] for i in idx], dtype=float)
    lo, hi = 1.0, float(k)

    def xpos(r):
        return (r - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(9.0, 0.55 * k + 2.2))
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.05, 1.25)
    ax.axis("off")

    axis_y = 1.0
    ax.plot([0, 1], [axis_y, axis_y], color="black", lw=1.6)
    for r in range(int(lo), int(hi) + 1):
        x = xpos(r)
        ax.plot([x, x], [axis_y, axis_y + 0.025], color="black", lw=1.2)
        ax.text(x, axis_y + 0.05, str(r), ha="center", va="bottom", fontsize=10)
    ax.text(0.5, axis_y + 0.135, "Mean rank  (1 = best / lowest MRT)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

    cd_x = xpos(lo + cd) - xpos(lo)
    bar_y = axis_y + 0.085
    ax.plot([0, cd_x], [bar_y, bar_y], color="black", lw=2.2)
    for ex in (0.0, cd_x):
        ax.plot([ex, ex], [bar_y - 0.013, bar_y + 0.013], color="black", lw=1.2)
    ax.text(cd_x / 2.0, bar_y + 0.02, f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=10)

    half = (k + 1) // 2
    row_gap = 0.115
    for pos, (nm, r) in enumerate(zip(names, rsorted)):
        x = xpos(r)
        if pos < half:
            level = axis_y - 0.14 - row_gap * pos
            ax.plot([x, x], [axis_y, level], color="tab:blue", lw=1.5)
            ax.plot([x, -0.03], [level, level], color="tab:blue", lw=1.5)
            ax.text(-0.05, level, f"{nm}  ({r:.2f})",
                    ha="right", va="center", fontsize=10)
        else:
            j = pos - half
            level = axis_y - 0.14 - row_gap * (k - 1 - pos)
            ax.plot([x, x], [axis_y, level], color="tab:red", lw=1.5)
            ax.plot([x, 1.03], [level, level], color="tab:red", lw=1.5)
            ax.text(1.05, level, f"{nm}  ({r:.2f})",
                    ha="left", va="center", fontsize=10)

    clique_y = axis_y - 0.055
    for n, (a, b) in enumerate(_cliques(rsorted, cd)):
        yy = clique_y - 0.022 * n
        ax.plot([xpos(rsorted[a]) - 0.006, xpos(rsorted[b]) + 0.006], [yy, yy],
                color="black", lw=4.0, solid_capstyle="round")

    ax.text(0.5, -0.02,
            "Friedman + Nemenyi post-hoc (alpha = 0.05); methods joined by a "
            "bar are not significantly different.",
            ha="center", va="top", fontsize=9, color="#444444")
    _save(fig, name)


def fig_effect_sizes(baselines, deltas, holm_p, name="fig12_effect_sizes.png"):
    order = np.argsort(np.abs(deltas))[::-1]
    names = [baselines[i] for i in order]
    vals = [abs(deltas[i]) for i in order]
    ps = [holm_p[i] for i in order]

    def color(v):
        if v < 0.147:
            return "#d9d9d9"
        if v < 0.330:
            return "#9ecae1"
        if v < 0.474:
            return "#4292c6"
        return "#08519c"

    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(names) + 1.6))
    y = np.arange(len(names))
    ax.barh(y, vals, color=[color(v) for v in vals], edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("|Cliff's \u03b4|  (NF-DiagScale vs baseline, MRT)")
    ax.set_title("Effect size of NF-DiagScale's MRT advantage\n"
                 "(Holm-corrected Wilcoxon signed-rank)")

    for thr, lbl in ((0.147, "small"), (0.330, "medium"), (0.474, "large")):
        ax.axvline(thr, color="gray", ls=":", lw=0.8)
        ax.text(thr, -0.6, lbl, ha="center", va="bottom", fontsize=8,
                color="gray")

    for yi, (v, p) in enumerate(zip(vals, ps)):
        star = ("***" if p < 0.001 else "**" if p < 0.01
                else "*" if p < 0.05 else "ns")
        ax.text(v + 0.012, yi, f"{v:.2f} {star}", va="center", fontsize=9)

    _save(fig, name)


def main():
    scenarios, methods, mrt = load_mrt_matrix()
    n_scen, k = len(scenarios), len(methods)
    baselines = methods[:-1]
    nf = mrt[:, -1]

    if k not in _NEMENYI_Q05:
        raise SystemExit(f"No Nemenyi critical value tabulated for k={k} methods.")

    chi2, p_fried = stats.friedmanchisquare(*[mrt[:, j] for j in range(k)])
    ranks = mean_ranks(mrt)
    cd = _NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n_scen))

    order = np.argsort(ranks)
    rows6 = [(methods[i], f"{ranks[i]:.2f}") for i in order]
    _write_table(
        "table6_friedman_ranks",
        ["Method", "Mean rank (MRT, 1=best)"],
        rows6,
        title=(f"Friedman test over {k} methods x {n_scen} scenarios: "
               f"chi2={chi2:.2f}, p={p_fried:.2e}. Nemenyi CD={cd:.2f} "
               f"(alpha=0.05)."),
    )

    wil_p, deltas = [], []
    for j in range(len(baselines)):
        res = stats.wilcoxon(nf, mrt[:, j], alternative="two-sided")
        wil_p.append(float(res.pvalue))
        deltas.append(cliffs_delta(nf, mrt[:, j]))
    holm = holm_bonferroni(wil_p)

    rows7 = []
    for j, b in enumerate(baselines):
        wins = int(np.sum(nf < mrt[:, j]))
        rows7.append((
            b,
            f"{np.median(mrt[:, j]):.1f}",
            f"{wins}/{n_scen}",
            f"{wil_p[j]:.4f}",
            f"{holm[j]:.4f}",
            f"{deltas[j]:+.3f}",
            cliffs_magnitude(deltas[j]),
        ))
    _write_table(
        "table7_pairwise_significance",
        ["Baseline", "Median MRT (ms)", "NF wins", "Wilcoxon p",
         "Holm p", "Cliff's d", "Effect"],
        rows7,
        title=("NF-DiagScale vs each baseline (paired by scenario, MRT). "
               "Cliff's d < 0 favours NF-DiagScale; p two-sided."),
    )

    fig_cd_diagram(methods, ranks, cd)
    fig_effect_sizes(baselines, deltas, holm)

    print("\n" + "=" * 72)
    print(f"Scenarios (all methods present): {n_scen}  ->  {', '.join(scenarios)}")
    print(f"Friedman: chi2={chi2:.3f}  p={p_fried:.3e}  (k={k}, N={n_scen})")
    print(f"Nemenyi CD (alpha=0.05) = {cd:.3f} rank units")
    print("Mean ranks (1 = best):")
    for i in order:
        print(f"   {methods[i]:<14} {ranks[i]:.2f}")
    print("\nNF-DiagScale vs baselines (paired Wilcoxon, Holm-corrected):")
    for j, b in enumerate(baselines):
        sig = "significant" if holm[j] < 0.05 else "n.s."
        print(f"   vs {b:<11} p={wil_p[j]:.4f}  Holm={holm[j]:.4f} ({sig})  "
              f"Cliff d={deltas[j]:+.3f} [{cliffs_magnitude(deltas[j])}]")


if __name__ == "__main__":
    main()
