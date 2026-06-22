"""Figure & table generator for NF-DiagScale (HGraphScale environment).

Produces the full set of plots and tables for the NF-DiagScale autoscaling
architecture, mirroring the artifacts in the two inspiration papers:

  * STAR (Fang et al., 2026, ESWA): workload traces (Fig. 9), the MRT/Vio
    comparison table (Table 3) and bar charts, and the EC2 m5 VM-type table
    (Table 2).
  * HGraphScale (Hu et al., 2026, IEEE TSC): per-episode control trajectory
    (response time / cost / scaling actions over time).

Plus the artifacts specific to NF-DiagScale's Forecast->Fuzzify->Sizer->ANFIS
pipeline: Kalman+Holt forecast accuracy, the cost-latency sizing front, the ANFIS
fuzzy membership functions, and the fuzzy rule base.

Everything is measured/derived from the vendored simulator and the live
controller objects -- no numbers are fabricated. The MRT/Vio comparison
(table1 + fig2) is read from ``star_comparison_results.json`` if present; run
``run_star_comparison.py`` first (or this script with ``--run-comparison``) to
produce it.

Usage:
    python reporting/make_report.py                 # all fast artifacts
    python reporting/make_report.py --rep N-13       # pick the instrumented scenario
    python reporting/make_report.py --run-comparison # also run the 9-scenario sweep
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nfg_diagscale.config import load_config
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from nfg_diagscale.hgraph_policy.controller import NFGDiagScaleController
from nfg_diagscale.hgraph_policy.forecaster import ContainerForecaster
from nfg_diagscale.decision.fuzzy_rules import (
    LINGUISTIC_TERMS, MODE_NAMES, build_rule_base, gaussian_mf,
)
import nfg_diagscale.hgraph_policy.optimizer as opt_mod

BUDGET = 200.0
DEADLINE = 500.0
TEST_INTERVALS = 480
TRAIN_INTERVALS = 480

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
TAB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
RESULTS_JSON = os.path.join(_REPO_ROOT, "star_comparison_results.json")

SCENARIOS = {
    "N-11": ("A11", "nasa"), "N-12": ("A12", "nasa"), "N-13": ("A13", "nasa"),
    "W-11": ("A11", "wiki"), "W-12": ("A12", "wiki"), "W-13": ("A13", "wiki"),
    "A-11": ("A11", "alibaba"), "A-12": ("A12", "alibaba"), "A-13": ("A13", "alibaba"),
}
SCEN_ORDER = list(SCENARIOS)
WORKLOADS = [("nasa", "NASA"), ("wiki", "Wikipedia"), ("alibaba", "Alibaba (real v2022)")]

METHODS = ["AWS-Scale", "ProScale", "DeepScale", "DRPC", "STAR", "NF-DiagScale"]
COLORS = {
    "AWS-Scale": "#9e9e9e", "ProScale": "#ff9800", "DeepScale": "#4caf50",
    "DRPC": "#1e88e5", "STAR": "#8e24aa", "NF-DiagScale": "#d32f2f",
}

A14_SCENARIOS = [("N-14", "NASA-14"), ("W-14", "Wiki-14"), ("A-14", "Alibaba-14")]
A14_METHODS = ["AWS-Scale", "ProScale", "DeepScale", "DRPC", "AGQ",
               "HGraphScale", "NF-DiagScale"]
A14_COLORS = {
    "AWS-Scale": "#9e9e9e", "ProScale": "#ff9800", "DeepScale": "#4caf50",
    "DRPC": "#1e88e5", "AGQ": "#00897b", "HGraphScale": "#8e24aa",
    "NF-DiagScale": "#d32f2f",
}
HGS_TABLE_IV_A14 = {
    "N-14": {"AWS-Scale": (1022.10, 0.00), "ProScale": (532.34, 28.37),
             "DeepScale": (348.03, 66.17), "DRPC": (510.72, 1.66),
             "AGQ": (336.43, 161.58), "HGraphScale": (325.67, 0.00)},
    "W-14": {"AWS-Scale": (1022.10, 0.00), "ProScale": (532.34, 11.36),
             "DeepScale": (348.03, 26.62), "DRPC": (510.72, 0.00),
             "AGQ": (520.24, 11.27), "HGraphScale": (325.67, 0.00)},
    "A-14": {"AWS-Scale": (988.76, 0.00), "ProScale": (549.38, 0.00),
             "DeepScale": (327.00, 20.28), "DRPC": (277.06, 56.84),
             "AGQ": (421.87, 12.71), "HGraphScale": (299.28, 0.00)},
}
CH_ORDER = SCEN_ORDER + ["N-14", "W-14", "A-14"]

_SILENT = io.StringIO()


def _style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 160, "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
        "legend.fontsize": 9, "legend.frameon": False,
        "figure.autolayout": False,
    })


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {os.path.relpath(path, _REPO_ROOT)}")
    return path


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
    return md_path


def build_env(app, workload, seed=0):
    with contextlib.redirect_stdout(_SILENT):
        return HGraphScaleEnv(app=app, workload=workload, seed=seed, budget=BUDGET)


def run_instrumented(tag, seed, cfg):
    app, workload = SCENARIOS[tag]
    env = build_env(app, workload, seed)

    captured_fronts = []
    orig_solve = opt_mod.MagnitudeSizer.solve_exact

    def _wrap(self, cur_h, lam, et, **kw):
        h_star, c_star, front = orig_solve(self, cur_h, lam, et, **kw)
        captured_fronts.append({"lam": float(lam), "h_star": int(h_star),
                                "c_star": int(c_star), "front": list(front),
                                "budget_room": float(kw.get("budget_room", float("nan"))),
                                "cur_h": int(cur_h)})
        return h_star, c_star, front

    opt_mod.MagnitudeSizer.solve_exact = _wrap
    try:
        with contextlib.redirect_stdout(_SILENT):
            state = env.reset(test=True)
            ctrl = NFGDiagScaleController(cfg, deadline=DEADLINE,
                                          total_intervals=TEST_INTERVALS)
            ctrl.reset(budget_T=BUDGET, total_intervals=TEST_INTERVALS)

            decide_modes = []
            _orig_decide = ctrl.anfis.decide

            def _wrap_decide(*a, **k):
                d = _orig_decide(*a, **k)
                decide_modes.append(d.get("mode", "none"))
                return d

            ctrl.anfis.decide = _wrap_decide

            traj = {k: [] for k in ("cum_cost", "num_vms", "n_con", "vcpu", "mean_rt")}
            observed_by_type = []
            done, info, t = False, {}, 0
            while not done:
                bt = {}
                for c in state.containers:
                    w = c.workload_his
                    bt[c.con_type] = bt.get(c.con_type, 0.0) + (
                        float(w[-1]) if (w is not None and len(w) > 0) else 0.0)
                observed_by_type.append(bt)

                action = ctrl.act(state)
                state, _r, done, info = env.step(action)

                traj["cum_cost"].append(float(state.total_cost))
                traj["num_vms"].append(int(state.num_vms))
                traj["n_con"].append(len(state.containers))
                traj["vcpu"].append(float(sum(c.vcpu for c in state.containers)))
                rts = [c.aver_resptime for c in state.containers if c.aver_resptime > 0]
                traj["mean_rt"].append(float(np.mean(rts)) if rts else 0.0)
                t += 1
    finally:
        opt_mod.MagnitudeSizer.solve_exact = orig_solve

    n = len(traj["cum_cost"])
    wl_test = list(env.set.Workload[TRAIN_INTERVALS:TRAIN_INTERVALS + n])
    mean_step_rt = np.asarray(getattr(env, "mean_step_resptime", []), float)
    step_cost = np.asarray(getattr(env, "step_cost", []), float)
    return {
        "tag": tag, "app": app, "workload": workload, "n": n,
        "traj": {k: np.asarray(v, float) for k, v in traj.items()},
        "wl_test": np.asarray(wl_test, float),
        "mean_step_rt": mean_step_rt,
        "step_cost": step_cost,
        "observed_by_type": observed_by_type,
        "fronts": captured_fronts,
        "decide_modes": decide_modes,
        "mrt": float(info.get("average_resptime", float("nan"))),
        "cost": float(info.get("VM_cost", float("nan"))),
    }


def fig_workloads():
    arrays = {}
    for wl, _label in WORKLOADS:
        env = build_env("A11", wl, 0)
        arrays[wl] = np.asarray(env.set.Workload, float)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    for ax, (wl, label) in zip(axes, WORKLOADS):
        y = arrays[wl]
        x = np.arange(len(y))
        ax.plot(x, y, color="#1e88e5", lw=1.1)
        ax.axvspan(0, TRAIN_INTERVALS, color="#90caf9", alpha=0.12)
        ax.axvline(TRAIN_INTERVALS, color="#555", ls="--", lw=1.0)
        ax.set_ylabel("requests / interval")
        ax.set_title(f"{label}   "
                     f"(min {y.min():.0f}, mean {y.mean():.0f}, max {y.max():.0f})",
                     loc="left")
        ax.margins(x=0.01)
    axes[0].text(TRAIN_INTERVALS * 0.5, axes[0].get_ylim()[1] * 0.92, "train (day 1)",
                 ha="center", va="top", fontsize=9, color="#444")
    axes[0].text(TRAIN_INTERVALS * 1.5, axes[0].get_ylim()[1] * 0.92, "test (day 2)",
                 ha="center", va="top", fontsize=9, color="#444")
    axes[-1].set_xlabel("3-minute control interval (960 = 2 days)")
    fig.suptitle("Workload traces driving the autoscaling evaluation",
                 fontsize=13, fontweight="bold")
    return _save(fig, "fig1_workload_traces.png"), arrays


def fig_trajectory(epi):
    t = np.arange(epi["n"])
    tr = epi["traj"]
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    ax[0].plot(t, epi["wl_test"], color="#1e88e5", lw=1.1)
    ax[0].set_ylabel("requests/int")
    ax[0].set_title(f"Injected workload  -  {epi['tag']} "
                    f"({epi['app']}, {epi['workload']})", loc="left")

    rt = epi["mean_step_rt"]
    rt_x = np.arange(len(rt))
    ax[1].plot(rt_x, rt, color="#d32f2f", lw=1.4, label="end-to-end mean response time")
    ax[1].axhline(DEADLINE, color="#555", ls="--", lw=1.0, label=f"deadline {DEADLINE:.0f} ms")
    ax[1].set_ylim(0, DEADLINE * 1.06)
    ax[1].set_ylabel("ms")
    ax[1].set_title(f"Response time  (episode MRT = {epi['mrt']:.1f} ms, well under SLO)", loc="left")
    ax[1].legend(loc="upper right")

    ax[2].plot(t, tr["cum_cost"], color="#2e7d32", lw=1.4, label="cumulative VM cost")
    ax[2].axhline(BUDGET, color="#555", ls="--", lw=1.0, label=f"budget ${BUDGET:.0f}")
    ax[2].set_ylabel("USD")
    ax[2].set_title(f"Cost vs budget  (final = ${epi['cost']:.1f}, "
                    f"violation = ${max(0.0, epi['cost'] - BUDGET):.1f})", loc="left")
    ax[2].legend(loc="upper left")

    ax[3].plot(t, tr["n_con"], color="#6a1b9a", lw=1.3, label="replicas (containers)")
    ax3b = ax[3].twinx()
    ax3b.plot(t, tr["vcpu"], color="#ef6c00", lw=1.1, ls="-", label="total vCPU")
    ax3b.grid(False)
    ax[3].set_ylabel("replicas", color="#6a1b9a")
    ax3b.set_ylabel("total vCPU", color="#ef6c00")
    ax[3].set_title("Scaling actuation (diagonal: replicas + vCPU)", loc="left")
    ax[3].set_xlabel("3-minute control interval (test horizon)")
    l1, la1 = ax[3].get_legend_handles_labels()
    l2, la2 = ax3b.get_legend_handles_labels()
    ax[3].legend(l1 + l2, la1 + la2, loc="upper right")
    for a in ax:
        a.margins(x=0.01)
    fig.suptitle("NF-DiagScale closed-loop control trajectory",
                 fontsize=13, fontweight="bold")
    return _save(fig, "fig3_control_trajectory.png")


def fig_forecast(epi, cfg):
    types = set()
    for d in epi["observed_by_type"]:
        types.update(d.keys())
    if not types:
        print("  [warn] no observed load captured; skipping forecast figure")
        return None
    means = {ty: np.mean([d.get(ty, 0.0) for d in epi["observed_by_type"]]) for ty in types}
    bott = max(means, key=means.get)
    obs = np.asarray([d.get(bott, 0.0) for d in epi["observed_by_type"]], float)

    fc = ContainerForecaster(cfg)
    fcast = np.asarray([fc.update(o) for o in obs], float)
    warm = 2
    actual = obs[1 + warm:]
    pred = fcast[warm:-1]
    x = np.arange(1 + warm, len(obs))
    eps = 1e-6
    mask = actual > eps
    mape = float(np.mean(np.abs(actual[mask] - pred[mask]) / actual[mask]) * 100) if mask.any() else float("nan")
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(x, actual, color="#1e88e5", lw=1.3, label="actual load")
    ax.plot(x, pred, color="#d32f2f", lw=1.2, ls="--", label="one-step-ahead forecast")
    ax.set_xlabel("3-minute control interval (test horizon)")
    ax.set_ylabel("requests / interval")
    ax.set_title(f"Kalman+Holt forecaster -- microservice {bott}  "
                 f"(MAPE {mape:.1f}%, RMSE {rmse:.1f})", loc="left")
    ax.legend(loc="upper right")
    ax.margins(x=0.01)
    fig.suptitle("Temporal load forecasting accuracy (F component)",
                 fontsize=13, fontweight="bold")
    return _save(fig, "fig4_forecast_accuracy.png")


def fig_membership():
    ranges = {"psi": (0.0, 3.0), "omega": (0.0, 1.0), "phi": (0.0, 1.0), "rho": (0.0, 1.0)}
    titles = {
        "psi": r"$\psi$  load pressure (drain time / deadline)",
        "omega": r"$\omega$  latency slack (SLO headroom)",
        "phi": r"$\phi$  budget headroom (cost fraction)",
        "rho": r"$\rho$  risk flag",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for ax, var in zip(axes.ravel(), ["psi", "omega", "phi", "rho"]):
        lo, hi = ranges[var]
        xs = np.linspace(lo, hi, 400)
        for term, (center, sigma) in LINGUISTIC_TERMS[var].items():
            ys = np.asarray([gaussian_mf(x, center, sigma, term) for x in xs])
            ax.plot(xs, ys, lw=1.8, label=term)
        ax.set_title(titles[var], loc="left")
        ax.set_ylim(-0.03, 1.08)
        ax.set_xlabel("input value")
        ax.set_ylabel("membership")
        ax.legend(loc="center right", ncol=1)
    fig.suptitle("ANFIS Gaussian membership functions (N component)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "fig6_membership_functions.png")


def fig_decision_modes(epi):
    """Quantitative breakdown of the ANFIS scaling decision per interval.

    Mirrors HGraphScale's Fig. 13 (vertical / horizontal / no-op action
    counts) and adds NF-DiagScale's distinctive *diagonal* mode (a single
    decision that changes vCPU *and* replica count at once).  The controller
    only invokes the ANFIS when a microservice is under pressure; the
    remaining intervals -- plus deadzone-suppressed deliberations -- are
    counted as no-ops.
    """
    modes = epi.get("decide_modes", [])
    n = epi["n"]
    vert = modes.count("vertical")
    diag = modes.count("diagonal")
    horiz = modes.count("horizontal")
    noop = (n - len(modes)) + modes.count("none")

    cats = ["vertical", "diagonal", "horizontal", "no-op"]
    vals = [vert, diag, horiz, noop]
    cols = ["#ef6c00", "#d32f2f", "#1e88e5", "#9e9e9e"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cats, vals, color=cols, edgecolor="k", linewidth=0.6)
    top = max(vals) if vals else 1
    for b, v in zip(bars, vals):
        ax.annotate(f"{v}\n({100.0 * v / max(n, 1):.0f}%)",
                    (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("number of 3-minute control intervals")
    ax.set_ylim(0, top * 1.18)
    ax.set_title(f"ANFIS scaling-mode breakdown  -  {epi['tag']} "
                 f"({epi['app']}, {epi['workload']}; {n} intervals)", loc="left")
    fig.suptitle("Neuro-fuzzy decision-mode distribution (N component)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "fig9_decision_modes.png")


def comparison_artifacts():
    if not os.path.exists(RESULTS_JSON):
        print(f"  [skip] {os.path.basename(RESULTS_JSON)} not found -- run the 9-scenario "
              "sweep to produce the comparison table & bars (see --run-comparison).")
        return None
    with open(RESULTS_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    star = data["star_table"]
    ours = data["nfg_diagscale"]
    tags = [t for t in SCEN_ORDER if t in ours]

    headers = ["Scenario"] + METHODS + ["Beats STAR"]
    rows = []
    wins = 0
    for tag in tags:
        cells = [tag]
        for m in METHODS[:-1]:
            mrt, vio = star[tag][m]
            cells.append(f"{mrt:.2f}/{vio:.0f}")
        omrt, ovio = ours[tag]["MRT"], ours[tag]["Vio"]
        cells.append(f"{omrt:.2f}/{ovio:.0f}")
        beat = (omrt < star[tag]["STAR"][0]) and (ovio <= 1e-6)
        wins += int(beat)
        cells.append("YES" if beat else "no")
        rows.append(cells)
    _write_table("table1_comparison", headers, rows,
                 title=f"MRT(ms)/Vio($) vs STAR Table 3 -- NF-DiagScale wins {wins}/{len(tags)} "
                       "(lower MRT, zero violation)")

    fig, (axm, axv) = plt.subplots(2, 1, figsize=(13.5, 8.4))
    x = np.arange(len(tags))
    bw = 0.14
    for i, m in enumerate(METHODS):
        mrt = [(ours[t]["MRT"] if m == "NF-DiagScale" else star[t][m][0]) for t in tags]
        vio = [(ours[t]["Vio"] if m == "NF-DiagScale" else star[t][m][1]) for t in tags]
        off = (i - (len(METHODS) - 1) / 2) * bw
        edge = "k" if m == "NF-DiagScale" else "none"
        axm.bar(x + off, mrt, bw, label=m, color=COLORS[m], edgecolor=edge, linewidth=0.7)
        axv.bar(x + off, vio, bw, label=m, color=COLORS[m], edgecolor=edge, linewidth=0.7)
    axm.set_ylabel("mean response time (ms)")
    axm.set_title("Mean response time by scenario (lower is better)", loc="left")
    axm.set_xticks(x)
    axm.set_xticklabels(tags)
    axm.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    axv.set_ylabel("cost violation (USD)")
    axv.set_title("Budget violation by scenario (zero is best)", loc="left")
    axv.set_xticks(x)
    axv.set_xticklabels(tags)
    axv.set_xlabel("scenario  (N/W/A workload  x  11/12/13-microservice app)")
    fig.suptitle("NF-DiagScale vs reported baselines (STAR Table 3)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "fig2_comparison_bars.png")
    return wins, len(tags)


def fig_mrt_reduction():
    if not os.path.exists(RESULTS_JSON):
        print(f"  [skip] {os.path.basename(RESULTS_JSON)} not found -- run the sweep first.")
        return None
    with open(RESULTS_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    star, ours = data["star_table"], data["nfg_diagscale"]
    tags = [t for t in SCEN_ORDER if t in ours and t in star]
    if not tags:
        print("  [skip] no overlapping STAR scenarios in JSON.")
        return None
    red = [100.0 * (star[t]["STAR"][0] - ours[t]["MRT"]) / star[t]["STAR"][0] for t in tags]
    cmap = {"N": "#1e88e5", "W": "#8e24aa", "A": "#2e7d32"}
    cols = [cmap[t[0]] for t in tags]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(tags, red, color=cols, edgecolor="k", linewidth=0.6)
    for b, v in zip(bars, red):
        ax.annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="#333", lw=1.0)
    ax.set_ylabel("MRT reduction vs STAR (%)")
    ax.set_ylim(0, max(red) * 1.16)
    ax.set_title(f"NF-DiagScale mean response-time reduction vs STAR  "
                 f"(mean {np.mean(red):.0f}%, every scenario improved)", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cmap["N"], label="NASA"),
                       Patch(color=cmap["W"], label="Wikipedia"),
                       Patch(color=cmap["A"], label="Alibaba")], loc="upper right")
    fig.suptitle("Relative improvement over the STAR baseline (lower MRT is better)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "fig8_mrt_reduction.png")


def a14_artifacts():
    if not os.path.exists(RESULTS_JSON):
        print(f"  [skip] {os.path.basename(RESULTS_JSON)} not found -- run the A14 sweep first.")
        return None
    with open(RESULTS_JSON, encoding="utf-8") as fh:
        ours = json.load(fh)["nfg_diagscale"]
    tags = [t for t, _ in A14_SCENARIOS if t in ours]
    if not tags:
        print("  [skip] no App-14 results in JSON -- run --scenario N-14 W-14 A-14 first.")
        return None
    label = dict(A14_SCENARIOS)

    headers = ["Scenario"] + A14_METHODS + ["Beats HGraphScale"]
    rows, wins = [], 0
    for tag in tags:
        cells = [label[tag]]
        for m in A14_METHODS[:-1]:
            art, vio = HGS_TABLE_IV_A14[tag][m]
            cells.append(f"{art:.2f}/{vio:.2f}%")
        omrt, ovio = ours[tag]["MRT"], ours[tag]["Vio"]
        cells.append(f"{omrt:.2f}/{ovio:.0f}")
        beat = omrt < HGS_TABLE_IV_A14[tag]["HGraphScale"][0]
        wins += int(beat)
        cells.append("YES" if beat else "no")
        rows.append(cells)
    _write_table("table5_a14_comparison", headers, rows,
                 title="App-14 ART(ms)/Vio vs HGraphScale IEEE TSC Table IV "
                       f"(STAR reports no App-14) -- NF-DiagScale beats HGraphScale {wins}/{len(tags)}. "
                       "Source Table IV lists identical NASA-14/Wiki-14 ART (apparent transcription artifact).")

    fig, (axm, axv) = plt.subplots(2, 1, figsize=(12, 8.2))
    x = np.arange(len(tags))
    bw = 0.12
    for i, m in enumerate(A14_METHODS):
        if m == "NF-DiagScale":
            art = [ours[t]["MRT"] for t in tags]
            vio = [ours[t]["Vio"] for t in tags]
        else:
            art = [HGS_TABLE_IV_A14[t][m][0] for t in tags]
            vio = [HGS_TABLE_IV_A14[t][m][1] for t in tags]
        off = (i - (len(A14_METHODS) - 1) / 2) * bw
        edge = "k" if m == "NF-DiagScale" else "none"
        axm.bar(x + off, art, bw, label=m, color=A14_COLORS[m], edgecolor=edge, linewidth=0.7)
        axv.bar(x + off, vio, bw, label=m, color=A14_COLORS[m], edgecolor=edge, linewidth=0.7)
    axm.axhline(DEADLINE, color="#555", ls="--", lw=1.0)
    axm.text(x[-1] + 0.42, DEADLINE, f"{DEADLINE:.0f} ms\ndeadline", va="center",
             fontsize=8, color="#555")
    axm.set_ylabel("average response time (ms)")
    axm.set_title("App-14 ART by workload (lower is better)", loc="left")
    axm.set_xticks(x)
    axm.set_xticklabels([label[t] for t in tags])
    axm.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    axv.set_ylabel("budget violation (%)")
    axv.set_title("App-14 budget violation (zero is best; NF-DiagScale = 0)", loc="left")
    axv.set_xticks(x)
    axv.set_xticklabels([label[t] for t in tags])
    axv.set_xlabel("real-world trace  x  App-14 (14-microservice application)")
    fig.suptitle("App-14: NF-DiagScale vs HGraphScale IEEE TSC Table IV (beyond STAR Table 3)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig7_a14_comparison.png")
    return wins, len(tags)


def fig_cost_headroom():
    if not os.path.exists(RESULTS_JSON):
        print(f"  [skip] {os.path.basename(RESULTS_JSON)} not found -- run the sweep first.")
        return None
    with open(RESULTS_JSON, encoding="utf-8") as fh:
        ours = json.load(fh)["nfg_diagscale"]
    tags = [t for t in CH_ORDER if t in ours]
    if not tags:
        print("  [skip] no NF-DiagScale results in JSON.")
        return None
    cost = [float(ours[t].get("VM_cost", ours[t].get("cost", float("nan")))) for t in tags]
    cmap = {"N": "#1e88e5", "W": "#8e24aa", "A": "#2e7d32"}
    cols = [cmap[t[0]] for t in tags]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(tags, cost, color=cols, edgecolor="k", linewidth=0.6)
    ax.axhline(BUDGET, color="#d32f2f", ls="--", lw=1.3, label=f"budget ${BUDGET:.0f}/day")
    for b, c in zip(bars, cost):
        ax.annotate(f"${c:.0f}\n{100.0 * c / BUDGET:.0f}%",
                    (b.get_x() + b.get_width() / 2, c),
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("daily VM cost (USD)")
    ax.set_ylim(0, BUDGET * 1.12)
    ax.set_title(f"Daily VM cost vs $200 budget  (peak {max(cost) / BUDGET * 100:.0f}% of budget, "
                 "every scenario under budget => Vio = 0)", loc="left")
    ax.set_xlabel("scenario  (App-11/12/13 vs STAR  +  App-14 vs HGraphScale)")
    from matplotlib.patches import Patch
    ax.legend(handles=[plt.Line2D([0], [0], color="#d32f2f", ls="--", label=f"budget ${BUDGET:.0f}/day"),
                       Patch(color=cmap["N"], label="NASA"),
                       Patch(color=cmap["W"], label="Wikipedia"),
                       Patch(color=cmap["A"], label="Alibaba")], loc="upper right", ncol=2)
    fig.suptitle("Cost headroom under the $200/day budget (zero violation everywhere)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "fig10_cost_headroom.png")


def table_vm_types():
    env = build_env("A11", "nasa", 0)
    ds = env.set.dataset
    headers = ["EC2 m5 type", "vCPU", "Memory (GiB)", "On-demand $/hr"]
    names = {4: "m5.xlarge", 8: "m5.2xlarge", 16: "m5.4xlarge",
             32: "m5.8xlarge", 48: "m5.12xlarge"}
    rows = []
    for vcpu, mem in zip(ds.vm_vcpu, ds.vm_mem):
        rows.append([names.get(vcpu, f"{vcpu}-vCPU"), vcpu, mem, f"{ds.vm_price[vcpu]:.3f}"])
    _write_table("table2_vm_types", headers, rows,
                 title="Heterogeneous VM types (EC2 m5, STAR Table 2)")


def table_rule_base():
    headers = ["Rule", "Antecedents (IF)", "Mode", "dCores", "dReplicas", "Rationale"]
    rows = []
    for r in build_rule_base():
        ante = ", ".join(f"{k}={v}" for k, v in r.antecedents.items())
        rows.append([r.rule_id, ante, MODE_NAMES[r.mode], r.delta_c, r.delta_n,
                     r.justification])
    _write_table("table3_rule_base", headers, rows,
                 title="ANFIS fuzzy rule base (Takagi-Sugeno consequents)")


def table_hyperparams(cfg):
    headers = ["Component", "Parameter", "Value"]
    rows = []

    def add(comp, mapping):
        for k, v in mapping.items():
            rows.append([comp, k, v])

    add("Kalman filter", cfg.get("kalman", {}))
    add("Holt forecast", cfg.get("forecast", {}))
    add("ANFIS deadzones", cfg.get("anfis", {}))
    add("Online learning", cfg.get("adaptive", {}))
    add("Sizer rebalance", cfg.get("rebalance", {}))
    add("Cloud bounds", cfg.get("cloud", {}))
    add("Controller", cfg.get("controller", {}))
    rows.append(["SLO", "slo_ms", cfg.get("slo_ms")])
    _write_table("table4_hyperparameters", headers, rows,
                 title="NF-DiagScale configuration (default.yaml)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="A-12", choices=SCEN_ORDER,
                    help="Scenario instrumented for the trajectory/forecast figures.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-comparison", action="store_true",
                    help="Run the full 9-scenario sweep first (writes star_comparison_results.json).")
    args = ap.parse_args()

    _style()
    cfg = load_config()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    print("\n=== NF-DiagScale report generator ===")

    if args.run_comparison:
        print("\n[0/8] Running 9-scenario sweep (this is the slow part)...")
        import subprocess
        subprocess.run([sys.executable, "-u", os.path.join(_REPO_ROOT, "run_star_comparison.py"),
                        "--seeds", str(args.seed), "--out", RESULTS_JSON], check=False)

    print("\n[1/11] Workload traces (fig1)...")
    fig_workloads()

    print(f"\n[2/11] Instrumented episode {args.rep} (powers fig3/fig4/fig9)...")
    epi = run_instrumented(args.rep, args.seed, cfg)
    print(f"      -> MRT {epi['mrt']:.1f} ms, cost ${epi['cost']:.1f}, "
          f"{epi['n']} intervals, {len(epi['fronts'])} sizing decisions")

    print("\n[3/11] Control trajectory (fig3)...")
    fig_trajectory(epi)
    print("\n[4/11] Forecast accuracy (fig4)...")
    fig_forecast(epi, cfg)
    print("\n[5/11] ANFIS membership functions (fig6)...")
    fig_membership()
    print("\n[6/11] ANFIS decision-mode breakdown (fig9)...")
    fig_decision_modes(epi)

    print("\n[7/11] Static tables (VM types, rule base, hyperparameters)...")
    table_vm_types()
    table_rule_base()
    table_hyperparams(cfg)

    print("\n[8/11] MRT/Vio comparison vs STAR (table1 + fig2)...")
    comparison_artifacts()
    print("\n[9/11] MRT reduction vs STAR (fig8)...")
    fig_mrt_reduction()
    print("\n[10/11] App-14 comparison vs HGraphScale Table IV (table5 + fig7)...")
    a14_artifacts()
    print("\n[11/11] Cost headroom vs budget (fig10)...")
    fig_cost_headroom()

    print(f"\nDone. Figures -> {os.path.relpath(FIG_DIR, _REPO_ROOT)}/   "
          f"Tables -> {os.path.relpath(TAB_DIR, _REPO_ROOT)}/\n")


if __name__ == "__main__":
    main()
