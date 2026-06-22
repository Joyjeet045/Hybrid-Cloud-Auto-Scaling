"""Robust, validation-selected auto-tuning of a *context-scheduled* fuzzy partition.

This is the upgraded rec4 ("final approach"). It combines four changes over the
original static (1+lambda)-ES tuner, each targeting why the static version
overfit the calibration set and failed to generalise:

  (A) Robust objective -- minimise *relative regret vs the expert baseline*, with a
      worst-case (minimax) term, so a design that blows up any scenario is rejected
      even if it wins others. (Kills the static tuner's high-variance tail.)
  (B) Train/validation split -- the search is ranked on a TRAIN set but the returned
      design is selected by its score on a disjoint VALIDATION set (model selection /
      early stopping), so we keep the design that *generalises*, not the one that
      best fits the training scenarios.
  (C) Context-scheduled membership functions -- the genome encodes a static base
      partition (20 params) plus six per-variable load-schedule gains, so the
      partition self-adapts to the operating point instead of being a single static
      compromise. See :mod:`scheduled_anfis`.
  (D) Separable CMA-ES -- a covariance-adapting evolution strategy (Ros & Hansen,
      2008) replaces the fixed-sigma (1+lambda)-ES for better sample efficiency in
      the larger search space.

The expert design (base = expert terms, gains = 0) is always evaluated, so the
returned design is never worse than the expert on the validation set.

Run from the repo root:
    python -m ablations.rec4_mf_autotune.autotune --workers 12
    python -m ablations.rec4_mf_autotune.autotune --workers 12 --quick
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from nfg_diagscale.config import load_config
from nfg_diagscale.decision.fuzzy_rules import LINGUISTIC_TERMS
from nfg_diagscale.hgraph_env.simulator import HGraphScaleEnv
from run_star_comparison import BUDGET, DEADLINE, SCENARIOS
from ablations.rec4_mf_autotune.scheduled_controller import ScheduledMFController

HERE = os.path.dirname(os.path.abspath(__file__))
TERMS_PATH = os.path.join(HERE, "mf_terms.json")
INTERVALS = 480

VARS = ("psi", "omega", "phi")
DOMAIN = {"psi": (0.3, 3.0), "omega": (0.08, 1.0), "phi": (0.03, 1.0)}

# Schedule signal g = clip((psi - G_LO)/(G_HI - G_LO), 0, 1).
G_LO, G_HI = 0.4, 1.6
GAIN_MAX = 0.6

# (A) robust objective weights and (B) the train/validation split.
LAMBDA_TAIL = 1.0
VIO_PENALTY = 1.0e3
# Capacity control: L2 pull on the schedule gains toward 0 (expert), so the search
# only deviates from the expert partition where it robustly helps -- the key fix for
# the over-fitting seen with the full 26-parameter design.
REG_COEF = 0.05
TRAIN_TAGS = ("N-13", "W-13", "A-13")
VAL_TAGS = ("N-11", "W-12", "A-14")

# Expert baseline (main, seed 0) -- the regret reference.
BASELINE = {
    "N-11": 152.91, "N-12": 157.39, "N-13": 140.52,
    "W-11": 229.34, "W-12": 261.53, "W-13": 210.52,
    "A-11": 169.09, "A-12": 162.74, "A-13": 130.07,
    "N-14": 249.87, "W-14": 383.11, "A-14": 224.89,
}


# --------------------------------------------------------------------------- #
# Genome <-> design                                                            #
# --------------------------------------------------------------------------- #
def build_params():
    """Flat list of the static membership terms in canonical order (expert seeds)."""
    params = []
    for var in VARS:
        for term, (center, sigma) in LINGUISTIC_TERMS[var].items():
            params.append({"var": var, "term": term,
                           "c0": float(center), "s0": float(sigma)})
    return params


def _decode_static(theta_static, params):
    """Map the static block to a valid, monotone base-terms dict."""
    by_var = OrderedDict()
    for k, p in enumerate(params):
        by_var.setdefault(p["var"], []).append((k, p))
    terms = {}
    for var, items in by_var.items():
        lo, hi = DOMAIN[var]
        gap = (hi - lo) * 0.04
        n = len(items)
        prev = lo
        out = {}
        for j, (k, p) in enumerate(items):
            tc = float(np.clip(theta_static[2 * k], 0.5, 1.8))
            ts = float(np.clip(theta_static[2 * k + 1], 0.5, 2.0))
            center = p["c0"] * tc
            upper = hi - gap * (n - 1 - j)
            center = min(max(center, prev + gap), upper)
            center = float(max(center, lo))
            sigma = float(max(0.05, p["s0"] * ts))
            out[p["term"]] = [center, sigma]
            prev = center
        terms[var] = out
    return terms


def decode_design(theta, params):
    """Map a full genome to a context-scheduled design dict.

    Genome layout: [static multipliers (2 per term) | schedule gains (gc, gs per var)].
    Seed ``theta = [1]*static + [0]*gains`` reproduces the expert partition exactly.
    """
    n_static = 2 * len(params)
    base = _decode_static(theta[:n_static], params)
    raw = theta[n_static:]
    gains = {}
    for i, var in enumerate(VARS):
        gc = float(np.clip(raw[2 * i], -GAIN_MAX, GAIN_MAX))
        gs = float(np.clip(raw[2 * i + 1], -GAIN_MAX, GAIN_MAX))
        gains[var] = [gc, gs]
    return {"base_terms": base, "gains": gains, "g_lo": G_LO, "g_hi": G_HI}


def seed_genome(params):
    return np.concatenate([np.ones(2 * len(params)), np.zeros(2 * len(VARS))])


# --------------------------------------------------------------------------- #
# Parallel rollout evaluation + robust objective                              #
# --------------------------------------------------------------------------- #
def _eval_one(task):
    cand_id, tag, design, seed = task
    app, workload = SCENARIOS[tag]
    cfg = load_config()
    cfg["fuzzify"] = {"scheduled_mf": True, "design": design}
    env = HGraphScaleEnv(app=app, workload=workload, seed=seed, budget=BUDGET)
    state = env.reset(test=True)
    ctrl = ScheduledMFController(cfg, deadline=DEADLINE, total_intervals=INTERVALS)
    ctrl.reset(budget_T=BUDGET, total_intervals=INTERVALS)
    done = False
    info = {}
    with contextlib.redirect_stdout(io.StringIO()):
        while not done:
            action = ctrl.act(state)
            state, _r, done, info = env.step(action)
    mrt = float(info.get("average_resptime", float("nan")))
    cost = float(info.get("VM_cost", 0.0))
    return cand_id, tag, mrt, max(0.0, cost - BUDGET)


def evaluate_population(designs, tags, seed, workers):
    """Roll out every design on every tag; return per-design {tag: (mrt, vio)}."""
    tasks = []
    for ci, design in enumerate(designs):
        for tag in tags:
            tasks.append((ci, tag, design, seed))
    out = {ci: {} for ci in range(len(designs))}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for ci, tag, mrt, vio in ex.map(_eval_one, tasks):
            out[ci][tag] = (mrt, vio)
    return out


def robust_score(per_tag, tags):
    """(A) mean + worst-case relative regret vs expert, plus a hard vio penalty."""
    regrets = []
    vio_total = 0.0
    for t in tags:
        mrt, vio = per_tag[t]
        regrets.append((mrt - BASELINE[t]) / BASELINE[t])
        vio_total += vio
    regrets = np.asarray(regrets)
    mean_reg = float(np.mean(regrets))
    worst_reg = float(np.max(regrets))
    score = mean_reg + LAMBDA_TAIL * worst_reg + VIO_PENALTY * vio_total
    return score, mean_reg, worst_reg, vio_total


# --------------------------------------------------------------------------- #
# (D) Separable CMA-ES                                                         #
# --------------------------------------------------------------------------- #
class SepCMAES:
    """Minimal separable (diagonal-covariance) CMA-ES (Ros & Hansen, 2008)."""

    def __init__(self, x0, sigma0, popsize, seed):
        self.dim = len(x0)
        self.mean = np.asarray(x0, dtype=float)
        self.sigma = float(sigma0)
        self.lam = int(popsize)
        self.rng = np.random.default_rng(seed)

        mu = self.lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        self.weights = w
        self.mu = mu
        self.mu_eff = 1.0 / np.sum(w ** 2)

        d = self.dim
        self.cc = (4 + self.mu_eff / d) / (d + 4 + 2 * self.mu_eff / d)
        self.cs = (self.mu_eff + 2) / (d + self.mu_eff + 5)
        # separable acceleration factor on the rank-1 / rank-mu learning rates
        accel = (d + 2) / 3.0
        self.c1 = accel * 2.0 / ((d + 1.3) ** 2 + self.mu_eff)
        self.cmu = accel * min(
            1 - self.c1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((d + 2) ** 2 + self.mu_eff),
        )
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mu_eff - 1) / (d + 1)) - 1) + self.cs
        self.chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        self.pc = np.zeros(d)
        self.ps = np.zeros(d)
        self.C = np.ones(d)          # diagonal covariance
        self.gen = 0

    def ask(self):
        self.gen += 1
        std = np.sqrt(self.C)
        self.z = self.rng.standard_normal((self.lam, self.dim))
        self.y = self.z * std                                  # ~ N(0, C)
        return self.mean + self.sigma * self.y

    def tell(self, fitness):
        idx = np.argsort(fitness)
        y_sel = self.y[idx[: self.mu]]
        z_sel = self.z[idx[: self.mu]]
        y_w = self.weights @ y_sel
        z_w = self.weights @ z_sel

        self.ps = (1 - self.cs) * self.ps + \
            np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * z_w
        ps_norm = float(np.linalg.norm(self.ps))
        hs = 1.0 if ps_norm / np.sqrt(
            1 - (1 - self.cs) ** (2 * self.gen)) < (1.4 + 2 / (self.dim + 1)) * self.chiN else 0.0
        self.pc = (1 - self.cc) * self.pc + \
            hs * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y_w

        delta_hs = (1 - hs) * self.cc * (2 - self.cc)
        rank_mu = self.weights @ (y_sel ** 2)
        self.C = (1 - self.c1 - self.cmu) * self.C \
            + self.c1 * (self.pc ** 2 + delta_hs * self.C) \
            + self.cmu * rank_mu
        self.C = np.maximum(self.C, 1e-10)

        self.sigma *= np.exp((self.cs / self.damps) * (ps_norm / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-4, 1.0))
        self.mean = self.mean + self.weights @ (self.sigma * y_sel)


# --------------------------------------------------------------------------- #
# Search driver                                                               #
# --------------------------------------------------------------------------- #
def _gain_l2(design):
    return float(np.mean([g ** 2 for v in VARS for g in design["gains"][v]]))


def autotune(workers, seed, gens, popsize, sigma0, train_tags, val_tags,
             schedule_only=True):
    params = build_params()
    all_tags = list(train_tags) + list(val_tags)

    if schedule_only:
        # Genome = the 6 schedule gains only; base partition frozen at expert.
        n_static = 2 * len(params)
        x0 = np.zeros(2 * len(VARS))

        def to_design(theta):
            return decode_design(np.concatenate([np.ones(n_static), theta]), params)
    else:
        x0 = seed_genome(params)

        def to_design(theta):
            return decode_design(theta, params)

    # Expert is always in the running -- guarantees no-harm on validation.
    expert_design = to_design(x0)
    res = evaluate_population([expert_design], all_tags, seed, workers)[0]
    _, e_tr_mean, e_tr_worst, _ = robust_score(res, train_tags)
    e_val_score, e_val_mean, e_val_worst, e_val_vio = robust_score(res, val_tags)
    print(f"[gen 0] expert: train mean_reg={e_tr_mean:+.4f} worst={e_tr_worst:+.4f} "
          f"| val score={e_val_score:+.4f} mean={e_val_mean:+.4f} worst={e_val_worst:+.4f}")

    best = {"design": expert_design, "val_score": e_val_score,
            "val_mean": e_val_mean, "val_worst": e_val_worst, "val_vio": e_val_vio,
            "per_tag": res}

    es = SepCMAES(x0, sigma0, popsize, seed)
    for g in range(1, gens + 1):
        thetas = es.ask()
        designs = [to_design(th) for th in thetas]
        results = evaluate_population(designs, all_tags, seed, workers)

        train_fit = np.empty(len(designs))
        for ci in range(len(designs)):
            sc, _, _, _ = robust_score(results[ci], train_tags)
            train_fit[ci] = sc + REG_COEF * _gain_l2(designs[ci])

        # (B) select the incumbent by validation score (feasible designs first).
        for ci in range(len(designs)):
            v_score, v_mean, v_worst, v_vio = robust_score(results[ci], val_tags)
            better = (v_vio == 0 and best["val_vio"] > 0) or \
                     (v_vio == best["val_vio"] and v_score < best["val_score"] - 1e-9)
            if better:
                best = {"design": designs[ci], "val_score": v_score,
                        "val_mean": v_mean, "val_worst": v_worst, "val_vio": v_vio,
                        "per_tag": results[ci]}

        es.tell(train_fit)
        bi = int(np.argmin(train_fit))
        _, b_mean, b_worst, _ = robust_score(results[bi], train_tags)
        print(f"[gen {g}] train best mean_reg={b_mean:+.4f} worst={b_worst:+.4f} "
              f"sigma={es.sigma:.3f} | incumbent val score={best['val_score']:+.4f} "
              f"mean={best['val_mean']:+.4f} worst={best['val_worst']:+.4f}")

    return best, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--sigma0", type=float, default=0.25)
    ap.add_argument("--full", action="store_true",
                    help="optimise the full 26-param design (base + gains); "
                         "default is schedule-only (base frozen at expert, 6 gains)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke run (2 gens, popsize 4, 1 train/1 val tag)")
    args = ap.parse_args()

    train_tags, val_tags = TRAIN_TAGS, VAL_TAGS
    gens, popsize = args.gens, args.popsize
    if args.quick:
        gens, popsize = 2, 4
        train_tags, val_tags = ("A-13",), ("A-14",)

    schedule_only = not args.full
    mode = "schedule-only" if schedule_only else "full"
    print(f"Scheduled-MF auto-tune [{mode}]  train={list(train_tags)} "
          f"val={list(val_tags)} gens={gens} popsize={popsize} workers={args.workers}")
    best, _ = autotune(args.workers, args.seed, gens, popsize, args.sigma0,
                       train_tags, val_tags, schedule_only=schedule_only)

    design = best["design"]
    out = {
        "schema": "scheduled-v2",
        "mode": mode,
        "train_tags": list(train_tags),
        "val_tags": list(val_tags),
        "seed": args.seed,
        "val_mean_regret": best["val_mean"],
        "val_worst_regret": best["val_worst"],
        "val_total_vio": best["val_vio"],
        "g_lo": G_LO,
        "g_hi": G_HI,
        "base_terms": design["base_terms"],
        "gains": design["gains"],
    }
    with open(TERMS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved scheduled design -> {TERMS_PATH}")
    print(f"validation mean regret {best['val_mean']:+.4f}  "
          f"worst {best['val_worst']:+.4f}  vio {best['val_vio']:.3f}")
    print("gains: " + "  ".join(f"{v}=(gc {design['gains'][v][0]:+.3f}, "
                                f"gs {design['gains'][v][1]:+.3f})" for v in VARS))


if __name__ == "__main__":
    main()
