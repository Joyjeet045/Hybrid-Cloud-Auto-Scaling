"""Deterministic magnitude sizer over a bottleneck microservice's allocation.

The deterministic corrective ("feedforward") component of NF-DiagScale. It sizes
the 2-variable space ``(replicas, vCPU-per-replica)`` for the selected bottleneck
microservice, returning the cost-feasible knee checkpoint ``(h*, c*)`` that
anchors the magnitude of the ANFIS decision (``ANFISEngine.decide(corrective=...)``)
plus the non-dominated (Pareto) trade-off front for reporting.

The allocation grid is small (``max_replicas x max_cores`` points), so
:meth:`solve_exact` enumerates it exhaustively and returns the *globally optimal*
STAR-Eq.8 checkpoint deterministically -- reproducible and independent of any RNG.
Objectives are grounded in :mod:`queue_model`:
  f1 = predicted batch response time (ms)          [M/D/1 drain, Kleinrock 1975]
  f2 = marginal VM cost (USD)                       [HGraphScale Eq. 5]
  f3 = rebalance penalty (prefer vertical changes)  [diagonal-scaling bias]
"""
from __future__ import annotations

import numpy as np

from nfg_diagscale.hgraph_policy import queue_model


class _Ind:
    __slots__ = ("h", "c", "obj", "rank", "dom_count", "dom_set")

    def __init__(self, h: int, c: int):
        self.h = int(h)
        self.c = int(c)
        self.obj = (0.0, 0.0, 0.0)
        self.rank = 0
        self.dom_count = 0
        self.dom_set = []


class MagnitudeSizer:
    def __init__(self, config):
        cloud = config.get("cloud", {})
        self.min_h = int(cloud.get("min_replicas", 1))
        self.max_h = int(cloud.get("max_replicas", 15))
        self.min_c = int(cloud.get("min_cores", 1))
        self.max_c = int(cloud.get("max_cores", 16))
        reb = config.get("rebalance", {})
        self._reb_weight = float(reb.get("penalty_multiplier", 1.5))

    def _evaluate(self, ind: _Ind, ctx) -> None:
        lat = queue_model.batch_response_time(ctx["lam"], ctx["et"], ind.c, ind.h)
        new_total_vcpu = ctx["other_vcpu"] + ind.h * ind.c
        cost = queue_model.marginal_vm_cost(
            ctx["base_total_vcpu"], new_total_vcpu,
            ctx["vm_size"], ctx["vm_price_per_hr"], ctx["hours_remaining"],
        )
        reb = self._reb_weight * abs(ind.h - ctx["cur_h"])
        ind.obj = (lat, cost, reb)

    @staticmethod
    def _dominates(a: _Ind, b: _Ind) -> bool:
        better_or_equal = all(x <= y for x, y in zip(a.obj, b.obj))
        strictly_better = any(x < y for x, y in zip(a.obj, b.obj))
        return better_or_equal and strictly_better

    def _non_dominated_sort(self, pop):
        fronts = [[]]
        for p in pop:
            p.dom_count = 0
            p.dom_set = []
            for q in pop:
                if p is q:
                    continue
                if self._dominates(p, q):
                    p.dom_set.append(q)
                elif self._dominates(q, p):
                    p.dom_count += 1
            if p.dom_count == 0:
                p.rank = 0
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in p.dom_set:
                    q.dom_count -= 1
                    if q.dom_count == 0:
                        q.rank = i + 1
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        return fronts[:-1]

    def _select_checkpoint(self, pareto, ctx, penalty, budget_room):
        """Pick the STAR-Eq.8-aligned knee: min (MRT + penalty * cost-violation)."""
        best = None
        best_score = float("inf")
        for p in pareto:
            lat, cost, _ = p.obj
            violation = max(0.0, cost - budget_room)
            score = lat + penalty * violation
            if score < best_score:
                best_score = score
                best = p
        if best is None:
            return ctx["cur_h"], 1
        return best.h, best.c

    def solve_exact(self, cur_h, lam, et, *, base_total_vcpu, other_vcpu,
                    vm_size, vm_price_per_hr, hours_remaining, deadline,
                    penalty=100.0, budget_room=float("inf")):
        """Deterministic, globally-optimal cost-feasible knee sizing.

        The allocation grid ``[min_h..max_h] x [min_c..max_c]`` is small enough
        to enumerate exhaustively, so we evaluate every configuration and return
        the exact STAR-Eq.8 checkpoint (the scalarised latency/cost objective)
        plus the non-dominated front for reporting. Being a full enumeration,
        the result is reproducible and independent of any RNG.
        """
        ctx = {
            "lam": float(lam), "et": float(et), "cur_h": int(cur_h),
            "base_total_vcpu": float(base_total_vcpu), "other_vcpu": float(other_vcpu),
            "vm_size": float(vm_size), "vm_price_per_hr": float(vm_price_per_hr),
            "hours_remaining": float(hours_remaining), "deadline": float(deadline),
        }
        grid = []
        for h in range(self.min_h, self.max_h + 1):
            for c in range(self.min_c, self.max_c + 1):
                ind = _Ind(h, c)
                self._evaluate(ind, ctx)
                grid.append(ind)
        h_star, c_star = self._select_checkpoint(grid, ctx, penalty, budget_room)
        fronts = self._non_dominated_sort(grid)
        pareto = fronts[0] if fronts else grid
        return h_star, c_star, [(p.h, p.c, p.obj) for p in pareto]
