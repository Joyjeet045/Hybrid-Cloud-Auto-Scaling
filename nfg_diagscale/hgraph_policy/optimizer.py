"""NSGA-II magnitude optimizer over a bottleneck microservice's allocation.

This is the "Genetic" component of NFG-DiagScale. It searches the 2-gene space
``(replicas, vCPU-per-replica)`` for the selected bottleneck microservice and
returns the non-dominated (Pareto) trade-off between predicted response time and
incremental cost, plus a single "knee" checkpoint ``(h*, c*)`` that biases the
ANFIS decision (matching ``ANFISEngine.decide(ga_checkpoint=...)``).

Algorithm: NSGA-II (Deb et al., 2002) — fast non-dominated sorting + crowding-
distance selection. Objectives are grounded in :mod:`queue_model`:
  f1 = predicted batch response time (ms)          [M/D/1 drain, Kleinrock 1975]
  f2 = marginal VM cost (USD)                       [HGraphScale Eq. 5]
  f3 = rebalance penalty (prefer vertical changes)  [diagonal-scaling bias]
"""
from __future__ import annotations

import numpy as np

from nfg_diagscale.hgraph_policy import queue_model


class _Ind:
    __slots__ = ("h", "c", "obj", "rank", "crowd", "dom_count", "dom_set")

    def __init__(self, h: int, c: int):
        self.h = int(h)
        self.c = int(c)
        self.obj = (0.0, 0.0, 0.0)
        self.rank = 0
        self.crowd = 0.0
        self.dom_count = 0
        self.dom_set = []


class MagnitudeNSGA2:
    def __init__(self, config):
        ncfg = config.get("nsga2", {})
        self.pop_size = int(ncfg.get("population_size", 20))
        self.n_gen = int(ncfg.get("generations", 15))
        self.p_cross = float(ncfg.get("crossover_prob", 0.9))
        self.p_mut = float(ncfg.get("mutation_prob", 0.1))
        cloud = config.get("cloud", {})
        self.min_h = int(cloud.get("min_replicas", 1))
        self.max_h = int(cloud.get("max_replicas", 15))
        self.min_c = int(cloud.get("min_cores", 1))
        self.max_c = int(cloud.get("max_cores", 16))
        reb = config.get("rebalance", {})
        self._reb_weight = float(reb.get("penalty_multiplier", 1.5))

    # -- objective evaluation ------------------------------------------------ #
    def _evaluate(self, ind: _Ind, ctx) -> None:
        lat = queue_model.batch_response_time(ctx["lam"], ctx["et"], ind.c, ind.h)
        new_total_vcpu = ctx["other_vcpu"] + ind.h * ind.c
        cost = queue_model.marginal_vm_cost(
            ctx["base_total_vcpu"], new_total_vcpu,
            ctx["vm_size"], ctx["vm_price_per_hr"], ctx["hours_remaining"],
        )
        # Rebalance penalty: horizontal (replica) changes are costlier to enact
        # than vertical (vCPU) changes, so bias toward vertical-first diagonal
        # scaling (NFG-DiagScale design).
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

    @staticmethod
    def _crowding(front):
        n = len(front)
        for p in front:
            p.crowd = 0.0
        if n <= 2:
            for p in front:
                p.crowd = float("inf")
            return
        for m in range(3):
            front.sort(key=lambda ind: ind.obj[m])
            front[0].crowd = front[-1].crowd = float("inf")
            lo, hi = front[0].obj[m], front[-1].obj[m]
            span = (hi - lo) or 1.0
            for k in range(1, n - 1):
                front[k].crowd += (front[k + 1].obj[m] - front[k - 1].obj[m]) / span

    def _random(self) -> _Ind:
        return _Ind(np.random.randint(self.min_h, self.max_h + 1),
                    np.random.randint(self.min_c, self.max_c + 1))

    def _crossover(self, a: _Ind, b: _Ind):
        if np.random.random() < self.p_cross:
            return _Ind(a.h, b.c), _Ind(b.h, a.c)
        return _Ind(a.h, a.c), _Ind(b.h, b.c)

    def _mutate(self, ind: _Ind):
        if np.random.random() < self.p_mut:
            ind.h = int(np.clip(ind.h + np.random.choice([-1, 1]), self.min_h, self.max_h))
        if np.random.random() < self.p_mut:
            ind.c = int(np.clip(ind.c + np.random.choice([-1, 1]), self.min_c, self.max_c))

    def optimize(self, cur_h, cur_c, lam, et, *, base_total_vcpu, other_vcpu,
                 vm_size, vm_price_per_hr, hours_remaining, deadline,
                 penalty=100.0, budget_room=float("inf")):
        """Return ``(h_star, c_star, front)`` for the bottleneck microservice.

        ``cur_h``/``cur_c`` are current replicas / vCPU-per-replica; ``lam`` the
        forecast tasks-per-interval; ``et`` the microservice base processing time.
        ``other_vcpu`` is the vCPU used by *other* microservices (held fixed) so
        the cost objective sees the true marginal VM footprint.
        """
        ctx = {
            "lam": float(lam), "et": float(et), "cur_h": int(cur_h),
            "base_total_vcpu": float(base_total_vcpu), "other_vcpu": float(other_vcpu),
            "vm_size": float(vm_size), "vm_price_per_hr": float(vm_price_per_hr),
            "hours_remaining": float(hours_remaining), "deadline": float(deadline),
        }

        pop = [self._random() for _ in range(self.pop_size)]
        # Always seed the current configuration so "do nothing" is considered.
        pop[0] = _Ind(int(np.clip(cur_h, self.min_h, self.max_h)),
                      int(np.clip(cur_c, self.min_c, self.max_c)))
        for ind in pop:
            self._evaluate(ind, ctx)

        for _ in range(self.n_gen):
            children = []
            while len(children) < self.pop_size:
                a, b = np.random.choice(pop, 2, replace=False)
                c1, c2 = self._crossover(a, b)
                self._mutate(c1)
                self._mutate(c2)
                self._evaluate(c1, ctx)
                self._evaluate(c2, ctx)
                children.extend([c1, c2])
            combined = pop + children
            fronts = self._non_dominated_sort(combined)
            new_pop = []
            for front in fronts:
                self._crowding(front)
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    front.sort(key=lambda ind: ind.crowd, reverse=True)
                    new_pop.extend(front[: self.pop_size - len(new_pop)])
                    break
            pop = new_pop

        fronts = self._non_dominated_sort(pop)
        pareto = fronts[0] if fronts else pop
        h_star, c_star = self._select_checkpoint(pareto, ctx, penalty, budget_room)
        return h_star, c_star, [(p.h, p.c, p.obj) for p in pareto]

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
