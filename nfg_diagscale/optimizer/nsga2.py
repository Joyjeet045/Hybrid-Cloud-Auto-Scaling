"""
NSGA-II Multi-Objective Optimizer on the Diagonal Scaling Plane.

Replaces P3's local-search DiagonalScale with a Pareto-optimal trajectory
optimizer, as suggested in P3's future work (sect VIII-C).

[Deb02] Deb K. et al. (2002), "A Fast and Elitist Multiobjective Genetic
  Algorithm: NSGA-II", IEEE Trans. Evol. Comput. 6(2):182-197.

  Core operations:
    - Non-dominated sorting (Deb02 sect III-A)
    - Crowding distance assignment (Deb02 sect III-B)
    - Tournament selection based on rank and crowding (Deb02 sect III-C)

[P3] Abdullah & Zaman (2025), arXiv:2511.21612:
  - sect VIII-C: "Algorithm finds local, not global, optima"
  - sect VIII-D: "Future work: Learning real-time surface approximations via ML"
  - sect V-D: Rebalance penalty integrated into fitness evaluation
"""
import numpy as np
from nfg_diagscale.optimizer.scaling_plane import ScalingPlane
from nfg_diagscale.optimizer.rebalance_penalty import RebalancePenalty


class Individual:
    def __init__(self, H, c, r=8, b=1, s=1000):
        self.H = int(max(1, H))
        self.c = float(max(1, c))
        self.r = float(r)
        self.b = float(b)
        self.s = float(s)
        self.objectives = [0.0, 0.0, 0.0]
        self.rank = 0
        self.crowding_distance = 0.0
        self.domination_count = 0
        self.dominated_set = []

    def get_V(self):
        return (self.c, self.r, self.b, self.s)

    def copy(self):
        ind = Individual(self.H, self.c, self.r, self.b, self.s)
        ind.objectives = list(self.objectives)
        return ind


class NSGA2Optimizer:
    def __init__(self, config):
        self.config = config
        ncfg = config["nsga2"]
        self.pop_size = ncfg["population_size"]
        self.n_gen = ncfg["generations"]
        self.crossover_prob = ncfg["crossover_prob"]
        self.mutation_prob = ncfg["mutation_prob"]

        cloud = config["cloud"]
        self.min_H = cloud["min_replicas"]
        self.max_H = cloud["max_replicas"]
        self.min_c = cloud["min_cores"]
        self.max_c = cloud["max_cores"]
        self.ram = cloud["ram_gb"]
        self.bw = cloud["bandwidth_gbps"]
        self.storage = cloud["storage_iops"]

        self.scaling_plane = ScalingPlane(config)
        self.rebalance = RebalancePenalty(config)
        self.slo = config["themis"]["slo_ms"]

        self.pareto_front = []

    def _random_individual(self):
        H = np.random.randint(self.min_H, self.max_H + 1)
        c = np.random.randint(self.min_c, self.max_c + 1)
        return Individual(H, c, self.ram, self.bw, self.storage)

    def _evaluate(self, ind, current_H, current_V, predicted_rps):
        """
        [P3 sect III-H] Multi-objective evaluation on Scaling Plane:
          f1 = infrastructure cost  [P3 sect III-G]
          f2 = SLO violation risk   [P1 latency model]
          f3 = rebalance penalty    [P3 sect V-D]
        """
        # [P3 sect III-G] f1: cost
        f1 = self.scaling_plane.total_cost(ind.H, ind.c, self.ram)

        # [P3 sect III-D] f2: latency-based SLO risk
        lat = self.scaling_plane.total_latency(
            ind.H, ind.c, self.ram, self.bw, self.storage, predicted_rps
        )
        f2 = max(0, lat - self.slo) / self.slo

        # [P3 sect V-D] f3: rebalance transition cost
        f3 = self.rebalance.compute(current_H, current_V, ind.H, ind.get_V())

        ind.objectives = [f1, f2, f3]

    def _non_dominated_sort(self, population):
        """
        [Deb02 sect III-A] Fast non-dominated sorting procedure.
        """
        fronts = [[]]

        for p in population:
            p.domination_count = 0
            p.dominated_set = []

            for q in population:
                if self._dominates(p, q):
                    p.dominated_set.append(q)
                elif self._dominates(q, p):
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_set:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _dominates(self, p, q):
        """[Deb02] p dominates q iff p is no worse in all and strictly better in at least one."""
        at_least_one_better = False
        for i in range(len(p.objectives)):
            if p.objectives[i] > q.objectives[i]:
                return False
            if p.objectives[i] < q.objectives[i]:
                at_least_one_better = True
        return at_least_one_better

    def _crowding_distance(self, front):
        """
        [Deb02 sect III-B] Crowding distance assignment for diversity.
        """
        n = len(front)
        if n <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return

        for ind in front:
            ind.crowding_distance = 0.0

        n_obj = len(front[0].objectives)
        for m in range(n_obj):
            front.sort(key=lambda x: x.objectives[m])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            obj_range = front[-1].objectives[m] - front[0].objectives[m]
            if obj_range < 1e-12:
                continue

            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    (front[i + 1].objectives[m] - front[i - 1].objectives[m])
                    / obj_range
                )

    def _tournament_select(self, population):
        """
        [Deb02 sect III-C] Binary tournament selection:
        prefer lower rank, then higher crowding distance.
        """
        i, j = np.random.randint(0, len(population), 2)
        a, b = population[i], population[j]
        if a.rank < b.rank:
            return a.copy()
        elif b.rank < a.rank:
            return b.copy()
        elif a.crowding_distance > b.crowding_distance:
            return a.copy()
        else:
            return b.copy()

    def _crossover(self, p1, p2):
        """SBX-like crossover for H (integer) and BLX-alpha for c (continuous)."""
        if np.random.random() > self.crossover_prob:
            return p1.copy(), p2.copy()

        # BLX-alpha blend crossover for continuous variable c
        alpha = 0.5
        d = abs(p1.c - p2.c)
        low = min(p1.c, p2.c) - alpha * d
        high = max(p1.c, p2.c) + alpha * d
        c1 = np.clip(np.random.uniform(low, high), self.min_c, self.max_c)
        c2 = np.clip(np.random.uniform(low, high), self.min_c, self.max_c)

        # Single-point crossover for integer H
        H1 = p1.H if np.random.random() < 0.5 else p2.H
        H2 = p2.H if np.random.random() < 0.5 else p1.H

        child1 = Individual(H1, c1, self.ram, self.bw, self.storage)
        child2 = Individual(H2, c2, self.ram, self.bw, self.storage)
        return child1, child2

    def _mutate(self, ind):
        """Integer perturbation for H, Gaussian perturbation for c."""
        if np.random.random() < self.mutation_prob:
            ind.H = int(np.clip(
                ind.H + np.random.choice([-2, -1, 1, 2]),
                self.min_H, self.max_H
            ))
        if np.random.random() < self.mutation_prob:
            ind.c = float(np.clip(
                ind.c + np.random.normal(0, 1),
                self.min_c, self.max_c
            ))
            ind.c = round(ind.c)

    def optimize(self, current_H, current_c, predicted_rps):
        """
        Run NSGA-II to find Pareto-optimal (H, c) configurations.

        [P3 sect VIII-C] This extends DiagonalScale's local-only search
        with a global evolutionary search on the Scaling Plane.
        """
        current_V = (current_c, self.ram, self.bw, self.storage)

        # Initialize population
        pop = [self._random_individual() for _ in range(self.pop_size)]

        for ind in pop:
            self._evaluate(ind, current_H, current_V, predicted_rps)

        for gen in range(self.n_gen):
            # Create offspring via selection, crossover, mutation
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament_select(pop)
                p2 = self._tournament_select(pop)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                self._evaluate(c1, current_H, current_V, predicted_rps)
                self._evaluate(c2, current_H, current_V, predicted_rps)
                offspring.extend([c1, c2])

            # [Deb02] Combine parent and offspring
            combined = pop + offspring[:self.pop_size]

            # [Deb02 sect III-A] Non-dominated sorting
            fronts = self._non_dominated_sort(combined)

            # [Deb02 sect III-B] Crowding distance and selection
            new_pop = []
            for front in fronts:
                self._crowding_distance(front)
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    front.sort(key=lambda x: -x.crowding_distance)
                    remaining = self.pop_size - len(new_pop)
                    new_pop.extend(front[:remaining])
                    break

            pop = new_pop

        # Extract Pareto front (rank 0)
        self.pareto_front = [ind for ind in pop if ind.rank == 0]
        self.pareto_front.sort(key=lambda x: x.objectives[0])

        return self.pareto_front

    def get_nearest_checkpoint(self, current_H, current_c):
        """
        Find the Pareto-optimal point closest to current configuration.
        Used by ANFIS to constrain reactive decisions toward GA checkpoints.
        """
        if not self.pareto_front:
            return current_H, current_c

        best = None
        best_dist = float("inf")
        for ind in self.pareto_front:
            dist = abs(ind.H - current_H) + abs(ind.c - current_c)
            if dist < best_dist:
                best_dist = dist
                best = ind

        return best.H, best.c
