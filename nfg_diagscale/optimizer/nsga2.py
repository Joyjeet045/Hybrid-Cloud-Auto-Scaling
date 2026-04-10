"""
NSGA-II Multi-Objective Optimizer on the Diagonal Scaling Plane.

Trajectory Optimizer:
  Replaces single-step optimization with a T-step trajectory chromosome.
  Encodes [(H1, c1), (H2, c2), ..., (HT, cT)] to optimize the entire path on the plane.
"""
import numpy as np
from nfg_diagscale.optimizer.scaling_plane import ScalingPlane
from nfg_diagscale.optimizer.rebalance_penalty import RebalancePenalty


class Individual:
    """
    Chromosome encodes a complete T-step scaling trajectory.
    """
    def __init__(self, trajectory, ram=8, bw=1, s=1000):
        # trajectory is a list of (H, c) tuples for T steps
        self.trajectory = trajectory 
        self.ram = float(ram)
        self.bw = float(bw)
        self.s = float(s)
        self.objectives = [0.0, 0.0, 0.0]
        self.rank = 0
        self.crowding_distance = 0.0
        self.domination_count = 0
        self.dominated_set = []

    def copy(self):
        traj_copy = [list(step) for step in self.trajectory]
        ind = Individual(traj_copy, self.ram, self.bw, self.s)
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
        self.T = ncfg.get("horizon_steps", 4) # T-step horizon

        cloud = config["cloud"]
        self.min_H = cloud["min_replicas"]
        self.max_H = cloud["max_replicas"]
        self.min_c = cloud["min_cores"]
        self.max_c = cloud["max_cores"]

        self.scaling_plane = ScalingPlane(config)
        self.rebalance = RebalancePenalty(config)
        self.slo = config["themis"]["slo_ms"]

        self.pareto_front = []

    def _random_individual(self):
        """Create a random T-step trajectory."""
        trajectory = []
        for _ in range(self.T):
            H = np.random.randint(self.min_H, self.max_H + 1)
            c = np.random.randint(self.min_c, self.max_c + 1)
            trajectory.append([H, c])
        return Individual(trajectory, config=self.config)

    def _random_individual_alt(self):
        # Helper to avoid issues with config passing in Individual __init__
        trajectory = []
        for _ in range(self.T):
            H = np.random.randint(self.min_H, self.max_H + 1)
            c = np.random.randint(self.min_c, self.max_c + 1)
            trajectory.append([H, float(c)])
        return Individual(trajectory)

    def _evaluate(self, ind, current_H, current_cores, predicted_rps, low_load_mode=False):
        """
        Cumulative trajectory evaluation.
        f1: Cumulative infrastructure cost
        f2: Cumulative SLO violation risk
        f3: Cumulative rebalance penalty along the path
        """
        f1_sum = 0.0
        f2_sum = 0.0
        f3_sum = 0.0

        prev_H = current_H
        prev_V = (current_cores, 8.0, 1.0, 1000.0) # Simplified V for rebalance computation

        for H_step, c_step in ind.trajectory:
            # f1: Cost at this step
            f1_sum += self.scaling_plane.total_cost(H_step, c_step, 8.0)

            # f2: SLO risk at this step (approximate using predicted RPS)
            lat = self.scaling_plane.total_latency(
                H_step, c_step, 8.0, 1.0, 1000.0, predicted_rps
            )
            f2_sum += max(0, lat - self.slo) / self.slo

            # f3: Rebalance penalty for transition to this step
            # Prefer vertical (rebalance=0) over horizontal moves
            curr_V = (c_step, 8.0, 1.0, 1000.0)
            rebalance_penalty = self.rebalance.compute(prev_H, prev_V, H_step, curr_V)
            
            # Low-load mode extra penalty
            if getattr(self, "low_load_mode", False):
                rebalance_penalty *= 2.5 # Aggressively avoid new pods if possible
                
            f3_sum += rebalance_penalty * 1.2

            prev_H = H_step
            prev_V = curr_V

        ind.objectives = [f1_sum, f2_sum, f3_sum]

    def _non_dominated_sort(self, population):
        """Fast non-dominated sorting."""
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
        at_least_one_better = False
        for i in range(len(p.objectives)):
            if p.objectives[i] > q.objectives[i]:
                return False
            if p.objectives[i] < q.objectives[i]:
                at_least_one_better = True
        return at_least_one_better

    def _crowding_distance(self, front):
        """Diversity preservation."""
        n = len(front)
        if n <= 2:
            for ind in front: ind.crowding_distance = float("inf")
            return
        for ind in front: ind.crowding_distance = 0.0
        n_obj = len(front[0].objectives)
        for m in range(n_obj):
            front.sort(key=lambda x: x.objectives[m])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")
            obj_range = front[-1].objectives[m] - front[0].objectives[m]
            if obj_range < 1e-12: continue
            for i in range(1, n - 1):
                front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range

    def _tournament_select(self, population):
        i, j = np.random.randint(0, len(population), 2)
        a, b = population[i], population[j]
        if a.rank < b.rank: return a.copy()
        if b.rank < a.rank: return b.copy()
        return a.copy() if a.crowding_distance > b.crowding_distance else b.copy()

    def _crossover(self, p1, p2):
        """Single-point crossover on the trajectory time index."""
        if np.random.random() > self.crossover_prob:
            return p1.copy(), p2.copy()

        point = np.random.randint(1, self.T)
        traj1 = p1.trajectory[:point] + p2.trajectory[point:]
        traj2 = p2.trajectory[:point] + p1.trajectory[point:]

        return Individual(traj1), Individual(traj2)

    def _mutate(self, ind):
        """Perturb config at a random step in the trajectory."""
        if np.random.random() < self.mutation_prob:
            step_idx = np.random.randint(0, self.T)
            # Mutate H
            ind.trajectory[step_idx][0] = int(np.clip(
                ind.trajectory[step_idx][0] + np.random.choice([-1, 1]),
                self.min_H, self.max_H
            ))
            # Mutate cores
            ind.trajectory[step_idx][1] = float(np.clip(
                ind.trajectory[step_idx][1] + np.random.normal(0, 0.5),
                self.min_c, self.max_c
            ))

    def optimize(self, current_H, current_cores, predicted_rps, low_load_mode=False):
        """Run NSGA-II to find Pareto-optimal trajectories."""
        self.low_load_mode = low_load_mode
        pop = [self._random_individual_alt() for _ in range(self.pop_size)]
        for ind in pop:
            self._evaluate(ind, current_H, current_cores, predicted_rps)

        for gen in range(self.n_gen):
            offspring = []
            while len(offspring) < self.pop_size:
                p1, p2 = self._tournament_select(pop), self._tournament_select(pop)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                self._evaluate(c1, current_H, current_cores, predicted_rps, low_load_mode)
                self._evaluate(c2, current_H, current_cores, predicted_rps, low_load_mode)
                offspring.extend([c1, c2])

            combined = pop + offspring[:self.pop_size]
            fronts = self._non_dominated_sort(combined)
            new_pop = []
            for front in fronts:
                self._crowding_distance(front)
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    front.sort(key=lambda x: -x.crowding_distance)
                    new_pop.extend(front[:self.pop_size - len(new_pop)])
                    break
            pop = new_pop

        self.pareto_front = [ind for ind in pop if ind.rank == 0]
        return self.pareto_front

    def get_nearest_checkpoint(self, current_H, current_c):
        """Find the immediate next step from the best trajectory."""
        if not self.pareto_front:
            return current_H, current_c

        # Strategy: pick the trajectory with minimum rebalance + cost at step 1
        best_step_1 = None
        min_dist = float("inf")
        
        for ind in self.pareto_front:
            h1, c1 = ind.trajectory[0]
            dist = abs(h1 - current_H) + abs(c1 - current_c)
            if dist < min_dist:
                min_dist = dist
                best_step_1 = (h1, c1)

        return best_step_1 if best_step_1 else (current_H, current_c)
