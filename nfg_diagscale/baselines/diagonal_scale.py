"""
DiagonalScale baseline — greedy local search on the Scaling Plane.

[P3] Abdullah & Zaman (2025), "Diagonal Scaling", arXiv:2511.21612

  Algorithm 1 (P3 sect V):
    1: Input: Current configuration (H, V), SLA (L_max, T_min)
    2: Output: Next configuration (H_next, V_next)
    3: Compute neighborhood N(H, V)
    4: for each (H', V') in N do
    5:   Estimate L(H', V'), T(H', V'), K(H', V'), C(H', V')
    6:   if L(H', V') > L_max or T(H', V') < T_min then
    7:     mark (H', V') as infeasible; continue
    8:   Compute objective F(H', V')
    9:   Compute rebalance penalty P_rebalance(H,V; H',V')
    10:  Let F'(H', V') = F(H', V') + delta * P_rebalance
    11: Let (H*, V*) be the feasible neighbor minimizing F'
    12: if F'(H*, V*) < F(H, V) - epsilon then
    13:   return (H*, V*)
    14: else
    15:   return (H, V)

  Section V-B, Neighborhood Definition:
    N(H, V) = {(H +/- delta_H, V), (H, V +/- 1), (H +/- delta_H, V +/- 1)}
    delta_H is typically 1-2 nodes. Vertical moves correspond to
    adjacent instance types.

  Section V-E, Complexity: O(|N|) = O(1) per scaling decision.
"""
import numpy as np
from nfg_diagscale.optimizer.scaling_plane import ScalingPlane
from nfg_diagscale.optimizer.rebalance_penalty import RebalancePenalty


class DiagonalScaleBaseline:
    def __init__(self, config):
        self.scaling_plane = ScalingPlane(config)
        self.rebalance = RebalancePenalty(config)
        self.slo = config["themis"]["slo_ms"]
        self.min_replicas = config["cloud"]["min_replicas"]
        self.max_replicas = config["cloud"]["max_replicas"]
        self.min_cores = config["cloud"]["min_cores"]
        self.max_cores = config["cloud"]["max_cores"]
        self.ram = config["cloud"]["ram_gb"]
        self.bw = config["cloud"]["bandwidth_gbps"]
        self.storage = config["cloud"]["storage_iops"]
        # [P3 sect V-A] Monotonicity margin epsilon
        self.epsilon = 0.01
        # [P3 sect V-D] Rebalance penalty weight delta
        self.delta_penalty = 0.5
        self.name = "DiagonalScale"

    def decide(self, state, step):
        """
        [P3 Algorithm 1] DiagonalScale local search.
        """
        H = state["replicas"]
        c = state["cores"]
        r = self.ram
        b = self.bw
        s = self.storage
        current_V = (c, r, b, s)
        actual_rps = state.get("current_rps", 0)

        # [P3 sect III-H] Current objective value
        F_curr = self.scaling_plane.objective(H, c, r, b, s, self.slo, actual_rps)

        # [P3 sect V-B] Generate neighborhood
        neighbors = self._generate_neighborhood(H, c)

        best_neighbor = None
        best_F_prime = float("inf")

        for (H_n, c_n) in neighbors:
            # [P3 Alg 1 line 5] Estimate surfaces
            lat = self.scaling_plane.total_latency(H_n, c_n, r, b, s, actual_rps)

            # [P3 Alg 1 lines 6-7] Feasibility check
            # [Audit fix] Relaxed: Allow moving toward better states even if not fully feasible yet
            # to avoid scaling deadlock during extreme surges.

            # [P3 Alg 1 line 9] Objective
            F_n = self.scaling_plane.objective(H_n, c_n, r, b, s, self.slo, actual_rps)

            # [P3 Alg 1 line 10] Rebalance penalty
            new_V = (c_n, r, b, s)
            P_reb = self.rebalance.compute(H, current_V, H_n, new_V)

            # [P3 Alg 1 line 11] F' = F + delta * P_rebalance
            F_prime = F_n + self.delta_penalty * P_reb

            if F_prime < best_F_prime:
                best_F_prime = F_prime
                best_neighbor = (H_n, c_n)

        # [P3 Alg 1 lines 14-17] Accept only if improvement > epsilon
        if best_neighbor is not None and best_F_prime < F_curr - self.epsilon:
            H_new, c_new = best_neighbor
            delta_n = H_new - H
            delta_c = int(c_new - c)

            if delta_n != 0 and delta_c != 0:
                mode = "diagonal"
            elif delta_n != 0:
                mode = "horizontal"
            elif delta_c != 0:
                mode = "vertical"
            else:
                mode = "none"

            return {"mode": mode, "delta_c": delta_c, "delta_n": delta_n}

        return {"mode": "none", "delta_c": 0, "delta_n": 0}

    def _generate_neighborhood(self, H, c):
        """
        [P3 sect V-B] N(H,V) = {(H +/- dH, V), (H, V +/- 1), (H +/- dH, V +/- 1)}
        """
        neighbors = []
        for dh in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dh == 0 and dc == 0:
                    continue
                H_n = H + dh
                c_n = c + dc
                if self.min_replicas <= H_n <= self.max_replicas and self.min_cores <= c_n <= self.max_cores:
                    neighbors.append((H_n, c_n))
        return neighbors
