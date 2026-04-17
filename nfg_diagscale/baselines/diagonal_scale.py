"""
DiagonalScale baseline — greedy local search on the Scaling Plane.
"""
import numpy as np
from nfg_diagscale.optimizer.scaling_plane import ScalingPlane
from nfg_diagscale.optimizer.rebalance_penalty import RebalancePenalty
from nfg_diagscale.decision.themis_latency import ThemisLatencyModel


class DiagonalScaleBaseline:
    def __init__(self, config):
        self.scaling_plane = ScalingPlane(config)
        self.rebalance = RebalancePenalty(config)
        self.themis = ThemisLatencyModel(config)
        self.slo = config["themis"]["slo_ms"]
        self.batch_size = config["themis"]["batch_size"]
        self.min_replicas = config["cloud"]["min_replicas"]
        self.max_replicas = config["cloud"]["max_replicas"]
        self.min_cores = config["cloud"]["min_cores"]
        self.max_cores = config["cloud"]["max_cores"]
        self.ram = config["cloud"]["ram_gb"]
        self.bw = config["cloud"]["bandwidth_gbps"]
        self.storage = config["cloud"]["storage_iops"]

        ds_cfg = config.get("baselines", {}).get("diagonal_scale", {})
        # Monotonicity margin epsilon
        self.epsilon = ds_cfg.get("epsilon", 0.01)
        # Rebalance penalty weight delta
        self.delta_penalty = ds_cfg.get("delta_penalty", 0.5)
        # Objective function weights
        self._w_latency = ds_cfg.get("weight_latency", 0.4)
        self._w_cost = ds_cfg.get("weight_cost", 0.4)
        self._w_interaction = ds_cfg.get("weight_interaction", 0.2)

        # DiagonalScale uses its own stabilization interval
        self.cooldown_steps = ds_cfg.get("cooldown_steps", 8)
        self._last_scale_step = -100
        self.name = "DiagonalScale"

    def _objective(self, lat_penalty, cost):
        """Compute composite objective F from latency penalty and cost."""
        return (self._w_latency * lat_penalty
                + self._w_cost * cost
                + self._w_interaction * lat_penalty * cost)

    def decide(self, state, step):
        """
        DiagonalScale local search logic.
        """
        if step - self._last_scale_step < self.cooldown_steps:
            return {"mode": "none", "delta_c": 0, "delta_n": 0}

        H = state["replicas"]
        c = state["cores"]
        r = self.ram
        b = self.bw
        s = self.storage
        current_V = (c, r, b, s)
        actual_rps = state.get("current_rps", 0)

        # Current objective value
        lat_curr = self.themis.total_latency(self.batch_size, c, actual_rps, H)
        lat_penalty_curr = (lat_curr / self.slo) ** 2 if lat_curr > self.slo else lat_curr / self.slo
        cost_curr = self.scaling_plane.total_cost(H, c, r)
        F_curr = self._objective(lat_penalty_curr, cost_curr)

        # Generate neighborhood candidates
        neighbors = self._generate_neighborhood(H, c)

        best_neighbor = None
        best_F_prime = float("inf")

        for (H_n, c_n) in neighbors:
            # Estimate surfaces
            lat = self.themis.total_latency(self.batch_size, c_n, actual_rps, H_n)

            # Feasibility check
            if lat > self.slo:
                lat_penalty = (lat / self.slo) ** 2
            else:
                lat_penalty = lat / self.slo

            # Objective
            cost = self.scaling_plane.total_cost(H_n, c_n, r)
            F_n = self._objective(lat_penalty, cost)

            # Rebalance penalty
            new_V = (c_n, r, b, s)
            P_reb = self.rebalance.compute(H, current_V, H_n, new_V)

            # F' = F + delta * P_rebalance
            F_prime = F_n + self.delta_penalty * P_reb

            if F_prime < best_F_prime:
                best_F_prime = F_prime
                best_neighbor = (H_n, c_n)

        # Accept only if improvement > epsilon
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

            if delta_n != 0 or delta_c != 0:
                self._last_scale_step = step

            return {"mode": mode, "delta_c": delta_c, "delta_n": delta_n}

        return {"mode": "none", "delta_c": 0, "delta_n": 0}

    def _generate_neighborhood(self, H, c):
        """
        Generates moves: {(H +/- dH, V), (H, V +/- 1), (H +/- dH, V +/- 1)}
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
