"""
Themis baseline — Hybrid H+V scaling using profiling and cost-optimization.
"""
import numpy as np
from nfg_diagscale.decision.themis_latency import ThemisLatencyModel
from nfg_diagscale.optimizer.scaling_plane import ScalingPlane

class ThemisBaseline:
    def __init__(self, config):
        self.themis = ThemisLatencyModel(config)
        self.scaling_plane = ScalingPlane(config)
        self.min_replicas = config["cloud"]["min_replicas"]
        self.max_replicas = config["cloud"]["max_replicas"]
        self.min_cores = config["cloud"]["min_cores"]
        self.max_cores = config["cloud"]["max_cores"]
        self.ram = config["cloud"]["ram_gb"]
        self.slo = config["themis"]["slo_ms"]
        self.name = "Themis"
        self._last_scale_step = -100
        self.cooldown_steps = 10 

    def decide(self, state, step):
        """
        Choose configuration (H, c) that satisfies SLO and minimizes cost.
        Includes cooldown and hysteresis to prevent excessive scaling.
        """
        current_h = state["replicas"]
        current_c = state["cores"]
        actual_rps = state.get("current_rps", 0)
        current_cost = self.scaling_plane.total_cost(current_h, current_c, self.ram)
        batch = 1

        if step - self._last_scale_step < self.cooldown_steps:
            return {"mode": "none", "delta_c": 0, "delta_n": 0}

        # 1. Evaluate current feasibility
        current_lat = self.themis.total_latency(batch, current_c, actual_rps, current_h)
        is_feasible = current_lat <= (self.slo * 0.9)

        # 2. Search for the cheapest feasible config across the full plane
        best_config = (current_h, current_c)
        min_cost = current_cost
        
        found_better = False
        for h in range(self.min_replicas, self.max_replicas + 1):
            for c in range(self.min_cores, self.max_cores + 1):
                lat = self.themis.total_latency(batch, c, actual_rps, h)
                if lat <= (self.slo * 0.85):
                    cost = self.scaling_plane.total_cost(h, c, self.ram)
                    # Hysteresis: only switch if at least 5% cheaper or if current is infeasible
                    if not is_feasible:
                        if cost < min_cost:
                            min_cost = cost
                            best_config = (h, c)
                            found_better = True
                    else:
                        if cost < (min_cost * 0.95):
                            min_cost = cost
                            best_config = (h, c)
                            found_better = True
        
        # 3. If no feasible config found, pick the one with minimal violation
        if not found_better and not is_feasible:
            best_infeasible = (current_h, current_c)
            min_infeasible_lat = current_lat
            for h in range(self.min_replicas, self.max_replicas + 1):
                for c in range(self.min_cores, self.max_cores + 1):
                    lat = self.themis.total_latency(batch, c, actual_rps, h)
                    if lat < min_infeasible_lat:
                        min_infeasible_lat = lat
                        best_infeasible = (h, c)
            best_config = best_infeasible
            found_better = True

        # Return scaling mode for the selected config
        h_new, c_new = best_config
        delta_n = h_new - current_h
        delta_c = int(c_new - current_c)
        
        if delta_n == 0 and delta_c == 0:
            return {"mode": "none", "delta_c": 0, "delta_n": 0}

        self._last_scale_step = step
        if delta_n != 0 and delta_c != 0:
            mode = "diagonal"
        elif delta_n != 0:
            mode = "horizontal"
        else:
            mode = "vertical"
            
        return {"mode": mode, "delta_c": delta_c, "delta_n": delta_n}
