"""
Themis baseline — Hybrid H+V scaling using profiling and cost-optimization.

[P1] Razavi et al. (2024), "A Tale of Two Scales" (Themis), arXiv:2407.14843
  - sect 5.2: "Themis initially uses in-place vertical scaling to handle 
    workload surges, then switching to horizontal scaling as needed."
  - sect 4.3: "Optimizes total cost while satisfying SLO constraints."
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

    def decide(self, state, step):
        """
        [P1] Choose configuration (H, c) that satisfies SLO and minimizes cost.
        Prioritizes in-place vertical scaling if it satisfies the SLO.
        """
        current_h = state["replicas"]
        current_c = state["cores"]
        actual_rps = state.get("current_rps", 0)
        batch = 1

        # 1. Try vertical scaling first (stay at current replica count) [P1 sect 5.2]
        best_v_cores = None
        min_v_cost = float("inf")
        
        for cores in range(self.min_cores, self.max_cores + 1):
            lat = self.themis.total_latency(batch, cores, actual_rps, current_h)
            if lat <= self.slo:
                cost = self.scaling_plane.total_cost(current_h, cores, self.ram)
                if cost < min_v_cost:
                    min_v_cost = cost
                    best_v_cores = cores
        
        if best_v_cores is not None:
            delta_c = int(best_v_cores - current_c)
            # Only return vertical mode if a real change occurs
            if delta_c != 0:
                return {"mode": "vertical", "delta_c": delta_c, "delta_n": 0}

        # 2. If vertical is not enough, search the full (H, V) plane for the cheapest feasible config
        best_config = None
        min_cost = float("inf")
        best_infeasible = None
        min_infeasible_lat = float("inf")
        
        for h in range(self.min_replicas, self.max_replicas + 1):
            for c in range(self.min_cores, self.max_cores + 1):
                lat = self.themis.total_latency(batch, c, actual_rps, h)
                if lat <= self.slo:
                    cost = self.scaling_plane.total_cost(h, c, self.ram)
                    if cost < min_cost:
                        min_cost = cost
                        best_config = (h, c)
                else:
                    if lat < min_infeasible_lat:
                        min_infeasible_lat = lat
                        best_infeasible = (h, c)
        
        # [Audit fix] If no feasible config found, pick the one that mini-violation
        if not best_config:
            best_config = best_infeasible
            h_new, c_new = best_config
            delta_n = h_new - current_h
            delta_c = int(c_new - current_c)
            
            if delta_n == 0 and delta_c == 0:
                return {"mode": "none", "delta_c": 0, "delta_n": 0}

            if delta_n != 0 and delta_c != 0:
                mode = "diagonal"
            elif delta_n != 0:
                mode = "horizontal"
            else:
                mode = "vertical"
                
            return {"mode": mode, "delta_c": delta_c, "delta_n": delta_n}

        return {"mode": "none", "delta_c": 0, "delta_n": 0}
