"""
Kubernetes Vertical Pod Autoscaler (VPA) baseline.

Standard VPA behavior:
  Monitor CPU usage per container, recommend resource changes
  when usage consistently exceeds or drops below target range.
"""
import numpy as np


class VPABaseline:
    def __init__(self, config):
        self.target_cpu_upper = 0.8
        self.target_cpu_lower = 0.3
        self.min_cores = config["cloud"]["min_cores"]
        self.max_cores = config["cloud"]["max_cores"]
        self.cooldown_steps = 5
        self._last_scale_step = -100
        self.name = "VPA"

    def decide(self, state, step):
        """
        VPA: scale CPU cores based on per-container utilization.
        """
        cpu = state["cpu_utilization"]
        current_cores = state["cores"]

        if step - self._last_scale_step < self.cooldown_steps:
            return {"mode": "none", "delta_c": 0, "delta_n": 0}

        delta_c = 0
        if cpu > self.target_cpu_upper:
            delta_c = max(1, int(np.ceil((cpu - self.target_cpu_upper) * current_cores)))
        elif cpu < self.target_cpu_lower and current_cores > self.min_cores:
            delta_c = -1

        new_cores = int(np.clip(current_cores + delta_c, self.min_cores, self.max_cores))
        delta_c = new_cores - current_cores

        if delta_c != 0:
            self._last_scale_step = step

        return {
            "mode": "vertical" if delta_c != 0 else "none",
            "delta_c": delta_c,
            "delta_n": 0,
        }
