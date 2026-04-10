"""
Kubernetes Horizontal Pod Autoscaler (HPA) baseline.

Standard HPA formula (Kubernetes docs):
  desiredReplicas = ceil(currentReplicas * (currentMetricValue / desiredMetricValue))
"""
import numpy as np


class HPABaseline:
    def __init__(self, config):
        self.target_cpu = 0.7
        self.min_replicas = config["cloud"]["min_replicas"]
        self.max_replicas = config["cloud"]["max_replicas"]
        self.cooldown_steps = 3
        self._last_scale_step = -100
        self.name = "HPA"

    def decide(self, state, step):
        """
        Standard Kubernetes HPA: scale based on CPU utilization ratio.
        desiredReplicas = ceil(currentReplicas * (currentCPU / targetCPU))
        """
        cpu = state["cpu_utilization"]
        current = state["replicas"]

        if step - self._last_scale_step < self.cooldown_steps:
            return {"mode": "none", "delta_c": 0, "delta_n": 0}

        desired = int(np.ceil(current * (cpu / self.target_cpu)))
        desired = int(np.clip(desired, self.min_replicas, self.max_replicas))

        delta_n = desired - current

        if delta_n != 0:
            self._last_scale_step = step

        return {
            "mode": "horizontal" if delta_n != 0 else "none",
            "delta_c": 0,
            "delta_n": delta_n,
        }
