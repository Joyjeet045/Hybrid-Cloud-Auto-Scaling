"""
Kubernetes Horizontal Pod Autoscaler (HPA) baseline.

Standard HPA formula (Kubernetes docs):
  desiredReplicas = ceil(currentReplicas * (currentMetricValue / desiredMetricValue))

Realistic parameters:
  - Default stabilization window for scale-down: 5 minutes (K8s docs)
  - Default tolerance band: 10% (--horizontal-pod-autoscaler-tolerance)
  - Scale-up has no default stabilization but has container startup latency
"""
import numpy as np


class HPABaseline:
    def __init__(self, config):
        baselines_cfg = config.get("baselines", {}).get("hpa", {})
        self.target_cpu = baselines_cfg.get("target_cpu", 0.6)
        self.min_replicas = config["cloud"]["min_replicas"]
        self.max_replicas = config["cloud"]["max_replicas"]
        # K8s HPA uses its own stabilization window, not relying on MAPE-K cooldown.
        # Default K8s downscale stabilization: 5 minutes = ~10 steps at 30s/step
        self.cooldown_steps = baselines_cfg.get("cooldown_steps", 10)
        # K8s default tolerance: ratio must deviate by >10% to trigger scaling
        self.tolerance = baselines_cfg.get("tolerance", 0.1)
        self._last_scale_step = -100
        self.name = "HPA"

    def decide(self, state, step):
        """
        Standard Kubernetes HPA: scale based on CPU utilization ratio.
        desiredReplicas = ceil(currentReplicas * (currentCPU / targetCPU))
        Includes tolerance band: no action if ratio within 1 +/- tolerance.
        """
        cpu = state["cpu_utilization"]
        current = state["replicas"]

        if step - self._last_scale_step < self.cooldown_steps:
            return {"mode": "none", "delta_c": 0, "delta_n": 0}

        ratio = cpu / self.target_cpu

        # K8s tolerance band: skip if ratio is within 1.0 +/- tolerance
        if abs(ratio - 1.0) <= self.tolerance:
            return {"mode": "none", "delta_c": 0, "delta_n": 0}

        desired = int(np.ceil(current * ratio))
        desired = int(np.clip(desired, self.min_replicas, self.max_replicas))

        delta_n = desired - current

        if delta_n != 0:
            self._last_scale_step = step

        return {
            "mode": "horizontal" if delta_n != 0 else "none",
            "delta_c": 0,
            "delta_n": delta_n,
        }
