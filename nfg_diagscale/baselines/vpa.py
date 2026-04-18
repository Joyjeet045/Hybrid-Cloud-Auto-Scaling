"""
Kubernetes Vertical Pod Autoscaler (VPA) baseline.

Realistic VPA behavior:
  - "Auto" mode requires pod eviction and restart for resource changes
  - Restart introduces temporary capacity loss (pod unavailable during restart)
  - Recommendation latency: VPA stabilizes recommendations over minutes
  - Each scaling event restarts pods, causing a brief service disruption
"""
import numpy as np


class VPABaseline:
    def __init__(self, config):
        baselines_cfg = config.get("baselines", {}).get("vpa", {})
        self.target_cpu_upper = baselines_cfg.get("target_cpu_upper", 0.7)
        self.target_cpu_lower = baselines_cfg.get("target_cpu_lower", 0.3)
        self.min_cores = config["cloud"]["min_cores"]
        self.max_cores = config["cloud"]["max_cores"]
        self.min_replicas = config["cloud"]["min_replicas"]
        # VPA requires pod restart → longer stabilization than HPA.
        # Realistic: pod eviction + restart + stabilization = ~6 minutes
        self.cooldown_steps = baselines_cfg.get("cooldown_steps", 12)
        # Number of steps where capacity is degraded during pod restart
        self.restart_delay_steps = baselines_cfg.get("restart_delay_steps", 3)
        self._last_scale_step = -100
        self._restart_remaining = 0
        # Track pod eviction/restore cycle
        self._pending_restore = False
        self.name = "VPA"

    def decide(self, state, step):
        """
        VPA: scale CPU cores based on per-container utilization.
        Models realistic pod eviction: when cores change, a pod is evicted
        (capacity drops), then restored after restart_delay with new resources.
        """
        cpu = state["cpu_utilization"]
        current_cores = state["cores"]
        current_replicas = state["replicas"]

        # During pod restart, check if pod should be restored
        if self._restart_remaining > 0:
            self._restart_remaining -= 1
            if self._restart_remaining == 0 and self._pending_restore:
                # Pod comes back online with new resources
                self._pending_restore = False
                return {"mode": "horizontal", "delta_c": 0, "delta_n": 1}
            return {"mode": "none", "delta_c": 0, "delta_n": 0}

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
            self._restart_remaining = self.restart_delay_steps

            # Model pod eviction: VPA evicts a pod to apply new resources
            # This temporarily reduces capacity (realistic K8s VPA behavior)
            if current_replicas > self.min_replicas:
                self._pending_restore = True
                return {
                    "mode": "diagonal",
                    "delta_c": delta_c,
                    "delta_n": -1,  # Pod evicted for restart
                }
            else:
                # Can't evict if at min replicas — just apply core change
                return {"mode": "vertical", "delta_c": delta_c, "delta_n": 0}

        return {"mode": "none", "delta_c": 0, "delta_n": 0}
