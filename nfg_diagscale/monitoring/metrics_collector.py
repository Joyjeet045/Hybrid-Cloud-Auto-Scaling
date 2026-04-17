"""Multi-level metrics collector and composite stress signal."""
import numpy as np


class MetricsCollector:
    def __init__(self, config):
        # Stress signal weights
        scfg = config["stress"]
        self.w1 = scfg["w1_cpu"]
        self.w2 = scfg["w2_latency"]
        self.w3 = scfg["w3_queue"]
        self.q_max = scfg["q_max"]
        self.slo = config["themis"]["slo_ms"]
        # Stress thresholds for violation detection
        self._upper_threshold = scfg.get("upper_threshold", 0.5)
        self._lower_threshold = scfg.get("lower_threshold", 0.4)

    def collect_from_state(self, cloud_state):
        """Collect metrics from current environment state."""
        metrics = {
            # Host level
            "cpu_utilization": cloud_state.get("cpu_utilization", 0.0),
            # Container level
            "per_container_cpu": cloud_state.get("per_container_cpu", 0.0),
            # Platform level
            "app_latency": cloud_state.get("app_latency", 0.0),
            "queue_depth": cloud_state.get("queue_depth", 0.0),
            # Raw measurements
            "current_rps": cloud_state.get("current_rps", 0.0),
            "replicas": cloud_state.get("replicas", 1),
            "cores": cloud_state.get("cores", 1),
        }
        return metrics

    def compute_stress(self, metrics):
        """
        Composite stress signal (Equation 1 in the paper):
        Σ_stress = w1 * (CPU/CPU_max) + w2 * (L_app/SLO) + w3 * (Q_depth/Q_max)
        Weighted sum aggregates multi-level telemetry into a single scalar.
        """
        cpu_frac = min(metrics["cpu_utilization"], 1.0)
        lat_frac = min(metrics["app_latency"] / self.slo, 1.5)
        q_frac = min(metrics["queue_depth"] / self.q_max, 1.5)

        sigma = self.w1 * cpu_frac + self.w2 * lat_frac + self.w3 * q_frac
        return sigma

    def detect_violation(self, sigma_stress):
        """Determine transition direction based on stress."""
        if sigma_stress > self._upper_threshold:
            return "UP"
        elif sigma_stress < self._lower_threshold:
            return "DOWN"
        else:
            return "NONE"
