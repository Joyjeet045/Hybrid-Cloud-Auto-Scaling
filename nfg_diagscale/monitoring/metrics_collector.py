"""
Multi-level metrics collector and composite stress signal.

[P4] Solino, Batista & Cavalcante (2025), ACM UCC'25

  Section 2.1: Three monitoring levels:
    - Host: CPU%, Memory% (gathered by Metricbeat)
    - Container: per-container CPU, Memory (gathered by Metricbeat)
    - Platform: internal operation execution time (gathered by AspectJ+Filebeat)

  Section 3: "We accomplish this through AspectJ, an AOP-compliant Java
    extension. We define aspects that capture the start and end timestamps
    of critical internal methods or operations."

  Algorithm 1 (P4): SymptomDetection procedure checks thresholds at
  all three levels.

Composite stress signal (author synthesis of P4's multi-level data):
  Sigma_stress(t) = w1 * CPU(t)/CPU_max + w2 * L_app(t)/SLO + w3 * Q(t)/Q_max
"""
import numpy as np


class MetricsCollector:
    def __init__(self, config):
        # Stress signal weights (our synthesis of P4 multi-level monitoring)
        scfg = config["stress"]
        self.w1 = scfg["w1_cpu"]
        self.w2 = scfg["w2_latency"]
        self.w3 = scfg["w3_queue"]
        self.q_max = scfg["q_max"]
        self.slo = config["themis"]["slo_ms"]

    def collect_from_state(self, cloud_state):
        """
        [P4 sect 2.1] Collect metrics at three levels from cloud environment state.

        [P4 Algorithm 1] SymptomDetection:
          - clusterSymptomDetection: host-level CPU/Mem
          - containerSymptomDetection: per-container metrics
          - platformSymptomDetection: application-level metrics
        """
        metrics = {
            # [P4 sect 2.1] Host level
            "cpu_utilization": cloud_state.get("cpu_utilization", 0.0),
            "memory_utilization": cloud_state.get("memory_utilization", 0.0),
            # [P4 sect 2.1] Container level
            "per_container_cpu": cloud_state.get("per_container_cpu", 0.0),
            # [P4 sect 2.1, sect 3] Platform level (AOP-intercepted)
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
        Composite stress signal (our synthesis of P4's three monitoring levels).
        Sigma_stress = w1 * CPU/CPU_max + w2 * L_app/SLO + w3 * Q/Q_max
        """
        cpu_frac = min(metrics["cpu_utilization"], 1.0)
        lat_frac = min(metrics["app_latency"] / self.slo, 2.0)
        q_frac = min(metrics["queue_depth"] / self.q_max, 2.0)

        sigma = self.w1 * cpu_frac + self.w2 * lat_frac + self.w3 * q_frac
        return sigma

    def detect_violation(self, sigma_stress, upper=0.65, lower=0.35):
        """
        [P4 sect 2.2] "When a metric exceeds a predefined threshold,
        it is flagged as a symptom."
        Upper set to 0.65 to allow proactive scaling BEFORE latency hits 1.0 (SLO limit).
        the need to add more resources, while DOWN suggests eliminating."
        """
        if sigma_stress > upper:
            return "UP"
        elif sigma_stress < lower:
            return "DOWN"
        else:
            return "NONE"
