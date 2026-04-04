"""
Themis latency model for SLO risk assessment.

[P1] Razavi et al. (2024), "A Tale of Two Scales" (Themis), arXiv:2407.14843

  Processing latency (P1 Equation 1):
    l(b, c) = gamma * b/c + epsilon/c + delta * b + eta

  Where:
    b = batch size
    c = number of CPU cores allocated to the pod
    gamma = compute sensitivity to batch-core ratio
    epsilon = fixed per-core overhead
    delta = per-item processing cost
    eta = constant baseline latency

  Queuing latency - P1 simplified operational form (sect 4.2):
    q(b) = (b - 1) / lambda

  Where lambda = arrival rate (RPS). In operational regime where
  n * h(b,c) >= lambda, the queue is stable.

  End-to-end pipeline latency (P1 Equation 5):
    L_total = sum_s [ l_s(b_s, c_s) + q_s(b_s) ]

  [P1 sect 3] Vertical-first strategy:
    "initially using in-place vertical scaling to handle workload surges,
     then switching to horizontal scaling"
"""
import numpy as np


class ThemisLatencyModel:
    def __init__(self, config):
        tcfg = config["themis"]
        # [P1 Eq. 1] Processing latency parameters
        self.gamma = tcfg["gamma"]
        self.epsilon = tcfg["epsilon"]
        self.delta = tcfg["delta"]
        self.eta = tcfg["eta"]
        self.default_batch = tcfg["batch_size"]
        self.slo = tcfg["slo_ms"]

    def processing_latency(self, batch_size, cores):
        """
        [P1 Eq. 1] l(b, c) = gamma * b/c + epsilon/c + delta * b + eta
        """
        b = max(batch_size, 1)
        c = max(cores, 0.5)
        return self.gamma * b / c + self.epsilon / c + self.delta * b + self.eta

    def queuing_latency(self, batch_size, arrival_rate):
        """
        [P1 sect 4.2] q(b) = (b - 1) / lambda
        In the operational regime where n * h(b,c) >= lambda.
        """
        if arrival_rate <= 0:
            return 0.0
        b = max(batch_size, 1)
        return (b - 1) / arrival_rate

    def total_latency(self, batch_size, cores, arrival_rate, num_replicas):
        """
        [P1 Eq. 5] L_total = sum_s [ l_s(b_s, c_s) + q_s(b_s) ]

        For a homogeneous service with num_replicas pods, the effective
        per-pod arrival rate is lambda / num_replicas.
        """
        if num_replicas <= 0:
            return float("inf")

        per_pod_rate = arrival_rate / num_replicas
        l = self.processing_latency(batch_size, cores)
        q = self.queuing_latency(batch_size, per_pod_rate)
        return l + q

    def slo_risk(self, batch_size, cores, arrival_rate, num_replicas):
        """
        [P1] rho = 1[L_total > SLO]
        Binary SLO violation indicator.
        """
        lat = self.total_latency(batch_size, cores, arrival_rate, num_replicas)
        return 1.0 if lat > self.slo else 0.0

    def latency_headroom(self, batch_size, cores, arrival_rate, num_replicas):
        """
        Omega = (SLO - L_curr) / SLO
        Remaining latency budget as a fraction.
        """
        lat = self.total_latency(batch_size, cores, arrival_rate, num_replicas)
        return (self.slo - lat) / self.slo

    def max_rps_per_pod(self, cores):
        """
        Compute maximum sustainable RPS for a single pod with given cores,
        derived from the constraint L_total <= SLO.

        From [P1 Eq. 1]: l(b,c) is constant for given b,c.
        From [P1 sect 4.2]: q(b) = (b-1)/lambda.
        So SLO = l(b,c) + (b-1)/lambda_max  =>  lambda_max = (b-1)/(SLO - l(b,c))
        """
        b = self.default_batch
        l = self.processing_latency(b, cores)
        remaining = self.slo - l
        if remaining <= 0:
            return 1.0
        if b <= 1:
            return remaining * 100
        return (b - 1) / remaining * 1000
