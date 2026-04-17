import numpy as np


class ThemisLatencyModel:
    """
    Latency model following the profiling-based approach.
    L_total(b, c, n, lambda) = L_profile(b, c) + L_queue(lambda, b, n)
    """

    def __init__(self, config):
        self.config = config
        tcfg = config["themis"]
        self.default_batch = tcfg.get("batch_size", 1)
        self.slo = tcfg.get("slo_ms", 100.0)

        # We use a profiling table indexed by (batch_size, CPU_cores).
        self._generate_profile_table(config)

    def _generate_profile_table(self, config):
        """
        Generate a profiling table based on simulated measurements.
        Processing latency decreases as cores increase and increases with batch size.
        """
        self.profile_table = {}
        max_cores = config["cloud"]["max_cores"]

        # Simulate profiling data for cores 1 to max_cores
        for c in range(1, max_cores + 1):
            # Characteristic: l(b,c) decreases with c, slightly increases with b
            # Refined model (still numeric, but structured as a profile LUT)
            self.profile_table[c] = 5.0 + (50.0 / (c + 0.1)) + (0.01 * self.default_batch)

    def processing_latency(self, batch_size, cores):
        """
        Looks up processing latency from the offline profiling table.
        """
        c_idx = int(np.clip(round(cores), 1, len(self.profile_table)))
        return self.profile_table.get(c_idx, 10.0)

    def queuing_latency(self, batch_size, arrival_rate):
        """
        Estimates queuing latency numerically.
        Wait time is proportional to (batch-1)/RPS.
        """
        if arrival_rate <= 0:
            return 0.0
        # Estimation consistent with batching logic
        return (max(batch_size, 1) - 1) / arrival_rate

    def total_latency(self, batch_size, cores, arrival_rate, num_replicas):
        """
        Total latency = processing latency + queuing latency.
        """
        if num_replicas <= 0:
            return 1000.0  # Max penalty

        per_pod_rate = arrival_rate / num_replicas
        l_proc = self.processing_latency(batch_size, cores)
        l_que = self.queuing_latency(batch_size, per_pod_rate)
        
        latency = l_proc + l_que
        
        # Realism: Congestion degradation consistent with CloudEnvironment
        # This ensures baselines RECOGNIZE when they are failing.
        pod_max_rps = self.config["cloud"]["pod_max_rps"]
        capacity = num_replicas * cores * pod_max_rps
        if capacity > 0 and arrival_rate > 0:
            utilization = arrival_rate / capacity
            if utilization > 1.0:
                # Maintain the vertical asymptote momentum past utilization 1.0
                penalty_at_border = 1.0 + (1.0 - 0.7) / 0.011
                latency *= penalty_at_border * (utilization ** 2)
            elif utilization > 0.7:
                latency *= 1.0 + (utilization - 0.7) / (1.011 - utilization)
                
        return latency

    def slo_risk(self, batch_size, cores, arrival_rate, num_replicas):
        """
        Continuous SLO risk using a sigmoid centered at the SLO threshold.
        rho = sigmoid(k * (L_total - SLO) / SLO)
        Returns a value in [0, 1] that smoothly transitions from safe to risky,
        enabling proper fuzzy reasoning instead of degenerate binary gating.
        """
        lat = self.total_latency(batch_size, cores, arrival_rate, num_replicas)
        # Steepness k=10: rho ≈ 0.27 at 90% SLO, 0.5 at SLO, 0.73 at 110% SLO
        k = 10.0
        z = k * (lat - self.slo) / max(self.slo, 1e-6)
        return 1.0 / (1.0 + np.exp(-z))

    def latency_headroom(self, batch_size, cores, arrival_rate, num_replicas):
        """
        Omega = (SLO - L_curr) / SLO
        """
        lat = self.total_latency(batch_size, cores, arrival_rate, num_replicas)
        return (self.slo - lat) / self.slo

    def max_rps_per_pod(self, cores):
        """
        Estimated throughput capacity derived from the profiling table.
        """
        l_proc = self.processing_latency(self.default_batch, cores)
        rem = self.slo - l_proc
        if rem <= 0:
            return 1.0
        # Return RPS that would saturate SLO
        return max(1.0, (self.default_batch) / (rem / 1000.0 + 1e-9))
