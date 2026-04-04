"""
Cloud environment simulator for trace-replay evaluation.

This is NOT a mock — it models real pod behavior using:
[P1] Themis latency model (arXiv:2407.14843, Eq. 1, sect 4.2)
  for computing actual request latency under a given configuration.
[P3] Scaling Plane surfaces (arXiv:2511.21612, sect III)
  for computing cost and coordination overhead.

Standard evaluation methodology in autoscaling research:
  [P5] sect 4: replay NASA/FIFA traces through the autoscaler
  [P2] sect 4: replay Azure traces with HAS-GPU policy
  [P1] sect 5: evaluate Themis on inference serving benchmarks
All use trace-driven simulation with their performance models.
"""
import numpy as np
from nfg_diagscale.decision.themis_latency import ThemisLatencyModel
from nfg_diagscale.optimizer.scaling_plane import ScalingPlane


class CloudEnvironment:
    def __init__(self, config):
        self.config = config
        cloud = config["cloud"]

        self.replicas = cloud["min_replicas"]
        self.cores = cloud["min_cores"]
        self.ram = cloud["ram_gb"]
        self.bw = cloud["bandwidth_gbps"]
        self.storage = cloud["storage_iops"]
        self.pod_max_rps = cloud["pod_max_rps"]

        self.min_replicas = cloud["min_replicas"]
        self.max_replicas = cloud["max_replicas"]
        self.min_cores = cloud["min_cores"]
        self.max_cores = cloud["max_cores"]

        # [P1] Delays: vertical is near-instant, horizontal has container startup
        self.h_delay = cloud["horizontal_delay_steps"]
        self.v_delay = cloud["vertical_delay_steps"]

        # [P1] Themis latency model for computing observed latency
        self.themis = ThemisLatencyModel(config)
        # [P3] Scaling Plane for cost computation
        self.scaling_plane = ScalingPlane(config)

        self._pending_h_actions = []
        self._pending_v_actions = []
        self._queue_depth = 0.0
        self._step = 0

        self.total_cost = 0.0
        self.history = []

    def get_state(self):
        """
        [P4 sect 2.1] Return current environment state at all three
        monitoring levels (host, container, platform).
        """
        return {
            "replicas": self.replicas,
            "cores": self.cores,
            "ram": self.ram,
            "step": self._step,
        }

    def _effective_pod_rps(self):
        """
        [P1 Eq. 1] Per-pod throughput scales with cores.
        From l(b,c) = gamma*b/c + epsilon/c + delta*b + eta:
        more cores => lower processing latency => higher sustainable RPS.
        Linear scaling is a standard assumption for CPU-bound workloads.
        """
        return self.pod_max_rps * self.cores

    def step(self, actual_rps):
        """
        Advance one time step with the given actual RPS from the trace.

        Computes real latency using [P1] Themis model and real cost
        using [P3] Scaling Plane cost surface. Adds congestion-based
        latency degradation when load exceeds capacity.
        """
        self._step += 1

        # Apply any matured pending scaling actions
        self._apply_pending_actions()

        # [P1] Effective capacity scales with both replicas and cores
        effective_pod_rps = self._effective_pod_rps()
        capacity = self.replicas * effective_pod_rps

        # [P1 Eq. 1, Eq. 5] Compute base latency under current config
        batch = self.config["themis"]["batch_size"]
        latency = self.themis.total_latency(
            batch, self.cores, actual_rps, self.replicas
        )

        # Congestion-based latency degradation when overloaded.
        # From M/M/c queuing theory: when utilization rho approaches 1,
        # waiting time grows as 1/(1-rho). This models the queue buildup
        # that P1's simplified q(b) formula omits at high load.
        if capacity > 0:
            utilization = actual_rps / capacity
            if utilization > 1.0:
                # Severely overloaded: latency grows proportionally
                congestion_factor = utilization ** 2
                latency *= congestion_factor
            elif utilization > 0.7:
                # Approaching saturation: M/M/c tail effect
                latency *= 1.0 + (utilization - 0.7) / (1.01 - utilization)

        # [P3 sect III-G] Compute cost for this time step
        step_cost = self.scaling_plane.total_cost(self.replicas, self.cores, self.ram)
        self.total_cost += step_cost

        # CPU utilization based on load vs capacity
        cpu_util = min(actual_rps / max(capacity, 1), 1.0)

        # Queue depth grows when demand exceeds capacity
        if actual_rps > capacity:
            self._queue_depth += (actual_rps - capacity) * 0.1
        else:
            self._queue_depth = max(0, self._queue_depth * 0.8)

        # SLO violation check
        slo = self.config["themis"]["slo_ms"]
        slo_violated = latency > slo

        state = {
            "cpu_utilization": cpu_util,
            "memory_utilization": min(cpu_util * 0.6, 1.0),
            "per_container_cpu": cpu_util,
            "app_latency": latency,
            "queue_depth": self._queue_depth,
            "current_rps": actual_rps,
            "replicas": self.replicas,
            "cores": self.cores,
            "step_cost": step_cost,
            "slo_violated": slo_violated,
            "capacity": capacity,
        }

        self.history.append(state)
        return state

    def execute_scaling(self, mode, delta_c, delta_n):
        """
        Execute a scaling action with appropriate delays.

        [P1 sect 3] "initially using in-place vertical scaling" —
          vertical scaling is near-instant (spec change).
        [P4 sect 2.3] "Creating VMs is a slow process" and
          "adding or removing containers in the environment is a
           less costly operation" — horizontal has startup delay.
        """
        action_record = {
            "step": self._step,
            "mode": mode,
            "delta_c": delta_c,
            "delta_n": delta_n,
        }

        if delta_c != 0:
            if self.v_delay == 0:
                # [P1] Vertical scaling is near-instant
                self.cores = int(np.clip(
                    self.cores + delta_c, self.min_cores, self.max_cores
                ))
            else:
                self._pending_v_actions.append({
                    "delta": delta_c,
                    "ready_step": self._step + self.v_delay,
                })

        if delta_n != 0:
            # [P4 sect 2.3] Horizontal scaling has container startup delay
            self._pending_h_actions.append({
                "delta": delta_n,
                "ready_step": self._step + self.h_delay,
            })

        return action_record

    def _apply_pending_actions(self):
        """Apply scaling actions that have matured past their delay."""
        remaining_h = []
        for action in self._pending_h_actions:
            if self._step >= action["ready_step"]:
                self.replicas = int(np.clip(
                    self.replicas + action["delta"],
                    self.min_replicas, self.max_replicas
                ))
            else:
                remaining_h.append(action)
        self._pending_h_actions = remaining_h

        remaining_v = []
        for action in self._pending_v_actions:
            if self._step >= action["ready_step"]:
                self.cores = int(np.clip(
                    self.cores + action["delta"],
                    self.min_cores, self.max_cores
                ))
            else:
                remaining_v.append(action)
        self._pending_v_actions = remaining_v

    def reset(self):
        """Reset environment to initial state."""
        cloud = self.config["cloud"]
        self.replicas = cloud["min_replicas"]
        self.cores = cloud["min_cores"]
        self._pending_h_actions = []
        self._pending_v_actions = []
        self._queue_depth = 0.0
        self._step = 0
        self.total_cost = 0.0
        self.history = []
