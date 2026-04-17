"""Cloud environment simulator for trace-replay evaluation."""
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

        # Delays: vertical is near-instant, horizontal has container startup
        self.h_delay = cloud["horizontal_delay_steps"]
        self.v_delay = cloud["vertical_delay_steps"]

        # Themis latency model for computing observed latency
        self.themis = ThemisLatencyModel(config)
        # Scaling Plane for cost computation
        self.scaling_plane = ScalingPlane(config)

        self._pending_h_actions = []
        self._pending_v_actions = []
        self._queue_depth = 0.0
        self._step = 0

        self.total_cost = 0.0
        self.history = []

    def get_state(self):
        """Return current environment state."""
        return {
            "replicas": self.replicas,
            "cores": self.cores,
            "ram": self.ram,
            "step": self._step,
        }

    def _effective_pod_rps(self):
        """
        Per-pod throughput scales with cores.
        """
        return self.pod_max_rps * self.cores

    def step(self, actual_rps):
        """Advance one time step and compute latency/cost."""
        self._step += 1

        # Apply any matured pending scaling actions
        self._apply_pending_actions()

        # Effective capacity scales with both replicas and cores
        effective_pod_rps = self._effective_pod_rps()
        capacity = self.replicas * effective_pod_rps

        # Compute base latency under current config
        batch = self.config["themis"]["batch_size"]
        latency = self.themis.total_latency(
            batch, self.cores, actual_rps, self.replicas
        )

        # Compute cost for this time step
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
        """Execute a scaling action with configured delays."""
        action_record = {
            "step": self._step,
            "mode": mode,
            "delta_c": delta_c,
            "delta_n": delta_n,
        }

        if delta_c != 0:
            if self.v_delay == 0:
                # Vertical scaling is near-instant
                self.cores = int(np.clip(
                    self.cores + delta_c, self.min_cores, self.max_cores
                ))
            else:
                self._pending_v_actions.append({
                    "delta": delta_c,
                    "ready_step": self._step + self.v_delay,
                })

        if delta_n != 0:
            # Horizontal scaling has container startup delay
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
