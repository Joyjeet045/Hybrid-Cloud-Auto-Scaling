"""Cloud environment simulator for trace-replay evaluation."""
import numpy as np
from nfg_diagscale.decision.themis_latency import ThemisLatencyModel
from nfg_diagscale.optimizer.scaling_plane import ScalingPlane

VM_TYPES = [
    {"type": "m5.xlarge", "cores": 4, "mem": 16, "price": 0.192},
    {"type": "m5.2xlarge", "cores": 8, "mem": 32, "price": 0.384},
    {"type": "m5.4xlarge", "cores": 16, "mem": 64, "price": 0.768},
    {"type": "m5.8xlarge", "cores": 32, "mem": 128, "price": 1.536},
    {"type": "m5.12xlarge", "cores": 48, "mem": 192, "price": 2.304},
]


class SimPM:
    def __init__(self, pm_id):
        self.pm_id = pm_id
        self.max_cores = 64
        self.vms = []
        self.active = True

    @property
    def allocated_cores(self):
        return sum(vm.max_cores for vm in self.vms if vm.active)

    @property
    def remaining_cores(self):
        return self.max_cores - self.allocated_cores


class SimVM:
    def __init__(self, vm_id, vm_type, max_cores, price):
        self.vm_id = vm_id
        self.vm_type = vm_type
        self.max_cores = max_cores
        self.price = price
        self.containers = []
        self.pm = None
        self.active = True

    @property
    def allocated_cores(self):
        return sum(c.cores for c in self.containers if c.active)

    @property
    def remaining_cores(self):
        return self.max_cores - self.allocated_cores


class SimContainer:
    def __init__(self, container_id, cores):
        self.container_id = container_id
        self.cores = cores
        self.vm = None
        self.active = True


class CloudEnvironment:
    def __init__(self, config):
        self.config = config
        cloud = config["cloud"]

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

        # Internal backing variables for initial/fallback property values
        self._replicas = cloud["min_replicas"]
        self._cores = cloud["min_cores"]

        # Infrastructure tracking
        self.pms = []
        self.vms = []
        self.containers = []

        self._init_deployment()

    @property
    def replicas(self):
        """Return number of active containers."""
        active_conts = [c for c in self.containers if c.active]
        return len(active_conts) if active_conts else self._replicas

    @replicas.setter
    def replicas(self, val):
        self._replicas = val

    @property
    def cores(self):
        """Return average core count of active containers."""
        active_conts = [c for c in self.containers if c.active]
        if active_conts:
            return int(np.mean([c.cores for c in active_conts]))
        return self._cores

    @cores.setter
    def cores(self, val):
        self._cores = val

    def get_state(self):
        """Return current environment state."""
        return {
            "replicas": self.replicas,
            "cores": self.cores,
            "ram": self.ram,
            "step": self._step,
        }

    def step(self, actual_rps):
        """Advance one time step and compute latency/cost."""
        self._step += 1

        # Apply any matured pending scaling actions
        self._apply_pending_actions()

        active_containers = [c for c in self.containers if c.active]

        if not active_containers:
            # Fallback handling in case of no active containers
            capacity = 0
            latency = 1000.0
            step_cost = 0.0
            cpu_util = 1.0
        else:
            # Capacity-based Weighted Round-Robin (CWRR) load distribution
            total_cores = sum(c.cores for c in active_containers)
            capacity = sum(c.cores * self.pod_max_rps for c in active_containers)

            # Compute latency using CWRR routing weights
            batch = self.config["themis"]["batch_size"]
            total_lat = 0.0
            for c in active_containers:
                weight = c.cores / total_cores
                lambda_c = actual_rps * weight
                lat_c = self.themis.total_latency(batch, c.cores, lambda_c, num_replicas=1)
                total_lat += weight * lat_c
            latency = total_lat

            # Compute actual VM rental cost for this time step
            cycle_seconds = self.config.get("mape_k", {}).get("cycle_seconds", 30)
            steps_per_hour = 3600.0 / cycle_seconds
            step_cost = sum(vm.price for vm in self.vms if vm.active) / steps_per_hour

        self.total_cost += step_cost

        # CPU utilization based on load vs capacity
        cpu_util = min(actual_rps / max(capacity, 1), 1.0)

        # Queue depth grows when demand exceeds capacity
        if actual_rps > capacity:
            self._queue_depth += (actual_rps - capacity) * 0.1
        else:
            self._queue_depth = max(0.0, self._queue_depth * 0.8)

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
                self._apply_vertical_action(delta_c)
            else:
                self._pending_v_actions.append({
                    "delta": delta_c,
                    "ready_step": self._step + self.v_delay,
                })

        if delta_n != 0:
            if self.h_delay == 0:
                self._apply_horizontal_action(delta_n)
            else:
                self._pending_h_actions.append({
                    "delta": delta_n,
                    "ready_step": self._step + self.h_delay,
                })

        return action_record

    def _apply_horizontal_action(self, delta):
        current_replicas = self.replicas
        new_replicas = int(np.clip(current_replicas + delta, self.min_replicas, self.max_replicas))
        actual_delta = new_replicas - current_replicas

        if actual_delta > 0:
            target_cores = max(1, self.cores)
            for _ in range(actual_delta):
                c = SimContainer(len(self.containers), target_cores)
                self.containers.append(c)
                self._place_container(c)
        elif actual_delta < 0:
            active_conts = [c for c in self.containers if c.active]
            active_conts.sort(key=lambda c: c.vm.allocated_cores)
            num_to_delete = min(len(active_conts), -actual_delta)
            for i in range(num_to_delete):
                c = active_conts[i]
                c.active = False
                if c in c.vm.containers:
                    c.vm.containers.remove(c)
                # Shutdown VM if empty
                if not any(cont.active for cont in c.vm.containers):
                    c.vm.active = False
                    if c.vm in c.vm.pm.vms:
                        c.vm.pm.vms.remove(c.vm)
                    # Shutdown PM if empty
                    if not any(vm.active for vm in c.vm.pm.vms):
                        c.vm.pm.active = False

    def _apply_vertical_action(self, delta):
        active_conts = [c for c in self.containers if c.active]
        if not active_conts:
            return

        if delta > 0:
            for c in active_conts:
                new_cores = int(np.clip(c.cores + delta, self.min_cores, self.max_cores))
                actual_add = new_cores - c.cores
                if actual_add <= 0:
                    continue

                vm = c.vm
                max_add = vm.remaining_cores
                if max_add >= actual_add:
                    c.cores += actual_add
                else:
                    c.cores += max_add
                    rem_c = actual_add - max_add
                    # Spawn helper container with remaining required cores
                    new_cont = SimContainer(len(self.containers), rem_c)
                    self.containers.append(new_cont)
                    self._place_container(new_cont)
        elif delta < 0:
            for c in active_conts:
                if c.cores + delta >= self.min_cores:
                    c.cores += delta
                else:
                    # Enforce min_replicas constraint during deletion
                    if len([cont for cont in self.containers if cont.active]) > self.min_replicas:
                        c.active = False
                        if c in c.vm.containers:
                            c.vm.containers.remove(c)
                        if not any(cont.active for cont in c.vm.containers):
                            c.vm.active = False
                            if c.vm in c.vm.pm.vms:
                                c.vm.pm.vms.remove(c.vm)
                            if not any(vm.active for vm in c.vm.pm.vms):
                                c.vm.pm.active = False
                    else:
                        c.cores = self.min_cores

    def _place_container(self, container):
        # Best-Fit VM Placement
        eligible_vms = [vm for vm in self.vms if vm.active and vm.remaining_cores >= container.cores]
        if eligible_vms:
            best_vm = min(eligible_vms, key=lambda vm: vm.remaining_cores)
            best_vm.containers.append(container)
            container.vm = best_vm
            return

        # Find cheapest eligible VM type
        eligible_types = [t for t in VM_TYPES if t["cores"] >= container.cores]
        selected_type = eligible_types[0] if eligible_types else VM_TYPES[-1]

        # Boot new VM
        new_vm_id = len(self.vms)
        new_vm = SimVM(
            new_vm_id,
            selected_type["type"],
            selected_type["cores"],
            selected_type["price"]
        )
        self.vms.append(new_vm)

        # Place VM on PM using Best-Fit
        eligible_pms = [pm for pm in self.pms if pm.active and pm.remaining_cores >= new_vm.max_cores]
        if eligible_pms:
            best_pm = min(eligible_pms, key=lambda pm: pm.remaining_cores)
            best_pm.vms.append(new_vm)
            new_vm.pm = best_pm
        else:
            # Boot new PM
            new_pm_id = len(self.pms)
            new_pm = SimPM(new_pm_id)
            self.pms.append(new_pm)
            new_pm.vms.append(new_vm)
            new_vm.pm = new_pm

        # Deploy container on VM
        new_vm.containers.append(container)
        container.vm = new_vm

    def _apply_pending_actions(self):
        """Apply scaling actions that have matured past their delay."""
        remaining_h = []
        for action in self._pending_h_actions:
            if self._step >= action["ready_step"]:
                self._apply_horizontal_action(action["delta"])
            else:
                remaining_h.append(action)
        self._pending_h_actions = remaining_h

        remaining_v = []
        for action in self._pending_v_actions:
            if self._step >= action["ready_step"]:
                self._apply_vertical_action(action["delta"])
            else:
                remaining_v.append(action)
        self._pending_v_actions = remaining_v

    def _init_deployment(self):
        self.pms = []
        self.vms = []
        self.containers = []

        # Boot 2 initial PMs
        pm0 = SimPM(0)
        pm1 = SimPM(1)
        self.pms.extend([pm0, pm1])

        # Boot 3 initial VMs of type m5.4xlarge
        vm0 = SimVM(0, "m5.4xlarge", 16, 0.768)
        vm1 = SimVM(1, "m5.4xlarge", 16, 0.768)
        vm2 = SimVM(2, "m5.4xlarge", 16, 0.768)

        pm0.vms.extend([vm0, vm1])
        vm0.pm = pm0
        vm1.pm = pm0

        pm1.vms.append(vm2)
        vm2.pm = pm1

        self.vms.extend([vm0, vm1, vm2])

        # Deploy initial containers
        for i in range(self._replicas):
            c = SimContainer(i, self._cores)
            self.containers.append(c)
            target_vm = self.vms[i % 3]
            target_vm.containers.append(c)
            c.vm = target_vm

    def reset(self):
        """Reset environment to initial state."""
        cloud = self.config["cloud"]
        self._replicas = cloud["min_replicas"]
        self._cores = cloud["min_cores"]
        self._pending_h_actions = []
        self._pending_v_actions = []
        self._queue_depth = 0.0
        self._step = 0
        self.total_cost = 0.0
        self.history = []

        self._init_deployment()
