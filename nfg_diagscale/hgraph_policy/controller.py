"""NFG-DiagScale controller for the HGraphScale environment.

This is the closed-loop autoscaling policy that plugs NFG-DiagScale (Neuro-Fuzzy-
Genetic Diagonal Scaling) into the vendored HGraphScale simulator. One scaling
decision is emitted per 3-minute control interval, matching the simulator's
one-action-per-step contract.

Pipeline (each interval):

1. Forecast (F).  Per microservice, a Kalman+Holt forecaster
   (:class:`ContainerForecaster`) predicts next-interval request count from
   ``workload_his`` (Kalman 1960; Holt 1957).

2. Fuzzify (F).  Four grounded inputs per microservice:
     psi   = predicted batch-drain time / deadline   (load pressure; queue_model)
     omega = latency slack vs. deadline               (SLO headroom)
     phi   = remaining budget fraction                (cost headroom; Eq. 5/7)
     rho   = binary risk flag (overload / near-SLO)
   The DAG upward rank (HEFT; Topcuoglu 2002) weights pressure so critical-path
   microservices are prioritized — this is the "spatial dependency" signal.

3. Optimize (G).  NSGA-II (:class:`MagnitudeNSGA2`, Deb 2002) searches the
   bottleneck's ``(replicas, vCPU)`` allocation for the Pareto trade-off between
   predicted response time and marginal VM cost, returning a knee checkpoint.

4. Decide (N).  ANFIS (:class:`ANFISEngine`, Jang 1993) defuzzifies the fuzzy
   inputs, biased toward the NSGA-II checkpoint, into ``(mode, delta_c, delta_n)``.

5. Actuate (Diagonal).  The target vCPU change is applied to the hottest replica.
   The simulator fills vertical headroom first and overflows into a new replica —
   i.e. native vertical-first *diagonal* scaling. A budget guard blocks new-VM
   spawns that would breach the per-day budget, keeping cost violation at zero.
"""
from __future__ import annotations

import numpy as np

from nfg_diagscale.decision.anfis import ANFISEngine
from nfg_diagscale.hgraph_policy import queue_model
from nfg_diagscale.hgraph_policy.forecaster import ContainerForecaster
from nfg_diagscale.hgraph_policy.optimizer import MagnitudeNSGA2


class NFGDiagScaleController:
    """Forecast -> fuzzify -> NSGA-II -> ANFIS -> diagonal actuation."""

    def __init__(self, config, *, deadline: float = 500.0, vm_size: float = 16.0,
                 vm_price_per_hr: float = 0.768, interval_minutes: float = 3.0,
                 total_intervals: int = 485, penalty: float = 100.0):
        self.config = config
        self.deadline = float(deadline)
        self.vm_size = float(vm_size)
        self.vm_price = float(vm_price_per_hr)
        self.interval_hours = float(interval_minutes) / 60.0
        self.total_intervals = int(total_intervals)
        self.penalty = float(penalty)

        self.anfis = ANFISEngine(config)
        self.optimizer = MagnitudeNSGA2(config)

        cloud = config.get("cloud", {})
        self.max_cores = int(cloud.get("max_cores", 16))
        self.min_cores = int(cloud.get("min_cores", 1))
        self.max_replicas = int(cloud.get("max_replicas", 15))
        self.min_replicas = int(cloud.get("min_replicas", 1))

        ctrl = config.get("controller", {})
        # Target utilization for the optimizer's sizing heuristic.
        self.target_util = float(ctrl.get("target_util", 0.5))
        # Below this criticality, suppress actions (fuzzy stability deadzone).
        self.idle_pressure = float(ctrl.get("idle_pressure", 0.25))
        # Safety fraction of the budget we are willing to commit to VMs.
        self.budget_safety = float(ctrl.get("budget_safety", 0.97))
        # Per-decision resource-change cap, matching STAR's Scale Generator, which
        # emits res in [-m, +m] with m=4 (Fang et al., 2026). Caps |delta_total|
        # so our action space is identical to the baseline's for a fair comparison.
        self.max_res = int(ctrl.get("max_res", 4))

        self.budget_T = float(config.get("cloud", {}).get("budget", 200.0))
        self._forecasters: dict[int, ContainerForecaster] = {}

    # ------------------------------------------------------------------ #
    def reset(self, budget_T: float, total_intervals: int | None = None) -> None:
        self.budget_T = float(budget_T)
        if total_intervals is not None:
            self.total_intervals = int(total_intervals)
        self._forecasters = {}

    # ------------------------------------------------------------------ #
    def _forecaster(self, con_type: int) -> ContainerForecaster:
        fc = self._forecasters.get(con_type)
        if fc is None:
            fc = ContainerForecaster(self.config)
            self._forecasters[con_type] = fc
        return fc

    @staticmethod
    def _type_last_load(replicas) -> float:
        """Total tasks served by a microservice in the last interval."""
        total = 0.0
        for c in replicas:
            if c.workload_his is not None and len(c.workload_his) > 0:
                total += float(c.workload_his[-1])
        return total

    # ------------------------------------------------------------------ #
    def act(self, state):
        """Return ``(con_id, vcpu_delta)`` or ``None`` (no-op) for this interval."""
        if not state.containers:
            return None

        by_type: dict[int, list] = {}
        for c in state.containers:
            by_type.setdefault(c.con_type, []).append(c)

        hours_remaining = max(0.0, (self.total_intervals - state.slot_index)) * self.interval_hours
        base_total_vcpu = sum(c.vcpu for c in state.containers)

        # ---- 1-2. Forecast + fuzzify, pick the critical-path bottleneck ---- #
        best_type = None
        best_score = -1.0
        feats: dict[int, dict] = {}
        for t, replicas in by_type.items():
            cur_h = len(replicas)
            cur_c = max(1, int(round(np.mean([c.vcpu for c in replicas]))))
            et = float(state.proc_time.get(t, 0.0))

            observed = self._type_last_load(replicas)
            predicted_lam = self._forecaster(t).update(observed)

            psi = queue_model.load_factor(predicted_lam, et, cur_c, cur_h, self.deadline)
            max_resp = max((c.aver_resptime for c in replicas), default=0.0)
            lat_risk = max_resp / max(self.deadline, 1e-6)
            pressure = max(psi, lat_risk)
            rank_t = float(state.rank.get(t, 0.0))
            score = (0.5 + 0.5 * rank_t) * pressure

            feats[t] = {
                "cur_h": cur_h, "cur_c": cur_c, "et": et, "lam": predicted_lam,
                "psi": psi, "lat_risk": lat_risk, "pressure": pressure,
                "max_resp": max_resp, "replicas": replicas,
            }
            if score > best_score:
                best_score = score
                best_type = t

        if best_type is None:
            return None

        f = feats[best_type]
        # If nothing is under pressure, stay put (avoid cost-raising churn).
        if f["pressure"] < self.idle_pressure:
            return None

        cur_h, cur_c, et, lam = f["cur_h"], f["cur_c"], f["et"], f["lam"]
        psi = f["psi"]
        omega = float(np.clip((self.deadline - f["max_resp"]) / self.deadline, 0.0, 1.0))
        phi = float(np.clip((self.budget_T - state.total_cost) / max(self.budget_T, 1e-6), 0.0, 1.0))
        rho = 1.0 if (psi >= 1.0 or f["lat_risk"] >= 0.8) else 0.0

        # ---- 3. NSGA-II magnitude optimization (Genetic) ------------------ #
        type_vcpu = sum(c.vcpu for c in f["replicas"])
        other_vcpu = base_total_vcpu - type_vcpu
        remaining_budget = self.budget_T - state.total_cost
        existing_vm_future = state.num_vms * self.vm_price * hours_remaining
        budget_room = max(0.0, remaining_budget * self.budget_safety - existing_vm_future)

        h_star, c_star, _front = self.optimizer.optimize(
            cur_h, cur_c, lam, et,
            base_total_vcpu=base_total_vcpu, other_vcpu=other_vcpu,
            vm_size=self.vm_size, vm_price_per_hr=self.vm_price,
            hours_remaining=hours_remaining, deadline=self.deadline,
            penalty=self.penalty, budget_room=budget_room,
        )

        # ---- 4. ANFIS decision (Neuro-Fuzzy), biased by the checkpoint ---- #
        decision = self.anfis.decide(
            psi=psi, omega=omega, phi=phi, rho=rho,
            n_current=cur_h, cores_current=cur_c, predicted_rps=lam,
            ga_checkpoint=(h_star, c_star),
        )
        new_cores = cur_c + decision["delta_c"]
        new_replicas = cur_h + decision["delta_n"]
        target_total = new_replicas * new_cores
        delta_total = int(round(target_total - cur_h * cur_c))
        # Fairness cap: clamp the per-decision resource change to STAR's action
        # range res in [-m, +m] (m=4) so the comparison isolates the policy, not a
        # wider action space (Fang et al., 2026).
        delta_total = int(np.clip(delta_total, -self.max_res, self.max_res))
        if delta_total == 0:
            return None

        # ---- 5. Diagonal actuation + budget guard ------------------------- #
        if delta_total > 0:
            return self._scale_up(f["replicas"], delta_total, cur_c, budget_room, hours_remaining)
        return self._scale_down(f["replicas"], delta_total)

    # ------------------------------------------------------------------ #
    def _scale_up(self, replicas, delta_total, replica_vcpu, budget_room, hours_remaining):
        """Apply a positive vCPU delta to the hottest replica (diagonal scaling).

        The simulator fills the host VM's free vCPU first (vertical, no new cost)
        and overflows the remainder into one new replica (horizontal). We cap the
        overflow to one replica of ``replica_vcpu`` so a single action never rents
        an oversized VM, and block the overflow entirely when the budget cannot
        absorb a new VM for the rest of the horizon (keeps cost violation at 0).
        """
        target = max(replicas, key=lambda c: (c.aver_resptime, c.qlen))
        headroom = int(target.max_scal_vcpu)
        if delta_total > headroom:
            # Overflow spawns at most one replica sized like the current ones.
            overflow = min(delta_total - headroom, max(self.min_cores, int(replica_vcpu)))
            new_vm_future_cost = self.vm_price * hours_remaining
            if budget_room < new_vm_future_cost:
                # Cannot afford a new VM: take the free vertical headroom only.
                delta_total = headroom
            else:
                delta_total = headroom + overflow
        if delta_total <= 0:
            return None
        return (target.con_id, int(delta_total))

    def _scale_down(self, replicas, delta_total):
        """Apply a negative vCPU delta to the coldest replica (scale-in)."""
        target = min(replicas, key=lambda c: (c.aver_resptime, c.qlen))
        # Never scale a single-vCPU container below its floor with no benefit.
        room_down = -(int(target.vcpu) - self.min_cores)
        if len(replicas) > 1:
            # Allow removing an idle replica entirely.
            room_down = -int(target.vcpu)
        delta_total = max(delta_total, room_down)
        if delta_total >= 0:
            return None
        return (target.con_id, int(delta_total))
