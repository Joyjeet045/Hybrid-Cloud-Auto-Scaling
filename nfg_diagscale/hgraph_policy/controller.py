"""NF-DiagScale controller for the HGraphScale environment.

This is the closed-loop autoscaling policy that plugs NF-DiagScale (a self-tuning
Neuro-Fuzzy Diagonal Scaler) into the vendored HGraphScale simulator. One scaling
decision is emitted per 3-minute control interval, matching the simulator's
one-action-per-step contract. (The Python package keeps the legacy ``nfg_``
prefix for import stability; the magnitude sizer is a *deterministic* queue-model
enumeration.)

Pipeline (each interval):

1. Forecast (F).  Per microservice, a Kalman+Holt forecaster
   (:class:`ContainerForecaster`) predicts next-interval request count from
   ``workload_his`` (Kalman 1960; Holt 1957). A graph-aware residual corrector
   (:mod:`nfg_diagscale.hgraph_policy.gnn_forecast`, Kipf & Welling 2017) then
   adds a DAG-propagation correction learned from upstream load; with no trained
   weights it is a no-op and the raw Kalman+Holt forecast is used.

2. Fuzzify (F).  Four grounded inputs per microservice:
     psi   = CWRR-weighted batch-drain time / deadline (load pressure; Eq. 15)
     omega = latency slack vs. deadline               (SLO headroom)
     phi   = remaining budget fraction                (cost headroom; Eq. 5/7)
     rho   = binary risk flag (overload / near-SLO)
   The DAG upward rank (HEFT; Topcuoglu 2002) weights pressure so critical-path
   microservices are prioritized -- this is the "spatial dependency" signal.

3. Size (deterministic feedforward).  An exact queue-model sizer
   (:meth:`MagnitudeSizer.solve_exact`) enumerates the bounded
   ``(replicas, vCPU)`` grid and returns the cost-feasible knee ``(h*, c*)`` that
   minimises predicted response time subject to the budget (STAR Eq. 8) -- a
   reproducible, globally-optimal exact solution.

4. Decide (N).  An online self-tuning ANFIS (:class:`ANFISEngine`, Jang 1993;
   adaptive fuzzy control, Wang 1993) blends the deterministic anchor with its
   fuzzy output into ``(mode, delta_c, delta_n)``. Its rule consequents are tuned
   online from the realised SLO/cost outcome of the previous decision (step 6),
   so the fuzzy layer corrects the anchor's systematic model error from feedback.

5. Actuate (Diagonal).  The target vCPU change is applied to the hottest replica.
   The simulator fills vertical headroom first and overflows into a new replica --
   i.e. native vertical-first *diagonal* scaling. A budget guard blocks new-VM
   spawns that would breach the per-day budget, keeping cost violation at zero.

6. Learn (online).  At the next interval, before deciding, the controller reads
   the realised response time of the microservice it last scaled and the realised
   cumulative cost, forms the SLO and budget-pacing errors, and adapts the fired
   ANFIS rules' singleton consequents (:meth:`ANFISEngine.adapt`). The premise
   membership functions stay fixed, keeping the fuzzy partition interpretable.
"""
from __future__ import annotations

import os

import numpy as np

from nfg_diagscale.decision.anfis import ANFISEngine
from nfg_diagscale.hgraph_policy import gnn_forecast, queue_model
from nfg_diagscale.hgraph_policy.forecaster import ContainerForecaster
from nfg_diagscale.hgraph_policy.optimizer import MagnitudeSizer

_DEFAULT_FORECAST_WEIGHTS = os.path.join(os.path.dirname(__file__), "forecast_weights.pt")


class NFGDiagScaleController:
    """Forecast -> fuzzify -> deterministic sizing + adaptive ANFIS -> diagonal -> learn."""

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
        self.sizer = MagnitudeSizer(config)

        cloud = config.get("cloud", {})
        self.max_cores = int(cloud.get("max_cores", 16))
        self.min_cores = int(cloud.get("min_cores", 1))
        self.max_replicas = int(cloud.get("max_replicas", 15))
        self.min_replicas = int(cloud.get("min_replicas", 1))

        ctrl = config.get("controller", {})
        self.idle_pressure = float(ctrl.get("idle_pressure", 0.25))
        self.budget_safety = float(ctrl.get("budget_safety", 0.97))
        self.max_res = int(ctrl.get("max_res", 4))

        crit = config.get("criticality", {})
        self.crit_weight = float(crit.get("weight", 0.0))
        self.crit_alpha = float(crit.get("alpha", 0.4))
        self.crit_beta = float(crit.get("beta", 0.4))
        self.crit_gamma = float(crit.get("gamma", 0.2))

        prop = config.get("propagation", {})
        self.prop_weight = float(prop.get("weight", 0.0))
        self.prop_hops = int(prop.get("hops", 1))

        self.budget_T = float(config.get("cloud", {}).get("budget", 200.0))
        self._forecasters: dict[int, ContainerForecaster] = {}

        # GNN load-forecast residual corrector (default-on; transparent fallback
        # to the raw Kalman+Holt forecast when no trained weights are present).
        fcfg = config.get("forecast", {})
        self._fc_enabled = bool(fcfg.get("gnn_residual", True))
        weights = fcfg.get("gnn_weights", _DEFAULT_FORECAST_WEIGHTS)
        self._fc_model = None
        self._fc_record: list | None = None
        self._fc_cache: dict = {}
        if self._fc_enabled and weights and os.path.exists(weights):
            self._fc_model = gnn_forecast.ForecastGCN.load(weights)

        adp = config.get("adaptive", {})
        self.kappa_slo = float(adp.get("kappa_slo", 1.0))
        self.kappa_cost = float(adp.get("kappa_cost", 0.5))
        self.adapt_beta = float(adp.get("beta", 0.9))
        self.budget_pacing = bool(adp.get("budget_pacing", True))
        self._pending: dict | None = None
        self.learn_trace: list[dict] = []

    def reset(self, budget_T: float, total_intervals: int | None = None) -> None:
        self.budget_T = float(budget_T)
        if total_intervals is not None:
            self.total_intervals = int(total_intervals)
        self._forecasters = {}
        self._fc_cache = {}
        self._pending = None
        self.learn_trace = []
        self.anfis.reset_consequents()

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

    def _learn_from_outcome(self, state) -> None:
        """Adapt the previously-fired ANFIS rules from their realised outcome.

        Called at the start of every interval, *before* the new decision. The
        ``self._pending`` buffer holds the microservice we last scaled and the
        normalised rule firing strengths of that decision. We read the realised
        response time of that microservice and the realised cumulative cost,
        turn them into a normalised SLO error and a budget-pacing error, combine
        them into a single control signal, and push it through
        :meth:`ANFISEngine.adapt`. The reference (deadline + daily budget) is a
        problem constraint, not another controller, so the loop tunes itself
        against the true objective with no surrogate target.
        """
        pend = self._pending
        if pend is None:
            return
        self._pending = None

        replicas = [c for c in state.containers if c.con_type == pend["type"]]
        realized_resp = max((c.aver_resptime for c in replicas), default=0.0)
        if realized_resp <= 0.0:
            return

        e_slo = (realized_resp - self.adapt_beta * self.deadline) / max(self.deadline, 1e-6)
        if self.budget_pacing and self.total_intervals > 0:
            pace = min(1.0, max(0.0, state.slot_index / self.total_intervals))
            e_cost = max(0.0, state.total_cost - self.budget_T * pace) / max(self.budget_T, 1e-6)
        else:
            e_cost = max(0.0, state.total_cost - self.budget_T) / max(self.budget_T, 1e-6)

        signal = self.kappa_slo * e_slo - self.kappa_cost * e_cost
        self.anfis.adapt(pend["firing_strengths"], signal)

        cons = self.anfis.get_consequents()
        self.learn_trace.append({
            "slot": int(state.slot_index),
            "type": int(pend["type"]),
            "realized_resp": float(realized_resp),
            "e_slo": float(e_slo),
            "e_cost": float(e_cost),
            "signal": float(signal),
            "mean_s_dc": float(np.mean(cons["s_dc"])),
            "mean_s_dn": float(np.mean(cons["s_dn"])),
        })

    def act(self, state):
        """Return ``(con_id, vcpu_delta)`` or ``None`` (no-op) for this interval."""
        if not state.containers:
            return None

        self._learn_from_outcome(state)

        by_type: dict[int, list] = {}
        for c in state.containers:
            by_type.setdefault(c.con_type, []).append(c)

        hours_remaining = max(0.0, (self.total_intervals - state.slot_index)) * self.interval_hours
        base_total_vcpu = sum(c.vcpu for c in state.containers)

        best_type = None
        best_score = -1.0
        feats: dict[int, dict] = {}
        for t, replicas in by_type.items():
            cur_h = len(replicas)
            cur_c = max(1, int(round(np.mean([c.vcpu for c in replicas]))))
            type_total_vcpu = float(sum(c.vcpu for c in replicas))
            et = float(state.proc_time.get(t, 0.0))

            observed = self._type_last_load(replicas)
            predicted_lam = self._forecaster(t).update(observed)
            predicted_lam = self._refine_forecast(t, predicted_lam, observed, state, by_type)

            psi = queue_model.load_factor_cwrr(predicted_lam, et, type_total_vcpu, self.deadline)
            max_resp = max((c.aver_resptime for c in replicas), default=0.0)
            lat_risk = max_resp / max(self.deadline, 1e-6)
            pressure = max(psi, lat_risk)
            rank_t = float(state.rank.get(t, 0.0))
            score = (0.5 + 0.5 * rank_t) * pressure
            if self.crit_weight > 0.0:
                score = self._criticality_score(score, psi, rank_t, lat_risk, pressure)

            feats[t] = {
                "cur_h": cur_h, "cur_c": cur_c, "et": et, "lam": predicted_lam,
                "psi": psi, "lat_risk": lat_risk, "pressure": pressure,
                "max_resp": max_resp, "replicas": replicas,
                "type_total_vcpu": type_total_vcpu, "rank": rank_t, "score": score,
            }
            if score > best_score:
                best_score = score
                best_type = t

        if best_type is None:
            return None

        best_type = self._select_bottleneck(state, feats, best_type)

        if self.prop_weight > 0.0:
            best_type = self._propagate_and_select(state, feats)

        f = feats[best_type]
        if f["pressure"] < self.idle_pressure:
            return None

        cur_h, cur_c, et, lam = f["cur_h"], f["cur_c"], f["et"], f["lam"]
        psi = f["psi"]
        omega = float(np.clip((self.deadline - f["max_resp"]) / self.deadline, 0.0, 1.0))
        phi = float(np.clip((self.budget_T - state.total_cost) / max(self.budget_T, 1e-6), 0.0, 1.0))
        rho = 1.0 if (psi >= 1.0 or f["lat_risk"] >= 0.8) else 0.0

        remaining_budget = self.budget_T - state.total_cost
        existing_vm_future = state.num_vms * self.vm_price * hours_remaining
        budget_room = max(0.0, remaining_budget * self.budget_safety - existing_vm_future)

        other_vcpu = base_total_vcpu - f["type_total_vcpu"]
        h_star, c_star, _front = self.sizer.solve_exact(
            cur_h, lam, et,
            base_total_vcpu=base_total_vcpu, other_vcpu=other_vcpu,
            vm_size=self.vm_size, vm_price_per_hr=self.vm_price,
            hours_remaining=hours_remaining, deadline=self.deadline,
            penalty=self.penalty, budget_room=budget_room,
        )

        decision = self.anfis.decide(
            psi=psi, omega=omega, phi=phi, rho=rho,
            n_current=cur_h, cores_current=cur_c,
            corrective=(h_star, c_star),
        )
        self._pending = {"type": best_type,
                         "firing_strengths": decision["firing_strengths"]}

        new_cores = cur_c + decision["delta_c"]
        new_replicas = cur_h + decision["delta_n"]
        target_total = new_replicas * new_cores
        delta_total = int(round(target_total - cur_h * cur_c))
        delta_total = int(np.clip(delta_total, -self.max_res, self.max_res))
        if delta_total == 0:
            return None

        if delta_total > 0:
            return self._scale_up(f["replicas"], delta_total, cur_c, budget_room, hours_remaining)
        return self._scale_down(f["replicas"], delta_total)

    def _select_bottleneck(self, state, feats, default_type):
        """Bottleneck-selection extension point.

        The baseline returns the analytic critical-path argmax unchanged. Pluggable
        ablations (see the ``ablations`` package) override this to inject an
        alternative selector without modifying the core control loop.
        """
        return default_type

    def record_on(self):
        """Start buffering per-interval forecast records for offline labelling.

        Used by ``train_forecast.py`` to collect free residual labels from a
        baseline rollout (construct the controller with ``forecast.gnn_residual``
        off so the recorded base is the raw Kalman+Holt forecast).
        """
        self._fc_record = []

    def pop_records(self):
        """Return and clear the buffered forecast records."""
        rec = self._fc_record if self._fc_record is not None else []
        self._fc_record = None
        return rec

    def _build_interval_graph(self, state, by_type):
        types = list(by_type.keys())
        obs = {s: self._type_last_load(by_type[s]) for s in types}
        et = {s: float(state.proc_time.get(s, 0.0)) for s in types}
        vcpu = {s: float(sum(c.vcpu for c in by_type[s])) for s in types}
        rank = {s: float(state.rank.get(s, 0.0)) for s in types}
        return gnn_forecast.build_forecast_inputs(
            types, obs, et, vcpu, rank, state.succ, self.deadline)

    def _refine_forecast(self, con_type, predicted_lam, observed, state, by_type):
        """Graph-aware residual correction on the per-type load forecast.

        Adds the trained GCN's DAG-propagation residual to the Kalman+Holt
        forecast (Kipf & Welling 2017 propagation on an upstream adjacency). With
        no trained weights and no recording active this returns the baseline
        forecast unchanged, so the controller degrades gracefully to Kalman+Holt.
        When :meth:`record_on` is active it also buffers ``(graph, base, obs)``
        per interval for offline label construction.
        """
        if self._fc_model is None and self._fc_record is None:
            return predicted_lam

        slot = int(state.slot_index)
        cache = self._fc_cache
        if cache.get("slot") != slot:
            order, X, A_hat = self._build_interval_graph(state, by_type)
            cache.clear()
            cache["slot"] = slot
            cache["order"] = order
            cache["idx"] = {t: i for i, t in enumerate(order)}
            cache["X"] = X
            cache["A"] = A_hat
            if self._fc_model is not None and len(order) > 0:
                cache["resid"] = self._fc_model.predict(X, A_hat)
            if self._fc_record is not None:
                rec = {"order": order, "X": X, "A": A_hat, "base": {}, "obs": {}}
                self._fc_record.append(rec)
                cache["rec"] = rec

        if self._fc_record is not None and "rec" in cache:
            cache["rec"]["base"][int(con_type)] = float(predicted_lam)
            cache["rec"]["obs"][int(con_type)] = float(observed)

        if self._fc_model is not None:
            i = cache["idx"].get(con_type)
            if i is not None:
                predicted_lam = max(0.0, float(predicted_lam) + float(cache["resid"][i]))
        return predicted_lam

    def _criticality_score(self, base_score, psi, rank_t, lat_risk, pressure):
        """Blend the legacy bottleneck score with a criticality priority (Rec 3).

        ``CS_i = alpha*C_i + beta*D_i + gamma*L_i`` with C = load demand (psi),
        D = DAG centrality (HEFT upward rank), L = latency contribution; the
        selection priority is ``Priority_i = CS_i * Risk_i`` with Risk = pressure.
        The blend weight is ``crit_weight``; at ``crit_weight == 0`` this is never
        called, so the legacy score is used verbatim.
        """
        c_i, d_i, l_i = psi, rank_t, lat_risk
        cs = self.crit_alpha * c_i + self.crit_beta * d_i + self.crit_gamma * l_i
        priority = cs * pressure
        return (1.0 - self.crit_weight) * base_score + self.crit_weight * priority

    def _propagate_and_select(self, state, feats):
        """Re-select the bottleneck after upstream root-cause propagation (Rec 4).

        Each service inherits a fraction (``prop_weight``) of the pressure of its
        downstream DAG successors, so an upstream service that causes a downstream
        SLA violation is scaled instead of the symptom. ``prop_hops`` controls how
        many DAG levels are followed; contributions decay as ``1/depth`` so deeper
        successors count less. At ``prop_hops == 1`` this is exact single-hop
        inheritance. Only invoked when ``prop_weight > 0``; the baseline selection
        is left untouched otherwise.
        """
        succ = getattr(state, "succ", {}) or {}
        hops = self.prop_hops if self.prop_hops > 0 else 1
        best_t, best_s = None, -1.0
        for t, f in feats.items():
            propagated = 0.0
            seen = {t}
            frontier = [t]
            for depth in range(1, hops + 1):
                nxt = []
                for u in frontier:
                    for s in succ.get(u, []):
                        if s in seen:
                            continue
                        seen.add(s)
                        nxt.append(s)
                        sf = feats.get(s)
                        if sf is not None:
                            propagated += sf["pressure"] / depth
                if not nxt:
                    break
                frontier = nxt
            adj = f["score"] + self.prop_weight * propagated
            if adj > best_s:
                best_s = adj
                best_t = t
        if best_t is None:
            return max(feats, key=lambda k: feats[k]["score"])
        return best_t

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
            overflow = min(delta_total - headroom, max(self.min_cores, int(replica_vcpu)))
            new_vm_future_cost = self.vm_price * hours_remaining
            if budget_room < new_vm_future_cost:
                delta_total = headroom
            else:
                delta_total = headroom + overflow
        if delta_total <= 0:
            return None
        return (target.con_id, int(delta_total))

    def _scale_down(self, replicas, delta_total):
        """Apply a negative vCPU delta to the coldest replica (scale-in)."""
        target = min(replicas, key=lambda c: (c.aver_resptime, c.qlen))
        room_down = -(int(target.vcpu) - self.min_cores)
        if len(replicas) > 1:
            room_down = -int(target.vcpu)
        delta_total = max(delta_total, room_down)
        if delta_total >= 0:
            return None
        return (target.con_id, int(delta_total))
