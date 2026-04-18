"""MAPE-K orchestrator implementing the NFG-DiagScale control loop."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from nfg_diagscale.monitoring.metrics_collector import MetricsCollector
from nfg_diagscale.monitoring.heat_accumulator import HeatAccumulator
from nfg_diagscale.decision.themis_latency import ThemisLatencyModel
from nfg_diagscale.decision.anfis import ANFISEngine
from nfg_diagscale.optimizer.nsga2 import NSGA2Optimizer
from nfg_diagscale.simulation.cloud_env import CloudEnvironment


def _compute_initial_state(config, initial_rps):
    """Compute consistent initial replicas and cores for a given RPS load.

    Shared by NFG-DiagScale and BaselineRunner so all autoscalers start
    from exactly the same resource allocation — essential for fair comparison.

    Strategy: target ~70% initial utilization with MODERATE core allocation.
    This prevents horizontal-only scalers (HPA) from coasting on max cores,
    and forces all autoscalers to actively manage resources.
    """
    cloud = config["cloud"]
    pod_max_rps = cloud["pod_max_rps"]

    # Use moderate cores: half of max, capped at 8.
    # This ensures HPA (horizontal-only) must actually scale replicas,
    # and VPA (vertical-only) has room to scale cores up.
    initial_cores = max(cloud["min_cores"], min(cloud["max_cores"] // 2, 8))

    # Compute replicas needed to handle initial load at ~95% utilization.
    # Tight provisioning forces all autoscalers to actively adapt —
    # critical for VPA which can't shed excess replicas.
    target_util = 0.95
    needed_capacity = initial_rps / target_util
    capacity_per_pod = initial_cores * pod_max_rps
    initial_pods = max(1, int(np.ceil(needed_capacity / max(capacity_per_pod, 1))))
    replicas = int(np.clip(
        initial_pods, cloud["min_replicas"], cloud["max_replicas"]
    ))

    return replicas, initial_cores


class NFGDiagScaleOrchestrator:
    def __init__(self, config, predictor):
        self.config = config
        self.predictor = predictor

        # Monitor: multi-level metrics
        self.metrics_collector = MetricsCollector(config)

        # Heat-based oscillation suppression
        self.heat_acc = HeatAccumulator(config)

        # Themis latency model for SLO risk
        self.themis = ThemisLatencyModel(config)

        # ANFIS decision engine
        self.anfis = ANFISEngine(config)

        # NSGA-II global optimizer
        self.nsga2 = NSGA2Optimizer(config)

        self.ga_run_interval = config["nsga2"]["run_interval_steps"]
        self.slo = config["themis"]["slo_ms"]
        self.pod_max_rps = config["cloud"]["pod_max_rps"]
        self.batch_size = config["themis"]["batch_size"]

        self.ga_checkpoint = None
        self.name = "NFG-DiagScale"

        # Cooldown: prevent action stacking while pending actions mature.
        # Minimum steps between consecutive scaling decisions.
        self._cooldown_remaining = 0
        self._cooldown_steps = config.get("mape_k", {}).get("cooldown_steps", 5)

        # SLO emergency bypass threshold (fraction of SLO)
        self._slo_emergency_frac = config.get("mape_k", {}).get(
            "slo_emergency_fraction", 0.92
        )

        # Proactive scaling thresholds from config
        proactive = config.get("proactive", {})
        self._cap_ratio_up = proactive.get("capacity_ratio_up", 0.70)
        self._cap_ratio_down = proactive.get("capacity_ratio_down", 0.45)
        self._lat_pressure_up = proactive.get("latency_pressure_up", 0.75)

        # ANFIS online learning update interval
        self._anfis_update_interval = config.get("anfis", {}).get(
            "update_interval", 20
        )

    def run_evaluation(self, test_df):
        """
        Run the full MAPE-K loop on a test trace.
        Returns history and action log for KPI evaluation.
        """
        env = CloudEnvironment(self.config)
        
        # Warm-up/Initialization: Right-size to initial load with generous
        # headroom to avoid SLO violations during cold-start stabilization.
        initial_rps = float(test_df["y"].iloc[0])
        env.replicas, env.cores = _compute_initial_state(
            self.config, initial_rps
        )
        action_log = []

        rps_values = test_df["y"].values
        n_steps = len(rps_values)

        lookback = self.config["lstm"]["lookback_window"]
        ds_values = test_df["ds"].values if "ds" in test_df.columns else None

        for step in tqdm(range(n_steps), desc="NFG-DiagScale", leave=False):
            actual_rps = float(rps_values[step])

            # ── EXECUTE environment step ──
            state = env.step(actual_rps)

            # MONITOR
            metrics = self.metrics_collector.collect_from_state(state)
            sigma_stress = self.metrics_collector.compute_stress(metrics)

            # ANALYZE
            violation = self.metrics_collector.detect_violation(sigma_stress)

            # EMERGENCY SLO BYPASS: force immediate action when latency
            # is critically close to the SLO threshold.
            force_action = False
            slo_emergency = state["app_latency"] > self.slo * self._slo_emergency_frac
            if slo_emergency:
                self._cooldown_remaining = 0
                force_action = True

            # Enforce cooldown to prevent action-on-action thrashing.
            # Heat is NOT updated during cooldown so the system observes
            # the effect of the last scaling action before deciding again.
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                action_log.append({"step": step, "mode": "none", "delta_c": 0, "delta_n": 0})
                continue

            # Kalman filter and Prophet-LSTM prediction
            row_df = None
            if ds_values is not None:
                row_df = pd.DataFrame({"ds": [ds_values[step]], "y": [actual_rps]})
            pred = self.predictor.predict_next(actual_rps, row_df)
            lambda_hat = pred["lambda_hat"]
            lambda_kf = pred["lambda_kf"]

            # Proactive signal uses capacity-based ratio
            current_capacity = env.replicas * env.cores * self.pod_max_rps
            capacity_ratio = lambda_hat / max(current_capacity, 1.0)
            latency_pressure = state["app_latency"] / self.slo

            # ── SCALE-UP path: requires sustained heat evidence ──
            # Translate forecast into a virtual violation for the heat accumulator
            if (capacity_ratio > self._cap_ratio_up) or (latency_pressure > self._lat_pressure_up):
                proactive_violation = "UP"
            else:
                proactive_violation = "NONE"

            # Merge reactive and proactive for scale-UP only
            if violation == "UP" or proactive_violation == "UP":
                combined_signal = "UP"
            else:
                combined_signal = "NONE"

            self.heat_acc.update(combined_signal)

            # ── SCALE-DOWN path: direct capacity check, bypasses heat ──
            # Scale-down fires when BOTH conditions hold simultaneously:
            #   1) Predicted load uses < cap_ratio_down of current capacity
            #   2) Latency is comfortably below SLO (ample headroom)
            force_scale_down = (
                capacity_ratio < self._cap_ratio_down
                and latency_pressure < 0.5
                and not force_action
            )

            # Decide whether to act this step
            heat_triggered = self.heat_acc.should_trigger()
            if not force_action and not heat_triggered and not force_scale_down:
                action_log.append({"step": step, "mode": "none", "delta_c": 0, "delta_n": 0})
                continue

            # Reset heat only after we decide to act (scale-up path)
            if heat_triggered or force_action:
                self.heat_acc.reset()

            if force_scale_down and not force_action and not heat_triggered:
                # ── DIRECT RIGHT-SIZING (bypasses ANFIS) ──
                # Find the cheapest (replicas, cores) that handles predicted load
                # with safety headroom. Vertical-first: start from min replicas.
                target_util = 0.75
                needed_capacity = lambda_hat / target_util

                # Search for cheapest feasible config (replicas-first ascending)
                best_cost = float("inf")
                target_replicas = env.replicas
                target_cores = env.cores
                sp = self.config["scaling_plane"]
                ram = self.config["cloud"]["ram_gb"]

                for h in range(self.config["cloud"]["min_replicas"],
                               env.replicas + 1):  # Only search AT or BELOW current
                    # Min cores needed for this replica count
                    c_needed = max(
                        self.config["cloud"]["min_cores"],
                        int(np.ceil(needed_capacity / max(h * self.pod_max_rps, 1)))
                    )
                    if c_needed > self.config["cloud"]["max_cores"]:
                        continue  # Not feasible at this replica count

                    # Check SLO feasibility via Themis model
                    lat = self.themis.total_latency(
                        self.batch_size, c_needed, lambda_hat, h
                    )
                    if lat > self.slo * 0.80:
                        continue  # Too risky

                    # Compute cost
                    cost = h * (sp["cost_per_core"] * c_needed
                                + sp["cost_per_gb_ram"] * ram
                                + sp["cost_per_replica"])
                    if cost < best_cost:
                        best_cost = cost
                        target_replicas = h
                        target_cores = c_needed

                # Move toward target gradually (max 2 steps at a time)
                delta_c = int(np.clip(target_cores - env.cores, -2, 2))
                delta_n = int(np.clip(target_replicas - env.replicas, -2, 0))

                # Final safety: never scale down if projected latency is risky
                if delta_c != 0 or delta_n != 0:
                    new_c = env.cores + delta_c
                    new_n = env.replicas + delta_n
                    projected_lat = self.themis.total_latency(
                        self.batch_size, new_c, lambda_hat, new_n
                    )
                    if projected_lat > self.slo * 0.85:
                        delta_c = 0
                        delta_n = 0
            else:
                # ── ANFIS DECISION (scale-up path) ──
                # ANFIS input variables
                psi = min(lambda_hat / max(lambda_kf, 1.0), 3.0)
                actual_latency = state["app_latency"]
                omega = (self.slo - actual_latency) / self.slo
                phi = 1.0 - env.cores / self.config["cloud"]["max_cores"]
                rho = self.themis.slo_risk(
                    self.batch_size, env.cores, lambda_hat, env.replicas
                )

                # Periodic NSGA-II optimization
                if step > 0 and step % self.ga_run_interval == 0:
                    single_pod_capacity = env.cores * self.pod_max_rps
                    low_load_mode = lambda_hat < single_pod_capacity
                    pareto = self.nsga2.optimize(env.replicas, env.cores, lambda_hat, low_load_mode)
                    if pareto:
                        self.ga_checkpoint = self.nsga2.get_nearest_checkpoint(
                            env.replicas, env.cores
                        )

                # ANFIS decision
                decision = self.anfis.decide(
                    psi=psi, omega=omega, phi=phi, rho=rho,
                    n_current=env.replicas, cores_current=env.cores,
                    predicted_rps=lambda_hat, ga_checkpoint=self.ga_checkpoint,
                )
                self.ga_checkpoint = None

                delta_c = decision["delta_c"]
                delta_n = decision["delta_n"]

                # Online learning (only for ANFIS path)
                self.anfis.record_outcome(
                    inputs={"psi": psi, "omega": omega, "phi": phi, "rho": rho},
                    action=decision,
                    latency_observed=state["app_latency"],
                    cost_observed=state["step_cost"],
                )
                if step > 0 and step % self._anfis_update_interval == 0:
                    self.anfis.update_parameters()

            # Derive mode from final deltas
            if delta_c != 0 and delta_n != 0:
                mode = "diagonal"
            elif delta_c != 0:
                mode = "vertical"
            elif delta_n != 0:
                mode = "horizontal"
            else:
                mode = "none"

            if mode != "none":
                env.execute_scaling(mode, delta_c, delta_n)
                # Activate cooldown to let the scaling action mature
                self._cooldown_remaining = self._cooldown_steps

            action_record = {
                "step": step,
                "mode": mode,
                "delta_c": delta_c,
                "delta_n": delta_n,
                "lambda_hat": lambda_hat,
            }
            action_log.append(action_record)

        return env.history, action_log


class BaselineRunner:
    """Run a baseline autoscaler on the same trace for comparison."""

    def __init__(self, config, baseline):
        self.config = config
        self.baseline = baseline
        self.name = baseline.name

    def run_evaluation(self, test_df):
        env = CloudEnvironment(self.config)
        
        # Consistent initialization with NFG-DiagScale (shared function)
        initial_rps = float(test_df["y"].iloc[0])
        env.replicas, env.cores = _compute_initial_state(
            self.config, initial_rps
        )
        action_log = []
        rps_values = test_df["y"].values

        for step in tqdm(range(len(rps_values)), desc=self.name, leave=False):
            actual_rps = float(rps_values[step])
            state = env.step(actual_rps)

            decision = self.baseline.decide(state, step)

            mode = decision["mode"]
            delta_c = decision.get("delta_c", 0)
            delta_n = decision.get("delta_n", 0)

            if mode != "none" and (delta_c != 0 or delta_n != 0):
                env.execute_scaling(mode, delta_c, delta_n)

            action_log.append({
                "step": step,
                "mode": mode,
                "delta_c": delta_c,
                "delta_n": delta_n,
            })

        return env.history, action_log
