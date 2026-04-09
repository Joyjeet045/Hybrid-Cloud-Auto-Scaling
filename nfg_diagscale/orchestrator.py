"""
MAPE-K orchestrator implementing the NFG-DiagScale control loop.

[P4] Solino, Batista & Cavalcante (2025), ACM UCC'25, Section 2:
  "MAPE-K operates in four phases in a continuous feedback cycle:
   (1) Monitor: continuously observes the system
   (2) Analyze: evaluates monitored data to determine if adaptation is required
   (3) Plan: builds an action plan
   (4) Execute: applies the planned changes"

Full algorithm pseudocode from proposal Section 8:
  MONITOR  -> collect metrics at 3 levels [P4]
  ANALYZE  -> heat check [P4 Alg 3], Kalman [P2], Prophet-LSTM [P5], Themis [P1]
  PLAN     -> ANFIS decision [Jang93], NSGA-II checkpoint [P3+GA]
  EXECUTE  -> scaling action [P1 vertical, P3 diagonal, P5 horizontal]
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from nfg_diagscale.monitoring.metrics_collector import MetricsCollector
from nfg_diagscale.monitoring.heat_accumulator import HeatAccumulator
from nfg_diagscale.decision.themis_latency import ThemisLatencyModel
from nfg_diagscale.decision.anfis import ANFISEngine
from nfg_diagscale.optimizer.nsga2 import NSGA2Optimizer
from nfg_diagscale.simulation.cloud_env import CloudEnvironment


class NFGDiagScaleOrchestrator:
    def __init__(self, config, predictor):
        self.config = config
        self.predictor = predictor

        # [P4 sect 2.1] Monitor: multi-level metrics
        self.metrics_collector = MetricsCollector(config)

        # [P4 sect 2.2, Alg 3] Heat-based oscillation suppression
        self.heat_acc = HeatAccumulator(config)

        # [P1] Themis latency model for SLO risk
        self.themis = ThemisLatencyModel(config)

        # [Jang93] ANFIS decision engine
        self.anfis = ANFISEngine(config)

        # [P3+GA] NSGA-II global optimizer
        self.nsga2 = NSGA2Optimizer(config)

        self.ga_run_interval = config["nsga2"]["run_interval_steps"]
        self.slo = config["themis"]["slo_ms"]
        self.pod_max_rps = config["cloud"]["pod_max_rps"]
        self.batch_size = config["themis"]["batch_size"]

        self.ga_checkpoint = None
        self.name = "NFG-DiagScale"

        # [P4 Alg 4] Cooldown: prevent action stacking while pending actions mature.
        # Minimum steps between consecutive scaling decisions.
        self._cooldown_remaining = 0
        self._cooldown_steps = config.get("mape_k", {}).get("cooldown_steps", 5)

    def run_evaluation(self, test_df):
        """
        Run the full MAPE-K loop on a test trace.
        Returns history and action log for KPI evaluation.
        """
        env = CloudEnvironment(self.config)
        action_log = []

        rps_values = test_df["y"].values
        n_steps = len(rps_values)

        lookback = self.config["lstm"]["lookback_window"]
        ds_values = test_df["ds"].values if "ds" in test_df.columns else None

        for step in tqdm(range(n_steps), desc="NFG-DiagScale", leave=False):
            actual_rps = float(rps_values[step])

            # ── EXECUTE environment step ──
            state = env.step(actual_rps)

            # ── MONITOR [P4 sect 2.1] ──
            metrics = self.metrics_collector.collect_from_state(state)
            sigma_stress = self.metrics_collector.compute_stress(metrics)

            # ── ANALYZE [P4 sect 2.2] ──
            violation = self.metrics_collector.detect_violation(sigma_stress)

            # [P4 Alg 3] Heat accumulator check
            self.heat_acc.update(violation)

            # [P4 Alg 4] Enforce cooldown to prevent action-on-action thrashing
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                action_log.append({"step": step, "mode": "none", "delta_c": 0, "delta_n": 0})
                continue

            # [P2 sect 3.3] Kalman filter + [P5 sect 3.1] Prophet-LSTM prediction
            row_df = None
            if ds_values is not None:
                row_df = pd.DataFrame({"ds": [ds_values[step]], "y": [actual_rps]})
            pred = self.predictor.predict_next(actual_rps, row_df)
            lambda_hat = pred["lambda_hat"]
            lambda_kf = pred["lambda_kf"]
            
            current_capacity = env.replicas * env.cores * self.config["cloud"]["pod_max_rps"]
            predicted_psi = lambda_hat / max(current_capacity, 1.0)
            
            proactive_trigger = (predicted_psi > 0.8) or (predicted_psi < 0.4)

            if not self.heat_acc.should_trigger() and not proactive_trigger:
                action_log.append({"step": step, "mode": "none", "delta_c": 0, "delta_n": 0})
                continue
            self.heat_acc.reset()

            # [P1 Eq. 1, Eq. 5] Themis SLO risk from predicted workload
            rho = self.themis.slo_risk(
                self.batch_size, env.cores, lambda_hat, env.replicas
            )

            # Use ACTUAL observed latency (includes congestion) for headroom
            # This is more realistic than the base Themis model alone
            actual_latency = state["app_latency"]
            omega = (self.slo - actual_latency) / self.slo

            # If actual latency already violates SLO, override rho
            if actual_latency > self.slo:
                rho = 1.0

            # ANFIS input variables
            # psi = predicted utilization ratio (demand / current capacity)
            # Clamped to [0.0, 3.0]: values above 3.0 are all treated as critical.
            current_capacity = env.replicas * env.cores * self.config["cloud"]["pod_max_rps"]
            psi = min(lambda_hat / max(current_capacity, 1.0), 3.0)

            phi = 1.0 - env.cores / self.config["cloud"]["max_cores"]

            # ── PLAN ──
            # [P3+GA] Periodic NSGA-II optimization
            if step > 0 and step % self.ga_run_interval == 0:
                pareto = self.nsga2.optimize(env.replicas, env.cores, lambda_hat)
                if pareto:
                    self.ga_checkpoint = self.nsga2.get_nearest_checkpoint(
                        env.replicas, env.cores
                    )

            # [Jang93] ANFIS decision
            decision = self.anfis.decide(
                psi=psi, omega=omega, phi=phi, rho=rho,
                n_current=env.replicas, cores_current=env.cores,
                predicted_rps=lambda_hat, ga_checkpoint=self.ga_checkpoint,
            )

            # ── EXECUTE [P1 vertical, P3 diagonal, P4 horizontal] ──
            mode = decision["mode"]
            delta_c = decision["delta_c"]
            delta_n = decision["delta_n"]

            # Only record and execute if a real change is produced
            if delta_c == 0 and delta_n == 0:
                # ANFIS voted stable — log as no-op, don't count as action
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
                "psi": psi,
                "omega": omega,
                "phi": phi,
                "rho": rho,
                "lambda_hat": lambda_hat,
            }
            action_log.append(action_record)

            # ── ONLINE LEARNING ──
            self.anfis.record_outcome(
                inputs={"psi": psi, "omega": omega, "phi": phi, "rho": rho},
                action=decision,
                latency_observed=state["app_latency"],
                cost_observed=state["step_cost"],
            )

            # [Jang93 sect IV] Periodic ANFIS parameter update
            if step > 0 and step % 60 == 0:
                self.anfis.update_parameters()

        return env.history, action_log


class BaselineRunner:
    """Run a baseline autoscaler on the same trace for comparison."""

    def __init__(self, config, baseline):
        self.config = config
        self.baseline = baseline
        self.name = baseline.name

    def run_evaluation(self, test_df):
        env = CloudEnvironment(self.config)
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
