"""
MAPE-K orchestrator implementing the NFG-DiagScale control loop.

MAPE-K operates in four phases in a continuous feedback cycle:
 (1) Monitor: continuously observes the system
 (2) Analyze: evaluates monitored data to determine if adaptation is required
 (3) Plan: builds an action plan
 (4) Execute: applies the planned changes
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

    def run_evaluation(self, test_df):
        """
        Run the full MAPE-K loop on a test trace.
        Returns history and action log for KPI evaluation.
        """
        env = CloudEnvironment(self.config)
        
        # Warm-up/Initialization: Start with capacity matching the initial load
        # to avoid astronomical latencies on the very first step.
        initial_rps = float(test_df["y"].iloc[0])
        initial_pods = int(np.ceil(initial_rps / (self.config["cloud"]["max_cores"] * self.pod_max_rps)))
        env.replicas = int(np.clip(initial_pods, self.config["cloud"]["min_replicas"], self.config["cloud"]["max_replicas"]))
        env.cores = self.config["cloud"]["max_cores"] # Start with full cores vertically to be safe
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

            # Heat accumulator check
            self.heat_acc.update(violation)

            # Enforce cooldown to prevent action-on-action thrashing
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
            
            current_capacity = env.replicas * env.cores * self.config["cloud"]["pod_max_rps"]
            predicted_psi = lambda_hat / max(current_capacity, 1.0)
            
            proactive_trigger = (predicted_psi > 0.8) or (predicted_psi < 0.4)

            if not self.heat_acc.should_trigger() and not proactive_trigger:
                action_log.append({"step": step, "mode": "none", "delta_c": 0, "delta_n": 0})
                continue
            self.heat_acc.reset()

            # Themis SLO risk from predicted workload
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
            # Periodic NSGA-II optimization
            if step > 0 and step % self.ga_run_interval == 0:
                # Low-load vertical bias
                # If RPS is low, we want to stay vertical to avoid rebalance delays
                low_load_mode = lambda_hat < 150.0
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

            # EXECUTE scaling action
            mode = decision["mode"]
            delta_c = decision["delta_c"]
            delta_n = decision["delta_n"]

            # Scale-down Hysteresis for stability
            # If we scaled UP (N) recently, don't scale DOWN (N) for 5 steps
            if delta_n > 0:
                self._last_scale_up_step = step
            elif delta_n < 0:
                cooldown = 6
                if step - getattr(self, "_last_scale_up_step", -100) < cooldown:
                    delta_n = 0 # Block the scale down to prevent jitter

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

            # Periodic ANFIS parameter update
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
        
        # Consistent initialization with NFG-DiagScale
        initial_rps = float(test_df["y"].iloc[0])
        initial_pods = int(np.ceil(initial_rps / (self.config["cloud"]["max_cores"] * self.config["cloud"]["pod_max_rps"])))
        env.replicas = int(np.clip(initial_pods, self.config["cloud"]["min_replicas"], self.config["cloud"]["max_replicas"]))
        env.cores = self.config["cloud"]["max_cores"]
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
