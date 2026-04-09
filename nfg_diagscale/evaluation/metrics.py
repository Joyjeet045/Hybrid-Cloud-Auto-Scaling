"""
KPI computation for autoscaler evaluation.

Metrics from:
[P5] Guruge & Priyadarshana (2025), sect 4.2.2:
  MSE  (Eq. 10): (1/n) * sum(a_i - f_i)^2
  RMSE (Eq. 11): sqrt(MSE)
  MAE  (Eq. 12): (1/n) * sum|a_i - f_i|
  R^2  (Eq. 13): 1 - sum(a_i - f_i)^2 / sum(a_i - f_bar)^2

[P4] Solino et al. (2025), UCC'25, sect 4.1:
  Reaction Time (Eq. 1): RT = CRD - DFTV
  Tolerance to Processing Peaks (Eq. 2): TPP = ACR / (ACR + SCR)

NFG-DiagScale KPIs from proposal:
  SVR  = #{t: L_t > SLO} / T       (SLO Violation Rate)
  CER  = Cost_baseline / Cost_ours  (Cost Efficiency Ratio)
  MAPE = (1/T) * sum|actual - pred| / actual  (Forecast MAPE)
"""
import numpy as np


def forecast_mse(actual, predicted):
    """[P5 Eq. 10] MSE = (1/n) * sum(a_i - f_i)^2"""
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean((actual - predicted) ** 2)


def forecast_rmse(actual, predicted):
    """[P5 Eq. 11] RMSE = sqrt(MSE)"""
    return np.sqrt(forecast_mse(actual, predicted))


def forecast_mae(actual, predicted):
    """[P5 Eq. 12] MAE = (1/n) * sum|a_i - f_i|"""
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs(actual - predicted))


def forecast_r2(actual, predicted):
    """[P5 Eq. 13] R^2 = 1 - sum(a_i - f_i)^2 / sum(a_i - a_bar)^2"""
    actual, predicted = np.array(actual), np.array(predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def forecast_mape(actual, predicted):
    """MAPE = (1/T) * sum|actual - predicted| / actual"""
    actual, predicted = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    mask = actual > 1.0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100


def slo_violation_rate(history, slo):
    """SVR = #{t: L_t > SLO} / T"""
    violations = sum(1 for s in history if s["app_latency"] > slo)
    return violations / max(len(history), 1) * 100


def total_cost(history):
    """Total infrastructure cost over evaluation period."""
    return sum(s["step_cost"] for s in history)


def cost_efficiency_ratio(baseline_cost, our_cost):
    """CER = Cost_baseline / Cost_ours"""
    if our_cost < 1e-8:
        return float("inf")
    return baseline_cost / our_cost


def avg_latency(history):
    """Average application latency."""
    lats = [s["app_latency"] for s in history]
    return np.mean(lats) if lats else 0.0


def p99_latency(history):
    """99th percentile latency."""
    lats = [s["app_latency"] for s in history]
    return np.percentile(lats, 99) if lats else 0.0


def scaling_action_count(action_log):
    """Total number of scaling actions executed."""
    return sum(1 for a in action_log if a.get("mode", "none") != "none")


def rebalance_overhead(action_log):
    """Sum of replica changes (proxy for rebalance cost)."""
    return sum(abs(a.get("delta_n", 0)) + abs(a.get("delta_c", 0)) for a in action_log)


def reaction_lag(history, slo):
    """
    [P4 Eq. 1] RT = CRD - DFTV
    Tracking the steps between SLO violation start and scaled resolution.
    Approximate: Average duration of SLO violation streaks.
    """
    if not history:
        return 0.0
    
    violation_streaks = []
    current_streak = 0
    
    for s in history:
        if s["app_latency"] > slo:
            current_streak += 1
        else:
            if current_streak > 0:
                violation_streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        violation_streaks.append(current_streak)
        
    return np.mean(violation_streaks) if violation_streaks else 0.0


def compute_all_metrics(history, action_log, slo, name=""):
    """Compute all KPIs for one autoscaler run."""
    results = {
        "name": name,
        "svr_pct": slo_violation_rate(history, slo),
        "total_cost": total_cost(history),
        "avg_latency_ms": avg_latency(history),
        "p99_latency_ms": p99_latency(history),
        "scaling_actions": scaling_action_count(action_log),
        "rebalance_overhead": rebalance_overhead(action_log),
        "reaction_lag_steps": reaction_lag(history, slo),
        "steps": len(history),
    }
    return results
