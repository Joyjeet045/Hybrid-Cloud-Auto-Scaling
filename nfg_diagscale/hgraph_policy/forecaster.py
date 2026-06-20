"""Per-container temporal load forecasting (Kalman + Holt).

NFG-DiagScale's forecasting identity is a Kalman filter (Kalman, 1960) for noise
suppression. We keep that and add Holt's linear (double-exponential) smoothing
(Holt, 1957) to extrapolate the *trend* one control interval ahead, which makes
the autoscaler proactive about ramps (the "temporal workload variations" that
STAR, Fang et al., 2026, highlights). Both are numpy-only, replacing the heavy
Prophet-LSTM stack.

The signal is each container's per-slot request count history
(``Container.workload_his``), updated once per 3-min interval by the simulator.
"""
from __future__ import annotations

from nfg_diagscale.forecasting.kalman_filter import KalmanFilterRPS


class HoltForecaster:
    """Holt's linear trend method (double-exponential smoothing).

    level_t  = alpha * y_t       + (1 - alpha) * (level_{t-1} + trend_{t-1})
    trend_t  = beta  * (level_t - level_{t-1}) + (1 - beta) * trend_{t-1}
    forecast = level_t + trend_t          (one-step-ahead)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._level = None
        self._trend = 0.0

    def update(self, y: float) -> float:
        y = float(y)
        if self._level is None:
            self._level = y
            self._trend = 0.0
            return y
        prev_level = self._level
        self._level = self.alpha * y + (1.0 - self.alpha) * (self._level + self._trend)
        self._trend = self.beta * (self._level - prev_level) + (1.0 - self.beta) * self._trend
        return self._level + self._trend

    def forecast(self) -> float:
        if self._level is None:
            return 0.0
        return self._level + self._trend


class ContainerForecaster:
    """Kalman-smoothed, Holt-trended one-step-ahead request forecaster."""

    def __init__(self, config):
        self._kf = KalmanFilterRPS(config)
        self._holt = HoltForecaster(
            alpha=config.get("forecast", {}).get("holt_alpha", 0.5),
            beta=config.get("forecast", {}).get("holt_beta", 0.3),
        )
        self._last = 0.0

    def update(self, observed: float) -> float:
        """Ingest one observed per-interval request count, return the forecast."""
        smoothed = self._kf.update(float(observed))
        forecast = self._holt.update(smoothed)
        # The next interval cannot have negative demand.
        self._last = max(0.0, forecast)
        return self._last

    def predict(self) -> float:
        return self._last


def seed_from_history(config, history) -> ContainerForecaster:
    """Build a forecaster and warm it up on an existing per-slot history array."""
    fc = ContainerForecaster(config)
    for y in history:
        fc.update(y)
    return fc
