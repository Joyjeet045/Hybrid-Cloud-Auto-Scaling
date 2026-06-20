"""Kalman filter for short-term request-rate (RPS) estimation.

Scalar linear Kalman filter (Kalman, 1960): the predict/update recursion below is
the standard form, with state-transition ``A``, observation ``H``, process-noise
``Q`` and measurement-noise ``D`` read from config. Used to denoise each
container's per-interval request count before Holt trend extrapolation.
"""
import numpy as np


class KalmanFilterRPS:
    def __init__(self, config):
        kf_cfg = config["kalman"]
        # State transition model
        self.A = float(kf_cfg["A"])
        # Observation model
        self.H = float(kf_cfg["H"])
        # Process noise covariance
        self.Q = float(kf_cfg["Q"])
        # Measurement noise covariance
        self.D = float(kf_cfg["D"])

        self._initial_P = float(kf_cfg["initial_P"])
        self.R_est = 0.0
        self.P = self._initial_P
        self._initialized = False

    def update(self, R_observed):
        """
        Updates the estimate by integrating predictions with observations.
        """
        if not self._initialized:
            self.R_est = R_observed
            self._initialized = True
            return self.R_est

        # State prediction: R'_t = A * R_{t-1}
        R_pred = self.A * self.R_est

        # Covariance prediction: P'_t = A * P_{t-1} * A^T + Q
        P_pred = self.A * self.P * self.A + self.Q

        # Kalman gain: K = P'_t * H / (H * P'_t * H^T + D)
        denom = self.H * P_pred * self.H + self.D
        K = P_pred * self.H / denom

        # State update: R = R'_t + K * (R_t - H * R'_t)
        innovation = R_observed - self.H * R_pred
        self.R_est = R_pred + K * innovation

        # Covariance update: P = (1 - K * H) * P'_t
        self.P = (1.0 - K * self.H) * P_pred

        return self.R_est

    def get_estimate(self):
        return self.R_est

    def reset(self):
        self.R_est = 0.0
        self.P = self._initial_P
        self._initialized = False
