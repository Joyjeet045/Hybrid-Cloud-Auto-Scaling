"""
Kalman Filter for short-term RPS estimation.

[P2] Gu et al. (2025), "HAS-GPU", arXiv:2505.01968, Section 3.3:
  "HAS-GPU proposes a Kalman filter-based short-term estimation approach
   that predicts the next request workload R by the current measured
   request load Rt."

  Equations from P2 sect 3.3:
    R'_t = A * R_{t-1}                         (state prediction)
    P'_t = A * R_{t-1} * A^T + Q               (covariance prediction)
    K    = P'_t * H / (H * P'_t * H^T + D)     (Kalman gain)
    R    = R'_t + K * (R_t - H * R'_t)          (state update)
    P    = (1 - K * H) * P'_t                   (covariance update)

  Where:
    A = state transition (scalar, ~1.0 for RPS tracking)
    H = observation model (scalar, 1.0)
    Q = process noise covariance
    D = measurement noise covariance
    K = Kalman gain, balancing predicted vs observed
"""
import numpy as np


class KalmanFilterRPS:
    def __init__(self, config):
        kf_cfg = config["kalman"]
        # [P2 sect 3.3] State transition model
        self.A = float(kf_cfg["A"])
        # [P2 sect 3.3] Observation model
        self.H = float(kf_cfg["H"])
        # [P2 sect 3.3] Process noise covariance
        self.Q = float(kf_cfg["Q"])
        # [P2 sect 3.3] Measurement noise covariance
        self.D = float(kf_cfg["D"])

        self.R_est = 0.0
        self.P = float(kf_cfg["initial_P"])
        self._initialized = False

    def update(self, R_observed):
        """
        [P2 sect 3.3] "By integrating predictions with observations,
        the request predictor can efficiently adapt to fluctuating workloads."
        """
        if not self._initialized:
            self.R_est = R_observed
            self._initialized = True
            return self.R_est

        # [P2 sect 3.3] R'_t = A * R_{t-1}
        R_pred = self.A * self.R_est

        # [P2 sect 3.3] P'_t = A * P_{t-1} * A^T + Q
        P_pred = self.A * self.P * self.A + self.Q

        # [P2 sect 3.3] K = P'_t * H / (H * P'_t * H^T + D)
        denom = self.H * P_pred * self.H + self.D
        K = P_pred * self.H / denom

        # [P2 sect 3.3] R = R'_t + K * (R_t - H * R'_t)
        innovation = R_observed - self.H * R_pred
        self.R_est = R_pred + K * innovation

        # [P2 sect 3.3] P = (1 - K * H) * P'_t
        self.P = (1.0 - K * self.H) * P_pred

        return self.R_est

    def get_estimate(self):
        return self.R_est

    def reset(self):
        self.R_est = 0.0
        self.P = 1.0
        self._initialized = False
