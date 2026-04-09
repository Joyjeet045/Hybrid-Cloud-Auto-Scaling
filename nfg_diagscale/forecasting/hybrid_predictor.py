"""
Hybrid Prophet-LSTM predictor combining seasonal decomposition with
residual correction.

[P5] Guruge & Priyadarshana (2025), Front. Comput. Sci. 7:1509165, Section 3.1:
  "The proposed model has two layers with 50 units and one dense layer."

  Fused prediction (P5 sect 3.1):
    hat_lambda_{t+k} = hat_lambda^(P)_{t+k} + hat_r_{t+k}

  Where:
    hat_lambda^(P) = Prophet seasonal prediction
    hat_r          = LSTM residual correction

  [P5 sect 3.2.2, Eq. 7] Pod count formula:
    pods_{t+1} = workload_{t+1} / workload_{pod}

  [P5 Eq. 14] Total prediction time:
    TPT = Prophet_Prediction_Time + LSTM_Prediction_Time
"""
import numpy as np
import time
from nfg_diagscale.forecasting.prophet_model import ProphetForecaster
from nfg_diagscale.forecasting.lstm_model import LSTMResidualModel
from nfg_diagscale.forecasting.kalman_filter import KalmanFilterRPS


class HybridPredictor:
    def __init__(self, config):
        self.config = config
        self.prophet = ProphetForecaster(config)
        self.lstm = LSTMResidualModel(config)
        self.kalman = KalmanFilterRPS(config)
        self.lookback = config["lstm"]["lookback_window"]

        self._residual_buffer = []
        self._prophet_predictions = None
        self._trained = False

    def train(self, train_df):
        """
        [P5 sect 3.1] Two-phase training:
        Phase 1: Train Prophet on raw time series to capture seasonality.
        Phase 2: Compute residuals, train LSTM on residuals.
        """
        # Cap training data to last 15,000 points for performance
        # (roughly 10 days of NASA traffic at 1min aggregation)
        if len(train_df) > 15000:
            train_df = train_df.iloc[-15000:].copy()

        print(f"[HybridPredictor] Phase 1: Training Prophet on {len(train_df)} samples...")
        t0 = time.time()
        self.prophet.train(train_df)
        t_prophet = time.time() - t0

        print("[HybridPredictor] Phase 2: Computing residuals and training LSTM...")
        # [P5 sect 3.1] r_t = lambda_t - hat_lambda^(P)_t
        residuals, prophet_pred = self.prophet.compute_residuals(train_df)
        self._residual_buffer = list(residuals)

        t0 = time.time()
        # The LSTM is trained on all available residues from our subsetted train_df
        self.lstm.train(residuals)
        t_lstm = time.time() - t0

        # [P5 Eq. 14] TPT = Prophet_Prediction_Time + LSTM_Prediction_Time
        print(f"[HybridPredictor] Training complete. "
              f"Prophet: {t_prophet:.1f}s, LSTM: {t_lstm:.1f}s")
        self._trained = True

    def predict_next(self, current_rps, current_df_row=None):
        """
        [P5 sect 3.1] Fused prediction:
          hat_lambda_{t+k} = hat_lambda^(P)_{t+k} + hat_r_{t+k}

        [P2 sect 3.3] Also update Kalman filter with observed RPS.
        """
        # [P2 sect 3.3] Kalman filter update for smoothed current RPS
        lambda_kf = self.kalman.update(current_rps)

        if not self._trained or len(self._residual_buffer) < self.lookback:
            return {
                "lambda_hat": current_rps,
                "lambda_kf": lambda_kf,
                "prophet_component": current_rps,
                "lstm_residual": 0.0,
            }

        # [P5 sect 3.1] Get Prophet seasonal prediction
        prophet_val = current_rps
        if current_df_row is not None:
            prophet_vals = self.prophet.get_seasonal_prediction(current_df_row)
            if len(prophet_vals) > 0:
                prophet_val = prophet_vals[-1]

        # [P5 sect 3.1] LSTM residual prediction
        recent_residuals = self._residual_buffer[-self.lookback:]
        lstm_residual = self.lstm.predict(recent_residuals)

        # [P5 sect 3.1] hat_lambda_{t+k} = hat_lambda^(P)_{t+k} + hat_r_{t+k}
        lambda_hat = prophet_val + lstm_residual

        # Update residual buffer with observed residual
        observed_residual = current_rps - prophet_val
        self._residual_buffer.append(observed_residual)

        return {
            "lambda_hat": max(lambda_hat, 0.0),
            "lambda_kf": lambda_kf,
            "prophet_component": prophet_val,
            "lstm_residual": lstm_residual,
        }

    def predict_batch(self, test_df):
        """
        [P5 sect 3.1] Batch prediction for evaluation.
        Returns array of fused predictions aligned with test data.
        """
        if not self._trained:
            raise RuntimeError("Model not trained.")

        residuals, prophet_pred = self.prophet.compute_residuals(test_df)

        all_residuals = np.concatenate([
            np.array(self._residual_buffer),
            residuals
        ])

        offset = len(self._residual_buffer)
        lstm_pred = np.zeros(len(test_df))

        for i in range(len(test_df)):
            idx = offset + i
            if idx >= self.lookback:
                window = all_residuals[idx - self.lookback:idx]
                lstm_pred[i] = self.lstm.predict(window)

        # [P5 sect 3.1] hat_lambda = hat_lambda^(P) + hat_r
        fused = prophet_pred + lstm_pred
        fused = np.maximum(fused, 0.0)
        return fused, prophet_pred, lstm_pred

    @staticmethod
    def compute_pod_count(predicted_rps, pod_max_rps):
        """
        [P5 sect 3.2.2, Eq. 7] pods_{t+1} = workload_{t+1} / workload_{pod}
        """
        if pod_max_rps <= 0:
            return 1
        return max(1, int(np.ceil(predicted_rps / pod_max_rps)))
