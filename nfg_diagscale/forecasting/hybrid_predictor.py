"""Hybrid Prophet-LSTM predictor."""
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
        """Two-phase training: Prophet (seasonality) then LSTM (residuals)."""
        # Cap training data to last 50,000 points for performance
        # (roughly 35 days of traffic - enough for seasonality and multiple peak events)
        if len(train_df) > 50000:
            train_df = train_df.iloc[-50000:].copy()

        print(f"[HybridPredictor] Phase 1: Training Prophet on {len(train_df)} samples...")
        t0 = time.time()
        self.prophet.train(train_df)
        t_prophet = time.time() - t0

        print("[HybridPredictor] Phase 2: Computing residuals and training LSTM...")
        # r_t = lambda_t - hat_lambda^(P)_t
        residuals, prophet_pred = self.prophet.compute_residuals(train_df)
        self._residual_buffer = list(residuals)

        t0 = time.time()
        # The LSTM is trained on all available residues from our subsetted train_df
        self.lstm.train(residuals)
        t_lstm = time.time() - t0

        # TPT = Prophet_Prediction_Time + LSTM_Prediction_Time
        print(f"[HybridPredictor] Training complete. "
              f"Prophet: {t_prophet:.1f}s, LSTM: {t_lstm:.1f}s")
        self._trained = True

    def predict_next(self, current_rps, current_df_row=None):
        """Fused prediction with Kalman update."""
        # Kalman filter update for smoothed current RPS
        lambda_kf = self.kalman.update(current_rps)

        if not self._trained or len(self._residual_buffer) < self.lookback:
            return {
                "lambda_hat": current_rps,
                "lambda_kf": lambda_kf,
                "prophet_component": current_rps,
                "lstm_residual": 0.0,
            }

        # Get Prophet seasonal prediction
        prophet_val = current_rps
        if current_df_row is not None:
            prophet_vals = self.prophet.get_seasonal_prediction(current_df_row)
            if len(prophet_vals) > 0:
                prophet_val = prophet_vals[-1]

        # LSTM residual prediction
        recent_residuals = self._residual_buffer[-self.lookback:]
        lstm_residual = self.lstm.predict(recent_residuals)

        # hat_lambda_{t+k} = hat_lambda^(P)_{t+k} + hat_r_{t+k}
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
        Batch prediction for evaluation.
        Returns array of fused predictions aligned with test data.
        """
        if not self._trained:
            raise RuntimeError("Model not trained.")

        residuals, prophet_pred = self.prophet.compute_residuals(test_df)

        all_residuals = np.concatenate([
            np.array(self._residual_buffer),
            residuals
        ])

        # Efficient vectorized LSTM residual prediction
        lstm_pred_full = self.lstm.predict_batch(all_residuals)
        
        # Align with test_df length
        lstm_pred = np.zeros(len(test_df))
        lstm_pred[:] = lstm_pred_full[-len(test_df):]

        # hat_lambda = hat_lambda^(P) + hat_r
        fused = prophet_pred + lstm_pred
        fused = np.maximum(fused, 0.0)
        return fused, prophet_pred, lstm_pred

    @staticmethod
    def compute_pod_count(predicted_rps, pod_max_rps):
        """Compute required pods based on workload prediction."""
        if pod_max_rps <= 0:
            return 1
        return max(1, int(np.ceil(predicted_rps / pod_max_rps)))
