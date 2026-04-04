"""
Prophet seasonal decomposition for workload forecasting.

[P5] Guruge & Priyadarshana (2025), Front. Comput. Sci. 7:1509165, Section 3.1:
  "The Prophet model captures the seasonality of data for time
   series forecasting..."

  Decomposition (P5 sect 3.1):
    y(t) = g(t) + s(t) + h(t) + epsilon_t

  Where:
    g(t) = piecewise-linear growth trend
    s(t) = Fourier-series seasonality
    h(t) = holiday/event effects
    epsilon_t ~ N(0, sigma^2)

  [P5 Table 2] Prophet configuration:
    Growth = Linear
    Changepoint prior scale = 5.1
    Yearly seasonality = False
    Weekly seasonality = 20
    Daily seasonality = 50
    Seasonality prior scale = 30

  [P5 sect 3.1.4] Time complexity:
    O(T * (k + m + n)) where k=Fourier order, m=changepoints, n=iterations
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetForecaster:
    def __init__(self, config):
        pcfg = config["prophet"]
        self.config = pcfg
        self.model = None
        self._trained = False

    def _create_model(self):
        """
        [P5 Table 2] Model configuration as specified in the paper.
        """
        model = Prophet(
            growth=self.config["growth"],
            changepoint_prior_scale=self.config["changepoint_prior_scale"],
            yearly_seasonality=self.config["yearly_seasonality"],
            weekly_seasonality=self.config["weekly_seasonality"],
            daily_seasonality=self.config["daily_seasonality"],
            seasonality_prior_scale=self.config["seasonality_prior_scale"],
        )
        return model

    def train(self, train_df):
        """
        Train Prophet on historical workload data.

        [P5 sect 3.1] Prophet expects DataFrame with columns 'ds' and 'y'.
        [P5 sect 4.2] "took 70% for training... preserving the time order"
        """
        df = train_df[["ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])

        self.model = self._create_model()
        self.model.fit(df)
        self._trained = True
        print(f"[Prophet] Trained on {len(df)} data points")

    def predict(self, periods, freq="min"):
        """
        [P5 sect 3.1] Generate seasonal forecast for future periods.
        Returns DataFrame with 'ds', 'yhat' (and components trend, seasonal).
        """
        if not self._trained:
            raise RuntimeError("Prophet model not trained. Call train() first.")

        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat", "trend", "yhat_lower", "yhat_upper"]]

    def get_seasonal_prediction(self, full_df, periods_ahead=0):
        """
        [P5 sect 3.1] Get Prophet prediction for given timestamps.
        Returns the seasonal component hat_lambda_P for residual calculation.
        """
        if not self._trained:
            raise RuntimeError("Prophet model not trained.")

        df_input = pd.DataFrame({"ds": pd.to_datetime(full_df["ds"])})
        forecast = self.model.predict(df_input)
        return forecast["yhat"].values

    def compute_residuals(self, df):
        """
        [P5 sect 3.1] r_t = lambda_t - hat_lambda^(P)_t
        "LSTM model is used for processing the residuals after seasonality removal"
        """
        prophet_pred = self.get_seasonal_prediction(df)
        actual = df["y"].values
        residuals = actual - prophet_pred
        return residuals, prophet_pred
