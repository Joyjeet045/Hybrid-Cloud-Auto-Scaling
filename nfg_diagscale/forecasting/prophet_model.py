"""
Prophet seasonal decomposition for workload forecasting.
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
        Model configuration for seasonal decomposition.
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
        Prophet expects DataFrame with columns 'ds' and 'y'.
        """
        df = train_df[["ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])

        self.model = self._create_model()
        self.model.fit(df)
        self._trained = True
        print(f"[Prophet] Trained on {len(df)} data points")

    def predict(self, periods, freq="min"):
        """
        Generate seasonal forecast for future periods.
        Returns DataFrame with 'ds', 'yhat' (and components trend, seasonal).
        """
        if not self._trained:
            raise RuntimeError("Prophet model not trained. Call train() first.")

        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat", "trend", "yhat_lower", "yhat_upper"]]

    def get_seasonal_prediction(self, full_df, periods_ahead=0):
        """
        Get Prophet prediction for given timestamps.
        Returns the seasonal component for residual calculation.
        """
        if not self._trained:
            raise RuntimeError("Prophet model not trained.")

        df_input = pd.DataFrame({"ds": pd.to_datetime(full_df["ds"])})
        forecast = self.model.predict(df_input)
        return forecast["yhat"].values

    def compute_residuals(self, df):
        """
        Computes residuals after seasonality removal for further processing.
        """
        prophet_pred = self.get_seasonal_prediction(df)
        actual = df["y"].values
        residuals = actual - prophet_pred
        return residuals, prophet_pred
