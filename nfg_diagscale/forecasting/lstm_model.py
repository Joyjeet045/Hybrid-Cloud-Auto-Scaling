"""
LSTM model for residual correction after Prophet seasonal decomposition.

The LSTM model is capable of learning from the temporal dependencies
within the residuals.
"""
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


class LSTMResidualModel:
    def __init__(self, config):
        lcfg = config["lstm"]
        self.n_layers = lcfg["layers"]
        self.n_units = lcfg["units"]
        self.loss = lcfg["loss"]
        self.lr = lcfg["learning_rate"]
        self.epochs = lcfg["epochs"]
        self.batch_size = lcfg["batch_size"]
        self.patience = lcfg["early_stopping_patience"]
        self.lookback = lcfg["lookback_window"]

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._trained = False

    def _build_model(self):
        """
        Build the model with 50 hidden units per layer.
        """
        model = Sequential()

        # First LSTM layer with return_sequences for stacking
        model.add(LSTM(
            self.n_units,
            return_sequences=True,
            input_shape=(self.lookback, 1)
        ))

        # Second LSTM layer
        model.add(LSTM(self.n_units, return_sequences=False))

        # Dense output layer
        model.add(Dense(1))

        # Adam optimizer with lr=0.001, MSE loss
        optimizer = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss=self.loss)
        return model

    def _create_sequences(self, data):
        """Create lookback window sequences for LSTM input."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def train(self, residuals):
        """
        Train LSTM on residuals.
        """
        residuals = np.array(residuals).reshape(-1, 1)
        scaled = self.scaler.fit_transform(residuals)

        X, y = self._create_sequences(scaled)
        if len(X) == 0:
            raise ValueError(
                f"Not enough data for LSTM. Need > {self.lookback} points, got {len(residuals)}"
            )

        # Early stopping
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True
        )

        self.model = self._build_model()

        # Validation split from training data
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        self._trained = True
        print(f"[LSTM] Trained on {len(X)} sequences (lookback={self.lookback})")

    def predict(self, recent_residuals):
        """
        Predict the next residual value given the recent residual window.
        """
        if not self._trained:
            raise RuntimeError("LSTM model not trained.")

        window = np.array(recent_residuals[-self.lookback:]).reshape(-1, 1)
        scaled = self.scaler.transform(window)
        X = scaled.reshape(1, self.lookback, 1)

        pred_scaled = self.model.predict(X, verbose=0)[0, 0]
        pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
        return pred

    def predict_batch(self, residual_series):
        """Predict residuals for an entire series using sliding window."""
        if not self._trained:
            raise RuntimeError("LSTM model not trained.")

        residual_series = np.array(residual_series).reshape(-1, 1)
        scaled_series = self.scaler.transform(residual_series)

        predictions = []
        for i in range(self.lookback, len(scaled_series)):
            window = scaled_series[i - self.lookback:i].reshape(1, self.lookback, 1)
            pred = self.model.predict(window, verbose=0)[0, 0]
            predictions.append(pred)

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        return predictions
