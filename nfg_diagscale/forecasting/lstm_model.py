"""
LSTM model for residual correction after Prophet seasonal decomposition.

[P5] Guruge & Priyadarshana (2025), Front. Comput. Sci. 7:1509165, Section 3.1:
  "The LSTM model is capable of learning from the temporal dependencies
   within the residuals."

  LSTM gate equations (P5 Equations 2-4):
    i_t = sigma(omega_i * [h_{t-1}, x_t] + b_i)    (input gate, Eq. 2)
    f_t = sigma(omega_f * [h_{t-1}, x_t] + b_f)    (forget gate, Eq. 3)
    o_t = sigma(omega_o * [h_{t-1}, x_t] + b_o)    (output gate, Eq. 4)

  [P5 Table 2] LSTM configuration:
    Layers = 2 LSTM, 1 dense
    Hidden units = 50
    Loss function = MSE
    Early stopping = 5
    Epochs = 50
    Batch size = 16
    Optimizer = Adam
    Learning rate = 0.001

  [P5 sect 3.1.4] Time complexity (Eq. 6):
    O(T * n * m) where T=sequence length, n=input features, m=hidden units
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
        [P5 Table 2] "2 LSTM, 1 dense" with 50 hidden units per layer.
        [P5 Eq. 2-4] Each LSTM layer implements input, forget, output gates.
        """
        model = Sequential()

        # [P5 Table 2] First LSTM layer with return_sequences for stacking
        model.add(LSTM(
            self.n_units,
            return_sequences=True,
            input_shape=(self.lookback, 1)
        ))

        # [P5 Table 2] Second LSTM layer
        model.add(LSTM(self.n_units, return_sequences=False))

        # [P5 Table 2] Dense output layer
        model.add(Dense(1))

        # [P5 Table 2] Adam optimizer with lr=0.001, MSE loss
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
        [P5 sect 3.1] Train LSTM on residuals r_t = lambda_t - hat_lambda^(P)_t
        [P5 Table 2] All hyperparameters from paper configuration.
        """
        residuals = np.array(residuals).reshape(-1, 1)
        scaled = self.scaler.fit_transform(residuals)

        X, y = self._create_sequences(scaled)
        if len(X) == 0:
            raise ValueError(
                f"Not enough data for LSTM. Need > {self.lookback} points, got {len(residuals)}"
            )

        # [P5 sect 3.1] Reshape for LSTM: [samples, timesteps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # [P5 Table 2] Early stopping with patience=5
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True
        )

        self.model = self._build_model()

        # [P5 sect 4.2] Validation split from training data
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
        [P5 sect 3.1] hat_r_{t+k} = LSTM([r_{t-tau}, ..., r_t]; Theta_LSTM)
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
