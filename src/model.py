"""
LSTM model definition, saving, and loading utilities.

Architecture (from project report):
    LSTM(50, return_sequences=True) → Dropout(0.2) →
    LSTM(50, return_sequences=False) → Dropout(0.2) →
    Dense(1)
"""

import json
import logging
from pathlib import Path

import joblib
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    DROPOUT_RATE,
    LOOK_BACK,
    LSTM_UNITS,
    METADATA_FILENAME,
    MODEL_FILENAME,
    MODELS_DIR,
    SCALER_FILENAME,
)

logger = logging.getLogger(__name__)


def build_lstm_model(look_back: int = LOOK_BACK) -> "keras.Model":
    """
    Build and compile the LSTM model.

    Returns:
        Compiled Keras Sequential model.
    """
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential()

    model.add(
        LSTM(
            units=LSTM_UNITS,
            return_sequences=True,
            input_shape=(look_back, 1),
        )
    )
    model.add(Dropout(DROPOUT_RATE))

    model.add(
        LSTM(
            units=LSTM_UNITS,
            return_sequences=False,
        )
    )
    model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def get_model_dir(ticker: str) -> Path:
    """Return the directory where a ticker's model artifacts are stored."""
    return MODELS_DIR / ticker.strip().upper()


def save_model_artifacts(
    model,
    scaler: MinMaxScaler,
    metadata: dict,
    ticker: str,
) -> Path:
    """
    Save model, scaler, and metadata to disk.

    Args:
        model: Trained Keras model.
        scaler: Fitted MinMaxScaler.
        metadata: Dictionary of training metadata.
        ticker: Stock ticker symbol.

    Returns:
        Path to the model directory.
    """
    model_dir = get_model_dir(ticker)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / MODEL_FILENAME
    scaler_path = model_dir / SCALER_FILENAME
    metadata_path = model_dir / METADATA_FILENAME

    model.save(str(model_path))
    joblib.dump(scaler, str(scaler_path))

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Model artifacts saved to %s", model_dir)
    return model_dir


def load_model_artifacts(ticker: str):
    """
    Load model, scaler, and metadata for a given ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Tuple of (model, scaler, metadata) or (None, None, None) if unavailable.
    """
    model_dir = get_model_dir(ticker)

    model_path = model_dir / MODEL_FILENAME
    scaler_path = model_dir / SCALER_FILENAME
    metadata_path = model_dir / METADATA_FILENAME

    if not model_path.exists() or not scaler_path.exists():
        logger.warning("Trained model not found for %s in %s", ticker, model_dir)
        return None, None, None

    try:
        from tensorflow.keras.models import load_model as keras_load_model

        model = keras_load_model(str(model_path))
        scaler = joblib.load(str(scaler_path))

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        logger.info("Loaded model artifacts for %s", ticker)
        return model, scaler, metadata

    except Exception as exc:
        logger.error("Error loading model for %s: %s", ticker, exc)
        return None, None, None


def model_exists(ticker: str) -> bool:
    """Check whether a trained model exists for the given ticker."""
    model_dir = get_model_dir(ticker)
    return (
        (model_dir / MODEL_FILENAME).exists()
        and (model_dir / SCALER_FILENAME).exists()
    )
