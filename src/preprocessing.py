"""
Data preprocessing module.

Handles MinMaxScaler fitting/transforming and supervised-learning
dataset creation for the LSTM model.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import LOOK_BACK


def extract_close_prices(data: pd.DataFrame) -> np.ndarray:
    """
    Extract Close prices from a DataFrame, drop NaNs, and reshape to (-1, 1).

    Args:
        data: DataFrame containing a 'Close' column.

    Returns:
        NumPy array of shape (n, 1).

    Raises:
        ValueError: If 'Close' column is missing or data is empty.
    """
    if "Close" not in data.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    close = data["Close"].dropna().values.astype(float)

    if close.size == 0:
        raise ValueError("No valid Close prices available.")

    return close.reshape(-1, 1)


def fit_scaler(close_prices: np.ndarray) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the close prices.

    Args:
        close_prices: Array of shape (n, 1).

    Returns:
        Fitted MinMaxScaler instance.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_prices)
    return scaler


def scale_data(close_prices: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Transform close prices using an already-fitted scaler.

    Args:
        close_prices: Array of shape (n, 1).
        scaler: Fitted MinMaxScaler.

    Returns:
        Scaled array of shape (n, 1).
    """
    return scaler.transform(close_prices)


def preprocess_data(data: pd.DataFrame):
    """
    Full preprocessing pipeline: extract Close → fit scaler → scale.

    Args:
        data: DataFrame with a 'Close' column.

    Returns:
        Tuple of (scaled_data, scaler, close_prices).
    """
    close_prices = extract_close_prices(data)
    scaler = fit_scaler(close_prices)
    scaled_data = scale_data(close_prices, scaler)
    return scaled_data, scaler, close_prices


def create_dataset(data: np.ndarray, look_back: int = LOOK_BACK):
    """
    Convert a time-series array into supervised-learning sequences.

    Each sample consists of `look_back` consecutive values as features
    and the immediately following value as the target.

    Args:
        data: Scaled price array of shape (n, 1).
        look_back: Number of past timesteps per sample.

    Returns:
        Tuple of (X, y) as NumPy arrays.

    Raises:
        ValueError: If there are not enough data points.
    """
    if len(data) <= look_back:
        raise ValueError(
            f"Need more than {look_back} data points to create sequences. "
            f"Got {len(data)}."
        )

    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : (i + look_back), 0])
        y.append(data[i + look_back, 0])

    return np.array(X), np.array(y)


def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
    """
    Reshape feature array to LSTM-expected 3-D shape: (samples, timesteps, 1).

    Args:
        X: Array of shape (samples, timesteps).

    Returns:
        Array of shape (samples, timesteps, 1).
    """
    return np.reshape(X, (X.shape[0], X.shape[1], 1))
