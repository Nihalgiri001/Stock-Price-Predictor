"""
Forecasting module.

Implements recursive multi-step future price prediction and
business-day-aware future date generation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import LOOK_BACK


def predict_future_prices(
    model,
    last_sequence: np.ndarray,
    scaler: MinMaxScaler,
    future_steps: int,
    look_back: int = LOOK_BACK,
) -> np.ndarray:
    """
    Recursively predict future stock prices.

    Starting from the last `look_back` scaled closing prices, each
    prediction is fed back as input for the next step.

    Args:
        model: Trained Keras LSTM model.
        last_sequence: Array of shape (look_back,) with scaled values.
        scaler: Fitted MinMaxScaler for inverse transformation.
        future_steps: Number of future trading days to predict.
        look_back: Sequence length (default 60).

    Returns:
        NumPy array of predicted prices in original scale, shape (future_steps,).
    """
    current_sequence = last_sequence.copy().flatten()[-look_back:]
    predictions_scaled = []

    for _ in range(future_steps):
        # Shape: (1, look_back, 1)
        input_seq = current_sequence.reshape(1, look_back, 1)
        predicted_value = model.predict(input_seq, verbose=0)[0, 0]

        predictions_scaled.append(predicted_value)

        # Slide window: drop oldest, append new prediction
        current_sequence = np.append(current_sequence[1:], predicted_value)

    # Inverse-transform to original price scale
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()

    return predictions


def generate_future_dates(
    last_date: pd.Timestamp,
    num_days: int,
) -> pd.DatetimeIndex:
    """
    Generate future business (trading) dates starting after `last_date`.

    Args:
        last_date: The last historical date in the dataset.
        num_days: Number of business days to generate.

    Returns:
        DatetimeIndex of future business dates.
    """
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=num_days,
    )
    return future_dates


def calculate_expected_change(
    current_price: float,
    predicted_price: float,
) -> float:
    """
    Calculate expected percentage change.

    Args:
        current_price: Current/latest closing price.
        predicted_price: Model's predicted price at the forecast horizon.

    Returns:
        Percentage change as a float (e.g. 5.25 for +5.25%).
    """
    if current_price == 0:
        return 0.0
    return ((predicted_price - current_price) / current_price) * 100
