"""
Stock analysis module.

Provides high-level functions for single-stock prediction and
top-stock scanning that the Streamlit app uses.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import LOOK_BACK, SCAN_FORECAST_DAYS
from src.data import fetch_stock_data, get_default_date_range
from src.forecasting import (
    calculate_expected_change,
    generate_future_dates,
    predict_future_prices,
)
from src.model import load_model_artifacts, model_exists
from src.preprocessing import extract_close_prices, scale_data
from src.sentiment import get_news_sentiment

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for a single stock prediction result."""

    ticker: str
    current_price: float
    predicted_price: float
    expected_change: float
    future_prices: np.ndarray
    future_dates: pd.DatetimeIndex
    historical_data: pd.DataFrame
    sentiment_label: str
    sentiment_score: float
    headlines: list
    horizon_days: int
    success: bool = True
    error_message: str = ""


@dataclass
class ScanResult:
    """Container for a stock-scan row."""

    ticker: str
    current_price: float
    predicted_price: float
    expected_change: float
    horizon_days: int
    sentiment_label: str
    sentiment_score: float
    available: bool = True
    error_message: str = ""


def predict_stock(
    ticker: str,
    forecast_days: int,
) -> PredictionResult:
    """
    Run full prediction pipeline for a single stock.

    Args:
        ticker: Stock ticker symbol.
        forecast_days: Number of future trading days to predict.

    Returns:
        PredictionResult instance.
    """
    ticker = ticker.strip().upper()

    # --- Load model ---
    model, scaler, metadata = load_model_artifacts(ticker)
    if model is None or scaler is None:
        return PredictionResult(
            ticker=ticker,
            current_price=0,
            predicted_price=0,
            expected_change=0,
            future_prices=np.array([]),
            future_dates=pd.DatetimeIndex([]),
            historical_data=pd.DataFrame(),
            sentiment_label="",
            sentiment_score=0,
            headlines=[],
            horizon_days=forecast_days,
            success=False,
            error_message=(
                f"Trained model not found for {ticker}. "
                f"Please train the model first using:\n"
                f"```\npython train.py --ticker {ticker}\n```"
            ),
        )

    look_back = metadata.get("look_back", LOOK_BACK)

    # --- Fetch data ---
    start_date, end_date = get_default_date_range()
    data = fetch_stock_data(ticker, start_date, end_date)

    if data is None or data.empty:
        return PredictionResult(
            ticker=ticker,
            current_price=0,
            predicted_price=0,
            expected_change=0,
            future_prices=np.array([]),
            future_dates=pd.DatetimeIndex([]),
            historical_data=pd.DataFrame(),
            sentiment_label="",
            sentiment_score=0,
            headlines=[],
            horizon_days=forecast_days,
            success=False,
            error_message=f"Could not fetch data for {ticker} from Yahoo Finance.",
        )

    # --- Preprocess ---
    try:
        close_prices = extract_close_prices(data)
    except ValueError as exc:
        return PredictionResult(
            ticker=ticker,
            current_price=0,
            predicted_price=0,
            expected_change=0,
            future_prices=np.array([]),
            future_dates=pd.DatetimeIndex([]),
            historical_data=data,
            sentiment_label="",
            sentiment_score=0,
            headlines=[],
            horizon_days=forecast_days,
            success=False,
            error_message=str(exc),
        )

    if len(close_prices) < look_back:
        return PredictionResult(
            ticker=ticker,
            current_price=0,
            predicted_price=0,
            expected_change=0,
            future_prices=np.array([]),
            future_dates=pd.DatetimeIndex([]),
            historical_data=data,
            sentiment_label="",
            sentiment_score=0,
            headlines=[],
            horizon_days=forecast_days,
            success=False,
            error_message=(
                f"Insufficient data for {ticker}. "
                f"Need at least {look_back} data points, got {len(close_prices)}."
            ),
        )

    scaled = scale_data(close_prices, scaler)
    last_sequence = scaled[-look_back:].flatten()

    current_price = float(close_prices[-1, 0])

    # --- Predict ---
    try:
        future_prices = predict_future_prices(
            model, last_sequence, scaler, forecast_days, look_back
        )
    except Exception as exc:
        logger.error("Prediction failed for %s: %s", ticker, exc)
        return PredictionResult(
            ticker=ticker,
            current_price=current_price,
            predicted_price=0,
            expected_change=0,
            future_prices=np.array([]),
            future_dates=pd.DatetimeIndex([]),
            historical_data=data,
            sentiment_label="",
            sentiment_score=0,
            headlines=[],
            horizon_days=forecast_days,
            success=False,
            error_message=f"Prediction failed for {ticker}: {exc}",
        )

    predicted_price = float(future_prices[-1])
    expected_change = calculate_expected_change(current_price, predicted_price)
    last_date = data.index[-1]
    future_dates = generate_future_dates(last_date, forecast_days)

    # --- Sentiment ---
    sentiment = get_news_sentiment(ticker)

    return PredictionResult(
        ticker=ticker,
        current_price=current_price,
        predicted_price=predicted_price,
        expected_change=expected_change,
        future_prices=future_prices,
        future_dates=future_dates,
        historical_data=data,
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
        headlines=sentiment.headlines,
        horizon_days=forecast_days,
        success=True,
    )


def scan_top_stocks(
    tickers: list[str] | None = None,
    forecast_days: int = SCAN_FORECAST_DAYS,
) -> list[ScanResult]:
    """
    Scan a list of stocks and return prediction summaries.

    Models must already be trained. Stocks without trained models are
    marked as unavailable (no training is triggered).

    Args:
        tickers: List of ticker symbols. Defaults to DEMO_TOP_STOCKS.
        forecast_days: Forecast horizon for the scan.

    Returns:
        List of ScanResult instances.
    """
    from src.config import DEMO_TOP_STOCKS

    if tickers is None:
        tickers = DEMO_TOP_STOCKS

    results = []
    for ticker in tickers:
        ticker = ticker.strip().upper()

        if not model_exists(ticker):
            results.append(
                ScanResult(
                    ticker=ticker,
                    current_price=0,
                    predicted_price=0,
                    expected_change=0,
                    horizon_days=forecast_days,
                    sentiment_label="N/A",
                    sentiment_score=0,
                    available=False,
                    error_message=f"Model not trained for {ticker}.",
                )
            )
            continue

        prediction = predict_stock(ticker, forecast_days)

        if prediction.success:
            results.append(
                ScanResult(
                    ticker=prediction.ticker,
                    current_price=prediction.current_price,
                    predicted_price=prediction.predicted_price,
                    expected_change=prediction.expected_change,
                    horizon_days=prediction.horizon_days,
                    sentiment_label=prediction.sentiment_label,
                    sentiment_score=prediction.sentiment_score,
                    available=True,
                )
            )
        else:
            results.append(
                ScanResult(
                    ticker=ticker,
                    current_price=0,
                    predicted_price=0,
                    expected_change=0,
                    horizon_days=forecast_days,
                    sentiment_label="N/A",
                    sentiment_score=0,
                    available=False,
                    error_message=prediction.error_message,
                )
            )

    return results
