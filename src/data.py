"""
Data fetching module.

Retrieves historical stock data from Yahoo Finance using yfinance.
Handles errors, validation, and column normalization.
"""

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from src.config import TRAINING_PERIOD_YEARS

logger = logging.getLogger(__name__)


def get_default_date_range(years: int = TRAINING_PERIOD_YEARS):
    """Return (start_date, end_date) spanning the given number of years up to today."""
    end = date.today()
    start = end - timedelta(days=years * 365)
    return start, end


def fetch_stock_data(
    ticker_symbol: str,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> pd.DataFrame | None:
    """
    Fetch historical OHLCV data from Yahoo Finance.

    Args:
        ticker_symbol: Stock ticker (e.g. 'AAPL').
        start_date: Start date for data retrieval.
        end_date: End date for data retrieval.

    Returns:
        A pandas DataFrame with at least a 'Close' column, sorted
        chronologically and with NaN Close values dropped.
        Returns None on error or if data is unavailable.
    """
    if not ticker_symbol or not ticker_symbol.strip():
        logger.warning("Empty ticker symbol provided.")
        return None

    ticker_symbol = ticker_symbol.strip().upper()

    if start_date is None or end_date is None:
        start_date, end_date = get_default_date_range()

    try:
        data = yf.download(
            ticker_symbol,
            start=str(start_date),
            end=str(end_date),
            progress=False,
        )
    except Exception as exc:
        logger.error("Failed to download data for %s: %s", ticker_symbol, exc)
        return None

    if data is None or data.empty:
        logger.warning("No data returned for ticker %s.", ticker_symbol)
        return None

    # Handle MultiIndex columns (happens when downloading a single ticker
    # with newer versions of yfinance)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        logger.warning("'Close' column missing for ticker %s.", ticker_symbol)
        return None

    # Sort chronologically and drop rows with missing Close
    data = data.sort_index()
    data = data.dropna(subset=["Close"])

    if data.empty:
        logger.warning("All Close values are NaN for ticker %s.", ticker_symbol)
        return None

    return data
