"""
Visualization module.

Creates matplotlib figures for historical and predicted stock prices.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def create_price_chart(
    historical_data: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    future_prices: np.ndarray,
    ticker: str,
) -> plt.Figure:
    """
    Create a combined historical + predicted price chart.

    Args:
        historical_data: DataFrame with DatetimeIndex and 'Close' column.
        future_dates: DatetimeIndex of predicted dates.
        future_prices: Array of predicted prices.
        ticker: Stock ticker for the title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Use a clean style
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    # Historical prices (solid blue)
    ax.plot(
        historical_data.index,
        historical_data["Close"],
        color="#4FC3F7",
        linewidth=1.5,
        label="Historical Price",
    )

    # Predicted prices (red dashed)
    if len(future_dates) > 0 and len(future_prices) > 0:
        # Connect from last historical point
        connect_dates = pd.DatetimeIndex(
            [historical_data.index[-1]]
        ).append(future_dates)
        connect_prices = np.concatenate(
            [[historical_data["Close"].iloc[-1]], future_prices]
        )

        ax.plot(
            connect_dates,
            connect_prices,
            color="#EF5350",
            linewidth=2,
            linestyle="--",
            label="Predicted Price",
        )

    ax.set_title(
        f"{ticker} — Historical & Predicted Prices",
        fontsize=16,
        fontweight="bold",
        color="white",
        pad=15,
    )
    ax.set_xlabel("Date", fontsize=12, color="white")
    ax.set_ylabel("Price ($)", fontsize=12, color="white")

    ax.legend(fontsize=11, loc="upper left", facecolor="#1E1E1E", edgecolor="gray",
              labelcolor="white")

    ax.tick_params(colors="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()

    ax.grid(True, alpha=0.15, color="white")

    for spine in ax.spines.values():
        spine.set_color("#333333")

    fig.tight_layout()
    return fig
