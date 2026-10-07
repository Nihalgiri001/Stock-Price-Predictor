#!/usr/bin/env python3
"""
Standalone prediction utility.

Usage:
    python predict.py --ticker AAPL --days 15

Loads a pre-trained model and prints a future price forecast.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import SHORT_TERM_DAYS
from src.stock_analysis import predict_stock


def parse_args():
    parser = argparse.ArgumentParser(description="Predict future stock prices.")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker.")
    parser.add_argument(
        "--days",
        type=int,
        default=SHORT_TERM_DAYS,
        help=f"Forecast horizon in trading days (default: {SHORT_TERM_DAYS}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.strip().upper()

    print(f"\nPredicting {args.days} days ahead for {ticker} ...\n")

    result = predict_stock(ticker, args.days)

    if not result.success:
        print(f"✗ {result.error_message}")
        sys.exit(1)

    print(f"  Ticker:          {result.ticker}")
    print(f"  Current Price:   ${result.current_price:.2f}")
    print(f"  Predicted Price: ${result.predicted_price:.2f}")
    print(f"  Expected Change: {result.expected_change:+.2f}%")
    print(f"  Horizon:         {result.horizon_days} trading days")
    print(f"  Sentiment:       {result.sentiment_label} ({result.sentiment_score:.4f})")
    print()

    print("Future Prices:")
    for d, p in zip(result.future_dates, result.future_prices):
        print(f"  {d.strftime('%Y-%m-%d')}  ${p:.2f}")


if __name__ == "__main__":
    main()
