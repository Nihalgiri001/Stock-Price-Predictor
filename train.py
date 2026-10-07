#!/usr/bin/env python3
"""
LSTM Stock Model — Manual Training Script.

Usage:
    python train.py --ticker AAPL
    python train.py --ticker MSFT --epochs 20 --years 5

This script downloads historical data, preprocesses it, builds the LSTM,
trains it, and saves the model artifacts.  It is intended to be run
**manually** by the user — the Streamlit app never calls this script.
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LOOK_BACK,
    OUTPUTS_DIR,
    TRAINING_PERIOD_YEARS,
    VALIDATION_SPLIT,
)
from src.data import fetch_stock_data
from src.model import build_lstm_model, save_model_artifacts
from src.preprocessing import (
    create_dataset,
    preprocess_data,
    reshape_for_lstm,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an LSTM model for stock price prediction."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, TSLA, RELIANCE.NS).",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=TRAINING_PERIOD_YEARS,
        help=f"Years of historical data to use (default: {TRAINING_PERIOD_YEARS}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Training batch size (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--look-back",
        type=int,
        default=LOOK_BACK,
        help=f"Look-back window size (default: {LOOK_BACK}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ticker = args.ticker.strip().upper()
    years = args.years
    epochs = args.epochs
    batch_size = args.batch_size
    look_back = args.look_back

    print("=" * 50)
    print("       LSTM STOCK MODEL TRAINING")
    print("=" * 50)
    print()
    print(f"  Ticker:           {ticker}")
    print(f"  Training period:  {years} years")
    print(f"  Look-back:        {look_back}")
    print(f"  Epochs:           {epochs}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Validation split: {VALIDATION_SPLIT}")
    print(f"  Early stopping:   patience={EARLY_STOPPING_PATIENCE}")
    print()
    print("=" * 50)
    print()

    # ------------------------------------------------------------------
    # 1. Fetch historical data
    # ------------------------------------------------------------------
    end_date = date.today()
    start_date = end_date - timedelta(days=years * 365)

    print(f"[1/7] Fetching data for {ticker} from {start_date} to {end_date} ...")
    data = fetch_stock_data(ticker, start_date, end_date)

    if data is None or data.empty:
        print(f"\n✗ ERROR: No data returned for {ticker}. Check the ticker symbol.")
        sys.exit(1)

    print(f"       Retrieved {len(data)} data points.")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print("[2/7] Preprocessing Close prices ...")
    scaled_data, scaler, close_prices = preprocess_data(data)
    print(f"       Scaled data shape: {scaled_data.shape}")

    # ------------------------------------------------------------------
    # 3. Create supervised sequences
    # ------------------------------------------------------------------
    print(f"[3/7] Creating {look_back}-day sequences ...")

    if len(scaled_data) <= look_back:
        print(
            f"\n✗ ERROR: Not enough data ({len(scaled_data)} points) "
            f"for look-back of {look_back}."
        )
        sys.exit(1)

    X, y = create_dataset(scaled_data, look_back)
    X = reshape_for_lstm(X)
    print(f"       X shape: {X.shape}  |  y shape: {y.shape}")

    # ------------------------------------------------------------------
    # 4. Build LSTM model
    # ------------------------------------------------------------------
    print("[4/7] Building LSTM model ...")
    model = build_lstm_model(look_back)
    model.summary()
    print()

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("[5/7] Training ...")
    from tensorflow.keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping],
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 6. Save artifacts
    # ------------------------------------------------------------------
    print("\n[6/7] Saving model artifacts ...")

    metadata = {
        "ticker": ticker,
        "look_back": look_back,
        "epochs": epochs,
        "batch_size": batch_size,
        "training_period_years": years,
        "validation_split": VALIDATION_SPLIT,
        "trained_at": datetime.now().isoformat(),
        "data_points": len(data),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }

    model_dir = save_model_artifacts(model, scaler, metadata, ticker)

    print()
    print(f"  Model saved to:    {model_dir / 'lstm_model.keras'}")
    print(f"  Scaler saved to:   {model_dir / 'scaler.pkl'}")
    print(f"  Metadata saved to: {model_dir / 'metadata.json'}")

    # ------------------------------------------------------------------
    # 7. Optional: save training history
    # ------------------------------------------------------------------
    print("\n[7/7] Saving training history ...")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = OUTPUTS_DIR / f"{ticker}_training_history.json"

    history_data = {
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
    }
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)

    print(f"  History saved to: {history_path}")

    print()
    print("=" * 50)
    print("  ✓ Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
