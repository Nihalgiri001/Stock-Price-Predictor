"""
Centralized configuration for the Stock Price Predictor project.

All configurable parameters are defined here to avoid scattering
magic numbers throughout the codebase.
"""

from pathlib import Path

# ==================================================
# Project Paths
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ==================================================
# LSTM Model Parameters
# ==================================================
LOOK_BACK = 60          # Number of past days used as input sequence
LSTM_UNITS = 50         # Units per LSTM layer
DROPOUT_RATE = 0.2      # Dropout rate after each LSTM layer

# ==================================================
# Training Parameters
# ==================================================
EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 5

# ==================================================
# Data Parameters
# ==================================================
TRAINING_PERIOD_YEARS = 3

# ==================================================
# Forecast Horizons
# ==================================================
SHORT_TERM_DAYS = 15
LONG_TERM_DAYS = 30
SCAN_FORECAST_DAYS = 30  # Used for top-stock scanner

# ==================================================
# Sentiment Thresholds (VADER)
# ==================================================
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# ==================================================
# Demo Top Stocks
# ==================================================
DEMO_TOP_STOCKS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "TSLA",
    "META",
    "JPM",
]

# ==================================================
# Model File Names
# ==================================================
MODEL_FILENAME = "lstm_model.keras"
SCALER_FILENAME = "scaler.pkl"
METADATA_FILENAME = "metadata.json"
