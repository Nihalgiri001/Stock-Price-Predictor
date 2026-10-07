# AI-Powered Stock Price Prediction and Sentiment-Driven Market Analysis

An interactive **Streamlit** web application that uses an **LSTM neural network** to predict future stock prices and **VADER** sentiment analysis on financial headlines — built for educational and analytical exploration.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-red)

---

## Features

- **Dynamic data retrieval** from Yahoo Finance via `yfinance`
- **LSTM model** with two stacked layers (50 units each) + Dropout
- **MinMaxScaler** preprocessing with 60-day sliding window
- **Recursive multi-step forecasting** (15 or 30 trading days)
- **VADER sentiment analysis** on simulated financial headlines
- **Top-8 stock scanner** for comparative analysis
- **Interactive Streamlit dashboard** with charts, metrics, and tables
- **Separate training script** — models are trained offline

---

## Project Structure

```
stock-price-predictor/
├── app.py                  # Streamlit application (entry point)
├── train.py                # Manual training script (CLI)
├── predict.py              # Standalone prediction utility
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Centralized configuration
│   ├── data.py             # Yahoo Finance data fetching
│   ├── preprocessing.py    # Scaling & dataset creation
│   ├── model.py            # LSTM architecture, save/load
│   ├── forecasting.py      # Recursive prediction & date utils
│   ├── sentiment.py        # VADER sentiment analysis
│   ├── stock_analysis.py   # High-level prediction pipeline
│   └── visualization.py    # Matplotlib chart generation
│
├── models/                 # Trained model artifacts (per ticker)
│   └── .gitkeep
│
├── outputs/                # Training history logs
│   └── .gitkeep
│
└── tests/
    ├── test_preprocessing.py
    ├── test_forecasting.py
    └── test_sentiment.py
```

---

## Prerequisites

- **Python 3.10, 3.11, or 3.12** (recommended for TensorFlow compatibility)
- **macOS / Linux / Windows** (cross-platform)
- **Internet connection** (for Yahoo Finance data retrieval)

---

## Setup

### 1. Create a virtual environment

```bash
cd stock-price-predictor

python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import yfinance as yf; print('yfinance:', yf.__version__)"
python -c "import streamlit as st; print('Streamlit:', st.__version__)"
```

### 4. VADER lexicon

The application automatically downloads the VADER lexicon on first use. Manual fallback:

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## Training

> **Important:** Training is a manual, offline step. The Streamlit app does not train models.

### Train a single stock

```bash
python train.py --ticker AAPL
```

### Train with custom parameters

```bash
python train.py --ticker TSLA --years 5 --epochs 20 --batch-size 64 --look-back 60
```

### CLI arguments

| Argument       | Default | Description                          |
|----------------|---------|--------------------------------------|
| `--ticker`     | *required* | Stock ticker (e.g. AAPL, RELIANCE.NS) |
| `--years`      | 3       | Years of historical data             |
| `--epochs`     | 10      | Training epochs                      |
| `--batch-size` | 32      | Batch size                           |
| `--look-back`  | 60      | Sliding window length                |

### Training output

After training completes, the following files are created:

```
models/AAPL/
├── lstm_model.keras    # Trained Keras model
├── scaler.pkl          # Fitted MinMaxScaler
└── metadata.json       # Training metadata

outputs/
└── AAPL_training_history.json   # Loss history (optional)
```

### Train all top stocks (for the scanner)

```bash
python train.py --ticker AAPL
python train.py --ticker MSFT
python train.py --ticker GOOGL
python train.py --ticker AMZN
python train.py --ticker NVDA
python train.py --ticker TSLA
python train.py --ticker META
python train.py --ticker JPM
```

---

## Running the Application

After training at least one model:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

### What happens if a model is missing?

The app displays a clear message:

> "Trained model not found for TICKER. Please train the model first using:
> `python train.py --ticker TICKER`"

No automatic training is triggered.

---

## Usage Guide

1. **Sidebar** — Select prediction duration (Short Term: 15 days, Long Term: 30 days)
2. **Individual Prediction** — Enter a ticker and click "Predict"
3. **Top Stock Scanner** — Click "Scan Top 8 Stocks" for a comparative overview
4. **Detailed Analysis** — Select a stock from the scan results for in-depth view
5. **Sentiment** — View VADER analysis on simulated financial headlines

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use synthetic data and do not require TensorFlow training.

---

## Architecture

```
USER → Streamlit UI → Stock Ticker + Forecast Duration
                        │
                        ▼
                    yfinance → Historical Close Prices
                        │
                        ▼
                    MinMaxScaler → 60-Day Window → LSTM Model
                        │
                        ▼
                    Recursive Prediction → Inverse Scaling
                        │
                        ├── Historical + Predicted Chart
                        │
                        └── VADER Sentiment → Headlines → Score → Label
                                │
                                ▼
                         Integrated Dashboard
```

---

## Model Architecture

```
Sequential
  ├── LSTM(50, return_sequences=True, input_shape=(60, 1))
  ├── Dropout(0.2)
  ├── LSTM(50, return_sequences=False)
  ├── Dropout(0.2)
  └── Dense(1)

Optimizer: Adam
Loss: Mean Squared Error
Early Stopping: patience=5, monitor=val_loss, restore_best_weights=True
```

---

## Disclaimer

This tool is intended for **educational and analytical purposes only** and does not constitute financial advice. Stock markets are highly volatile — predictions should not be interpreted as guaranteed future prices.

---

## License

This project is for academic / educational use.
