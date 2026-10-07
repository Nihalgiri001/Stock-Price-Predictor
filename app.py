#!/usr/bin/env python3
"""
AI-Powered Stock Analyzer & Predictor — Streamlit Application.

Entry point:
    streamlit run app.py

This app loads pre-trained LSTM models and presents an interactive
dashboard for stock-price prediction and sentiment analysis.
It does NOT train models — see train.py for that.
"""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    DEMO_TOP_STOCKS,
    LONG_TERM_DAYS,
    LOOK_BACK,
    SCAN_FORECAST_DAYS,
    SHORT_TERM_DAYS,
)
from src.data import fetch_stock_data, get_default_date_range
from src.model import model_exists
from src.sentiment import get_news_sentiment
from src.stock_analysis import predict_stock, scan_top_stocks
from src.visualization import create_price_chart

# =====================================================================
# Page Config
# =====================================================================
st.set_page_config(
    layout="wide",
    page_title="Stock Price Predictor",
    page_icon="📈",
)

# =====================================================================
# Session State
# =====================================================================
if "selected_stock_ticker" not in st.session_state:
    st.session_state.selected_stock_ticker = None
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.header("⚙️ Prediction Settings")

    duration_option = st.radio(
        "Prediction Duration",
        options=["Short Term", "Long Term"],
        index=0,
        help=f"Short Term = {SHORT_TERM_DAYS} trading days, "
             f"Long Term = {LONG_TERM_DAYS} trading days.",
    )

    forecast_days = SHORT_TERM_DAYS if duration_option == "Short Term" else LONG_TERM_DAYS

    st.markdown("---")
    st.markdown(
        "📊 **About This Tool**\n\n"
        "This application uses an LSTM neural network trained on "
        "historical closing prices to generate multi-step stock-price "
        "forecasts, combined with VADER-based sentiment analysis on "
        "simulated financial headlines."
    )

    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer**\n\n"
        "This tool is intended for **educational and analytical purposes only** "
        "and does not constitute financial advice. Stock markets are highly "
        "volatile — predictions should not be interpreted as guaranteed future prices."
    )

# =====================================================================
# Main Header
# =====================================================================
st.title("📈 AI-Powered Stock Analyzer & Predictor")
st.caption(
    "LSTM-based price forecasting · VADER sentiment analysis · "
    "Educational use only"
)

# =====================================================================
# Section 1 — Individual Stock Prediction
# =====================================================================
st.markdown("---")
st.subheader("🔍 Individual Stock Prediction")

col_input, col_button = st.columns([3, 1])

with col_input:
    ticker_input = st.text_input(
        "Enter Stock Ticker",
        placeholder="e.g. AAPL, TSLA, RELIANCE.NS, NVDA",
        help="Enter any valid Yahoo Finance ticker symbol.",
    )

with col_button:
    st.write("")  # spacer
    st.write("")  # spacer
    predict_clicked = st.button("🚀 Predict", use_container_width=True)

if predict_clicked and ticker_input:
    ticker = ticker_input.strip().upper()

    if not ticker:
        st.error("Please enter a valid stock ticker.")
    else:
        with st.spinner(f"Analyzing {ticker} ..."):
            result = predict_stock(ticker, forecast_days)

        if not result.success:
            st.error(result.error_message)
        else:
            # ---- Metrics row ----
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Price", f"${result.current_price:.2f}")
            with col2:
                st.metric(
                    f"Predicted Price ({result.horizon_days}d)",
                    f"${result.predicted_price:.2f}",
                )
            with col3:
                delta_color = "normal"
                st.metric(
                    "Expected Change",
                    f"{result.expected_change:+.2f}%",
                    delta=f"{result.expected_change:+.2f}%",
                )
            with col4:
                st.metric("Forecast Horizon", f"{result.horizon_days} days")

            # ---- Chart ----
            st.markdown("#### 📉 Historical & Predicted Prices")
            fig = create_price_chart(
                result.historical_data,
                result.future_dates,
                result.future_prices,
                ticker,
            )
            st.pyplot(fig)

            # ---- Sentiment ----
            st.markdown("#### 💬 Sentiment Analysis")
            st.info(
                "ℹ️ Sentiment is computed using **VADER** on **simulated demo headlines**. "
                "These are not sourced from real-time news."
            )

            sent_col1, sent_col2 = st.columns(2)
            with sent_col1:
                color_map = {
                    "Positive": "🟢",
                    "Negative": "🔴",
                    "Neutral": "🟡",
                }
                icon = color_map.get(result.sentiment_label, "⚪")
                st.metric(
                    "Overall Sentiment",
                    f"{icon} {result.sentiment_label}",
                )
            with sent_col2:
                st.metric("Sentiment Score", f"{result.sentiment_score:.4f}")

            with st.expander("📰 Headlines Analyzed", expanded=False):
                for h in result.headlines:
                    st.markdown(f"- {h}")

            # ---- Prediction Table ----
            with st.expander("📅 Day-by-Day Forecast", expanded=False):
                forecast_df = pd.DataFrame({
                    "Date": result.future_dates.strftime("%Y-%m-%d"),
                    "Predicted Price ($)": [f"{p:.2f}" for p in result.future_prices],
                })
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)


# =====================================================================
# Section 2 — Top Stock Scanner
# =====================================================================
st.markdown("---")
st.subheader("🏆 Top Stock Scanner")

st.caption(
    f"Scans {len(DEMO_TOP_STOCKS)} predefined stocks with a "
    f"{SCAN_FORECAST_DAYS}-day forecast horizon. "
    f"Models must be trained before scanning."
)

if st.button("📡 Scan Top 8 Stocks", use_container_width=False):
    with st.spinner("Scanning stocks — this may take a moment …"):
        scan_results = scan_top_stocks(forecast_days=SCAN_FORECAST_DAYS)
        st.session_state.scan_results = scan_results

if st.session_state.scan_results is not None:
    scan_results = st.session_state.scan_results

    available = [r for r in scan_results if r.available]
    unavailable = [r for r in scan_results if not r.available]

    if unavailable:
        missing_tickers = ", ".join(r.ticker for r in unavailable)
        st.warning(
            f"⚠️ Models not available for: **{missing_tickers}**. "
            f"Train them with `python train.py --ticker TICKER`."
        )

    if available:
        # Build results table
        table_data = []
        for r in sorted(available, key=lambda x: x.expected_change, reverse=True):
            table_data.append({
                "Ticker": r.ticker,
                "Current Price ($)": f"{r.current_price:.2f}",
                "Predicted Price ($)": f"{r.predicted_price:.2f}",
                "Expected Change (%)": f"{r.expected_change:+.2f}",
                "Horizon (days)": r.horizon_days,
                "Sentiment": r.sentiment_label,
            })

        results_df = pd.DataFrame(table_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Selection mechanism
        st.markdown("**Select a stock for detailed analysis:**")
        ticker_options = [r.ticker for r in sorted(available, key=lambda x: x.expected_change, reverse=True)]
        selected = st.selectbox(
            "Choose ticker",
            options=ticker_options,
            index=0,
            label_visibility="collapsed",
        )
        if st.button("📊 View Detailed Analysis"):
            st.session_state.selected_stock_ticker = selected

    else:
        st.info(
            "No trained models found for any of the top stocks. "
            "Please train at least one model first."
        )


# =====================================================================
# Section 3 — Detailed Stock View
# =====================================================================
if st.session_state.selected_stock_ticker:
    ticker = st.session_state.selected_stock_ticker
    st.markdown("---")
    st.subheader(f"📋 Detailed Analysis — {ticker}")

    with st.spinner(f"Loading detailed analysis for {ticker} ..."):
        detail_result = predict_stock(ticker, SCAN_FORECAST_DAYS)

    if not detail_result.success:
        st.error(detail_result.error_message)
    else:
        # Metrics
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.metric("Current Price", f"${detail_result.current_price:.2f}")
        with d_col2:
            st.metric(
                f"Predicted Price ({detail_result.horizon_days}d)",
                f"${detail_result.predicted_price:.2f}",
            )
        with d_col3:
            st.metric(
                "Expected Change",
                f"{detail_result.expected_change:+.2f}%",
                delta=f"{detail_result.expected_change:+.2f}%",
            )

        # Chart
        fig = create_price_chart(
            detail_result.historical_data,
            detail_result.future_dates,
            detail_result.future_prices,
            ticker,
        )
        st.pyplot(fig)

        # Sentiment
        st.markdown("#### 💬 Sentiment Analysis")
        st.info(
            "ℹ️ Headlines below are **simulated for demonstration**. "
            "They do not represent real-time financial news."
        )

        s1, s2 = st.columns(2)
        with s1:
            color_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
            icon = color_map.get(detail_result.sentiment_label, "⚪")
            st.metric("Overall Sentiment", f"{icon} {detail_result.sentiment_label}")
        with s2:
            st.metric("Sentiment Score", f"{detail_result.sentiment_score:.4f}")

        with st.expander("📰 Headlines Analyzed", expanded=True):
            for h in detail_result.headlines:
                st.markdown(f"- {h}")

        # Day-by-day forecast
        with st.expander("📅 Day-by-Day Forecast", expanded=False):
            forecast_df = pd.DataFrame({
                "Date": detail_result.future_dates.strftime("%Y-%m-%d"),
                "Predicted Price ($)": [f"{p:.2f}" for p in detail_result.future_prices],
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)


# =====================================================================
# Section 4 — Model Limitations & Info
# =====================================================================
st.markdown("---")
with st.expander("ℹ️ Model Limitations & Disclaimers", expanded=False):
    st.markdown("""
**Important considerations:**

- **Market volatility:** Stock markets are inherently volatile and influenced by
  countless unpredictable factors. Historical patterns do not guarantee future
  performance.

- **Recursive forecasting errors:** The LSTM model predicts one day at a time and
  feeds each prediction back as input. Errors can accumulate over longer horizons,
  making long-term forecasts less reliable.

- **Simulated headlines:** The sentiment analysis uses demonstrative, simulated
  financial headlines — not real-time news. Sentiment scores are illustrative only.

- **Educational purpose:** This system is designed for learning and analytical
  exploration. Predictions should **never** be interpreted as financial advice.

- **No guarantee of accuracy:** Model performance depends on training data quality,
  market conditions, and many other factors. Past accuracy does not imply future
  reliability.
    """)

st.markdown(
    "<div style='text-align: center; color: gray; padding: 20px;'>"
    "AI-Powered Stock Analyzer & Predictor · Educational Use Only · "
    "Built with Streamlit, TensorFlow, yfinance & VADER"
    "</div>",
    unsafe_allow_html=True,
)
