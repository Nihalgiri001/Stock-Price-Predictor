"""
Unit tests for the forecasting module.

Uses synthetic data — no TensorFlow or trained model required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forecasting import calculate_expected_change, generate_future_dates


class TestExpectedChange:
    def test_positive_change(self):
        result = calculate_expected_change(100.0, 110.0)
        assert result == pytest.approx(10.0)

    def test_negative_change(self):
        result = calculate_expected_change(200.0, 180.0)
        assert result == pytest.approx(-10.0)

    def test_no_change(self):
        result = calculate_expected_change(50.0, 50.0)
        assert result == pytest.approx(0.0)

    def test_zero_current_price(self):
        result = calculate_expected_change(0.0, 50.0)
        assert result == 0.0


class TestFutureDates:
    def test_count(self):
        last = pd.Timestamp("2024-01-05")  # Friday
        dates = generate_future_dates(last, 5)
        assert len(dates) == 5

    def test_skips_weekends(self):
        last = pd.Timestamp("2024-01-05")  # Friday
        dates = generate_future_dates(last, 3)
        # Next business days: Mon 8, Tue 9, Wed 10
        assert dates[0] == pd.Timestamp("2024-01-08")
        assert dates[1] == pd.Timestamp("2024-01-09")
        assert dates[2] == pd.Timestamp("2024-01-10")

    def test_empty(self):
        last = pd.Timestamp("2024-01-05")
        dates = generate_future_dates(last, 0)
        assert len(dates) == 0


class TestModelPathConstruction:
    """Test that model directory paths are constructed correctly."""

    def test_model_dir(self):
        from src.model import get_model_dir

        path = get_model_dir("AAPL")
        assert path.name == "AAPL"
        assert "models" in str(path)

    def test_uppercasing(self):
        from src.model import get_model_dir

        path = get_model_dir("  aapl  ")
        assert path.name == "AAPL"
