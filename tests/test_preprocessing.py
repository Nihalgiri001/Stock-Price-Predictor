"""
Unit tests for the preprocessing module.

Uses small synthetic data — no TensorFlow required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    create_dataset,
    extract_close_prices,
    fit_scaler,
    reshape_for_lstm,
    scale_data,
)


class TestExtractClosePrices:
    def test_basic(self):
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
        result = extract_close_prices(df)
        assert result.shape == (3, 1)
        assert result[0, 0] == 100.0

    def test_drops_nan(self):
        df = pd.DataFrame({"Close": [100.0, np.nan, 102.0]})
        result = extract_close_prices(df)
        assert result.shape == (2, 1)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"Open": [100.0]})
        with pytest.raises(ValueError, match="Close"):
            extract_close_prices(df)

    def test_empty_raises(self):
        df = pd.DataFrame({"Close": pd.Series([], dtype=float)})
        with pytest.raises(ValueError, match="No valid"):
            extract_close_prices(df)


class TestScaler:
    def test_fit_and_scale(self):
        prices = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
        scaler = fit_scaler(prices)
        scaled = scale_data(prices, scaler)
        assert scaled.min() == pytest.approx(0.0)
        assert scaled.max() == pytest.approx(1.0)

    def test_inverse(self):
        prices = np.array([[100.0], [200.0], [300.0]])
        scaler = fit_scaler(prices)
        scaled = scale_data(prices, scaler)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, prices, atol=1e-6)


class TestCreateDataset:
    def test_shapes(self):
        data = np.arange(100).reshape(-1, 1).astype(float)
        X, y = create_dataset(data, look_back=10)
        assert X.shape == (90, 10)
        assert y.shape == (90,)

    def test_values(self):
        data = np.arange(15).reshape(-1, 1).astype(float)
        X, y = create_dataset(data, look_back=5)
        np.testing.assert_array_equal(X[0], [0, 1, 2, 3, 4])
        assert y[0] == 5
        np.testing.assert_array_equal(X[1], [1, 2, 3, 4, 5])
        assert y[1] == 6

    def test_insufficient_data_raises(self):
        data = np.arange(5).reshape(-1, 1).astype(float)
        with pytest.raises(ValueError, match="Need more than"):
            create_dataset(data, look_back=10)

    def test_exact_boundary(self):
        # Exactly look_back + 1 points → 1 sample
        data = np.arange(11).reshape(-1, 1).astype(float)
        X, y = create_dataset(data, look_back=10)
        assert X.shape == (1, 10)
        assert y.shape == (1,)


class TestReshapeForLSTM:
    def test_shape(self):
        X = np.zeros((50, 60))
        result = reshape_for_lstm(X)
        assert result.shape == (50, 60, 1)
