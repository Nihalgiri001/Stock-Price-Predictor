"""
Unit tests for the sentiment analysis module.

No external API or TensorFlow required.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sentiment import (
    SentimentResult,
    get_demo_headlines,
    get_news_sentiment,
)


class TestDemoHeadlines:
    def test_non_empty(self):
        headlines = get_demo_headlines("AAPL")
        assert len(headlines) > 0

    def test_contains_ticker(self):
        headlines = get_demo_headlines("TSLA")
        for h in headlines:
            assert "TSLA" in h

    def test_different_tickers(self):
        h1 = get_demo_headlines("AAPL")
        h2 = get_demo_headlines("MSFT")
        assert h1 != h2


class TestSentimentAnalysis:
    def test_returns_sentiment_result(self):
        result = get_news_sentiment("AAPL")
        assert isinstance(result, SentimentResult)
        assert result.label in ("Positive", "Negative", "Neutral")
        assert isinstance(result.score, float)
        assert len(result.headlines) > 0
        assert len(result.scores) == len(result.headlines)

    def test_score_range(self):
        result = get_news_sentiment("GOOGL")
        assert -1.0 <= result.score <= 1.0
        for s in result.scores:
            assert -1.0 <= s <= 1.0

    def test_classification_thresholds(self):
        """Verify the classification logic matches the report thresholds."""
        from src.config import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD

        result = get_news_sentiment("META")
        if result.score >= POSITIVE_THRESHOLD:
            assert result.label == "Positive"
        elif result.score <= NEGATIVE_THRESHOLD:
            assert result.label == "Negative"
        else:
            assert result.label == "Neutral"
