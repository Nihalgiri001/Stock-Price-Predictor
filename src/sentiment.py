"""
Sentiment analysis module using VADER.

Generates simulated financial headlines for a given ticker and
computes an overall sentiment score and label.
"""

import logging
from typing import NamedTuple

import nltk
import numpy as np

from src.config import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VADER Lexicon Setup
# ---------------------------------------------------------------------------


def ensure_vader_lexicon():
    """Download the VADER lexicon if it is not already available."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


# ---------------------------------------------------------------------------
# Simulated Headlines
# ---------------------------------------------------------------------------


def get_demo_headlines(ticker: str) -> list[str]:
    """
    Return simulated financial headlines for the given ticker.

    These are illustrative headlines used for demonstration purposes.
    They are NOT sourced from real-time news feeds.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        List of headline strings.
    """
    return [
        f"{ticker} reports strong earnings, exceeding expectations.",
        f"Analysts positive on {ticker}'s new product launch.",
        f"{ticker} faces minor supply chain disruptions.",
        f"Market reacts cautiously to {ticker}'s quarterly outlook.",
        f"{ticker} stock falls slightly on broader market concerns.",
        f"{ticker} announces new strategic partnership.",
        f"Concerns about {ticker}'s competition in the market.",
        f"{ticker} CEO optimistic about future growth.",
        f"Regulatory challenges could impact {ticker} next quarter.",
        f"{ticker} beats revenue forecasts, shares rise in after-hours trading.",
        f"Investors remain bullish on {ticker} despite sector headwinds.",
        f"{ticker} expands into new international markets.",
    ]


# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------


class SentimentResult(NamedTuple):
    """Container for sentiment analysis results."""

    label: str              # "Positive", "Negative", or "Neutral"
    score: float            # Average compound score
    headlines: list[str]    # Headlines that were analyzed
    scores: list[float]     # Individual compound scores


def get_news_sentiment(ticker: str) -> SentimentResult:
    """
    Analyze sentiment for the given ticker using simulated headlines and VADER.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        SentimentResult with label, average score, headlines, and per-headline scores.
    """
    ensure_vader_lexicon()

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    headlines = get_demo_headlines(ticker.strip().upper())

    if not headlines:
        return SentimentResult(
            label="Neutral",
            score=0.0,
            headlines=[],
            scores=[],
        )

    compound_scores = []
    for headline in headlines:
        scores = analyzer.polarity_scores(headline)
        compound_scores.append(scores["compound"])

    average_score = float(np.mean(compound_scores))

    if average_score >= POSITIVE_THRESHOLD:
        label = "Positive"
    elif average_score <= NEGATIVE_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    return SentimentResult(
        label=label,
        score=average_score,
        headlines=headlines,
        scores=compound_scores,
    )
