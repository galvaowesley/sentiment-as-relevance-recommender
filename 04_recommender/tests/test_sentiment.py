"""Unit tests for the mock sentiment scorer."""

from __future__ import annotations

import pandas as pd
import pytest

from reranking.sentiment import MockSentimentScorer, SentimentScorer


def _reviews(ratings: list) -> pd.DataFrame:
    return pd.DataFrame({"overall_rating": ratings})


def test_positive_fraction():
    scorer = MockSentimentScorer(positive_threshold=4)
    assert scorer.score_product(_reviews([5, 4, 1, 2])) == pytest.approx(0.5)
    assert scorer.score_product(_reviews([5, 5, 4])) == pytest.approx(1.0)
    assert scorer.score_product(_reviews([1, 2, 3])) == pytest.approx(0.0)


def test_custom_threshold():
    scorer = MockSentimentScorer(positive_threshold=3)
    assert scorer.score_product(_reviews([1, 2, 3])) == pytest.approx(1 / 3)


def test_empty_and_nan_handling():
    scorer = MockSentimentScorer()
    assert scorer.score_product(_reviews([])) == 0.0
    # Non-numeric / missing ratings are ignored: only the valid 5 counts -> 1.0.
    assert scorer.score_product(_reviews([None, 5])) == pytest.approx(1.0)
    # A NaN rating among negatives leaves only negatives -> 0.0.
    assert scorer.score_product(_reviews([None, 1, 2])) == pytest.approx(0.0)


def test_satisfies_protocol():
    assert isinstance(MockSentimentScorer(), SentimentScorer)
