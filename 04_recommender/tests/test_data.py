"""Unit tests for split loading and product-catalog construction."""

from __future__ import annotations

import pandas as pd
import pytest

from data import (
    CATALOG_COLUMNS,
    build_catalog,
    catalog_to_frame,
    compute_sentiment_scores,
    load_reviews,
)
from reranking.sentiment import MockSentimentScorer


def test_build_catalog_dedup_and_scores(corpus_df):
    records = build_catalog(corpus_df, MockSentimentScorer(), min_reviews=1)
    by_id = {r.product_id: r for r in records}

    assert set(by_id) == {"C1", "C2", "C3"}
    assert by_id["C1"].num_reviews == 3
    assert by_id["C1"].sentiment_score == pytest.approx(1.0)
    assert by_id["C2"].sentiment_score == pytest.approx(0.0)
    assert by_id["C3"].sentiment_score == pytest.approx(0.5)
    assert by_id["C1"].product_name == "Smartphone Samsung Galaxy S10"


def test_min_reviews_filter(corpus_df):
    records = build_catalog(corpus_df, MockSentimentScorer(), min_reviews=3)
    assert {r.product_id for r in records} == {"C1", "C3"}  # C2 has only 2 reviews


def test_sorted_by_review_count_and_limit(corpus_df):
    records = build_catalog(corpus_df, MockSentimentScorer(), min_reviews=1, limit=2)
    # C3 (4 reviews) then C1 (3 reviews); C2 (2) dropped by the limit.
    assert [r.product_id for r in records] == ["C3", "C1"]


def test_catalog_to_frame_columns(corpus_df):
    records = build_catalog(corpus_df, MockSentimentScorer())
    frame = catalog_to_frame(records)
    assert list(frame.columns) == list(CATALOG_COLUMNS)


def test_compute_sentiment_scores():
    df = pd.DataFrame(
        {
            "product_id": ["P1", "P1", "P1", "P1", "P2", "P2"],
            "inferencia_bertimbau": [1, 1, 1, 0, 0, 0],
        }
    )
    scores = compute_sentiment_scores(df).set_index("product_id")

    assert set(scores.columns) == {"num_reviews", "num_positive", "sentiment_score"}
    assert scores.loc["P1", "num_reviews"] == 4
    assert scores.loc["P1", "num_positive"] == 3
    assert scores.loc["P1", "sentiment_score"] == pytest.approx(0.75)
    assert scores.loc["P2", "sentiment_score"] == pytest.approx(0.0)


def test_compute_sentiment_scores_missing_column():
    df = pd.DataFrame({"product_id": ["P1"]})
    with pytest.raises(ValueError):
        compute_sentiment_scores(df)


def test_load_reviews_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reviews([tmp_path / "nope.csv"])


def test_load_reviews_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"product_id": ["1"], "product_name": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_reviews([path])
