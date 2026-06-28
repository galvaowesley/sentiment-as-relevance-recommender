"""Per-product sentiment score S(p).

``S(p)`` is the proportion of *positive* reviews of a product and is the second
term of the final re-ranking score. The real value will come from the Pipeline 1
sentiment classifier; until that model is ready we use a rating-based placeholder.

Any scorer just needs to implement the :class:`SentimentScorer` protocol, so the
real classifier can be dropped in without touching the rest of the engine.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class SentimentScorer(Protocol):
    """Computes S(p) in ``[0, 1]`` from a product's reviews."""

    def score_product(self, reviews: pd.DataFrame) -> float:
        """Return the positive-review proportion for one product's reviews."""
        ...


class MockSentimentScorer:
    """Rating-based stand-in for the Pipeline 1 sentiment classifier.

    ``S(p)`` is approximated by the fraction of a product's reviews whose
    ``overall_rating`` is at least ``positive_threshold``. This keeps re-ranking
    meaningful and testable today; the production scorer will implement the same
    ``score_product`` contract over ``review_text`` instead of the star rating.
    """

    def __init__(self, positive_threshold: int = 4) -> None:
        self.positive_threshold = positive_threshold

    def score_product(self, reviews: pd.DataFrame) -> float:
        ratings = pd.to_numeric(reviews["overall_rating"], errors="coerce").dropna()
        if ratings.empty:
            return 0.0
        return float((ratings >= self.positive_threshold).mean())
