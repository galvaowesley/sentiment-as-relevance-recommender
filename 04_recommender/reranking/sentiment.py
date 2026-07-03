"""Per-product sentiment score S(p).

``S(p)`` is the proportion of *positive* reviews of a product and is the second
term of the final re-ranking score. In production it comes from the Pipeline 1
classifier's per-review predictions via :class:`PredictedLabelSentimentScorer`; a
rating-based :class:`MockSentimentScorer` is kept for tests and for the storefront
split, which has no classifier inference.

Any scorer just needs to implement the :class:`SentimentScorer` protocol, so a
different classifier can be dropped in without touching the rest of the engine.
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


class PredictedLabelSentimentScorer:
    """Real S(p) from the Pipeline 1 classifier's per-review predictions.

    ``S(p)`` is the fraction of a product's reviews whose predicted label equals
    ``positive_label``. It reads the hard label column produced during inference
    (e.g. ``inferencia_bertimbau`` with ``1`` = positive), so the score is the
    literal "proportion of positive reviews" from the README/report spec.

    The column is configurable because each classifier writes its own label column
    (``inferencia_tfidf`` for the TF-IDF models, ``llm_predicted_polarity`` for the
    LLMs), letting the same scorer serve any of them.
    """

    def __init__(
        self,
        label_column: str = "inferencia_bertimbau",
        positive_label: int = 1,
    ) -> None:
        self.label_column = label_column
        self.positive_label = positive_label

    def score_product(self, reviews: pd.DataFrame) -> float:
        labels = pd.to_numeric(reviews[self.label_column], errors="coerce").dropna()
        if labels.empty:
            return 0.0
        return float((labels == self.positive_label).mean())
