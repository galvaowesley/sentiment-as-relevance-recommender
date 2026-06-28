"""Sentiment-weighted re-ranking.

Combines semantic similarity with the per-product sentiment score:

    Score_final(p) = alpha * sim(q, p) + (1 - alpha) * S(p)

The function is pure (no I/O, no model), which keeps it trivial to unit-test.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """A retrieved product awaiting re-ranking."""

    product_id: str
    similarity: float  # cosine similarity sim(q, p) in [-1, 1]
    sentiment_score: float  # S(p) in [0, 1]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Recommendation:
    """A re-ranked product returned to the caller."""

    product_id: str
    score: float  # Score_final
    similarity: float
    sentiment_score: float
    metadata: dict[str, Any]


def rerank(
    candidates: Iterable[Candidate],
    alpha: float,
    top_n: int | None = None,
    exclude_id: str | None = None,
) -> list[Recommendation]:
    """Re-rank candidates by ``alpha*sim + (1-alpha)*S(p)``.

    Args:
        candidates: retrieved products with similarity and sentiment scores.
        alpha: trade-off between semantic relevance (1.0) and sentiment (0.0).
        top_n: cap on the number of results; ``None`` returns all.
        exclude_id: product to drop (e.g. the query product itself).

    Returns:
        Recommendations sorted by descending final score (ties broken by id).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    ranked: list[Recommendation] = []
    for cand in candidates:
        if exclude_id is not None and cand.product_id == exclude_id:
            continue
        final = alpha * cand.similarity + (1.0 - alpha) * cand.sentiment_score
        ranked.append(
            Recommendation(
                product_id=cand.product_id,
                score=final,
                similarity=cand.similarity,
                sentiment_score=cand.sentiment_score,
                metadata=cand.metadata,
            )
        )

    ranked.sort(key=lambda rec: (-rec.score, rec.product_id))
    return ranked[:top_n] if top_n is not None else ranked
