"""Shared test fixtures.

Heavy components (the Qwen3 model) are replaced by a deterministic ``FakeEmbedder``
so the wiring — retrieval, re-ranking, persistence, HTTP — is exercised end-to-end
without downloading weights. The real model is covered by the ``integration`` tests.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the 04_recommender modules importable regardless of pytest's invocation dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RecommenderConfig  # noqa: E402
from reranking.sentiment import MockSentimentScorer  # noqa: E402


class FakeEmbedder:
    """Deterministic stand-in for ``Qwen3Embedder``.

    Identical text always maps to the same unit vector, so a query whose title
    matches a corpus title retrieves it with similarity ~1. Queries and documents
    are embedded symmetrically (the prompt is ignored), which is all the wiring
    tests need.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dimension = dim

    def _vec(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def encode_documents(self, texts) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_queries(self, texts) -> np.ndarray:
        return self.encode_documents(texts)


def _reviews(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=16)


@pytest.fixture
def corpus_df() -> pd.DataFrame:
    """Three corpus products with known titles and sentiment proportions."""
    rows: list[dict] = []
    # C1: identical title to storefront S1; all positive -> S(p) = 1.0
    for rating in (5, 4, 5):
        rows.append(_corpus_row("C1", "Smartphone Samsung Galaxy S10", "samsung", rating))
    # C2: all negative -> S(p) = 0.0
    for rating in (1, 2):
        rows.append(_corpus_row("C2", "Geladeira Brastemp Frost Free", "brastemp", rating))
    # C3: half positive -> S(p) = 0.5
    for rating in (5, 1, 4, 2):
        rows.append(_corpus_row("C3", "Notebook Dell Inspiron 15", "dell", rating))
    return _reviews(rows)


@pytest.fixture
def storefront_df() -> pd.DataFrame:
    """Two storefront products; S1 shares C1's title."""
    rows = [
        _corpus_row("S1", "Smartphone Samsung Galaxy S10", "samsung", 5),
        _corpus_row("S2", "Cafeteira Nespresso Inissia", "nespresso", 4),
        _corpus_row("S2", "Cafeteira Nespresso Inissia", "nespresso", 3),
    ]
    return _reviews(rows)


@pytest.fixture
def built_artifacts(tmp_path, corpus_df, storefront_df, fake_embedder):
    """Build corpus + storefront artifacts on disk with the fake embedder."""
    from build_index import build_corpus, build_storefront

    corpus_csv = tmp_path / "corpus.csv"
    train_csv = tmp_path / "train.csv"
    corpus_df.to_csv(corpus_csv, index=False)
    storefront_df.to_csv(train_csv, index=False)

    config = RecommenderConfig(
        corpus_splits=(corpus_csv,),
        storefront_split=train_csv,
        artifacts_dir=tmp_path / "artifacts",
        min_reviews=1,
        recommend_min_reviews=1,  # fixture products have 2-4 reviews; don't filter them
        collection_name="test_corpus",
    )
    scorer = MockSentimentScorer()
    build_corpus(config, fake_embedder, scorer)
    build_storefront(config, fake_embedder, scorer)
    return config, fake_embedder


def _corpus_row(product_id: str, name: str, brand: str, rating: int) -> dict:
    return {
        "product_id": product_id,
        "product_name": name,
        "product_brand": brand,
        "site_category_lv1": "Eletronicos",
        "site_category_lv2": "Diversos",
        "overall_rating": rating,
        "review_text": "texto de avaliacao",
    }
