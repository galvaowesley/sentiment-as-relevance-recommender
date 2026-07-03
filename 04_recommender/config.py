"""Central configuration for the recommendation engine.

All tunable knobs (data splits, embedding model, retrieval/re-rank parameters and
on-disk artifact locations) live here so the library, the build script and the
web service share a single source of truth.

Paths are resolved relative to the repository layout, so the engine works
regardless of the machine it runs on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_DATA_DIR = _REPO_ROOT / "data" / "processed"

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_QUERY_INSTRUCTION = (
    "Given a retail product title, retrieve titles of similar products"
)


def _corpus_default_splits() -> tuple[Path, ...]:
    """No-neutral val + test splits make up the searchable corpus."""
    return (
        _DATA_DIR / "B2W-Reviews01_no_neutral_val.csv",
        _DATA_DIR / "B2W-Reviews01_no_neutral_test.csv",
    )


@dataclass
class RecommenderConfig:
    """Runtime configuration for building and serving recommendations."""

    # --- data ---
    corpus_splits: tuple[Path, ...] = field(default_factory=_corpus_default_splits)
    storefront_split: Path = _DATA_DIR / "B2W-Reviews01_train.csv"
    artifacts_dir: Path = _THIS_DIR / "artifacts"

    # --- embedding ---
    model_name: str = DEFAULT_MODEL
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    device: str | None = None  # None -> auto-detect (cuda > mps > cpu)
    batch_size: int = 32

    # --- corpus construction ---
    min_reviews: int = 1  # report spec uses 5; kept configurable
    positive_rating_threshold: int = 4  # overall_rating >= -> "positive" (mock S(p))

    # --- retrieval / re-rank ---
    retrieve_k: int = 50  # ANN candidates pulled before re-ranking
    top_n: int = 10  # recommendations returned
    alpha: float = 0.7  # Score_final = alpha*sim + (1-alpha)*S(p)

    # --- vector store ---
    collection_name: str = "product_corpus"

    @classmethod
    def from_env(cls) -> "RecommenderConfig":
        """Build a config, applying ``RECOMMENDER_*`` environment overrides."""
        cfg = cls()
        if value := os.getenv("RECOMMENDER_DEVICE"):
            cfg.device = value
        if value := os.getenv("RECOMMENDER_MODEL"):
            cfg.model_name = value
        if value := os.getenv("RECOMMENDER_ARTIFACTS"):
            cfg.artifacts_dir = Path(value)
        if value := os.getenv("RECOMMENDER_ALPHA"):
            cfg.alpha = float(value)
        if value := os.getenv("RECOMMENDER_TOP_N"):
            cfg.top_n = int(value)
        return cfg

    # --- artifact paths (derived from ``artifacts_dir``) ---
    @property
    def corpus_collection_dir(self) -> Path:
        return self.artifacts_dir / "corpus"

    @property
    def corpus_catalog_path(self) -> Path:
        return self.artifacts_dir / "corpus_catalog.csv"

    @property
    def corpus_embeddings_path(self) -> Path:
        return self.artifacts_dir / "corpus_embeddings.npy"

    @property
    def storefront_catalog_path(self) -> Path:
        return self.artifacts_dir / "storefront_catalog.csv"

    @property
    def storefront_embeddings_path(self) -> Path:
        return self.artifacts_dir / "storefront_embeddings.npy"

    @property
    def meta_path(self) -> Path:
        return self.artifacts_dir / "meta.json"
