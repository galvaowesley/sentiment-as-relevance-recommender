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
_RESULTS_DIR = _REPO_ROOT / "data" / "results_from_best_model"

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_QUERY_INSTRUCTION = (
    "Given a retail product title, retrieve titles of similar products"
)
# Best-model (BERTimbau) inference over val+test; carries ``inferencia_bertimbau``
# (0/1) so S(p) is a real proportion of positive reviews, not the rating-based mock.
DEFAULT_INFERRED_CSV = _RESULTS_DIR / "B2W-Reviews01_inferred_bertimbau.csv"
DEFAULT_LABEL_COLUMN = "inferencia_bertimbau"


def _corpus_default_splits() -> tuple[Path, ...]:
    """The BERTimbau-inferred val+test file is the searchable corpus source."""
    return (DEFAULT_INFERRED_CSV,)


def _load_dotenv() -> None:
    """Populate ``os.environ`` from a versioned ``.env`` at the repo root.

    Only fills keys that are not already set, so a real environment variable always
    wins. Dependency-free ``KEY=value`` parsing (``#`` comments and quotes allowed).
    The ``.env`` holds paths/params only — never secrets — and is committed.
    """
    path = _REPO_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _resolve_path(value: str) -> Path:
    """Resolve a config path, treating relatives as repo-root-relative."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (_REPO_ROOT / p)


@dataclass
class RecommenderConfig:
    """Runtime configuration for building and serving recommendations."""

    # --- data ---
    # Corpus and storefront share the same BERTimbau-inferred val+test source, so
    # both sides carry the real S(p) and stay coherent (no train data involved).
    corpus_splits: tuple[Path, ...] = field(default_factory=_corpus_default_splits)
    storefront_split: Path = DEFAULT_INFERRED_CSV
    artifacts_dir: Path = _THIS_DIR / "artifacts"

    # --- embedding ---
    model_name: str = DEFAULT_MODEL
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    device: str | None = None  # None -> auto-detect (cuda > mps > cpu)
    batch_size: int = 32

    # --- corpus construction ---
    min_reviews: int = 1  # every product is vectorized; recommend-time cut is separate
    sentiment_label_column: str = DEFAULT_LABEL_COLUMN  # predicted-label column for S(p)
    positive_rating_threshold: int = 4  # overall_rating >= -> "positive" (mock S(p))

    # --- retrieval / re-rank ---
    recommend_min_reviews: int = 5  # a product needs >= this many reviews to be recommended
    retrieve_k: int = 50  # ANN candidates pulled before re-ranking
    top_n: int = 10  # recommendations returned
    alpha: float = 0.7  # Score_final = alpha*sim + (1-alpha)*S(p)

    # --- vector store ---
    collection_name: str = "product_corpus"

    @classmethod
    def from_env(cls) -> "RecommenderConfig":
        """Build a config, applying ``.env`` then ``RECOMMENDER_*`` env overrides."""
        _load_dotenv()
        cfg = cls()
        # A single inferred CSV feeds both corpus and storefront, keeping them coherent.
        if value := os.getenv("RECOMMENDER_INFERRED_CSV"):
            source = _resolve_path(value)
            cfg.corpus_splits = (source,)
            cfg.storefront_split = source
        if value := os.getenv("RECOMMENDER_DEVICE"):
            cfg.device = value
        if value := os.getenv("RECOMMENDER_MODEL"):
            cfg.model_name = value
        if value := os.getenv("RECOMMENDER_ARTIFACTS"):
            cfg.artifacts_dir = _resolve_path(value)
        if value := os.getenv("RECOMMENDER_ALPHA"):
            cfg.alpha = float(value)
        if value := os.getenv("RECOMMENDER_TOP_N"):
            cfg.top_n = int(value)
        if value := os.getenv("RECOMMENDER_LABEL_COLUMN"):
            cfg.sentiment_label_column = value
        if value := os.getenv("RECOMMENDER_MIN_REVIEWS"):
            cfg.recommend_min_reviews = int(value)
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

    @property
    def sentiment_scores_path(self) -> Path:
        """Per-product S(p) table (Pipeline 1 output), shared under data/processed."""
        return _DATA_DIR / "sentiment_scores.parquet"
