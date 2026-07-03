"""High-level product recommender used by the web service.

Loads the artifacts produced by ``build_index.py`` and exposes a small API:

* :meth:`ProductRecommender.recommend` — semantic retrieval + sentiment re-ranking
* :meth:`ProductRecommender.list_storefront` / :meth:`get_storefront_product` —
  browse the simulated retail catalog (the pages that issue queries)

Recommendations are drawn from the **corpus** while the **storefront** supplies the
browsable product pages. Both come from the same BERTimbau-inferred val+test source
(same products and S(p)); they differ only in embedding encoding (document vs query).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import RecommenderConfig
from reranking.reranker import Candidate, Recommendation, rerank
from vector_store.store import ZvecVectorStore


_STRING_COLUMNS = (
    "product_id",
    "product_name",
    "product_brand",
    "site_category_lv1",
    "site_category_lv2",
)


def _load_catalog(path) -> pd.DataFrame:
    """Read a catalog CSV, keeping string columns as clean (non-NaN) strings."""
    df = pd.read_csv(path, dtype={"product_id": str})
    for column in _STRING_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)
    return df


def _row_meta(row: pd.Series) -> dict[str, Any]:
    """Project a catalog row to the metadata dict returned to callers."""
    return {
        "product_id": str(row["product_id"]),
        "product_name": row["product_name"],
        "product_brand": row["product_brand"],
        "site_category_lv1": row["site_category_lv1"],
        "site_category_lv2": row["site_category_lv2"],
        "num_reviews": int(row["num_reviews"]),
        "sentiment_score": float(row["sentiment_score"]),
    }


class ProductRecommender:
    """Serves recommendations from prebuilt corpus and storefront artifacts."""

    def __init__(
        self,
        config: RecommenderConfig,
        store: ZvecVectorStore,
        corpus_catalog: pd.DataFrame,
        corpus_embeddings: np.ndarray,
        storefront_catalog: pd.DataFrame | None = None,
        storefront_embeddings: np.ndarray | None = None,
        embedder: Any = None,
    ) -> None:
        self.config = config
        self.store = store
        self.corpus_catalog = corpus_catalog.reset_index(drop=True)
        self.corpus_embeddings = corpus_embeddings
        self.storefront_catalog = (
            storefront_catalog.reset_index(drop=True)
            if storefront_catalog is not None
            else None
        )
        self.storefront_embeddings = storefront_embeddings
        self._embedder = embedder  # lazily created for free-text queries

        self._corpus_pos = {
            pid: i for i, pid in enumerate(self.corpus_catalog["product_id"])
        }
        self._storefront_pos = (
            {pid: i for i, pid in enumerate(self.storefront_catalog["product_id"])}
            if self.storefront_catalog is not None
            else {}
        )

    # --- construction ---
    @classmethod
    def load(
        cls,
        config: RecommenderConfig | None = None,
        embedder: Any = None,
    ) -> "ProductRecommender":
        """Open the persisted store and load catalogs/embeddings from disk."""
        config = config or RecommenderConfig.from_env()
        store = ZvecVectorStore.open(config.corpus_collection_dir)

        corpus_catalog = _load_catalog(config.corpus_catalog_path)
        corpus_embeddings = np.load(config.corpus_embeddings_path)

        storefront_catalog = None
        storefront_embeddings = None
        if config.storefront_catalog_path.exists():
            storefront_catalog = _load_catalog(config.storefront_catalog_path)
            storefront_embeddings = np.load(config.storefront_embeddings_path)

        return cls(
            config=config,
            store=store,
            corpus_catalog=corpus_catalog,
            corpus_embeddings=corpus_embeddings,
            storefront_catalog=storefront_catalog,
            storefront_embeddings=storefront_embeddings,
            embedder=embedder,
        )

    # --- recommendation ---
    def recommend(
        self,
        product_id: str | None = None,
        query_text: str | None = None,
        top_n: int | None = None,
        alpha: float | None = None,
    ) -> list[Recommendation]:
        """Recommend corpus products for a storefront product or free-text query."""
        alpha = self.config.alpha if alpha is None else alpha
        top_n = self.config.top_n if top_n is None else top_n

        query_vec, exclude_id = self._resolve_query(product_id, query_text)
        # Every product is indexed, but only well-reviewed ones are recommendable.
        # Push the cut into the ANN so the retrieved candidates already qualify;
        # the Python guard below is a backstop if the backend ignores the filter.
        min_reviews = self.config.recommend_min_reviews
        review_filter = f"num_reviews >= {min_reviews}" if min_reviews > 1 else None
        hits = self.store.query(
            query_vec, topk=self.config.retrieve_k, filter=review_filter
        )

        candidates: list[Candidate] = []
        for pid, _retrieval_sim in hits:
            pos = self._corpus_pos.get(pid)
            if pos is None:
                continue
            row = self.corpus_catalog.iloc[pos]
            if int(row["num_reviews"]) < min_reviews:
                continue
            # Recompute exact cosine sim from stored normalised embeddings so the
            # final score is independent of the ANN backend's scoring convention.
            sim = float(np.dot(self.corpus_embeddings[pos], query_vec))
            candidates.append(
                Candidate(
                    product_id=pid,
                    similarity=sim,
                    sentiment_score=float(row["sentiment_score"]),
                    metadata=_row_meta(row),
                )
            )

        return rerank(candidates, alpha=alpha, top_n=top_n, exclude_id=exclude_id)

    def _resolve_query(
        self, product_id: str | None, query_text: str | None
    ) -> tuple[np.ndarray, str | None]:
        """Return a normalised query vector and the id to exclude from results."""
        if product_id is not None:
            pos = self._storefront_pos.get(product_id)
            if pos is not None:
                return self.storefront_embeddings[pos], product_id
            pos = self._corpus_pos.get(product_id)
            if pos is not None:
                return self.corpus_embeddings[pos], product_id
            raise KeyError(f"Unknown product_id: {product_id}")

        if query_text:
            return self._embed_query_text(query_text), None

        raise ValueError("Provide either product_id or query_text")

    def _embed_query_text(self, text: str) -> np.ndarray:
        """Embed an arbitrary search string, loading the model on first use."""
        if self._embedder is None:
            from vectorizer.embedder import Qwen3Embedder

            self._embedder = Qwen3Embedder(
                model_name=self.config.model_name,
                device=self.config.device,
                query_instruction=self.config.query_instruction,
                batch_size=self.config.batch_size,
            )
        return self._embedder.encode_queries([text])[0]

    # --- storefront browsing ---
    def list_storefront(
        self,
        limit: int = 20,
        offset: int = 0,
        category: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a page of storefront products for the simulated catalog."""
        if self.storefront_catalog is None:
            return []
        df = self.storefront_catalog
        if category:
            df = df[df["site_category_lv1"] == category]
        if query:
            df = df[df["product_name"].str.contains(query, case=False, na=False)]
        page = df.iloc[offset : offset + limit]
        return [_row_meta(row) for _, row in page.iterrows()]

    def get_storefront_product(self, product_id: str) -> dict[str, Any] | None:
        """Return a single storefront product, or ``None`` if unknown."""
        if self.storefront_catalog is None:
            return None
        pos = self._storefront_pos.get(product_id)
        if pos is None:
            return None
        return _row_meta(self.storefront_catalog.iloc[pos])
