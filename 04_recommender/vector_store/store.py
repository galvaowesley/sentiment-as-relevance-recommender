"""Vector store backed by the zvec database.

Wraps the small slice of the zvec API we need: create/open a persisted HNSW
cosine collection, bulk-insert product vectors, and run Top-K ANN queries.

zvec reports cosine *distance* (0 = identical, 1 = orthogonal); we convert it to a
similarity with ``sim = 1 - distance``.
"""

from __future__ import annotations

import gc
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np


class ZvecVectorStore:
    """Thin wrapper over a single-vector zvec collection."""

    VECTOR_FIELD = "embedding"
    # zvec rejects an insert() larger than this many docs in one call.
    MAX_WRITE_BATCH = 1024

    def __init__(self, collection) -> None:  # noqa: ANN001 - zvec.Collection
        self._collection = collection

    @classmethod
    def create(
        cls,
        path: str | Path,
        dim: int,
        collection_name: str = "product_corpus",
    ) -> "ZvecVectorStore":
        """Create and open a fresh collection on disk."""
        import zvec

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # zvec.create_and_open refuses an existing path; clear it so rebuilds
        # into an existing artifacts dir are idempotent (matches the CSV/npy
        # artifacts, which overwrite in place).
        if path.exists():
            shutil.rmtree(path)
        schema = zvec.CollectionSchema(
            name=collection_name,
            fields=[
                zvec.FieldSchema(name="num_reviews", data_type=zvec.DataType.INT32),
            ],
            vectors=[
                zvec.VectorSchema(
                    name=cls.VECTOR_FIELD,
                    data_type=zvec.DataType.VECTOR_FP32,
                    dimension=dim,
                    index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
                ),
            ],
        )
        collection = zvec.create_and_open(path=str(path), schema=schema)
        return cls(collection)

    @classmethod
    def open(cls, path: str | Path) -> "ZvecVectorStore":
        """Open an existing collection."""
        import zvec

        return cls(zvec.open(str(path)))

    def add(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        num_reviews: Sequence[int],
    ) -> None:
        """Insert a batch of product vectors with their review counts."""
        import zvec

        docs = [
            zvec.Doc(
                id=str(pid),
                vectors={self.VECTOR_FIELD: np.asarray(emb, dtype=np.float32).tolist()},
                fields={"num_reviews": int(nrev)},
            )
            for pid, emb, nrev in zip(ids, embeddings, num_reviews, strict=True)
        ]
        # Chunk so a large corpus stays under zvec's per-insert doc limit.
        for start in range(0, len(docs), self.MAX_WRITE_BATCH):
            self._collection.insert(docs[start : start + self.MAX_WRITE_BATCH])

    def optimize(self) -> None:
        """Build/compact the index for fast search after bulk loading."""
        self._collection.optimize()

    def query(
        self,
        embedding: np.ndarray,
        topk: int,
        filter: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(product_id, cosine_similarity)`` for the Top-K nearest titles."""
        import zvec

        results = self._collection.query(
            queries=zvec.Query(
                field_name=self.VECTOR_FIELD,
                vector=np.asarray(embedding, dtype=np.float32).tolist(),
            ),
            topk=topk,
            filter=filter,
        )
        return [(doc.id, 1.0 - float(doc.score)) for doc in results]

    def close(self) -> None:
        """Release the on-disk write lock (zvec has no explicit close)."""
        self._collection = None
        gc.collect()
