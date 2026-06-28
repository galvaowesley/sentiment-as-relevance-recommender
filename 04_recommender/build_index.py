"""Build the recommender artifacts from the B2W-Reviews01 splits.

Produces, under ``artifacts/``:

* ``corpus/``               — persisted zvec collection (recommendable products)
* ``corpus_catalog.csv``    — corpus metadata, row-aligned with the embeddings
* ``corpus_embeddings.npy`` — document embeddings of corpus titles
* ``storefront_catalog.csv`` / ``storefront_embeddings.npy`` — browsable pages
* ``meta.json``             — build metadata

Corpus = no-neutral val + test. Storefront = train (full, incl. neutral).

Usage:
    python build_index.py                      # full build
    python build_index.py --limit 2000         # quick smoke build
    python build_index.py --out artifacts_demo --min-reviews 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import RecommenderConfig
from data import build_catalog, catalog_to_frame, load_reviews
from reranking.sentiment import MockSentimentScorer, SentimentScorer
from vector_store.store import ZvecVectorStore


def build_corpus(
    config: RecommenderConfig,
    embedder: Any,
    scorer: SentimentScorer,
    limit: int | None = None,
) -> int:
    """Build the searchable corpus index and catalog. Returns product count."""
    df = load_reviews(config.corpus_splits)
    records = build_catalog(df, scorer, min_reviews=config.min_reviews, limit=limit)
    if not records:
        raise RuntimeError("Corpus is empty; check splits and --min-reviews")

    titles = [r.product_name for r in records]
    embeddings = embedder.encode_documents(titles)

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    store = ZvecVectorStore.create(
        config.corpus_collection_dir,
        dim=int(embeddings.shape[1]),
        collection_name=config.collection_name,
    )
    store.add(
        ids=[r.product_id for r in records],
        embeddings=embeddings,
        num_reviews=[r.num_reviews for r in records],
    )
    store.optimize()
    store.close()

    catalog_to_frame(records).to_csv(config.corpus_catalog_path, index=False)
    np.save(config.corpus_embeddings_path, embeddings)
    print(f"[corpus] {len(records)} products -> {config.corpus_collection_dir}")
    return len(records)


def build_storefront(
    config: RecommenderConfig,
    embedder: Any,
    scorer: SentimentScorer,
    limit: int | None = None,
) -> int:
    """Build the browsable storefront catalog + query embeddings. Returns count."""
    df = load_reviews([config.storefront_split])
    # Every browsable product is a valid page, so storefront keeps min_reviews=1.
    records = build_catalog(df, scorer, min_reviews=1, limit=limit)
    if not records:
        raise RuntimeError("Storefront is empty; check the train split")

    titles = [r.product_name for r in records]
    embeddings = embedder.encode_queries(titles)

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    catalog_to_frame(records).to_csv(config.storefront_catalog_path, index=False)
    np.save(config.storefront_embeddings_path, embeddings)
    print(f"[storefront] {len(records)} products -> {config.storefront_catalog_path}")
    return len(records)


def _write_meta(config: RecommenderConfig, counts: dict[str, int], dim: int) -> None:
    meta = {
        "model_name": config.model_name,
        "embedding_dim": dim,
        "query_instruction": config.query_instruction,
        "min_reviews": config.min_reviews,
        "positive_rating_threshold": config.positive_rating_threshold,
        "collection_name": config.collection_name,
        "counts": counts,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    config.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def build_all(
    config: RecommenderConfig,
    embedder: Any | None = None,
    scorer: SentimentScorer | None = None,
    limit: int | None = None,
    corpus: bool = True,
    storefront: bool = True,
) -> dict[str, int]:
    """Build the requested artifacts and write ``meta.json``."""
    if embedder is None:
        from vectorizer.embedder import Qwen3Embedder

        embedder = Qwen3Embedder(
            model_name=config.model_name,
            device=config.device,
            query_instruction=config.query_instruction,
            batch_size=config.batch_size,
        )
    scorer = scorer or MockSentimentScorer(config.positive_rating_threshold)

    counts: dict[str, int] = {}
    if corpus:
        counts["corpus"] = build_corpus(config, embedder, scorer, limit=limit)
    if storefront:
        counts["storefront"] = build_storefront(config, embedder, scorer, limit=limit)

    _write_meta(config, counts, dim=int(getattr(embedder, "dimension", 0)))
    return counts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="artifacts directory")
    parser.add_argument("--model", type=str, default=None, help="embedding model name")
    parser.add_argument("--device", type=str, default=None, help="cuda|mps|cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-reviews", type=int, default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap products per side (smoke builds)"
    )
    parser.add_argument("--corpus-only", action="store_true")
    parser.add_argument("--storefront-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = RecommenderConfig.from_env()
    if args.out is not None:
        config.artifacts_dir = args.out
    if args.model is not None:
        config.model_name = args.model
    if args.device is not None:
        config.device = args.device
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.min_reviews is not None:
        config.min_reviews = args.min_reviews

    counts = build_all(
        config,
        limit=args.limit,
        corpus=not args.storefront_only,
        storefront=not args.corpus_only,
    )
    print(f"Done: {counts} -> {config.artifacts_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
