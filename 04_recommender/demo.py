"""Print recommendations for one storefront product — a quick end-to-end smoke.

Usage:
    python demo.py                         # uses the first storefront product
    python demo.py --product-id 123456789  # a specific product
    python demo.py --artifacts artifacts_demo --top-n 5 --alpha 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import RecommenderConfig
from recommender import ProductRecommender


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--product-id", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = RecommenderConfig.from_env()
    if args.artifacts is not None:
        config.artifacts_dir = args.artifacts

    recommender = ProductRecommender.load(config)

    product_id = args.product_id
    if product_id is None:
        page = recommender.list_storefront(limit=1)
        if not page:
            print("No storefront products available.")
            return
        product_id = page[0]["product_id"]

    product = recommender.get_storefront_product(product_id)
    print("Product page:")
    if product:
        print(f"  [{product['product_id']}] {product['product_name']}")
        print(
            f"  brand={product['product_brand']} | "
            f"{product['site_category_lv1']} > {product['site_category_lv2']} | "
            f"S(p)={product['sentiment_score']:.3f} ({product['num_reviews']} reviews)"
        )
    print(f"\nTop {args.top_n} recommendations (alpha={config.alpha if args.alpha is None else args.alpha}):")
    recs = recommender.recommend(
        product_id=product_id, top_n=args.top_n, alpha=args.alpha
    )
    for rank, rec in enumerate(recs, start=1):
        meta = rec.metadata
        print(
            f"  {rank}. [{rec.product_id}] {meta['product_name'][:70]}\n"
            f"     score={rec.score:.4f}  sim={rec.similarity:.4f}  "
            f"S(p)={rec.sentiment_score:.3f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
