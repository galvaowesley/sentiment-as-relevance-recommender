"""Objective ranking evaluation (Precision@k, Recall@k, NDCG@k) for one query product.

Relevance is defined independently of the re-ranking formula: it is the product's
mean review rating (``overall_rating``), min-max normalised *within its category*
(Eq. rel(p) in the report draft), not the sentiment score S(p) used by the engine
itself. Using S(p) as relevance would be circular, since the engine already ranks
by S(p).

Usage (recommender API must be running, e.g. `uvicorn service.app:app --port 8000`
from `04_recommender/`):

    python objective_metrics.py --product-id 132444092 --category "Celulares e Smartphones"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = (
    _REPO_ROOT / "data" / "results_from_best_model" / "B2W-Reviews01_inferred_bertimbau.csv"
)
DEFAULT_API = "http://127.0.0.1:8000"
MIN_REVIEWS = 5  # same cut as recommend_min_reviews in 04_recommender/config.py
RELEVANCE_THRESHOLD = 0.7  # rel(p) >= this -> "relevant" (binary label for P/R)


def load_category_ratings(csv_path: Path, category: str) -> pd.DataFrame:
    """Aggregate avg_rating/num_reviews per product for one site_category_lv1."""
    df = pd.read_csv(csv_path, dtype={"product_id": str}, usecols=[
        "product_id", "product_name", "site_category_lv1", "overall_rating",
    ])
    df = df[df["site_category_lv1"] == category].copy()
    df["overall_rating"] = pd.to_numeric(df["overall_rating"], errors="coerce")

    grouped = df.groupby("product_id", sort=False).agg(
        product_name=("product_name", "first"),
        avg_rating=("overall_rating", "mean"),
        num_reviews=("product_id", "size"),
    ).reset_index()

    return grouped[grouped["num_reviews"] >= MIN_REVIEWS].reset_index(drop=True)


def add_relevance(catalog: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise avg_rating within the given catalog slice -> rel(p) in [0,1]."""
    lo, hi = catalog["avg_rating"].min(), catalog["avg_rating"].max()
    span = hi - lo
    catalog = catalog.copy()
    catalog["relevance"] = 0.0 if span == 0 else (catalog["avg_rating"] - lo) / span
    return catalog


def fetch_recommendations(api_base: str, product_id: str, top_n: int, alpha: float) -> list[dict]:
    resp = requests.get(
        f"{api_base}/recommend",
        params={"product_id": product_id, "top_n": top_n, "alpha": alpha},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def precision_at_k(rel_labels: list[int], k: int) -> float:
    top_k = rel_labels[:k]
    return sum(top_k) / k if top_k else 0.0


def recall_at_k(rel_labels: list[int], k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(rel_labels[:k]) / total_relevant


def ndcg_at_k(rel_grades: list[float], k: int) -> float:
    """Linear-gain NDCG@k (continuous relevance in [0, 1], not exponential gain)."""
    top_k = rel_grades[:k]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(top_k))
    ideal = sorted(rel_grades, reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-id", default="132444092")
    parser.add_argument("--category", default="Celulares e Smartphones")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--api-base", default=DEFAULT_API)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "objective_metrics_smartphone.json",
    )
    args = parser.parse_args()

    catalog = load_category_ratings(args.csv_path, args.category)
    catalog = add_relevance(catalog)
    rel_by_id = dict(zip(catalog["product_id"], catalog["relevance"], strict=True))

    query_row = catalog[catalog["product_id"] == args.product_id]
    if query_row.empty:
        raise SystemExit(f"product_id {args.product_id} not found in category {args.category}")
    query_meta = query_row.iloc[0].to_dict()

    # Universe of relevant items for Recall@k: all recommendable products in the
    # category (excluding the query itself) with rel(p) >= threshold.
    universe = catalog[catalog["product_id"] != args.product_id]
    total_relevant = int((universe["relevance"] >= RELEVANCE_THRESHOLD).sum())

    recs = fetch_recommendations(args.api_base, args.product_id, args.top_n, args.alpha)

    rel_grades: list[float] = []
    rel_labels: list[int] = []
    per_item = []
    for rec in recs:
        pid = rec["product_id"]
        rel = rel_by_id.get(pid)
        if rel is None:
            # Recommended product outside the evaluated category / rating table
            # (e.g. missing overall_rating after aggregation); treat as non-relevant.
            rel = 0.0
        rel_grades.append(rel)
        rel_labels.append(int(rel >= RELEVANCE_THRESHOLD))
        per_item.append({
            "product_id": pid,
            "product_name": rec["product_name"],
            "avg_rating": float(catalog.loc[catalog["product_id"] == pid, "avg_rating"].iloc[0])
            if pid in rel_by_id else None,
            "relevance": rel,
            "is_relevant": bool(rel >= RELEVANCE_THRESHOLD),
            "similarity": rec["similarity"],
            "sentiment_score": rec["sentiment_score"],
            "score": rec["score"],
        })

    metrics = {}
    for k in args.ks:
        metrics[f"precision@{k}"] = precision_at_k(rel_labels, k)
        metrics[f"recall@{k}"] = recall_at_k(rel_labels, k, total_relevant)
        metrics[f"ndcg@{k}"] = ndcg_at_k(rel_grades, k)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "top_n": args.top_n,
            "alpha": args.alpha,
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "min_reviews": MIN_REVIEWS,
            "ks": args.ks,
            "relevance_definition": "min-max normalised avg_rating within category",
        },
        "category": args.category,
        "category_recommendable_products": int(len(catalog)),
        "query_product": {
            "product_id": args.product_id,
            "product_name": query_meta["product_name"],
            "avg_rating": float(query_meta["avg_rating"]),
            "relevance": float(query_meta["relevance"]),
            "num_reviews": int(query_meta["num_reviews"]),
        },
        "total_relevant_in_category": total_relevant,
        "metrics": metrics,
        "recommendations": per_item,
    }

    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "total_relevant_in_category": total_relevant}, indent=2, ensure_ascii=False))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
