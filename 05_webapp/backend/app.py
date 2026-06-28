"""FastAPI backend for the SentiShop web demo.

Wraps the ProductRecommender from 04_recommender/ and adds a /categories
endpoint for the frontend sidebar. Run via run.sh (sets PYTHONPATH correctly).

Endpoints:
  GET /health                     liveness probe
  GET /categories                 sorted list of lv1 categories
  GET /categories/lv2             lv2 subcategories for a given lv1
  GET /products                   paginated + filterable storefront catalog
  GET /products/{id}              single product (with avg_rating, review_count, description)
  GET /products/{id}/reviews      all reviews for a product (val+test, incl. neutral)
  GET /recommend                  re-ranked recommendations for a product or query
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "04_recommender"))

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_recommender: Any = None
_reviews_by_product: dict[str, list[dict]] = {}
_product_stats: dict[str, dict] = {}   # product_id → {avg_rating, review_count}


# ── Data paths ────────────────────────────────────────────────────────────
_DATA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
_REVIEW_SPLITS = [
    _DATA_ROOT / "B2W-Reviews01_val.csv",
    _DATA_ROOT / "B2W-Reviews01_test.csv",
]


def _load_reviews() -> None:
    """Load val+test splits (including neutral ratings) into in-memory indexes."""
    global _reviews_by_product, _product_stats

    frames = []
    for path in _REVIEW_SPLITS:
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))

    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["product_id"]).copy()
    df["product_id"] = df["product_id"].astype(str)
    df["overall_rating"] = pd.to_numeric(df["overall_rating"], errors="coerce")

    # Build per-product stats
    stats: dict[str, dict] = {}
    reviews_idx: dict[str, list[dict]] = {}

    for pid, group in df.groupby("product_id", sort=False):
        pid = str(pid)
        valid_ratings = group["overall_rating"].dropna()
        avg = float(valid_ratings.mean()) if not valid_ratings.empty else 0.0
        stats[pid] = {
            "avg_rating": round(avg, 1),
            "review_count": len(group),
        }

        rev_list = []
        for _, row in group.iterrows():
            rev_list.append({
                "reviewer_id": str(row.get("reviewer_id", ""))[:8] + "...",
                "review_title": str(row.get("review_title", "") or ""),
                "overall_rating": int(row["overall_rating"]) if pd.notna(row.get("overall_rating")) else None,
                "recommend_to_a_friend": str(row.get("recommend_to_a_friend", "") or ""),
                "review_text": str(row.get("review_text", "") or ""),
                "reviewer_birth_year": int(row["reviewer_birth_year"]) if pd.notna(row.get("reviewer_birth_year")) else None,
                "reviewer_gender": str(row.get("reviewer_gender", "") or ""),
                "reviewer_state": str(row.get("reviewer_state", "") or ""),
                "submission_date": str(row.get("submission_date", "") or "")[:10],
                "sentiment_score": None,  # filled by classification pipeline
            })
        reviews_idx[pid] = rev_list

    _reviews_by_product = reviews_idx
    _product_stats = stats


def _enrich_product(product: dict) -> dict:
    """Add avg_rating, review_count, description from reviews index."""
    pid = product.get("product_id", "")
    stats = _product_stats.get(pid, {})
    product = dict(product)
    product["avg_rating"] = stats.get("avg_rating", None)
    product["review_count"] = stats.get("review_count", product.get("num_reviews", 0))

    # Synthetic description (no description column in dataset)
    reviews = _reviews_by_product.get(pid, [])
    product["description"] = _build_description(
        product.get("product_name", ""),
        product.get("product_brand", ""),
        product.get("site_category_lv1", ""),
        product.get("site_category_lv2", ""),
        reviews,
    )
    return product


def _build_description(name: str, brand: str, lv1: str, lv2: str, reviews: list[dict]) -> str:
    # Dataset has no description field — construct a factual summary from product metadata only.
    parts = []
    if brand:
        parts.append(f"Marca: {brand}.")
    if lv1 and lv2 and lv2 != lv1:
        parts.append(f"Categoria: {lv1} › {lv2}.")
    elif lv1:
        parts.append(f"Categoria: {lv1}.")
    return " ".join(parts) if parts else ""


# ── Recommender ────────────────────────────────────────────────────────────
def set_recommender(recommender: Any) -> None:
    global _recommender
    _recommender = recommender


def get_recommender() -> Any:
    global _recommender
    if _recommender is None:
        from recommender import ProductRecommender
        _recommender = ProductRecommender.load()
    return _recommender


# ── Pydantic models ────────────────────────────────────────────────────────
class ProductOut(BaseModel):
    product_id: str
    product_name: str
    product_brand: str
    site_category_lv1: str
    site_category_lv2: str
    num_reviews: int
    sentiment_score: float
    avg_rating: float | None = None
    review_count: int | None = None
    description: str | None = None


class RecommendationOut(ProductOut):
    score: float
    similarity: float


class ReviewOut(BaseModel):
    reviewer_id: str
    review_title: str
    overall_rating: int | None
    recommend_to_a_friend: str
    review_text: str
    reviewer_birth_year: int | None
    reviewer_gender: str
    reviewer_state: str
    submission_date: str
    sentiment_score: float | None = None


# ── App lifecycle ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_reviews()
    get_recommender()
    yield


app = FastAPI(title="SentiShop Recommender API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/categories")
def list_categories() -> list[str]:
    rec = get_recommender()
    catalog = getattr(rec, "storefront_catalog", None)
    if catalog is None:
        return []
    return sorted(catalog["site_category_lv1"].dropna().unique().tolist())


@app.get("/categories/lv2")
def list_subcategories(cat1: str | None = None) -> list[str]:
    """Return lv2 subcategories, optionally filtered by lv1."""
    rec = get_recommender()
    catalog = getattr(rec, "storefront_catalog", None)
    if catalog is None:
        return []
    sub = catalog
    if cat1:
        sub = sub[sub["site_category_lv1"] == cat1]
    return sorted(sub["site_category_lv2"].dropna().unique().tolist())


@app.get("/products/count")
def count_products(
    category: str | None = None,
    cat2: str | None = None,
    q: str | None = None,
) -> dict[str, int]:
    """Total number of products matching the given filters (for pagination)."""
    raw = get_recommender().list_storefront(limit=10_000, offset=0, category=category, query=q)
    if cat2:
        raw = [p for p in raw if p.get("site_category_lv2") == cat2]
    return {"total": len(raw)}


@app.get("/products", response_model=list[ProductOut])
def list_products(
    limit: int = Query(24, ge=1, le=200),
    offset: int = Query(0, ge=0),
    category: str | None = None,
    cat2: str | None = None,
    q: str | None = None,
) -> list[dict[str, Any]]:
    raw = get_recommender().list_storefront(
        limit=limit, offset=offset, category=category, query=q
    )
    # lv2 filter (not natively supported by recommender)
    if cat2:
        raw = [p for p in raw if p.get("site_category_lv2") == cat2]
    return [_enrich_product(p) for p in raw]


@app.get("/products/{product_id}", response_model=ProductOut)
def get_product(product_id: str) -> dict[str, Any]:
    product = get_recommender().get_storefront_product(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return _enrich_product(product)


@app.get("/products/{product_id}/reviews", response_model=list[ReviewOut])
def get_product_reviews(product_id: str) -> list[dict[str, Any]]:
    """All reviews for a product from val+test splits (including neutral ratings)."""
    reviews = _reviews_by_product.get(product_id)
    if reviews is None:
        raise HTTPException(status_code=404, detail="No reviews found for this product")
    return reviews


@app.get("/recommend", response_model=list[RecommendationOut])
def recommend(
    product_id: str | None = None,
    q: str | None = None,
    top_n: int | None = Query(None, ge=1, le=100),
    alpha: float | None = Query(None, ge=0.0, le=1.0),
) -> list[dict[str, Any]]:
    try:
        recs = get_recommender().recommend(
            product_id=product_id, query_text=q, top_n=top_n, alpha=alpha
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return [
        _enrich_product({**rec.metadata, "score": rec.score, "similarity": rec.similarity})
        for rec in recs
    ]


# ── Static frontend ────────────────────────────────────────────────────────
from fastapi.staticfiles import StaticFiles

_FRONTEND_DIST = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="static")
