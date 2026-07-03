"""FastAPI service exposing the recommender to the web app.

Endpoints (a retail product page can be simulated with these):

* ``GET /health``               — liveness probe
* ``GET /products``             — browse the storefront catalog (the pages)
* ``GET /products/{id}``        — a single product page
* ``GET /recommend``            — re-ranked recommendations for a product/query

Run from inside ``04_recommender/``:

    uvicorn service.app:app --app-dir . --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_recommender: Any = None


def set_recommender(recommender: Any) -> None:
    """Inject a recommender instance (used by tests and custom bootstraps)."""
    global _recommender
    _recommender = recommender


def get_recommender() -> Any:
    """Return the active recommender, lazily loading from disk on first use."""
    global _recommender
    if _recommender is None:
        from recommender import ProductRecommender

        _recommender = ProductRecommender.load()
    return _recommender


class ProductOut(BaseModel):
    product_id: str
    product_name: str
    product_brand: str
    site_category_lv1: str
    site_category_lv2: str
    num_reviews: int
    sentiment_score: float


class RecommendationOut(ProductOut):
    score: float
    similarity: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_recommender()  # warm-load store + catalogs (no-op if already injected)
    yield


app = FastAPI(title="Sentiment-as-Relevance Recommender", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/products", response_model=list[ProductOut])
def list_products(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    category: str | None = None,
    q: str | None = None,
) -> list[dict[str, Any]]:
    return get_recommender().list_storefront(
        limit=limit, offset=offset, category=category, query=q
    )


@app.get("/products/{product_id}", response_model=ProductOut)
def get_product(product_id: str) -> dict[str, Any]:
    product = get_recommender().get_storefront_product(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


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
        {**rec.metadata, "score": rec.score, "similarity": rec.similarity}
        for rec in recs
    ]
