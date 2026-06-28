"""Loading B2W-Reviews01 splits and building the product catalog.

Reviews are deduplicated into one record per ``product_id``. Each record carries
the metadata shown on a retail product page plus the sentiment score S(p) computed
by a :class:`~reranking.sentiment.SentimentScorer`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from reranking.sentiment import SentimentScorer

REQUIRED_COLUMNS = (
    "product_id",
    "product_name",
    "product_brand",
    "site_category_lv1",
    "site_category_lv2",
    "overall_rating",
)

CATALOG_COLUMNS = (
    "product_id",
    "product_name",
    "product_brand",
    "site_category_lv1",
    "site_category_lv2",
    "num_reviews",
    "sentiment_score",
)


@dataclass(frozen=True)
class ProductRecord:
    """One deduplicated product and its aggregated signals."""

    product_id: str
    product_name: str
    product_brand: str
    site_category_lv1: str
    site_category_lv2: str
    num_reviews: int
    sentiment_score: float


def load_reviews(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and concatenate one or more review-split CSVs."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Review split not found: {path}")
        frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError("No review splits provided")

    df = pd.concat(frames, ignore_index=True)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["product_id", "product_name"]).copy()
    df["product_id"] = df["product_id"].astype(str)
    return df


def _mode_or_blank(series: pd.Series) -> str:
    """Most frequent non-null value, or an empty string when none exists."""
    cleaned = series.dropna()
    if cleaned.empty:
        return ""
    mode = cleaned.mode()
    return str(mode.iloc[0]) if not mode.empty else str(cleaned.iloc[0])


def build_catalog(
    df: pd.DataFrame,
    scorer: SentimentScorer,
    min_reviews: int = 1,
    limit: int | None = None,
) -> list[ProductRecord]:
    """Deduplicate reviews into product records with sentiment scores.

    Products with fewer than ``min_reviews`` reviews are dropped. Records are
    sorted by review count (desc) so that ``limit`` keeps the best-supported
    products, which also makes builds deterministic.
    """
    records: list[ProductRecord] = []
    for product_id, group in df.groupby("product_id", sort=False):
        if len(group) < min_reviews:
            continue
        records.append(
            ProductRecord(
                product_id=str(product_id),
                product_name=_mode_or_blank(group["product_name"]),
                product_brand=_mode_or_blank(group["product_brand"]),
                site_category_lv1=_mode_or_blank(group["site_category_lv1"]),
                site_category_lv2=_mode_or_blank(group["site_category_lv2"]),
                num_reviews=int(len(group)),
                sentiment_score=scorer.score_product(group),
            )
        )

    records.sort(key=lambda r: (-r.num_reviews, r.product_id))
    if limit is not None:
        records = records[:limit]
    return records


def catalog_to_frame(records: list[ProductRecord]) -> pd.DataFrame:
    """Serialise records to a DataFrame with the canonical column order."""
    return pd.DataFrame(
        [
            {
                "product_id": r.product_id,
                "product_name": r.product_name,
                "product_brand": r.product_brand,
                "site_category_lv1": r.site_category_lv1,
                "site_category_lv2": r.site_category_lv2,
                "num_reviews": r.num_reviews,
                "sentiment_score": r.sentiment_score,
            }
            for r in records
        ],
        columns=list(CATALOG_COLUMNS),
    )
