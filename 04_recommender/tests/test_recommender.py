"""End-to-end wiring tests for ProductRecommender (fake embedder, real zvec)."""

from __future__ import annotations

import pandas as pd

from recommender import ProductRecommender, _load_catalog


def _load(built_artifacts) -> ProductRecommender:
    config, embedder = built_artifacts
    return ProductRecommender.load(config, embedder=embedder)


def test_load_catalog_fills_missing_strings(tmp_path):
    path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "product_id": ["1"],
            "product_name": ["Item"],
            "product_brand": [None],  # missing brand -> must become "" not NaN
            "site_category_lv1": ["Cat"],
            "site_category_lv2": ["Sub"],
            "num_reviews": [2],
            "sentiment_score": [0.5],
        }
    ).to_csv(path, index=False)

    df = _load_catalog(path)
    assert df.loc[0, "product_brand"] == ""
    assert isinstance(df.loc[0, "product_brand"], str)


def test_recommends_matching_title_first(built_artifacts):
    rec = _load(built_artifacts)
    # S1 shares C1's title, so C1 must be the top recommendation.
    results = rec.recommend(product_id="S1", top_n=3)
    assert results
    assert results[0].product_id == "C1"
    assert results[0].similarity > 0.99


def test_excludes_query_product_itself(built_artifacts):
    rec = _load(built_artifacts)
    results = rec.recommend(product_id="C1", top_n=10)
    assert "C1" not in {r.product_id for r in results}


def test_scores_sorted_descending(built_artifacts):
    rec = _load(built_artifacts)
    results = rec.recommend(product_id="S2", top_n=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_n_is_respected(built_artifacts):
    rec = _load(built_artifacts)
    assert len(rec.recommend(product_id="S1", top_n=1)) == 1


def test_free_text_query(built_artifacts):
    rec = _load(built_artifacts)
    results = rec.recommend(query_text="Smartphone Samsung Galaxy S10", top_n=1)
    assert results[0].product_id == "C1"


def test_unknown_product_raises(built_artifacts):
    rec = _load(built_artifacts)
    try:
        rec.recommend(product_id="does-not-exist")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown product_id")


def test_storefront_browsing(built_artifacts):
    rec = _load(built_artifacts)
    products = rec.list_storefront(limit=10)
    assert {p["product_id"] for p in products} == {"S1", "S2"}
    assert rec.get_storefront_product("S1")["product_name"].startswith("Smartphone")
    assert rec.get_storefront_product("nope") is None
