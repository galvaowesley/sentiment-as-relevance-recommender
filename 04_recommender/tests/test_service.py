"""Tests for the FastAPI service using an injected (fake-embedder) recommender."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from recommender import ProductRecommender
from service import app as app_module


@pytest.fixture
def client(built_artifacts):
    config, embedder = built_artifacts
    recommender = ProductRecommender.load(config, embedder=embedder)
    app_module.set_recommender(recommender)
    with TestClient(app_module.app) as test_client:
        yield test_client
    app_module.set_recommender(None)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_products(client):
    response = client.get("/products", params={"limit": 10})
    assert response.status_code == 200
    ids = {p["product_id"] for p in response.json()}
    assert ids == {"S1", "S2"}


def test_get_product_and_404(client):
    assert client.get("/products/S1").status_code == 200
    assert client.get("/products/missing").status_code == 404


def test_recommend(client):
    response = client.get("/recommend", params={"product_id": "S1", "top_n": 3})
    assert response.status_code == 200
    body = response.json()
    assert body[0]["product_id"] == "C1"
    assert {"score", "similarity", "sentiment_score"} <= body[0].keys()


def test_recommend_unknown_product_404(client):
    assert client.get("/recommend", params={"product_id": "nope"}).status_code == 404
