"""Tests for the zvec vector store wrapper (uses the real zvec backend)."""

from __future__ import annotations

import numpy as np
import pytest

from vector_store.store import ZvecVectorStore


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_build_query_returns_nearest_first(tmp_path):
    store = ZvecVectorStore.create(tmp_path / "col", dim=4, collection_name="test_corpus")
    embeddings = np.vstack(
        [_unit([1, 0, 0, 0]), _unit([0.9, 0.1, 0, 0]), _unit([0, 0, 1, 0])]
    )
    store.add(ids=["a", "b", "c"], embeddings=embeddings, num_reviews=[3, 3, 3])
    store.optimize()

    results = store.query(_unit([1, 0, 0, 0]), topk=3)
    ids = [pid for pid, _ in results]
    sims = {pid: sim for pid, sim in results}

    assert ids[0] == "a"
    assert sims["a"] > sims["b"] > sims["c"]
    assert sims["a"] == pytest.approx(1.0, abs=1e-5)


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "col"
    store = ZvecVectorStore.create(path, dim=4, collection_name="test_corpus")
    store.add(
        ids=["a", "b"],
        embeddings=np.vstack([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])]),
        num_reviews=[1, 1],
    )
    store.optimize()
    store.close()  # release the write lock

    reopened = ZvecVectorStore.open(path)
    top = reopened.query(_unit([0, 1, 0, 0]), topk=1)
    assert top[0][0] == "b"
