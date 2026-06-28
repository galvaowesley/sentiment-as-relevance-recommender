"""Unit tests for the pure re-ranking function."""

from __future__ import annotations

import pytest

from reranking.reranker import Candidate, rerank


def _cand(pid: str, sim: float, sent: float) -> Candidate:
    return Candidate(product_id=pid, similarity=sim, sentiment_score=sent, metadata={})


def test_weighted_score_and_order():
    candidates = [
        _cand("a", sim=0.9, sent=0.0),  # 0.5*0.9 + 0.5*0.0 = 0.45
        _cand("b", sim=0.4, sent=1.0),  # 0.5*0.4 + 0.5*1.0 = 0.70
        _cand("c", sim=0.5, sent=0.5),  # 0.50
    ]
    ranked = rerank(candidates, alpha=0.5)
    assert [r.product_id for r in ranked] == ["b", "c", "a"]
    assert ranked[0].score == pytest.approx(0.70)


def test_alpha_extremes():
    candidates = [_cand("a", 0.9, 0.0), _cand("b", 0.1, 1.0)]
    assert [r.product_id for r in rerank(candidates, alpha=1.0)] == ["a", "b"]
    assert [r.product_id for r in rerank(candidates, alpha=0.0)] == ["b", "a"]


def test_exclude_id_drops_self():
    candidates = [_cand("self", 1.0, 1.0), _cand("other", 0.2, 0.2)]
    ranked = rerank(candidates, alpha=0.7, exclude_id="self")
    assert [r.product_id for r in ranked] == ["other"]


def test_top_n_caps_results():
    candidates = [_cand(str(i), sim=i / 10, sent=0.0) for i in range(5)]
    assert len(rerank(candidates, alpha=1.0, top_n=2)) == 2


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        rerank([], alpha=1.5)


def test_deterministic_tie_break():
    candidates = [_cand("b", 0.5, 0.5), _cand("a", 0.5, 0.5)]
    assert [r.product_id for r in rerank(candidates, alpha=0.5)] == ["a", "b"]
