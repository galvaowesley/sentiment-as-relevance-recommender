"""Integration test that loads the real Qwen3-Embedding-0.6B model.

Run explicitly (downloads ~1.2 GB on first use):

    pytest -m integration
"""

from __future__ import annotations

import numpy as np
import pytest

from config import DEFAULT_EMBEDDING_DIM


@pytest.mark.integration
def test_qwen3_embeddings_shape_and_norm():
    from vectorizer.embedder import Qwen3Embedder

    embedder = Qwen3Embedder()
    docs = embedder.encode_documents(["Smartphone Samsung", "Geladeira Brastemp"])

    assert docs.shape == (2, DEFAULT_EMBEDDING_DIM)
    assert docs.dtype == np.float32
    norms = np.linalg.norm(docs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


@pytest.mark.integration
def test_query_and_document_prompts_differ():
    from vectorizer.embedder import Qwen3Embedder

    embedder = Qwen3Embedder()
    text = "Notebook Dell Inspiron"
    doc = embedder.encode_documents([text])[0]
    query = embedder.encode_queries([text])[0]
    # The instruction prefix should change the query representation.
    assert not np.allclose(doc, query, atol=1e-3)
