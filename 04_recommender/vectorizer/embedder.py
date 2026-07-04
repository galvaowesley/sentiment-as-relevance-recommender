"""Dense embeddings of product titles using Qwen3-Embedding-0.6B.

Corpus titles are encoded as *documents* (no instruction) while query products are
encoded with the model's instruction prefix, following the asymmetric retrieval
setup recommended for Qwen3-Embedding. All vectors are L2-normalised, so a dot
product equals cosine similarity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from config import DEFAULT_EMBEDDING_DIM, DEFAULT_MODEL, DEFAULT_QUERY_INSTRUCTION


def pick_device(preferred: str | None = None) -> str:
    """Select a torch device, preferring CUDA, then Apple MPS, then CPU.

    ``preferred`` (or the ``RECOMMENDER_DEVICE`` env var via the config) overrides
    auto-detection so the same code runs on any machine.
    """
    if preferred:
        return preferred
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    """Human-readable label for the torch device used to vectorize."""
    if device == "cuda":
        try:
            import torch

            return f"GPU (CUDA) — {torch.cuda.get_device_name(0)}"
        except Exception:
            return "GPU (CUDA)"
    if device == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


class Qwen3Embedder:
    """Wraps a ``sentence-transformers`` model to embed product titles."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = 32,
        attn_implementation: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = pick_device(device)
        print(f"[embedder] vectorizing on {describe_device(self.device)}")
        model_kwargs: dict[str, str] = {}
        # Qwen3 uses grouped-query attention; the fused SDPA matmul crashes on
        # Apple MPS ("Failed to infer result type"). Eager attention expands the
        # KV heads first and runs cleanly, so force it on MPS by default.
        if attn_implementation is None and self.device == "mps":
            attn_implementation = "eager"
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = SentenceTransformer(
            model_name, device=self.device, model_kwargs=model_kwargs or None
        )
        self.batch_size = batch_size
        # sentence-transformers prepends this prefix to every query string.
        self.query_prompt = f"Instruct: {query_instruction}\nQuery:"

    @property
    def dimension(self) -> int:
        dim = self.model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else DEFAULT_EMBEDDING_DIM

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed corpus titles (no instruction prefix)."""
        return self._encode(list(texts), prompt=None)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed query titles with the instruction prefix."""
        return self._encode(list(texts), prompt=self.query_prompt)

    def _encode(self, texts: list[str], prompt: str | None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        )
        # The model computes in bf16, so re-normalise in float32 to guarantee exact
        # unit vectors (a plain dot product then equals cosine similarity).
        emb = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return emb / norms
