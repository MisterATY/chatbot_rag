"""
Unified embedding interface. Supports BGE-M3 (SentenceTransformer) and
Qwen3-VL-Embedding-2B. Exposes embedding_model.encode(texts, normalize_embeddings=True).
"""
import numpy as np

from app.config import (
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    QWEN_EMBEDDING_MODEL,
)


class _QwenEncodeWrapper:
    """Wrapper so Qwen3VLEmbedder matches SentenceTransformer.encode(texts, normalize_embeddings=True)."""

    def __init__(self):
        from app.qwen3_vl_embedding import Qwen3VLEmbedder
        self._embedder = Qwen3VLEmbedder(model_name_or_path=QWEN_EMBEDDING_MODEL)

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        inputs = [{"text": t} for t in texts]
        out = self._embedder.process(inputs, normalize=normalize_embeddings)
        arr = out.cpu().float().numpy()
        if single:
            return arr[0]
        return arr


def _get_embedding_model():
    if EMBEDDING_BACKEND == "qwen3-vl":
        return _QwenEncodeWrapper()
    if EMBEDDING_BACKEND == "qwen3":
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(
            QWEN_EMBEDDING_MODEL,
            model_kwargs={"low_cpu_mem_usage": True},
        )
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        EMBEDDING_MODEL,
        model_kwargs={"low_cpu_mem_usage": True},
    )


embedding_model = _get_embedding_model()
