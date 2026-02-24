"""
Unified embedding interface. Supports BGE-M3 (SentenceTransformer), Qwen3-Embedding (HF or GGUF via llama-cpp-python), and Qwen3-VL-Embedding-2B. Exposes embedding_model.encode(texts, normalize_embeddings=True).
"""
import numpy as np

from app.config import (
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    QWEN_EMBEDDING_GGUF_PATH,
    QWEN_EMBEDDING_MODEL,
)


class _QwenGGUFWrapper:
    """Use Qwen3-Embedding-8B GGUF via llama-cpp-python. Same .encode(texts, normalize_embeddings=True) interface."""

    def __init__(self):
        from llama_cpp import Llama
        self._llama = Llama(
            model_path=QWEN_EMBEDDING_GGUF_PATH.strip(),
            embedding=True,
            n_ctx=8192,
            n_batch=512,
            verbose=False,
        )

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        out = self._llama.embed(texts, normalize=normalize_embeddings, truncate=True)
        arr = np.array(out, dtype=np.float32)
        if single:
            return arr[0]
        return arr


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
    if EMBEDDING_BACKEND == "qwen3" and QWEN_EMBEDDING_GGUF_PATH and QWEN_EMBEDDING_GGUF_PATH.strip():
        return _QwenGGUFWrapper()
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
