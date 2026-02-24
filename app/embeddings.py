"""
Unified embedding interface. Supports BGE-M3 (SentenceTransformer) and Qwen3-Embedding (HF or GGUF via llama-cpp-python). Exposes embedding_model.encode(texts, normalize_embeddings=True).
"""
import numpy as np

from app.config import (
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    QWEN_EMBEDDING_GGUF_PATH,
    QWEN_EMBEDDING_MODEL,
)


def _embedding_device():
    """Use GPU for embedding if available."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class _QwenGGUFWrapper:
    """Use Qwen3-Embedding-8B GGUF via llama-cpp-python. Same .encode(texts, normalize_embeddings=True) interface."""

    def __init__(self):
        from llama_cpp import Llama
        # n_gpu_layers=-1 offloads all layers to GPU when built with CUDA/Metal
        self._llama = Llama(
            model_path=QWEN_EMBEDDING_GGUF_PATH.strip(),
            embedding=True,
            n_ctx=8192,
            n_batch=512,
            n_gpu_layers=-1,
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


def _get_embedding_model():
    if EMBEDDING_BACKEND == "qwen3" and QWEN_EMBEDDING_GGUF_PATH and QWEN_EMBEDDING_GGUF_PATH.strip():
        return _QwenGGUFWrapper()
    device = _embedding_device()
    if EMBEDDING_BACKEND == "qwen3":
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(
            QWEN_EMBEDDING_MODEL,
            device=device,
            model_kwargs={"low_cpu_mem_usage": True},
        )
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
        model_kwargs={"low_cpu_mem_usage": True},
    )


embedding_model = _get_embedding_model()
