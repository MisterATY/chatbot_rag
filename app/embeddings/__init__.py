"""
Unified embedding interface. Exposes embedding_model.encode(texts, normalize_embeddings=True).
Backend is selected via config: bge-m3, qwen3 (HF), or qwen3 (GGUF).
"""
from app.config import (
    EMBEDDING_BACKEND,
    QWEN_EMBEDDING_GGUF_PATH,
)


def _load_embedding_model():
    if EMBEDDING_BACKEND == "qwen3" and QWEN_EMBEDDING_GGUF_PATH and QWEN_EMBEDDING_GGUF_PATH.strip():
        from app.embeddings.qwen3_gguf import get_model
        return get_model()
    if EMBEDDING_BACKEND == "qwen3":
        from app.embeddings.qwen3_hf import get_model
        return get_model()
    from app.embeddings.bge_m3 import get_model
    return get_model()


embedding_model = _load_embedding_model()
