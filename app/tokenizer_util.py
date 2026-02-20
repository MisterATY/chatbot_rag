"""
Shared token counting for ingestion and retrieval.
Uses the tokenizer of the active embedding backend (bge-m3 or qwen3-vl).
"""
from functools import lru_cache

from app.config import EMBEDDING_BACKEND, EMBEDDING_MODEL, QWEN_EMBEDDING_MODEL

# Max sequence length; avoid tokenizer overflow on very long text
MAX_MODEL_TOKENS = 8192


def _tokenizer_model():
    if EMBEDDING_BACKEND in ("qwen3", "qwen3-vl"):
        return QWEN_EMBEDDING_MODEL
    return EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_tokenizer():
    """Load and cache the tokenizer for the embedding model."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_tokenizer_model())


def count_tokens(text: str) -> int:
    """Return token count for text (deterministic, matches embedding model)."""
    if not text or not text.strip():
        return 0
    tokenizer = get_tokenizer()
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        max_length=MAX_MODEL_TOKENS,
        truncation=True,
    )
    return len(ids)
