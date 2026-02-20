"""
Shared token counting for ingestion and retrieval.
Uses the same tokenizer as the embedding model (BAAI/bge-m3) for consistency.
"""
from functools import lru_cache

from app.config import EMBEDDING_MODEL

# BGE-M3 max sequence length; avoid tokenizer overflow on very long text
MAX_MODEL_TOKENS = 8192


@lru_cache(maxsize=1)
def get_tokenizer():
    """Load and cache the tokenizer for the embedding model."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


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
