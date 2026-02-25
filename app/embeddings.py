"""Thin loader: embedding_model is implemented in app.embeddings (bge_m3, qwen3_hf, qwen3_gguf)."""
from app.embeddings import embedding_model

__all__ = ["embedding_model"]
