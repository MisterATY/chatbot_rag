"""Qwen3-Embedding via SentenceTransformer (HuggingFace)."""
import torch

from app.config import EMBEDDING_DEVICE, QWEN_EMBEDDING_MODEL


def _device():
    if EMBEDDING_DEVICE and str(EMBEDDING_DEVICE).strip():
        return str(EMBEDDING_DEVICE).strip().lower()
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    from sentence_transformers import SentenceTransformer
    model_kwargs = {"low_cpu_mem_usage": True}
    try:
        model = SentenceTransformer(
            QWEN_EMBEDDING_MODEL,
            device=_device(),
            model_kwargs={**model_kwargs, "use_safetensors": True},
        )
    except OSError as e:
        if "safetensors" in str(e).lower():
            model = SentenceTransformer(QWEN_EMBEDDING_MODEL, device=_device(), model_kwargs=model_kwargs)
        else:
            raise
    except ValueError as e:
        if "torch to at least v2.6" in str(e):
            raise RuntimeError(
                "Qwen3 HF loading failed (torch.load). Upgrade: pip install 'torch>=2.6'"
            ) from e
        raise
    model = model.to(torch.device(_device()))
    print(f"[embeddings] qwen3_hf: {QWEN_EMBEDDING_MODEL}, device={_device()}")
    return model
