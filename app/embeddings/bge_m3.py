"""BGE-M3 embedding via SentenceTransformer (BAAI/bge-m3)."""
import torch

from app.config import EMBEDDING_DEVICE, EMBEDDING_MODEL


def _device():
    if EMBEDDING_DEVICE and str(EMBEDDING_DEVICE).strip():
        return str(EMBEDDING_DEVICE).strip().lower()
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    from sentence_transformers import SentenceTransformer
    model_kwargs = {"low_cpu_mem_usage": True}
    try:
        model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=_device(),
            model_kwargs={**model_kwargs, "use_safetensors": True},
        )
    except OSError as e:
        if "safetensors" in str(e).lower():
            model = SentenceTransformer(EMBEDDING_MODEL, device=_device(), model_kwargs=model_kwargs)
        else:
            raise
    except ValueError as e:
        if "torch to at least v2.6" in str(e):
            raise RuntimeError(
                "BGE-M3 loading failed (torch.load). Upgrade: pip install 'torch>=2.6'"
            ) from e
        raise
    model = model.to(torch.device(_device()))
    print(f"[embeddings] bge_m3: device={_device()}")
    return model
