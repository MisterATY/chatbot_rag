"""Qwen3-Embedding GGUF via llama-cpp-python. Avoids llama_decode -1 by truncating text and encoding in small batches."""
import numpy as np

from app.config import QWEN_EMBEDDING_GGUF_PATH

# Avoid context overflow: max chars per text (llama_decode -1 when input too long)
GGUF_MAX_TEXT_CHARS = 6000
# Encode in small batches to avoid decode errors
GGUF_ENCODE_BATCH_SIZE = 16


def get_model():
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "QWEN_EMBEDDING_GGUF_PATH is set but 'llama-cpp-python' is not installed. "
            "CPU: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu "
            "CUDA: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
        ) from e

    path = QWEN_EMBEDDING_GGUF_PATH.strip()
    llama = Llama(
        model_path=path,
        embedding=True,
        n_ctx=8192,
        n_batch=256,
        n_gpu_layers=-1,
        verbose=False,
    )

    class _Wrapper:
        def encode(self, texts, normalize_embeddings=True, **kwargs):
            single = isinstance(texts, str)
            if single:
                texts = [texts]
            texts = [t if t else "" for t in texts]
            texts = [t[:GGUF_MAX_TEXT_CHARS] if len(t) > GGUF_MAX_TEXT_CHARS else t for t in texts]
            out = []
            for i in range(0, len(texts), GGUF_ENCODE_BATCH_SIZE):
                batch = texts[i : i + GGUF_ENCODE_BATCH_SIZE]
                batch_out = llama.embed(batch, normalize=normalize_embeddings, truncate=True)
                out.extend(batch_out)
            arr = np.array(out, dtype=np.float32)
            if single:
                return arr[0]
            return arr

    print(f"[embeddings] qwen3_gguf: {path}, max_chars={GGUF_MAX_TEXT_CHARS}, batch={GGUF_ENCODE_BATCH_SIZE}")
    return _Wrapper()
