"""Embedding via remote HTTP server.

- Batch endpoint (ingestion): POST {url}/embed_batch with {"texts": [...]}.
- Fast single-text endpoint (retriever): POST {url}/embed_fast with {"text": "..."}.
"""
import numpy as np
import requests

from app.config import EMBEDDING_SERVER_URL

# Max texts per request to avoid timeouts
REMOTE_EMBED_BATCH_SIZE = 100
REMOTE_TIMEOUT = 120


def get_model():
    url = (EMBEDDING_SERVER_URL or "").strip().rstrip("/")
    if not url:
        raise ValueError(
            "EMBEDDING_BACKEND is 'remote' but EMBEDDING_SERVER_URL is not set in config. "
            "Set EMBEDDING_SERVER_URL to e.g. 'http://192.168.224.95:9080'"
        )
    embed_batch_url = f"{url}/embed_batch"
    embed_fast_url = f"{url}/embed_fast"

    class _RemoteWrapper:
        def encode(self, texts, normalize_embeddings=True, **kwargs):
            # Single-text: use /embed_fast
            if isinstance(texts, str):
                text = texts if texts is not None else ""
                resp = requests.post(
                    embed_fast_url,
                    json={"text": text},
                    headers={"Content-Type": "application/json"},
                    timeout=REMOTE_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                # Accept common response shapes
                if isinstance(data, dict):
                    vec = (
                        data.get("embedding")
                        or data.get("vector")
                        or data.get("emb")
                        or data.get("embedding_vector")
                    )
                else:
                    vec = data
                v = np.array(vec, dtype=np.float32)
                if normalize_embeddings:
                    norm = np.linalg.norm(v)
                    if norm == 0:
                        norm = 1.0
                    v = v / norm
                return v

            # Batch texts: use /embed_batch
            texts = [t if t is not None else "" for t in texts]
            out = []
            for i in range(0, len(texts), REMOTE_EMBED_BATCH_SIZE):
                batch = texts[i : i + REMOTE_EMBED_BATCH_SIZE]
                resp = requests.post(
                    embed_batch_url,
                    json={"texts": batch},
                    headers={"Content-Type": "application/json"},
                    timeout=REMOTE_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                vectors = data.get("embeddings", data) if isinstance(data, dict) else data
                if normalize_embeddings and vectors:
                    vectors = np.array(vectors, dtype=np.float32)
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1, norms)
                    vectors = (vectors / norms).tolist()
                out.extend(vectors)
            return np.array(out, dtype=np.float32)

    print(
        f"[embeddings] remote: batch={embed_batch_url}, fast={embed_fast_url}, "
        f"batch_size={REMOTE_EMBED_BATCH_SIZE}"
    )
    return _RemoteWrapper()
