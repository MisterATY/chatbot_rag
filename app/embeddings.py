from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL

# low_cpu_mem_usage reduces peak RAM when loading large models (e.g. BGE-M3)
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    model_kwargs={"low_cpu_mem_usage": True},
)
