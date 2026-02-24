QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "customs_chatbot"

# Embedding backend: "bge-m3" (BAAI/bge-m3), "qwen3" (Qwen3-Embedding-*)
EMBEDDING_BACKEND = "qwen3"
EMBEDDING_MODEL = "BAAI/bge-m3"
QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
# Device for embedding: "cuda", "cpu", or None (auto: use cuda if available)
EMBEDDING_DEVICE = None
# GGUF embedding: set to a .gguf file path to use llama-cpp-python; leave "" to use HuggingFace model above. Only one is used.
QWEN_EMBEDDING_GGUF_PATH = ""  # e.g. "models/Qwen3-Embedding-8B-Q4_K_M.gguf"
# Vector dimension: bge-m3 1024; qwen3 0.6B 1024, 4B 2560, 8B 4096
VECTOR_SIZE = (
    4096 if EMBEDDING_BACKEND == "qwen3" and "8B" in QWEN_EMBEDDING_MODEL else
    2560 if EMBEDDING_BACKEND == "qwen3" and "4B" in QWEN_EMBEDDING_MODEL else
    1024
)

# LLM Server Configuration
# LLM_SERVER_URL = "http://192.168.224.146:8000/v1/chat/completions"
LLM_SERVER_URL = "http://192.168.224.95:9000/v1/chat/completions"
LLM_MODEL = "Qwen3-14B-Q5_K_M.gguf"

# API Server (this FastAPI app)
API_HOST = "0.0.0.0"
API_PORT = 8080
