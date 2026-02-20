QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "customs_chatbot"

# Embedding backend: "bge-m3" (BAAI/bge-m3), "qwen3" (Qwen3-Embedding-0.6B), or "qwen3-vl" (Qwen3-VL-Embedding-2B)
EMBEDDING_BACKEND = "qwen3"
EMBEDDING_MODEL = "BAAI/bge-m3"
QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Vector dimension: 1024 for bge-m3 and qwen3 (0.6B), 2048 for qwen3-vl (2B)
VECTOR_SIZE = 2048 if EMBEDDING_BACKEND == "qwen3-vl" else 1024

# LLM Server Configuration
# LLM_SERVER_URL = "http://192.168.224.146:8000/v1/chat/completions"
LLM_SERVER_URL = "http://192.168.224.95:9000/v1/chat/completions"
LLM_MODEL = "Qwen3-14B-Q5_K_M.gguf"

# API Server (this FastAPI app)
API_HOST = "0.0.0.0"
API_PORT = 8080
