QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "customs_chatbot_lang"

# Languages
# Supported language codes: "oz" (Uzbek Latin, default), "uz" (Uzbek Cyrillic), "ru", "en"
LANGUAGES = ["oz", "uz", "ru", "en"]
DEFAULT_LANG = "oz"

# Embedding backend: "bge-m3", "qwen3", or "remote" (HTTP embed server)
EMBEDDING_BACKEND = "remote"
EMBEDDING_MODEL = "BAAI/bge-m3"
QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
# Device for embedding: "cuda", "cpu", or None (auto: use cuda if available)
EMBEDDING_DEVICE = None
# GGUF embedding: set to a .gguf file path to use llama-cpp-python; leave "" to use HuggingFace model above.
QWEN_EMBEDDING_GGUF_PATH = ""  # e.g. "models/Qwen3-Embedding-8B-Q4_K_M.gguf"
# Remote embedding server: POST {url}/embed_batch with {"texts": ["A", "B", ...]}; set when EMBEDDING_BACKEND="remote"
EMBEDDING_SERVER_URL = "http://192.168.224.95:9080"  # e.g. "http://192.168.224.95:9080"
# Vector dimension: bge-m3 1024; qwen3 0.6B 1024, 4B 2560, 8B 4096; for remote set to your server's dimension
VECTOR_SIZE = (
    4096 if EMBEDDING_BACKEND == "qwen3" and "8B" in QWEN_EMBEDDING_MODEL else
    2560 if EMBEDDING_BACKEND == "qwen3" and "4B" in QWEN_EMBEDDING_MODEL else
    1024  # used for bge-m3 and remote (override for remote if your server uses different dim)
)

# LLM Server Configuration
# LLM_SERVER_URL = "http://192.168.224.146:8000/v1/chat/completions"
LLM_SERVER_URL = "http://192.168.224.146:8000/v1/chat/completions"
LLM_MODEL = "Qwen3-14B-Q5_K_M.gguf"

# API Server (this FastAPI app)
API_HOST = "0.0.0.0"
API_PORT = 8080
