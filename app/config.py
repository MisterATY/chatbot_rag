QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "customs_chatbot"

EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_SIZE = 1024

# LLM Server Configuration
# LLM_SERVER_URL = "http://192.168.224.146:8000/v1/chat/completions"
LLM_SERVER_URL = "http://192.168.224.95:9000/v1/chat/completions"
LLM_MODEL = "Qwen3-14B-Q5_K_M.gguf"

# API Server (this FastAPI app)
API_HOST = "0.0.0.0"
API_PORT = 8080
