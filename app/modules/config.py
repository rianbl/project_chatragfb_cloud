import os

DATABASE_HOST = os.getenv("DATABASE_HOST", "postgres")
DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
DATABASE_NAME = os.getenv("DATABASE_NAME", "llm_data")
DATABASE_USER = os.getenv("DATABASE_USER", "admin")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", os.getenv("POSTGRES_PASSWORD", "admin"))

CONTEXT_ROOT = os.getenv("CONTEXT_ROOT", os.getenv("UPLOAD_FOLDER", "uploads"))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", CONTEXT_ROOT)
SUPPORTED_EXTENSIONS = {".csv", ".txt", ".pdf"}

MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "3"))
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))
MAX_TOTAL_SIZE_BYTES = int(os.getenv("MAX_TOTAL_SIZE_BYTES", str(30 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "150"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "all-MiniLM-L6-v2")

MCP_SERVER_ENABLED = os.getenv("MCP_SERVER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8090").strip()
MCP_TIMEOUT = float(os.getenv("MCP_TIMEOUT", "10"))
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "").strip()
MCP_MEMORY_ENABLED = os.getenv("MCP_MEMORY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))
MEMORY_MAX_OBSERVATIONS = int(os.getenv("MEMORY_MAX_OBSERVATIONS", "3"))

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto").strip()
HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "60"))
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()

APP_ROOT = os.path.dirname(os.path.dirname(__file__))
PROMPT_STORE_PATH = os.getenv(
    "PROMPT_STORE_PATH",
    os.path.join(APP_ROOT, "config", "prompt_store.yaml"),
)
