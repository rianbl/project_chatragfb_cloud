import os

DATABASE_HOST = os.getenv("DATABASE_HOST", "postgres")
DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
DATABASE_NAME = os.getenv("DATABASE_NAME", "llm_data")
DATABASE_USER = os.getenv("DATABASE_USER", "admin")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", os.getenv("POSTGRES_PASSWORD", "admin"))

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
SUPPORTED_EXTENSIONS = {".csv", ".txt", ".pdf"}

MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "3"))
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))
MAX_TOTAL_SIZE_BYTES = int(os.getenv("MAX_TOTAL_SIZE_BYTES", str(30 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "150"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "all-MiniLM-L6-v2")

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto").strip()
HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "60"))
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()

APP_ROOT = os.path.dirname(os.path.dirname(__file__))
PROMPT_STORE_PATH = os.getenv(
    "PROMPT_STORE_PATH",
    os.path.join(APP_ROOT, "config", "prompt_store.yaml"),
)
