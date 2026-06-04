import io
import logging
import pathlib
import sys
import unittest

APP_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from application.use_cases import (
    AppLimits,
    ChatService,
    ContextService,
    HealthService,
    QueryService,
)


class FakeIngestion:
    def __init__(self):
        self.state = {
            "documents": [],
            "limits": {},
            "usage": {"document_count": 0, "total_size_bytes": 0},
            "is_upload_blocked": False,
            "blocked_reasons": [],
        }

    def create_schema(self):
        return None

    def is_supported_file(self, filename):
        return filename.endswith(".txt")

    def build_file_path(self, filename, upload_folder):
        return f"{upload_folder}/{filename}"

    def uploaded_file_size_bytes(self, file_obj):
        return 10

    def uploaded_pdf_page_count(self, file_obj):
        return 1

    def ingest_file(self, file_path, original_filename=None, size_bytes=0, page_count=None):
        return {"document_id": 1, "chunks_inserted": 2}

    def safe_remove_file(self, path, upload_folder):
        return None

    def load_context_state(
        self,
        max_documents,
        max_file_size_bytes,
        max_total_size_bytes,
        max_pdf_pages,
    ):
        return self.state

    def delete_document_by_id(self, document_id, upload_folder):
        return {"filename": "a.txt"}


class FakeRetrieval:
    def __init__(self):
        self.refresh_exception = None
        self.last_query = None
        self.status = {"embedding_model_id": "x", "embeddings_initialized": True, "vectorstore_cached": True}

    def initialize_embeddings(self):
        return None

    def refresh_vectorstore_cache(self):
        if self.refresh_exception:
            raise self.refresh_exception
        return None

    def query_context(self, query, k=None):
        self.last_query = (query, k)
        return [{"content": "ok", "metadata": {"k": k}}]

    def get_retrieval_status(self):
        return self.status


class FakeChat:
    def __init__(self):
        self.status = {
            "token_present": True,
            "dns": {"api_inference": {"ok": True}, "router": {"ok": False}},
        }

    def get_chat_status(self):
        return self.status

    def startup_check_chat_client(self):
        return None

    def process_chat_query(self, user_query):
        return {"query": user_query, "response": "ok"}


class FakeDbHealth:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def check_connection(self):
        if self.should_fail:
            raise RuntimeError("db down")
        return None


class FakeUploadFile:
    def __init__(self, filename):
        self.filename = filename
        self.stream = io.BytesIO(b"hello world")

    def save(self, dst):
        self.saved_to = dst


class ApplicationServicesPhase1Tests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test-phase1")
        self.limits = AppLimits(
            max_documents=3,
            max_file_size_bytes=1024,
            max_total_size_bytes=4096,
            max_pdf_pages=150,
            retrieval_top_k=4,
            embedding_model_id="embed-model",
            upload_folder="uploads",
        )

    def test_context_refresh_returns_empty_message_when_allow_empty(self):
        ingestion = FakeIngestion()
        retrieval = FakeRetrieval()
        retrieval.refresh_exception = ValueError("No chunks found in database.")
        service = ContextService(ingestion=ingestion, retrieval=retrieval, limits=self.limits, logger=self.logger)

        result = service.refresh_search_index(allow_empty=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["status_code"], 200)
        self.assertIn("empty corpus", result["message"])

    def test_query_service_uses_default_top_k(self):
        retrieval = FakeRetrieval()
        service = QueryService(retrieval=retrieval, default_top_k=7)

        payload = service.execute("teste", requested_k=None)

        self.assertEqual(payload["query"], "teste")
        self.assertEqual(retrieval.last_query, ("teste", 7))

    def test_chat_service_validates_empty_query(self):
        service = ChatService(chat=FakeChat())

        with self.assertRaises(ValueError):
            service.execute("")

    def test_health_service_degraded_when_db_is_down(self):
        health = HealthService(
            db_health=FakeDbHealth(should_fail=True),
            retrieval=FakeRetrieval(),
            chat=FakeChat(),
        )

        payload, status_code = health.execute()

        self.assertEqual(status_code, 503)
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["database"]["ok"])

    def test_upload_rejects_unsupported_extension(self):
        service = ContextService(
            ingestion=FakeIngestion(),
            retrieval=FakeRetrieval(),
            limits=self.limits,
            logger=self.logger,
        )

        payload, status_code = service.handle_upload(FakeUploadFile("arquivo.exe"))

        self.assertEqual(status_code, 400)
        self.assertIn("Unsupported file format", payload["error"])


if __name__ == "__main__":
    unittest.main()
