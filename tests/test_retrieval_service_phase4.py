import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from infrastructure.retrieval.faiss_service import FaissRetrievalService, RetrievalSettings


class _FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query):
        del query
        return self._docs


class _FakeVectorStore:
    def __init__(self, docs):
        self._docs = docs
        self.requested_k = None

    def as_retriever(self, search_kwargs):
        self.requested_k = search_kwargs.get("k")
        return _FakeRetriever(self._docs)


class _FakeChunkSource:
    def __init__(self, rows):
        self._rows = rows
        self.calls = 0

    def fetch_chunks(self):
        self.calls += 1
        return list(self._rows)


class RetrievalServicePhase4Tests(unittest.TestCase):
    def test_embeddings_initialized_once(self):
        calls = {"count": 0}

        def _embedding_factory(model_name):
            calls["count"] += 1
            return f"emb::{model_name}"

        service = FaissRetrievalService(
            settings=RetrievalSettings(embedding_model_id="m1", top_k=4),
            chunk_source=_FakeChunkSource([]),
            embedding_factory=_embedding_factory,
            document_factory=lambda content, metadata: _FakeDoc(content, metadata),
            vectorstore_builder=lambda docs, emb: _FakeVectorStore(docs),
        )

        first = service.initialize_embeddings()
        second = service.initialize_embeddings()

        self.assertEqual(first, "emb::m1")
        self.assertEqual(second, "emb::m1")
        self.assertEqual(calls["count"], 1)

    def test_query_context_clamps_k_and_returns_documents(self):
        source_rows = [
            {
                "chunk_id": 1,
                "content": "conteudo teste",
                "chunk_index": 0,
                "metadata": {"origin": "x"},
                "document_id": 10,
                "filename": "a.txt",
                "file_type": "txt",
            }
        ]
        source = _FakeChunkSource(source_rows)

        service = FaissRetrievalService(
            settings=RetrievalSettings(embedding_model_id="m1", top_k=4),
            chunk_source=source,
            embedding_factory=lambda _: "emb",
            document_factory=lambda content, metadata: _FakeDoc(content, metadata),
            vectorstore_builder=lambda docs, emb: _FakeVectorStore(docs),
        )

        service.refresh_vectorstore_cache()
        vectorstore = service.get_vectorstore()
        results = service.query_context("pergunta", k=99)

        self.assertEqual(source.calls, 1)
        self.assertEqual(vectorstore.requested_k, 20)
        self.assertEqual(results[0]["content"], "conteudo teste")
        self.assertEqual(results[0]["metadata"]["origin"], "x")
        self.assertEqual(results[0]["metadata"]["filename"], "a.txt")

    def test_services_do_not_share_cache_state(self):
        source_a = _FakeChunkSource(
            [
                {
                    "chunk_id": 1,
                    "content": "doc a",
                    "chunk_index": 0,
                    "metadata": {},
                    "document_id": 10,
                    "filename": "a.txt",
                    "file_type": "txt",
                }
            ]
        )
        source_b = _FakeChunkSource(
            [
                {
                    "chunk_id": 2,
                    "content": "doc b",
                    "chunk_index": 0,
                    "metadata": {},
                    "document_id": 20,
                    "filename": "b.txt",
                    "file_type": "txt",
                }
            ]
        )

        service_a = FaissRetrievalService(
            settings=RetrievalSettings(embedding_model_id="m1", top_k=4),
            chunk_source=source_a,
            embedding_factory=lambda _: "emb",
            document_factory=lambda content, metadata: _FakeDoc(content, metadata),
            vectorstore_builder=lambda docs, emb: _FakeVectorStore(docs),
        )
        service_b = FaissRetrievalService(
            settings=RetrievalSettings(embedding_model_id="m1", top_k=4),
            chunk_source=source_b,
            embedding_factory=lambda _: "emb",
            document_factory=lambda content, metadata: _FakeDoc(content, metadata),
            vectorstore_builder=lambda docs, emb: _FakeVectorStore(docs),
        )

        service_a.refresh_vectorstore_cache()
        service_b.refresh_vectorstore_cache()

        result_a = service_a.query_context("q")
        result_b = service_b.query_context("q")
        self.assertEqual(result_a[0]["content"], "doc a")
        self.assertEqual(result_b[0]["content"], "doc b")


if __name__ == "__main__":
    unittest.main()
