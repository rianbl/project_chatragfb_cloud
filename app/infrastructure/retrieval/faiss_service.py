from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


class ChunkDataSource(Protocol):
    def fetch_chunks(self) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class RetrievalSettings:
    embedding_model_id: str
    top_k: int


class PostgresChunkDataSource:
    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connection_factory = connection_factory

    def fetch_chunks(self) -> list[dict[str, Any]]:
        from psycopg2.extras import RealDictCursor

        conn = self._connection_factory()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT
                        c.id AS chunk_id,
                        c.content,
                        c.chunk_index,
                        c.metadata,
                        d.id AS document_id,
                        d.filename,
                        d.file_type
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    ORDER BY d.id, c.chunk_index;
                    """
                )
                rows = cursor.fetchall()
                logger.info("Loaded %s chunks from PostgreSQL for indexing/querying.", len(rows))
                return rows
        finally:
            conn.close()


class FaissRetrievalService:
    def __init__(
        self,
        *,
        settings: RetrievalSettings,
        chunk_source: ChunkDataSource,
        embedding_factory: Callable[[str], Any] | None = None,
        document_factory: Callable[[str, dict[str, Any]], Any] | None = None,
        vectorstore_builder: Callable[[list[Any], Any], Any] | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._chunk_source = chunk_source
        self._embedding_factory = embedding_factory or self._default_embedding_factory
        self._document_factory = document_factory or self._default_document_factory
        self._vectorstore_builder = vectorstore_builder or self._default_vectorstore_builder
        self._logger = logger_instance or logger
        self._embeddings = None
        self._vectorstore_cache = None

    @staticmethod
    def _default_embedding_factory(model_name: str):
        import sentence_transformers
        import torch
        import transformers
        from langchain_community.embeddings import SentenceTransformerEmbeddings

        logger.info(
            "Embedding runtime versions: sentence-transformers=%s transformers=%s torch=%s",
            sentence_transformers.__version__,
            transformers.__version__,
            torch.__version__,
        )
        logger.info(
            "Initializing embeddings model '%s' (this may download model files on first run).",
            model_name,
        )
        embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        logger.info("Embeddings model '%s' initialized successfully.", model_name)
        return embeddings

    @staticmethod
    def _default_vectorstore_builder(documents: list[Any], embeddings: Any):
        from langchain_community.vectorstores import FAISS

        return FAISS.from_documents(documents, embeddings)

    @staticmethod
    def _default_document_factory(content: str, metadata: dict[str, Any]):
        from langchain_core.documents import Document

        return Document(page_content=content, metadata=metadata)

    def initialize_embeddings(self):
        if self._embeddings is None:
            self._embeddings = self._embedding_factory(self._settings.embedding_model_id)
        return self._embeddings

    def fetch_chunks_from_db(self) -> list[dict[str, Any]]:
        return self._chunk_source.fetch_chunks()

    def build_vectorstore(self, chunks: list[dict[str, Any]]):
        embeddings = self.initialize_embeddings()

        documents = []
        for row in chunks:
            metadata = {
                "chunk_id": row["chunk_id"],
                "chunk_index": row["chunk_index"],
                "document_id": row["document_id"],
                "filename": row["filename"],
                "file_type": row["file_type"],
            }
            extra_metadata = row.get("metadata")
            if isinstance(extra_metadata, dict):
                metadata.update(extra_metadata)
            documents.append(self._document_factory(row["content"], metadata))

        if not documents:
            raise ValueError("No chunks available for indexing.")

        self._logger.info("Building FAISS index from %s chunk documents.", len(documents))
        vectorstore = self._vectorstore_builder(documents, embeddings)
        self._logger.info("FAISS index build completed.")
        return vectorstore

    def refresh_vectorstore_cache(self) -> None:
        chunks = self.fetch_chunks_from_db()
        if not chunks:
            self._vectorstore_cache = None
            raise ValueError("No chunks found in database.")

        self._vectorstore_cache = self.build_vectorstore(chunks)
        self._logger.info("Vectorstore cache refreshed successfully.")

    def get_vectorstore(self):
        if self._vectorstore_cache is None:
            self._logger.info("Vectorstore cache is empty. Refreshing from database.")
            self.refresh_vectorstore_cache()
        else:
            self._logger.info("Reusing cached vectorstore.")

        return self._vectorstore_cache

    def query_context(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("Query cannot be empty.")

        if k is None:
            k_value = self._settings.top_k
        else:
            k_value = int(k)

        k_value = max(1, min(k_value, 20))
        self._logger.info("Running retrieval query with k=%s. Query preview='%s'.", k_value, query[:80])
        retriever = self.get_vectorstore().as_retriever(search_kwargs={"k": k_value})
        docs = retriever.invoke(query)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

    def get_retrieval_status(self) -> dict[str, Any]:
        return {
            "embedding_model_id": self._settings.embedding_model_id,
            "embeddings_initialized": self._embeddings is not None,
            "vectorstore_cached": self._vectorstore_cache is not None,
        }


def build_default_retrieval_service(
    *,
    connection_factory: Callable[[], Any],
    embedding_model_id: str,
    top_k: int,
) -> FaissRetrievalService:
    settings = RetrievalSettings(embedding_model_id=embedding_model_id, top_k=top_k)
    data_source = PostgresChunkDataSource(connection_factory=connection_factory)
    return FaissRetrievalService(settings=settings, chunk_source=data_source)

