from __future__ import annotations

from .config import EMBEDDING_MODEL_ID, RETRIEVAL_TOP_K
from .db import get_db_connection
from .infrastructure.retrieval import FaissRetrievalService, build_default_retrieval_service

_DEFAULT_RETRIEVAL_SERVICE = build_default_retrieval_service(
    connection_factory=get_db_connection,
    embedding_model_id=EMBEDDING_MODEL_ID,
    top_k=RETRIEVAL_TOP_K,
)


def get_default_retrieval_service() -> FaissRetrievalService:
    return _DEFAULT_RETRIEVAL_SERVICE


def initialize_embeddings():
    return _DEFAULT_RETRIEVAL_SERVICE.initialize_embeddings()


def fetch_chunks_from_db():
    return _DEFAULT_RETRIEVAL_SERVICE.fetch_chunks_from_db()


def build_vectorstore(chunks):
    return _DEFAULT_RETRIEVAL_SERVICE.build_vectorstore(chunks)


def refresh_vectorstore_cache():
    _DEFAULT_RETRIEVAL_SERVICE.refresh_vectorstore_cache()


def get_vectorstore():
    return _DEFAULT_RETRIEVAL_SERVICE.get_vectorstore()


def query_context(query, k=None):
    return _DEFAULT_RETRIEVAL_SERVICE.query_context(query, k=k)


def get_retrieval_status():
    return _DEFAULT_RETRIEVAL_SERVICE.get_retrieval_status()

