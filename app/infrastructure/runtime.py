from __future__ import annotations

from modules.config import (
    DATABASE_HOST,
    DATABASE_NAME,
    DATABASE_PASSWORD,
    DATABASE_PORT,
    DATABASE_USER,
    EMBEDDING_MODEL_ID,
    RETRIEVAL_TOP_K,
)

from .db import PostgresSettings, build_default_postgres_factory
from .retrieval import FaissRetrievalService, build_default_retrieval_service

_DEFAULT_CONNECTION_FACTORY = build_default_postgres_factory(
    PostgresSettings(
        host=DATABASE_HOST,
        port=DATABASE_PORT,
        database=DATABASE_NAME,
        user=DATABASE_USER,
        password=DATABASE_PASSWORD,
    )
)

_DEFAULT_RETRIEVAL_SERVICE = build_default_retrieval_service(
    connection_factory=_DEFAULT_CONNECTION_FACTORY.create_connection,
    embedding_model_id=EMBEDDING_MODEL_ID,
    top_k=RETRIEVAL_TOP_K,
)


def get_default_connection_factory():
    return _DEFAULT_CONNECTION_FACTORY


def get_db_connection():
    return _DEFAULT_CONNECTION_FACTORY.create_connection()


def get_default_retrieval_service() -> FaissRetrievalService:
    return _DEFAULT_RETRIEVAL_SERVICE

