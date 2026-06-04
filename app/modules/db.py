from __future__ import annotations

from .config import (
    DATABASE_HOST,
    DATABASE_NAME,
    DATABASE_PASSWORD,
    DATABASE_PORT,
    DATABASE_USER,
)
from .infrastructure.db import PostgresSettings, build_default_postgres_factory

_DEFAULT_FACTORY = build_default_postgres_factory(
    PostgresSettings(
        host=DATABASE_HOST,
        port=DATABASE_PORT,
        database=DATABASE_NAME,
        user=DATABASE_USER,
        password=DATABASE_PASSWORD,
    )
)


def get_default_connection_factory():
    return _DEFAULT_FACTORY


def get_db_connection():
    return _DEFAULT_FACTORY.create_connection()

