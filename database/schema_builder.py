import os
import psycopg2

DB_CONFIG = {
    "host": os.getenv("DATABASE_HOST", "postgres"),
    "port": os.getenv("DATABASE_PORT", "5432"),
    "dbname": os.getenv("DATABASE_NAME", "llm_data"),
    "user": os.getenv("DATABASE_USER", "admin"),
    "password": os.getenv("DATABASE_PASSWORD", os.getenv("POSTGRES_PASSWORD", "admin")),
}


def create_schema():
    """Create the persistent schema used by the document+chunk RAG pipeline."""
    ddl = """
    CREATE TABLE IF NOT EXISTS documents (
        id BIGSERIAL PRIMARY KEY,
        filename TEXT NOT NULL,
        file_type TEXT NOT NULL,
        storage_path TEXT,
        size_bytes BIGINT NOT NULL DEFAULT 0,
        page_count INTEGER,
        chunk_count INTEGER NOT NULL DEFAULT 0,
        uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS chunks (
        id BIGSERIAL PRIMARY KEY,
        document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(document_id, chunk_index)
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

    ALTER TABLE documents ADD COLUMN IF NOT EXISTS storage_path TEXT;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS size_bytes BIGINT NOT NULL DEFAULT 0;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_count INTEGER;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER NOT NULL DEFAULT 0;
    """

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(ddl)
        conn.commit()
    finally:
        conn.close()


def create_table(_filepath=None):
    """
    Backward-compatible wrapper.
    The old pipeline called `create_table(filepath)` before population.
    """
    create_schema()

