import os
import re
from datetime import datetime

from infrastructure.runtime import get_db_connection
from psycopg2.extras import Json, RealDictCursor
from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE, SUPPORTED_EXTENSIONS, UPLOAD_FOLDER
from .ingestion_parsers import build_default_parser_registry, extract_units

PARSER_REGISTRY = build_default_parser_registry()


def create_schema():
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

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(ddl)
        conn.commit()
    finally:
        conn.close()


def is_supported_file(filename):
    extension = os.path.splitext(filename)[1].lower()
    return extension in SUPPORTED_EXTENSIONS


def build_file_path(filename, upload_folder=UPLOAD_FOLDER):
    from werkzeug.utils import secure_filename

    safe_name = secure_filename(filename)
    if not safe_name:
        raise ValueError("Invalid filename.")

    base_name, extension = os.path.splitext(safe_name)
    candidate = os.path.join(upload_folder, safe_name)
    if not os.path.exists(candidate):
        return candidate

    import time

    timestamp = int(time.time())
    deduped = f"{base_name}_{timestamp}{extension}"
    return os.path.join(upload_folder, deduped)


def uploaded_file_size_bytes(file_obj):
    stream = file_obj.stream
    current_pos = stream.tell()
    stream.seek(0, os.SEEK_END)
    size_bytes = stream.tell()
    stream.seek(current_pos)
    return int(size_bytes)


def uploaded_pdf_page_count(file_obj):
    stream = file_obj.stream
    current_pos = stream.tell()
    stream.seek(0)
    try:
        reader = PdfReader(stream)
        return len(reader.pages)
    finally:
        stream.seek(current_pos)


def file_size_bytes(file_path):
    return int(os.path.getsize(file_path))


def pdf_page_count(file_path):
    with open(file_path, "rb") as pdf_stream:
        reader = PdfReader(pdf_stream)
        return len(reader.pages)


def safe_remove_file(path, upload_folder=UPLOAD_FOLDER):
    if not path:
        return

    upload_root = os.path.normcase(os.path.abspath(upload_folder))
    absolute_path = os.path.normcase(os.path.abspath(path))
    if not absolute_path.startswith(upload_root):
        return

    if os.path.exists(absolute_path):
        os.remove(absolute_path)


def _normalize_text(text):
    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    content = _normalize_text(text)
    if not content:
        return []

    if len(content) <= chunk_size:
        return [content]

    effective_overlap = max(0, min(overlap, chunk_size - 1))
    step = max(1, chunk_size - effective_overlap)

    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        piece = content[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(content):
            break
        start += step
    return chunks


def register_file_parser(extension, parser):
    PARSER_REGISTRY.register(extension, parser)


def _build_chunks(units):
    chunks = []
    chunk_index = 0
    for unit in units:
        unit_text = unit["text"]
        unit_metadata = unit.get("metadata", {})
        for local_idx, content in enumerate(_chunk_text(unit_text), start=1):
            metadata = dict(unit_metadata)
            metadata["unit_chunk"] = local_idx
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "content": content,
                    "metadata": metadata,
                }
            )
            chunk_index += 1
    return chunks


def ingest_file(file_path, original_filename=None, size_bytes=0, page_count=None):
    create_schema()
    normalized_path = os.path.normcase(os.path.abspath(file_path))

    file_name = original_filename or os.path.basename(normalized_path)
    extension = os.path.splitext(file_name)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension: {extension}. Supported: {supported}")

    units = extract_units(normalized_path, extension, registry=PARSER_REGISTRY)
    chunks = _build_chunks(units)
    if not chunks:
        raise ValueError("No readable content found in file.")

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (filename, file_type, storage_path, size_bytes, page_count)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    file_name,
                    extension.lstrip("."),
                    normalized_path,
                    int(size_bytes or 0),
                    page_count,
                ),
            )
            document_id = cursor.fetchone()[0]

            insert_sql = """
                INSERT INTO chunks (document_id, chunk_index, content, metadata)
                VALUES (%s, %s, %s, %s);
            """
            for chunk in chunks:
                cursor.execute(
                    insert_sql,
                    (
                        document_id,
                        chunk["chunk_index"],
                        chunk["content"],
                        Json(chunk["metadata"]),
                    ),
                )

            cursor.execute(
                "UPDATE documents SET chunk_count = %s WHERE id = %s;",
                (len(chunks), document_id),
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "document_id": document_id,
        "filename": file_name,
        "file_type": extension.lstrip("."),
        "chunks_inserted": len(chunks),
        "size_bytes": int(size_bytes or 0),
        "page_count": page_count,
    }


def _to_jsonable_document(row):
    item = dict(row)
    uploaded_at = item.get("uploaded_at")
    if isinstance(uploaded_at, datetime):
        item["uploaded_at"] = uploaded_at.isoformat()
    return item


def load_context_state(max_documents, max_file_size_bytes, max_total_size_bytes, max_pdf_pages):
    create_schema()

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT
                    d.id,
                    d.filename,
                    d.file_type,
                    d.size_bytes,
                    d.page_count,
                    d.chunk_count,
                    d.uploaded_at,
                    COALESCE(stats.csv_rows, 0) AS csv_rows,
                    COALESCE(stats.txt_blocks, 0) AS txt_blocks,
                    COALESCE(stats.pdf_pages_detected, 0) AS pdf_pages_detected
                FROM documents d
                LEFT JOIN LATERAL (
                    SELECT
                        COUNT(DISTINCT (c.metadata->>'row_number')) FILTER (WHERE c.metadata ? 'row_number') AS csv_rows,
                        COUNT(DISTINCT (c.metadata->>'block_number')) FILTER (WHERE c.metadata ? 'block_number') AS txt_blocks,
                        COUNT(DISTINCT (c.metadata->>'page_number')) FILTER (WHERE c.metadata ? 'page_number') AS pdf_pages_detected
                    FROM chunks c
                    WHERE c.document_id = d.id
                ) stats ON TRUE
                ORDER BY d.uploaded_at DESC, d.id DESC;
                """
            )
            documents = [_to_jsonable_document(row) for row in cursor.fetchall()]
    finally:
        conn.close()

    document_count = len(documents)
    total_size_bytes = sum(int(doc.get("size_bytes") or 0) for doc in documents)

    blocked_reasons = []
    if document_count >= max_documents:
        blocked_reasons.append("document_limit")
    if total_size_bytes >= max_total_size_bytes:
        blocked_reasons.append("total_size_limit")

    return {
        "documents": documents,
        "limits": {
            "max_documents": max_documents,
            "max_file_size_bytes": max_file_size_bytes,
            "max_total_size_bytes": max_total_size_bytes,
            "max_pdf_pages": max_pdf_pages,
        },
        "usage": {
            "document_count": document_count,
            "total_size_bytes": total_size_bytes,
        },
        "is_upload_blocked": bool(blocked_reasons),
        "blocked_reasons": blocked_reasons,
    }


def delete_document_by_id(document_id, upload_folder=UPLOAD_FOLDER, remove_file=True):
    create_schema()

    conn = get_db_connection()
    deleted_file_path = None
    deleted_name = None
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, filename, storage_path
                FROM documents
                WHERE id = %s;
                """,
                (document_id,),
            )
            document = cursor.fetchone()
            if not document:
                raise ValueError("Document not found.")

            deleted_file_path = document.get("storage_path")
            deleted_name = document.get("filename")
            cursor.execute("DELETE FROM documents WHERE id = %s;", (document_id,))

        conn.commit()
    finally:
        conn.close()

    if remove_file:
        safe_remove_file(deleted_file_path, upload_folder=upload_folder)
    return {"document_id": document_id, "filename": deleted_name, "storage_path": deleted_file_path}


def delete_document_by_storage_path(storage_path, upload_folder=UPLOAD_FOLDER, remove_file=False):
    create_schema()
    normalized_path = os.path.normcase(os.path.abspath(storage_path))

    conn = get_db_connection()
    deleted_rows = []
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, filename, storage_path
                FROM documents
                WHERE storage_path = %s;
                """,
                (normalized_path,),
            )
            deleted_rows = cursor.fetchall() or []
            if deleted_rows:
                cursor.execute("DELETE FROM documents WHERE storage_path = %s;", (normalized_path,))
        conn.commit()
    finally:
        conn.close()

    if remove_file:
        safe_remove_file(normalized_path, upload_folder=upload_folder)

    return {
        "storage_path": normalized_path,
        "deleted_documents": [dict(row) for row in deleted_rows],
        "deleted_count": len(deleted_rows),
    }
