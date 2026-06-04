import csv
import os
import re

import psycopg2
from psycopg2.extras import Json
from pypdf import PdfReader

from schema_builder import create_schema

DB_CONFIG = {
    "host": os.getenv("DATABASE_HOST", "postgres"),
    "port": os.getenv("DATABASE_PORT", "5432"),
    "dbname": os.getenv("DATABASE_NAME", "llm_data"),
    "user": os.getenv("DATABASE_USER", "admin"),
    "password": os.getenv("DATABASE_PASSWORD", os.getenv("POSTGRES_PASSWORD", "admin")),
}

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".pdf"}
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))


def _read_text_with_fallback(file_path):
    encodings = ("utf-8-sig", "utf-8", "latin-1")
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding, errors="strict") as source:
                return source.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, "r", encoding="utf-8", errors="replace") as source:
        return source.read()


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


def _extract_units_from_csv(file_path):
    """
    Extract robustly from CSV without assuming strict tabular consistency.
    We keep each record as text so malformed delimiters don't break ingestion.
    """
    units = []
    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as source:
        reader = csv.reader(source)
        rows = [row for row in reader if any(cell.strip() for cell in row)]

    if not rows:
        return units

    header_text = " | ".join(cell.strip() for cell in rows[0] if cell.strip())
    for row_idx, row in enumerate(rows[1:], start=1):
        row_text = " | ".join(cell.strip() for cell in row if cell.strip())
        if not row_text:
            continue
        units.append(
            {
                "text": f"CSV header: {header_text}\nCSV row {row_idx}: {row_text}",
                "metadata": {"source_type": "csv_row", "row_number": row_idx},
            }
        )

    # Header-only CSVs still become searchable.
    if not units and header_text:
        units.append(
            {
                "text": f"CSV header: {header_text}",
                "metadata": {"source_type": "csv_header"},
            }
        )
    return units


def _extract_units_from_txt(file_path):
    text = _read_text_with_fallback(file_path)
    blocks = [segment.strip() for segment in re.split(r"\n\s*\n", text) if segment.strip()]
    units = []

    if blocks:
        for idx, block in enumerate(blocks, start=1):
            units.append(
                {
                    "text": block,
                    "metadata": {"source_type": "txt_block", "block_number": idx},
                }
            )
        return units

    normalized = _normalize_text(text)
    if normalized:
        units.append({"text": normalized, "metadata": {"source_type": "txt_raw"}})
    return units


def _extract_units_from_pdf(file_path):
    reader = PdfReader(file_path)
    units = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = _normalize_text(page.extract_text() or "")
        if not page_text:
            continue
        units.append(
            {
                "text": page_text,
                "metadata": {"source_type": "pdf_page", "page_number": page_number},
            }
        )
    return units


def _extract_units(file_path, extension):
    if extension == ".csv":
        return _extract_units_from_csv(file_path)
    if extension == ".txt":
        return _extract_units_from_txt(file_path)
    if extension == ".pdf":
        return _extract_units_from_pdf(file_path)
    raise ValueError(f"Unsupported file format: {extension}")


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
    file_name = original_filename or os.path.basename(file_path)
    extension = os.path.splitext(file_name)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension: {extension}. Supported: {supported}")

    units = _extract_units(file_path, extension)
    chunks = _build_chunks(units)
    if not chunks:
        raise ValueError("No readable content found in file.")

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (filename, file_type, storage_path, size_bytes, page_count)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (file_name, extension.lstrip("."), file_path, int(size_bytes or 0), page_count),
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


def populate_table(file_path=None, filename=None, size_bytes=0, page_count=None):
    """
    Backward-compatible wrapper used by the upload API.
    """
    if not file_path:
        raise ValueError("file_path is required for ingestion.")
    return ingest_file(
        file_path=file_path,
        original_filename=filename,
        size_bytes=size_bytes,
        page_count=page_count,
    )
