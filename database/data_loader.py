from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
import os
import time

import psycopg2
from psycopg2.extras import RealDictCursor
from pypdf import PdfReader
import requests

from schema_builder import create_schema
from populate import populate_table, SUPPORTED_EXTENSIONS

app = Flask(__name__)
CORS(app, origins=["http://localhost:8080"])

UPLOAD_FOLDER = "uploads"
SEARCH_ENGINE_API_URL = "http://search:5000"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DB_CONFIG = {
    "host": os.getenv("DATABASE_HOST", "postgres"),
    "port": os.getenv("DATABASE_PORT", "5432"),
    "dbname": os.getenv("DATABASE_NAME", "llm_data"),
    "user": os.getenv("DATABASE_USER", "admin"),
    "password": os.getenv("DATABASE_PASSWORD", os.getenv("POSTGRES_PASSWORD", "admin")),
}

MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "3"))
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))
MAX_TOTAL_SIZE_BYTES = int(os.getenv("MAX_TOTAL_SIZE_BYTES", str(30 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "150"))

# Set up logger
app.logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_handler.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s - %(message)s")
log_handler.setFormatter(log_formatter)
app.logger.addHandler(log_handler)

# Store logs in a list
log_messages = []


class ListHandler(logging.Handler):
    def emit(self, record):
        log_messages.append(self.format(record))


list_handler = ListHandler()
list_handler.setFormatter(log_formatter)
app.logger.addHandler(list_handler)

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _db_connection():
    return psycopg2.connect(**DB_CONFIG)


def _is_supported_file(filename):
    extension = os.path.splitext(filename)[1].lower()
    return extension in SUPPORTED_EXTENSIONS


def _build_file_path(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        raise ValueError("Invalid filename.")

    base_name, extension = os.path.splitext(safe_name)
    candidate = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    if not os.path.exists(candidate):
        return candidate

    timestamp = int(time.time())
    deduped = f"{base_name}_{timestamp}{extension}"
    return os.path.join(app.config["UPLOAD_FOLDER"], deduped)


def _uploaded_file_size_bytes(file_obj):
    stream = file_obj.stream
    current_pos = stream.tell()
    stream.seek(0, os.SEEK_END)
    size_bytes = stream.tell()
    stream.seek(current_pos)
    return int(size_bytes)


def _uploaded_pdf_page_count(file_obj):
    stream = file_obj.stream
    current_pos = stream.tell()
    stream.seek(0)
    try:
        reader = PdfReader(stream)
        return len(reader.pages)
    finally:
        stream.seek(current_pos)


def _to_jsonable_document(row):
    item = dict(row)
    uploaded_at = item.get("uploaded_at")
    if isinstance(uploaded_at, datetime):
        item["uploaded_at"] = uploaded_at.isoformat()
    return item


def _load_context_state():
    create_schema()

    conn = _db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, filename, file_type, size_bytes, page_count, chunk_count, uploaded_at
                FROM documents
                ORDER BY uploaded_at DESC, id DESC;
                """
            )
            documents = [_to_jsonable_document(row) for row in cursor.fetchall()]
    finally:
        conn.close()

    document_count = len(documents)
    total_size_bytes = sum(int(doc.get("size_bytes") or 0) for doc in documents)

    blocked_reasons = []
    if document_count >= MAX_DOCUMENTS:
        blocked_reasons.append("document_limit")
    if total_size_bytes >= MAX_TOTAL_SIZE_BYTES:
        blocked_reasons.append("total_size_limit")

    return {
        "documents": documents,
        "limits": {
            "max_documents": MAX_DOCUMENTS,
            "max_file_size_bytes": MAX_FILE_SIZE_BYTES,
            "max_total_size_bytes": MAX_TOTAL_SIZE_BYTES,
            "max_pdf_pages": MAX_PDF_PAGES,
        },
        "usage": {
            "document_count": document_count,
            "total_size_bytes": total_size_bytes,
        },
        "is_upload_blocked": bool(blocked_reasons),
        "blocked_reasons": blocked_reasons,
    }


def _refresh_search_index(allow_empty=False):
    try:
        response = requests.post(f"{SEARCH_ENGINE_API_URL}/refresh", timeout=60)
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 502,
            "message": f"Error calling search engine API: {exc}",
        }

    if response.status_code == 200:
        return {"ok": True, "status_code": 200, "message": "Vector database built successfully."}

    error = "Unknown error"
    try:
        error = response.json().get("error", error)
    except Exception:
        pass

    if allow_empty and response.status_code == 404:
        return {
            "ok": True,
            "status_code": 200,
            "message": "Search index refreshed with empty corpus.",
        }

    return {
        "ok": False,
        "status_code": 502,
        "message": f"Error building vector database: {error}",
    }


def _safe_remove_file(path):
    if not path:
        return

    upload_root = os.path.abspath(app.config["UPLOAD_FOLDER"])
    absolute_path = os.path.abspath(path)

    if not absolute_path.startswith(upload_root):
        app.logger.warning(f"Skipping delete outside upload directory: {absolute_path}")
        return

    if os.path.exists(absolute_path):
        os.remove(absolute_path)


@app.route("/documents", methods=["GET"])
def list_documents():
    state = _load_context_state()
    return jsonify(state), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        app.logger.error("No file provided in the request.")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        app.logger.error("No file selected.")
        return jsonify({"error": "No file selected"}), 400

    if not _is_supported_file(file.filename):
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        app.logger.error("Unsupported file format for upload.")
        return jsonify({"error": f"Unsupported file format. Supported: {supported}"}), 400

    file_size_bytes = _uploaded_file_size_bytes(file)
    if file_size_bytes <= 0:
        return jsonify({"error": "Uploaded file is empty."}), 400

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        return (
            jsonify(
                {
                    "error": (
                        f"File exceeds limit of {MAX_FILE_SIZE_BYTES} bytes. "
                        f"Current file size: {file_size_bytes} bytes."
                    )
                }
            ),
            400,
        )

    extension = os.path.splitext(file.filename)[1].lower()
    page_count = None
    if extension == ".pdf":
        try:
            page_count = _uploaded_pdf_page_count(file)
        except Exception as exc:
            return jsonify({"error": f"Could not parse PDF: {exc}"}), 400

        if page_count > MAX_PDF_PAGES:
            return (
                jsonify(
                    {
                        "error": (
                            f"PDF exceeds page limit of {MAX_PDF_PAGES}. "
                            f"Current PDF pages: {page_count}."
                        )
                    }
                ),
                400,
            )

    # Reset stream pointer before saving.
    file.stream.seek(0)

    current_state = _load_context_state()
    current_docs = current_state["usage"]["document_count"]
    current_total_size = current_state["usage"]["total_size_bytes"]

    if current_docs >= MAX_DOCUMENTS:
        return jsonify({"error": f"Document limit reached ({MAX_DOCUMENTS}/{MAX_DOCUMENTS})."}), 409

    if current_total_size >= MAX_TOTAL_SIZE_BYTES:
        return (
            jsonify({"error": "Total storage limit reached. Delete a file before uploading."}),
            409,
        )

    projected_total = current_total_size + file_size_bytes
    if projected_total > MAX_TOTAL_SIZE_BYTES:
        return (
            jsonify(
                {
                    "error": (
                        f"Upload exceeds total size limit of {MAX_TOTAL_SIZE_BYTES} bytes. "
                        f"Projected total: {projected_total} bytes."
                    )
                }
            ),
            409,
        )

    filepath = _build_file_path(file.filename)
    filename = os.path.basename(filepath)
    file.save(filepath)

    message_parts = [f"File '{filename}' uploaded successfully."]

    try:
        create_schema()
        ingestion_result = populate_table(
            file_path=filepath,
            filename=filename,
            size_bytes=file_size_bytes,
            page_count=page_count,
        )
        message_parts.append(
            f"Ingestion completed with {ingestion_result['chunks_inserted']} chunks."
        )
    except Exception as exc:
        _safe_remove_file(filepath)
        app.logger.error(f"Error ingesting file: {exc}")
        return jsonify({"error": f"Error ingesting file: {exc}"}), 500

    refresh_result = _refresh_search_index(allow_empty=False)
    message_parts.append(refresh_result["message"])

    updated_state = _load_context_state()

    return (
        jsonify(
            {
                "message": " ".join(message_parts),
                "file_path": filepath,
                "ingestion": ingestion_result,
                "context": updated_state,
            }
        ),
        200 if refresh_result["ok"] else refresh_result["status_code"],
    )


@app.route("/documents/<int:document_id>", methods=["DELETE"])
def delete_document(document_id):
    create_schema()

    conn = _db_connection()
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
                return jsonify({"error": "Document not found."}), 404

            deleted_file_path = document.get("storage_path")
            deleted_name = document.get("filename")

            cursor.execute("DELETE FROM documents WHERE id = %s;", (document_id,))
        conn.commit()
    finally:
        conn.close()

    try:
        _safe_remove_file(deleted_file_path)
    except Exception as exc:
        app.logger.warning(f"Failed to remove file from disk: {exc}")

    refresh_result = _refresh_search_index(allow_empty=True)
    updated_state = _load_context_state()

    return (
        jsonify(
            {
                "message": (
                    f"Document '{deleted_name}' removed successfully. {refresh_result['message']}"
                ),
                "deleted_document_id": document_id,
                "context": updated_state,
            }
        ),
        200 if refresh_result["ok"] else refresh_result["status_code"],
    )


@app.route("/logs", methods=["GET"])
def get_logs():
    """Endpoint to retrieve log messages."""
    return jsonify({"logs": log_messages}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
