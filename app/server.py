import logging
import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from modules.chat_module import get_chat_status, process_chat_query, startup_check_chat_client
from modules.config import (
    EMBEDDING_MODEL_ID,
    MAX_DOCUMENTS,
    MAX_FILE_SIZE_BYTES,
    MAX_PDF_PAGES,
    MAX_TOTAL_SIZE_BYTES,
    RETRIEVAL_TOP_K,
    UPLOAD_FOLDER,
)
from modules.db import get_db_connection
from modules.ingestion import (
    build_file_path,
    create_schema,
    delete_document_by_id,
    ingest_file,
    is_supported_file,
    load_context_state,
    uploaded_file_size_bytes,
    uploaded_pdf_page_count,
)
from modules.retrieval import (
    get_retrieval_status,
    initialize_embeddings,
    query_context,
    refresh_vectorstore_cache,
)

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

app.logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_handler.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s - %(message)s")
log_handler.setFormatter(log_formatter)
app.logger.addHandler(log_handler)

log_messages = []


class ListHandler(logging.Handler):
    def emit(self, record):
        log_messages.append(self.format(record))


list_handler = ListHandler()
list_handler.setFormatter(log_formatter)
app.logger.addHandler(list_handler)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _current_state():
    return load_context_state(
        max_documents=MAX_DOCUMENTS,
        max_file_size_bytes=MAX_FILE_SIZE_BYTES,
        max_total_size_bytes=MAX_TOTAL_SIZE_BYTES,
        max_pdf_pages=MAX_PDF_PAGES,
    )


def _refresh_search_index(allow_empty=False):
    try:
        app.logger.info("Refreshing FAISS index from persisted chunks.")
        refresh_vectorstore_cache()
        return {"ok": True, "status_code": 200, "message": "Vector index refreshed successfully."}
    except ValueError as exc:
        if allow_empty:
            return {"ok": True, "status_code": 200, "message": "Vector index refreshed with empty corpus."}
        return {"ok": False, "status_code": 404, "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status_code": 500, "message": str(exc)}


def _check_database_connection():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1;")
            cursor.fetchone()
    finally:
        conn.close()


def _startup_readiness_check():
    app.logger.info("Starting app readiness checks.")
    app.logger.info("Checking PostgreSQL connectivity.")
    _check_database_connection()
    app.logger.info("PostgreSQL connection is healthy.")

    app.logger.info("Ensuring documents/chunks schema exists.")
    create_schema()
    app.logger.info("Schema check completed.")

    app.logger.info("Initializing embedding model '%s'.", EMBEDDING_MODEL_ID)
    initialize_embeddings()
    app.logger.info("Embedding model is ready.")
    app.logger.info("Initializing chat inference client and DNS checks.")
    startup_check_chat_client()
    app.logger.info("Chat inference startup checks completed.")

    state = _current_state()
    app.logger.info(
        "Current corpus state: documents=%s total_size_bytes=%s blocked=%s",
        state["usage"]["document_count"],
        state["usage"]["total_size_bytes"],
        state["is_upload_blocked"],
    )
    app.logger.info("Readiness checks completed successfully.")


@app.route("/", methods=["GET"])
def root():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/documents", methods=["GET"])
def list_documents():
    app.logger.info("Listing context documents state.")
    return jsonify(_current_state()), 200


@app.route("/health", methods=["GET"])
def health():
    try:
        _check_database_connection()
        db_ok = True
        db_error = None
    except Exception as exc:  # noqa: BLE001
        db_ok = False
        db_error = str(exc)

    retrieval_status = get_retrieval_status()
    chat_status = get_chat_status()
    chat_ready = chat_status["token_present"] and (
        chat_status["dns"]["api_inference"]["ok"] or chat_status["dns"]["router"]["ok"]
    )
    ready = db_ok and retrieval_status["embeddings_initialized"] and chat_ready
    status_code = 200 if ready else 503

    return (
        jsonify(
            {
                "status": "ok" if ready else "degraded",
                "database": {"ok": db_ok, "error": db_error},
                "retrieval": retrieval_status,
                "chat": chat_status,
            }
        ),
        status_code,
    )


@app.route("/upload", methods=["POST"])
def upload_file():
    app.logger.info("Upload requested.")
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    app.logger.info("Incoming filename='%s'.", file.filename)
    if not is_supported_file(file.filename):
        return jsonify({"error": "Unsupported file format. Supported: .csv, .pdf, .txt"}), 400

    file_size_bytes = uploaded_file_size_bytes(file)
    app.logger.info("File size detected: %s bytes.", file_size_bytes)
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
            page_count = uploaded_pdf_page_count(file)
            app.logger.info("PDF page count detected: %s pages.", page_count)
        except Exception as exc:  # noqa: BLE001
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

    file.stream.seek(0)

    state = _current_state()
    current_docs = state["usage"]["document_count"]
    current_total_size = state["usage"]["total_size_bytes"]
    app.logger.info(
        "Current limits usage before upload: docs=%s/%s total_size=%s/%s.",
        current_docs,
        MAX_DOCUMENTS,
        current_total_size,
        MAX_TOTAL_SIZE_BYTES,
    )

    if current_docs >= MAX_DOCUMENTS:
        return jsonify({"error": f"Document limit reached ({MAX_DOCUMENTS}/{MAX_DOCUMENTS})."}), 409

    if current_total_size >= MAX_TOTAL_SIZE_BYTES:
        return jsonify({"error": "Total storage limit reached. Delete a file before uploading."}), 409

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

    filepath = build_file_path(file.filename, upload_folder=UPLOAD_FOLDER)
    filename = os.path.basename(filepath)
    app.logger.info("Saving upload to '%s'.", filepath)
    file.save(filepath)

    try:
        ingestion_result = ingest_file(
            file_path=filepath,
            original_filename=filename,
            size_bytes=file_size_bytes,
            page_count=page_count,
        )
        app.logger.info(
            "Ingestion done: document_id=%s chunks=%s.",
            ingestion_result.get("document_id"),
            ingestion_result.get("chunks_inserted"),
        )
    except Exception as exc:  # noqa: BLE001
        from modules.ingestion import safe_remove_file

        safe_remove_file(filepath, upload_folder=UPLOAD_FOLDER)
        app.logger.exception("Upload failed during ingestion.")
        return jsonify({"error": f"Error ingesting file: {exc}"}), 500

    refresh_result = _refresh_search_index(allow_empty=False)
    app.logger.info("Refresh result after upload: ok=%s msg='%s'.", refresh_result["ok"], refresh_result["message"])
    updated_state = _current_state()

    return (
        jsonify(
            {
                "message": refresh_result["message"],
                "file_path": filepath,
                "ingestion": ingestion_result,
                "context": updated_state,
            }
        ),
        200 if refresh_result["ok"] else refresh_result["status_code"],
    )


@app.route("/documents/<int:document_id>", methods=["DELETE"])
def delete_document(document_id):
    app.logger.info("Delete requested for document_id=%s.", document_id)
    try:
        deleted = delete_document_by_id(document_id, upload_folder=UPLOAD_FOLDER)
    except ValueError:
        return jsonify({"error": "Document not found."}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to delete document: {exc}"}), 500

    refresh_result = _refresh_search_index(allow_empty=True)
    app.logger.info(
        "Delete completed for document_id=%s. Refresh result: ok=%s msg='%s'.",
        document_id,
        refresh_result["ok"],
        refresh_result["message"],
    )
    updated_state = _current_state()

    return (
        jsonify(
            {
                "message": (
                    f"Document '{deleted['filename']}' removed successfully. "
                    f"{refresh_result['message']}"
                ),
                "deleted_document_id": document_id,
                "context": updated_state,
            }
        ),
        200 if refresh_result["ok"] else refresh_result["status_code"],
    )


@app.route("/refresh", methods=["POST"])
def refresh():
    refresh_result = _refresh_search_index(allow_empty=False)
    status = 200 if refresh_result["ok"] else refresh_result["status_code"]
    key = "message" if refresh_result["ok"] else "error"
    return jsonify({key: refresh_result["message"]}), status


@app.route("/query", methods=["POST"])
def query():
    payload = request.json or {}
    query_text = (payload.get("query") or "").strip()
    if not query_text:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        requested_k = payload.get("k", RETRIEVAL_TOP_K)
        results = query_context(query_text, k=requested_k)
        return jsonify({"query": query_text, "results": results}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.json or {}
    user_query = (payload.get("query") or "").strip()
    if not user_query:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        return jsonify(process_chat_query(user_query)), 200
    except ValueError as exc:
        app.logger.warning("Chat validation error: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Chat processing failed.")
        error_text = str(exc)
        lowered = error_text.lower()
        if "failed to resolve" in lowered or "nameresolutionerror" in lowered:
            error_text = (
                "Chat provider DNS resolution failed. "
                "Check container internet/DNS access and HF_PROVIDER configuration."
            )
        return jsonify({"error": f"Chat processing failed: {error_text}"}), 502


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    app.logger.info("Received feedback data: %s", data)
    return jsonify({"message": "Data received successfully"}), 200


@app.route("/_routes", methods=["GET"])
def routes():
    mapped = sorted(
        [
            {
                "rule": str(rule),
                "methods": sorted([m for m in rule.methods if m not in {"HEAD", "OPTIONS"}]),
            }
            for rule in app.url_map.iter_rules()
        ],
        key=lambda item: item["rule"],
    )
    return jsonify({"routes": mapped}), 200


@app.route("/logs", methods=["GET"])
def get_logs():
    return jsonify({"logs": log_messages}), 200


@app.route("/<path:path>", methods=["GET"])
def static_files(path):
    file_path = os.path.join(app.static_folder, path)
    if os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    _startup_readiness_check()
    app.run(host="0.0.0.0", port=8080)
