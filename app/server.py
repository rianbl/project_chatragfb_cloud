import logging
import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from modules.application.adapters import (
    DefaultChatAdapter,
    DefaultDatabaseHealthAdapter,
    DefaultIngestionAdapter,
    DefaultRetrievalAdapter,
)
from modules.application.services import (
    AppLimits,
    ChatService,
    ContextService,
    HealthService,
    QueryService,
    StartupService,
)
from modules.config import (
    EMBEDDING_MODEL_ID,
    MAX_DOCUMENTS,
    MAX_FILE_SIZE_BYTES,
    MAX_PDF_PAGES,
    MAX_TOTAL_SIZE_BYTES,
    RETRIEVAL_TOP_K,
    UPLOAD_FOLDER,
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

limits = AppLimits(
    max_documents=MAX_DOCUMENTS,
    max_file_size_bytes=MAX_FILE_SIZE_BYTES,
    max_total_size_bytes=MAX_TOTAL_SIZE_BYTES,
    max_pdf_pages=MAX_PDF_PAGES,
    retrieval_top_k=RETRIEVAL_TOP_K,
    embedding_model_id=EMBEDDING_MODEL_ID,
    upload_folder=UPLOAD_FOLDER,
)

ingestion_adapter = DefaultIngestionAdapter()
retrieval_adapter = DefaultRetrievalAdapter()
chat_adapter = DefaultChatAdapter()
db_health_adapter = DefaultDatabaseHealthAdapter()

context_service = ContextService(
    ingestion=ingestion_adapter,
    retrieval=retrieval_adapter,
    limits=limits,
    logger=app.logger,
)
query_service = QueryService(retrieval=retrieval_adapter, default_top_k=limits.retrieval_top_k)
chat_service = ChatService(chat=chat_adapter)
health_service = HealthService(
    db_health=db_health_adapter,
    retrieval=retrieval_adapter,
    chat=chat_adapter,
)
startup_service = StartupService(
    db_health=db_health_adapter,
    ingestion=ingestion_adapter,
    retrieval=retrieval_adapter,
    chat=chat_adapter,
    context=context_service,
    embedding_model_id=limits.embedding_model_id,
    logger=app.logger,
)


@app.route("/", methods=["GET"])
def root():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/documents", methods=["GET"])
def list_documents():
    app.logger.info("Listing context documents state.")
    return jsonify(context_service.current_state()), 200


@app.route("/health", methods=["GET"])
def health():
    payload, status_code = health_service.execute()
    return jsonify(payload), status_code


@app.route("/upload", methods=["POST"])
def upload_file():
    app.logger.info("Upload requested.")
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    payload, status_code = context_service.handle_upload(request.files["file"])
    return jsonify(payload), status_code


@app.route("/documents/<int:document_id>", methods=["DELETE"])
def delete_document(document_id):
    payload, status_code = context_service.delete_document(document_id)
    return jsonify(payload), status_code


@app.route("/refresh", methods=["POST"])
def refresh():
    refresh_result = context_service.refresh_search_index(allow_empty=False)
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
        requested_k = payload.get("k", limits.retrieval_top_k)
        return jsonify(query_service.execute(query_text, requested_k=requested_k)), 200
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
        return jsonify(chat_service.execute(user_query)), 200
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
    startup_service.run()
    app.run(host="0.0.0.0", port=8080)
