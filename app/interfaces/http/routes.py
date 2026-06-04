from __future__ import annotations

import os

from flask import jsonify, request, send_from_directory

from bootstrap.container import ServiceContainer


def register_routes(app, container: ServiceContainer, log_messages: list[str]) -> None:
    @app.route("/", methods=["GET"])
    def root():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/documents", methods=["GET"])
    def list_documents():
        app.logger.info("Listing context documents state.")
        return jsonify(container.context_service.current_state()), 200

    @app.route("/health", methods=["GET"])
    def health():
        payload, status_code = container.health_service.execute()
        return jsonify(payload), status_code

    @app.route("/upload", methods=["POST"])
    def upload_file():
        app.logger.info("Upload requested.")
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        payload, status_code = container.context_service.handle_upload(request.files["file"])
        return jsonify(payload), status_code

    @app.route("/documents/<int:document_id>", methods=["DELETE"])
    def delete_document(document_id):
        payload, status_code = container.context_service.delete_document(document_id)
        return jsonify(payload), status_code

    @app.route("/refresh", methods=["POST"])
    def refresh():
        refresh_result = container.context_service.refresh_search_index(allow_empty=False)
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
            requested_k = payload.get("k", container.limits.retrieval_top_k)
            return jsonify(container.query_service.execute(query_text, requested_k=requested_k)), 200
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
            return jsonify(container.chat_service.execute(user_query)), 200
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

