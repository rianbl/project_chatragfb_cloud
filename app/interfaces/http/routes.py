from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from threading import Lock
from typing import TYPE_CHECKING, Any

from flask import Response, jsonify, request, send_from_directory
from modules.config import INTERNAL_API_TOKEN

if TYPE_CHECKING:
    from bootstrap.container import ServiceContainer


def register_routes(app, container: "ServiceContainer", log_messages: list[str]) -> None:
    class _MemoryGraphEventHub:
        def __init__(self) -> None:
            self._subscribers: list[Queue[str]] = []
            self._lock = Lock()

        @staticmethod
        def encode_event(event_name: str, payload: dict[str, Any]) -> str:
            return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def has_subscribers(self) -> bool:
            with self._lock:
                return bool(self._subscribers)

        def subscribe(self) -> Queue[str]:
            subscriber: Queue[str] = Queue(maxsize=16)
            with self._lock:
                self._subscribers.append(subscriber)
            return subscriber

        def unsubscribe(self, subscriber: Queue[str]) -> None:
            with self._lock:
                self._subscribers = [item for item in self._subscribers if item is not subscriber]

        def publish(self, event_name: str, payload: dict[str, Any]) -> None:
            chunk = self.encode_event(event_name, payload)
            with self._lock:
                subscribers = list(self._subscribers)
            for subscriber in subscribers:
                try:
                    subscriber.put_nowait(chunk)
                except Full:
                    try:
                        subscriber.get_nowait()
                    except Empty:
                        pass
                    try:
                        subscriber.put_nowait(chunk)
                    except Full:
                        continue

    memory_graph_events = _MemoryGraphEventHub()
    INTERNAL_MEMORY_ENTITY_NAMES = {"session_memory"}

    def _as_memory_graph(raw: Any) -> dict[str, list[dict[str, Any]]]:
        if not isinstance(raw, dict):
            return {"entities": [], "relations": []}

        if isinstance(raw.get("entities"), list) and isinstance(raw.get("relations"), list):
            entities = [item for item in raw.get("entities", []) if isinstance(item, dict)]
            relations = [item for item in raw.get("relations", []) if isinstance(item, dict)]
            return {"entities": entities, "relations": relations}

        structured = raw.get("structuredContent")
        if isinstance(structured, dict):
            return _as_memory_graph(structured)

        content = raw.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                try:
                    parsed = json.loads(text)
                except ValueError:
                    continue
                graph = _as_memory_graph(parsed)
                if graph["entities"] or graph["relations"]:
                    return graph

        return {"entities": [], "relations": []}

    def _build_graph_view(graph: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        entities = graph.get("entities", [])
        relations = graph.get("relations", [])
        filtered_entities: list[dict[str, Any]] = []
        for entity in entities:
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            if name.lower() in INTERNAL_MEMORY_ENTITY_NAMES:
                continue
            filtered_entities.append(entity)

        filtered_relations: list[dict[str, Any]] = []
        for relation in relations:
            source = str(relation.get("from", "")).strip()
            target = str(relation.get("to", "")).strip()
            if not source or not target:
                continue
            if source.lower() in INTERNAL_MEMORY_ENTITY_NAMES or target.lower() in INTERNAL_MEMORY_ENTITY_NAMES:
                continue
            filtered_relations.append(relation)

        nodes: list[dict[str, Any]] = []
        node_names: set[str] = set()

        for entity in filtered_entities:
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            node_names.add(name)
            observations = entity.get("observations", [])
            normalized_observations = (
                [str(item) for item in observations if str(item).strip()]
                if isinstance(observations, list)
                else []
            )
            nodes.append(
                {
                    "id": name,
                    "label": name,
                    "type": str(entity.get("entityType", "unknown")).strip() or "unknown",
                    "observations": normalized_observations,
                    "observation_count": len(normalized_observations),
                }
            )

        edges: list[dict[str, Any]] = []
        for index, relation in enumerate(filtered_relations):
            source = str(relation.get("from", "")).strip()
            target = str(relation.get("to", "")).strip()
            relation_type = str(relation.get("relationType", "")).strip()
            if not source or not target:
                continue
            if source not in node_names:
                node_names.add(source)
                nodes.append(
                    {
                        "id": source,
                        "label": source,
                        "type": "unknown",
                        "observations": [],
                        "observation_count": 0,
                    }
                )
            if target not in node_names:
                node_names.add(target)
                nodes.append(
                    {
                        "id": target,
                        "label": target,
                        "type": "unknown",
                        "observations": [],
                        "observation_count": 0,
                    }
                )

            edges.append(
                {
                    "id": f"{source}|{relation_type}|{target}|{index}",
                    "source": source,
                    "target": target,
                    "label": relation_type or "related_to",
                }
            )

        return {
            "graph": {
                "entities": filtered_entities,
                "relations": filtered_relations,
            },
            "visualization": {
                "nodes": nodes,
                "edges": edges,
            },
            "meta": {
                "entity_count": len(nodes),
                "relation_count": len(edges),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def _load_memory_graph_payload() -> dict[str, Any]:
        mcp_service = getattr(container, "mcp_service", None)
        if mcp_service is None:
            raise RuntimeError("MCP service not configured.")

        result = mcp_service.execute_tool("memory.read_graph", arguments={})
        if not isinstance(result, dict) or not result.get("ok"):
            raise RuntimeError("Memory graph tool returned an unsuccessful response.")
        return _build_graph_view(_as_memory_graph(result.get("data")))

    def _publish_memory_graph_update() -> None:
        if not memory_graph_events.has_subscribers():
            return
        try:
            payload = _load_memory_graph_payload()
        except Exception as exc:  # noqa: BLE001
            app.logger.warning("Skipping memory graph SSE publish after update: %s", exc)
            return
        memory_graph_events.publish("update", payload)

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
        raw_context = payload.get("conversation_context", "")
        if isinstance(raw_context, list):
            conversation_context = "\n".join(str(item) for item in raw_context)
        elif raw_context is None:
            conversation_context = ""
        else:
            conversation_context = str(raw_context)
        if not user_query:
            return jsonify({"error": "Query cannot be empty."}), 400

        try:
            response_payload = container.chat_service.execute(user_query, conversation_context=conversation_context)
            _publish_memory_graph_update()
            return jsonify(response_payload), 200
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

    @app.route("/mcp/health", methods=["GET"])
    def mcp_health():
        mcp_service = getattr(container, "mcp_service", None)
        if mcp_service is None:
            return jsonify({"error": "MCP service not configured."}), 501
        return jsonify(mcp_service.status()), 200

    @app.route("/mcp/tools", methods=["GET"])
    def mcp_tools():
        mcp_service = getattr(container, "mcp_service", None)
        if mcp_service is None:
            return jsonify({"error": "MCP service not configured."}), 501
        return jsonify({"tools": mcp_service.list_tools()}), 200

    @app.route("/mcp/tools/<path:tool_name>", methods=["POST"])
    def mcp_execute_tool(tool_name: str):
        mcp_service = getattr(container, "mcp_service", None)
        if mcp_service is None:
            return jsonify({"error": "MCP service not configured."}), 501
        payload = request.json or {}
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            return jsonify({"error": "Field 'arguments' must be an object."}), 400
        try:
            tool_result = mcp_service.execute_tool(tool_name, arguments=arguments)
            normalized_name = str(tool_name or "").strip().lower()
            if normalized_name.startswith("memory.") and isinstance(tool_result, dict) and tool_result.get("ok"):
                _publish_memory_graph_update()
            return jsonify(tool_result), 200
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("MCP tool execution failed.")
            return jsonify({"error": str(exc)}), 502

    @app.route("/memory/graph", methods=["GET"])
    def memory_graph():
        try:
            payload = _load_memory_graph_payload()
            return jsonify(payload), 200
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("Memory graph load failed.")
            return jsonify({"error": f"Failed to read memory graph: {exc}"}), 502

    @app.route("/memory/graph/events", methods=["GET"])
    def memory_graph_events_stream():
        mcp_service = getattr(container, "mcp_service", None)
        if mcp_service is None:
            return jsonify({"error": "MCP service not configured."}), 501

        try:
            snapshot = _load_memory_graph_payload()
        except Exception as exc:  # noqa: BLE001
            app.logger.warning("Memory graph snapshot failed for SSE bootstrap: %s", exc)
            snapshot = _build_graph_view({"entities": [], "relations": []})

        subscriber = memory_graph_events.subscribe()

        def event_stream():
            yield _MemoryGraphEventHub.encode_event("snapshot", snapshot)
            try:
                while True:
                    try:
                        chunk = subscriber.get(timeout=25)
                    except Empty:
                        yield ": keep-alive\n\n"
                        continue
                    yield chunk
            finally:
                memory_graph_events.unsubscribe(subscriber)

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/internal/filesystem/events", methods=["POST"])
    def internal_filesystem_events():
        provided_token = request.headers.get("x-internal-token", "")
        if INTERNAL_API_TOKEN and provided_token != INTERNAL_API_TOKEN:
            return jsonify({"error": "Unauthorized internal call."}), 401

        payload = request.json or {}
        operation = payload.get("operation", "")
        relative_path = payload.get("path", "")
        if not isinstance(operation, str) or not isinstance(relative_path, str):
            return jsonify({"error": "Fields 'operation' and 'path' must be strings."}), 400

        try:
            response_payload, status_code = container.context_service.sync_filesystem_event(
                operation=operation,
                relative_path=relative_path,
            )
            return jsonify(response_payload), status_code
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("Internal filesystem event sync failed.")
            return jsonify({"error": str(exc)}), 500

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
