import pathlib
import sys
import unittest

from flask import Flask

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from interfaces.http.routes import register_routes


class _FakeChatService:
    def execute(self, user_query: str, conversation_context: str = ""):
        return {"query": user_query, "response": "ok"}


class _FakeMcpService:
    def __init__(self):
        self.tool_calls = []

    def status(self):
        return {"enabled": True, "reachable": True}

    def list_tools(self):
        return [{"name": "filesystem.read_file"}]

    def execute_tool(self, tool_name, arguments=None):
        self.tool_calls.append((tool_name, arguments or {}))
        if tool_name == "memory.read_graph":
            return {
                "ok": True,
                "data": {
                    "entities": [
                        {
                            "name": "alice",
                            "entityType": "person",
                            "observations": ["likes graphs"],
                        }
                    ],
                    "relations": [
                        {
                            "from": "alice",
                            "to": "project_chatragfb",
                            "relationType": "works_on",
                        }
                    ],
                },
            }
        return {"ok": True, "tool": tool_name}


class _FakeContextService:
    def __init__(self):
        self.events = []

    def sync_filesystem_event(self, operation: str, relative_path: str):
        self.events.append((operation, relative_path))
        return {"ok": True, "operation": operation, "path": relative_path}, 200


class _FakeContainer:
    def __init__(self):
        self.chat_service = _FakeChatService()
        self.mcp_service = _FakeMcpService()
        self.context_service = _FakeContextService()


class HttpMcpRouteTests(unittest.TestCase):
    def _build_client(self):
        app = Flask(__name__, static_folder=str(APP_ROOT / "static"), static_url_path="")
        container = _FakeContainer()
        register_routes(app, container, log_messages=[])
        return app.test_client(), container

    def test_mcp_tools_returns_registered_tools(self):
        client, _container = self._build_client()

        response = client.get("/mcp/tools")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["tools"][0]["name"], "filesystem.read_file")

    def test_mcp_execute_tool_calls_service(self):
        client, container = self._build_client()

        response = client.post(
            "/mcp/tools/filesystem.read_file",
            json={"arguments": {"path": "doc.txt"}},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            container.mcp_service.tool_calls[0], ("filesystem.read_file", {"path": "doc.txt"})
        )

    def test_internal_filesystem_event_route_calls_context_sync(self):
        client, container = self._build_client()

        response = client.post(
            "/internal/filesystem/events",
            json={"operation": "upsert", "path": "doc.txt"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(container.context_service.events[0], ("upsert", "doc.txt"))

    def test_memory_graph_route_returns_visualization_payload(self):
        client, container = self._build_client()

        response = client.get("/memory/graph")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["meta"]["entity_count"], 2)
        self.assertEqual(payload["meta"]["relation_count"], 1)
        self.assertEqual(payload["visualization"]["nodes"][0]["id"], "alice")
        self.assertEqual(container.mcp_service.tool_calls[0], ("memory.read_graph", {}))

    def test_memory_graph_sse_stream_returns_snapshot_event(self):
        client, _container = self._build_client()

        response = client.get("/memory/graph/events", buffered=False)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/event-stream")
        first_chunk = next(response.response).decode("utf-8")
        self.assertIn("event: snapshot", first_chunk)
        self.assertIn('"visualization"', first_chunk)
        response.close()


if __name__ == "__main__":
    unittest.main()
