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
    def __init__(self):
        self.calls = []

    def execute(self, user_query: str, conversation_context: str = ""):
        self.calls.append((user_query, conversation_context))
        return {"query": user_query, "response": "ok"}


class _FakeContainer:
    def __init__(self):
        self.chat_service = _FakeChatService()


class HttpChatRouteTests(unittest.TestCase):
    def _build_client(self):
        app = Flask(__name__, static_folder=str(APP_ROOT / "static"), static_url_path="")
        container = _FakeContainer()
        register_routes(app, container, log_messages=[])
        return app.test_client(), container

    def test_chat_accepts_list_context_and_joins_lines(self):
        client, container = self._build_client()

        response = client.post(
            "/chat",
            json={
                "query": "resuma",
                "conversation_context": ["linha 1", "linha 2"],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(container.chat_service.calls[0], ("resuma", "linha 1\nlinha 2"))

    def test_chat_requires_query(self):
        client, _ = self._build_client()

        response = client.post("/chat", json={"conversation_context": "x"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Query cannot be empty", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
