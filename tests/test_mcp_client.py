import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from infrastructure.mcp.http_client import McpHttpClient, McpServerSettings


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def request(self, method, url, json=None, timeout=None):
        self.calls.append((method, url, json, timeout))
        if not self._responses:
            raise RuntimeError("No fake response configured.")
        return self._responses.pop(0)


class McpHttpClientTests(unittest.TestCase):
    def test_list_tools_returns_payload_list(self):
        settings = McpServerSettings(enabled=True, base_url="http://mcp-server:8090", timeout_seconds=3.0)
        session = _FakeSession(
            [
                _FakeResponse(
                    200,
                    {
                        "tools": [
                            {"name": "filesystem.read_file"},
                            {"name": "filesystem.list_directory"},
                        ]
                    },
                )
            ]
        )
        client = McpHttpClient(settings, session=session)

        tools = client.list_tools()

        self.assertEqual(len(tools), 2)
        self.assertEqual(session.calls[0][0], "GET")
        self.assertIn("/tools", session.calls[0][1])

    def test_execute_tool_requires_name(self):
        settings = McpServerSettings(enabled=True, base_url="http://mcp-server:8090", timeout_seconds=3.0)
        session = _FakeSession([])
        client = McpHttpClient(settings, session=session)

        with self.assertRaises(ValueError):
            client.execute_tool("  ", arguments={})

    def test_disabled_client_raises(self):
        settings = McpServerSettings(enabled=False, base_url="http://mcp-server:8090", timeout_seconds=3.0)
        session = _FakeSession([])
        client = McpHttpClient(settings, session=session)

        with self.assertRaises(RuntimeError):
            client.health()


if __name__ == "__main__":
    unittest.main()
