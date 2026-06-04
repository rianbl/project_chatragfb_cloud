from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class McpServerSettings:
    enabled: bool
    base_url: str
    timeout_seconds: float


class McpHttpClient:
    def __init__(self, settings: McpServerSettings, session: requests.Session | None = None) -> None:
        self._settings = settings
        self._session = session or requests.Session()

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/health")

    def list_tools(self) -> list[dict[str, Any]]:
        payload = self._request_json("GET", "/tools")
        tools = payload.get("tools", [])
        if not isinstance(tools, list):
            raise RuntimeError("Unexpected MCP server response format for /tools.")
        return [item for item in tools if isinstance(item, dict)]

    def execute_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        safe_name = (tool_name or "").strip()
        if not safe_name:
            raise ValueError("Tool name cannot be empty.")
        payload = {
            "arguments": arguments or {},
        }
        result = self._request_json("POST", f"/tools/{safe_name}", json=payload)
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected MCP server response for tool execution.")
        return result

    def _request_json(self, method: str, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._settings.enabled:
            raise RuntimeError("MCP client is disabled by configuration.")
        url = f"{self._settings.base_url.rstrip('/')}{path}"
        try:
            response = self._session.request(
                method=method,
                url=url,
                json=json,
                timeout=self._settings.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"MCP request failed: {exc}") from exc

        if response.status_code >= 400:
            raise RuntimeError(f"MCP server returned {response.status_code}: {response.text[:300]}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("MCP server returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("MCP server response is not an object.")
        return payload
