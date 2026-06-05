from __future__ import annotations

import logging
from typing import Any

from modules.chat_module import get_chat_status, process_chat_query, startup_check_chat_client
from modules.config import MCP_SERVER_ENABLED, MCP_SERVER_URL, MCP_TIMEOUT
from modules.ingestion import (
    build_file_path,
    create_schema,
    delete_document_by_id,
    delete_document_by_storage_path,
    file_size_bytes,
    ingest_file,
    is_supported_file,
    load_context_state,
    pdf_page_count,
    safe_remove_file,
    uploaded_file_size_bytes,
    uploaded_pdf_page_count,
)

from .mcp import McpHttpClient, McpServerSettings
from .runtime import get_default_connection_factory, get_default_retrieval_service

logger = logging.getLogger(__name__)


class DefaultIngestionAdapter:
    def create_schema(self) -> None:
        create_schema()

    def is_supported_file(self, filename: str) -> bool:
        return is_supported_file(filename)

    def build_file_path(self, filename: str, upload_folder: str) -> str:
        return build_file_path(filename, upload_folder=upload_folder)

    def uploaded_file_size_bytes(self, file_obj: Any) -> int:
        return uploaded_file_size_bytes(file_obj)

    def uploaded_pdf_page_count(self, file_obj: Any) -> int:
        return uploaded_pdf_page_count(file_obj)

    def file_size_bytes(self, file_path: str) -> int:
        return file_size_bytes(file_path)

    def pdf_page_count(self, file_path: str) -> int:
        return pdf_page_count(file_path)

    def ingest_file(
        self,
        file_path: str,
        original_filename: str | None = None,
        size_bytes: int = 0,
        page_count: int | None = None,
    ) -> dict[str, Any]:
        return ingest_file(
            file_path=file_path,
            original_filename=original_filename,
            size_bytes=size_bytes,
            page_count=page_count,
        )

    def safe_remove_file(self, path: str | None, upload_folder: str) -> None:
        safe_remove_file(path, upload_folder=upload_folder)

    def load_context_state(
        self,
        max_documents: int,
        max_file_size_bytes: int,
        max_total_size_bytes: int,
        max_pdf_pages: int,
    ) -> dict[str, Any]:
        return load_context_state(
            max_documents=max_documents,
            max_file_size_bytes=max_file_size_bytes,
            max_total_size_bytes=max_total_size_bytes,
            max_pdf_pages=max_pdf_pages,
        )

    def delete_document_by_id(self, document_id: int, upload_folder: str) -> dict[str, Any]:
        return delete_document_by_id(document_id, upload_folder=upload_folder)

    def delete_document_by_storage_path(
        self, storage_path: str, upload_folder: str
    ) -> dict[str, Any]:
        return delete_document_by_storage_path(
            storage_path, upload_folder=upload_folder, remove_file=False
        )


class DefaultRetrievalAdapter:
    def __init__(self, retrieval_service=None) -> None:
        self._service = retrieval_service or get_default_retrieval_service()

    def initialize_embeddings(self) -> None:
        self._service.initialize_embeddings()

    def refresh_vectorstore_cache(self) -> None:
        self._service.refresh_vectorstore_cache()

    def query_context(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        return self._service.query_context(query, k=k)

    def get_retrieval_status(self) -> dict[str, Any]:
        return self._service.get_retrieval_status()


class DefaultChatAdapter:
    def get_chat_status(self) -> dict[str, Any]:
        return get_chat_status()

    def startup_check_chat_client(self) -> None:
        startup_check_chat_client()

    def process_chat_query(self, user_query: str, conversation_context: str = "") -> dict[str, Any]:
        return process_chat_query(user_query, conversation_context=conversation_context)


class DefaultDatabaseHealthAdapter:
    def __init__(self, connection_factory=None) -> None:
        factory = connection_factory or get_default_connection_factory()
        self._connection_factory = factory.create_connection

    def check_connection(self) -> None:
        conn = self._connection_factory()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                cursor.fetchone()
        finally:
            conn.close()


class DefaultMcpAdapter:
    def __init__(self, client: McpHttpClient | None = None) -> None:
        self._client = client or McpHttpClient(
            McpServerSettings(
                enabled=MCP_SERVER_ENABLED,
                base_url=MCP_SERVER_URL,
                timeout_seconds=MCP_TIMEOUT,
            )
        )

    def get_mcp_status(self) -> dict[str, Any]:
        if not self._client.enabled:
            return {
                "enabled": False,
                "reachable": False,
                "base_url": self._client.base_url,
                "error": None,
            }

        try:
            payload = self._client.health()
            return {
                "enabled": True,
                "reachable": payload.get("status") == "ok",
                "base_url": self._client.base_url,
                "error": None,
                "server": payload,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "enabled": True,
                "reachable": False,
                "base_url": self._client.base_url,
                "error": str(exc),
            }

    def startup_check_mcp_client(self) -> None:
        status = self.get_mcp_status()
        if not status.get("enabled"):
            logger.info("MCP integration is disabled.")
            return
        if status.get("reachable"):
            logger.info("MCP server connectivity check succeeded.")
        else:
            logger.warning("MCP server connectivity check failed: %s", status.get("error"))

    def list_tools(self) -> list[dict[str, Any]]:
        if not self._client.enabled:
            return []
        return self._client.list_tools()

    def execute_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._client.execute_tool(tool_name, arguments=arguments)
