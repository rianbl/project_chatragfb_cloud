from __future__ import annotations

from typing import Any, Protocol


class UploadedFile(Protocol):
    filename: str | None
    stream: Any

    def save(self, dst: str) -> None:
        ...


class IngestionPort(Protocol):
    def create_schema(self) -> None:
        ...

    def is_supported_file(self, filename: str) -> bool:
        ...

    def build_file_path(self, filename: str, upload_folder: str) -> str:
        ...

    def uploaded_file_size_bytes(self, file_obj: UploadedFile) -> int:
        ...

    def uploaded_pdf_page_count(self, file_obj: UploadedFile) -> int:
        ...

    def ingest_file(
        self,
        file_path: str,
        original_filename: str | None = None,
        size_bytes: int = 0,
        page_count: int | None = None,
    ) -> dict[str, Any]:
        ...

    def safe_remove_file(self, path: str | None, upload_folder: str) -> None:
        ...

    def load_context_state(
        self,
        max_documents: int,
        max_file_size_bytes: int,
        max_total_size_bytes: int,
        max_pdf_pages: int,
    ) -> dict[str, Any]:
        ...

    def delete_document_by_id(self, document_id: int, upload_folder: str) -> dict[str, Any]:
        ...


class RetrievalPort(Protocol):
    def initialize_embeddings(self) -> None:
        ...

    def refresh_vectorstore_cache(self) -> None:
        ...

    def query_context(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        ...

    def get_retrieval_status(self) -> dict[str, Any]:
        ...


class ChatPort(Protocol):
    def get_chat_status(self) -> dict[str, Any]:
        ...

    def startup_check_chat_client(self) -> None:
        ...

    def process_chat_query(self, user_query: str, conversation_context: str = "") -> dict[str, Any]:
        ...


class DatabaseHealthPort(Protocol):
    def check_connection(self) -> None:
        ...


class McpPort(Protocol):
    def get_mcp_status(self) -> dict[str, Any]:
        ...

    def startup_check_mcp_client(self) -> None:
        ...

    def list_tools(self) -> list[dict[str, Any]]:
        ...

    def execute_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        ...
