from __future__ import annotations

import os
from logging import Logger
from typing import Any

from domain.models import AppLimits

from .ports import ChatPort, DatabaseHealthPort, IngestionPort, McpPort, RetrievalPort, UploadedFile


class ContextService:
    def __init__(
        self,
        ingestion: IngestionPort,
        retrieval: RetrievalPort,
        limits: AppLimits,
        logger: Logger,
    ) -> None:
        self._ingestion = ingestion
        self._retrieval = retrieval
        self._limits = limits
        self._logger = logger

    def _resolve_storage_path(self, relative_path: str) -> str:
        base_root = os.path.normcase(os.path.abspath(self._limits.upload_folder))
        raw_relative = str(relative_path or "").strip().replace("\\", "/")
        normalized_relative = os.path.normpath(raw_relative)
        if normalized_relative in {"", "."}:
            raise ValueError("Path cannot be empty.")
        if os.path.isabs(normalized_relative):
            raise ValueError("Path must be relative to context root.")

        absolute_path = os.path.normcase(
            os.path.abspath(os.path.join(base_root, normalized_relative))
        )
        if absolute_path != base_root and not absolute_path.startswith(f"{base_root}{os.sep}"):
            raise ValueError("Path is outside context root.")
        return absolute_path

    def current_state(self) -> dict[str, Any]:
        return self._ingestion.load_context_state(
            max_documents=self._limits.max_documents,
            max_file_size_bytes=self._limits.max_file_size_bytes,
            max_total_size_bytes=self._limits.max_total_size_bytes,
            max_pdf_pages=self._limits.max_pdf_pages,
        )

    def refresh_search_index(self, allow_empty: bool = False) -> dict[str, Any]:
        try:
            self._logger.info("Refreshing FAISS index from persisted chunks.")
            self._retrieval.refresh_vectorstore_cache()
            return {
                "ok": True,
                "status_code": 200,
                "message": "Vector index refreshed successfully.",
            }
        except ValueError as exc:
            if allow_empty:
                return {
                    "ok": True,
                    "status_code": 200,
                    "message": "Vector index refreshed with empty corpus.",
                }
            return {"ok": False, "status_code": 404, "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "status_code": 500, "message": str(exc)}

    def handle_upload(self, file: UploadedFile) -> tuple[dict[str, Any], int]:
        filename = file.filename
        if not filename:
            return {"error": "No file selected"}, 400

        self._logger.info("Incoming filename='%s'.", filename)
        if not self._ingestion.is_supported_file(filename):
            return {"error": "Unsupported file format. Supported: .csv, .pdf, .txt"}, 400

        file_size_bytes = self._ingestion.uploaded_file_size_bytes(file)
        self._logger.info("File size detected: %s bytes.", file_size_bytes)
        if file_size_bytes <= 0:
            return {"error": "Uploaded file is empty."}, 400

        if file_size_bytes > self._limits.max_file_size_bytes:
            return (
                {
                    "error": (
                        f"File exceeds limit of {self._limits.max_file_size_bytes} bytes. "
                        f"Current file size: {file_size_bytes} bytes."
                    )
                },
                400,
            )

        extension = os.path.splitext(filename)[1].lower()
        page_count = None
        if extension == ".pdf":
            try:
                page_count = self._ingestion.uploaded_pdf_page_count(file)
                self._logger.info("PDF page count detected: %s pages.", page_count)
            except Exception as exc:  # noqa: BLE001
                return {"error": f"Could not parse PDF: {exc}"}, 400

            if page_count > self._limits.max_pdf_pages:
                return (
                    {
                        "error": (
                            f"PDF exceeds page limit of {self._limits.max_pdf_pages}. "
                            f"Current PDF pages: {page_count}."
                        )
                    },
                    400,
                )

        file.stream.seek(0)

        state = self.current_state()
        current_docs = state["usage"]["document_count"]
        current_total_size = state["usage"]["total_size_bytes"]
        self._logger.info(
            "Current limits usage before upload: docs=%s/%s total_size=%s/%s.",
            current_docs,
            self._limits.max_documents,
            current_total_size,
            self._limits.max_total_size_bytes,
        )

        if current_docs >= self._limits.max_documents:
            return (
                {
                    "error": (
                        f"Document limit reached ({self._limits.max_documents}/{self._limits.max_documents})."
                    )
                },
                409,
            )

        if current_total_size >= self._limits.max_total_size_bytes:
            return {"error": "Total storage limit reached. Delete a file before uploading."}, 409

        projected_total = current_total_size + file_size_bytes
        if projected_total > self._limits.max_total_size_bytes:
            return (
                {
                    "error": (
                        f"Upload exceeds total size limit of {self._limits.max_total_size_bytes} bytes. "
                        f"Projected total: {projected_total} bytes."
                    )
                },
                409,
            )

        filepath = self._ingestion.build_file_path(
            filename, upload_folder=self._limits.upload_folder
        )
        filename = os.path.basename(filepath)
        self._logger.info("Saving upload to '%s'.", filepath)
        file.save(filepath)

        try:
            ingestion_result = self._ingestion.ingest_file(
                file_path=filepath,
                original_filename=filename,
                size_bytes=file_size_bytes,
                page_count=page_count,
            )
            self._logger.info(
                "Ingestion done: document_id=%s chunks=%s.",
                ingestion_result.get("document_id"),
                ingestion_result.get("chunks_inserted"),
            )
        except Exception as exc:  # noqa: BLE001
            self._ingestion.safe_remove_file(filepath, upload_folder=self._limits.upload_folder)
            self._logger.exception("Upload failed during ingestion.")
            return {"error": f"Error ingesting file: {exc}"}, 500

        refresh_result = self.refresh_search_index(allow_empty=False)
        self._logger.info(
            "Refresh result after upload: ok=%s msg='%s'.",
            refresh_result["ok"],
            refresh_result["message"],
        )
        updated_state = self.current_state()

        return (
            {
                "message": refresh_result["message"],
                "file_path": filepath,
                "ingestion": ingestion_result,
                "context": updated_state,
            },
            200 if refresh_result["ok"] else refresh_result["status_code"],
        )

    def delete_document(self, document_id: int) -> tuple[dict[str, Any], int]:
        self._logger.info("Delete requested for document_id=%s.", document_id)
        try:
            deleted = self._ingestion.delete_document_by_id(
                document_id,
                upload_folder=self._limits.upload_folder,
            )
        except ValueError:
            return {"error": "Document not found."}, 404
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Failed to delete document: {exc}"}, 500

        refresh_result = self.refresh_search_index(allow_empty=True)
        self._logger.info(
            "Delete completed for document_id=%s. Refresh result: ok=%s msg='%s'.",
            document_id,
            refresh_result["ok"],
            refresh_result["message"],
        )
        updated_state = self.current_state()

        return (
            {
                "message": (
                    f"Document '{deleted['filename']}' removed successfully. "
                    f"{refresh_result['message']}"
                ),
                "deleted_document_id": document_id,
                "context": updated_state,
            },
            200 if refresh_result["ok"] else refresh_result["status_code"],
        )

    def sync_filesystem_event(
        self, operation: str, relative_path: str
    ) -> tuple[dict[str, Any], int]:
        op = str(operation or "").strip().lower()
        if op not in {"upsert", "delete"}:
            return {"error": "Unsupported operation. Use 'upsert' or 'delete'."}, 400

        try:
            storage_path = self._resolve_storage_path(relative_path)
        except ValueError as exc:
            return {"error": str(exc)}, 400

        if op == "upsert":
            filename = os.path.basename(storage_path)
            if not self._ingestion.is_supported_file(filename):
                return (
                    {
                        "message": f"Filesystem event ignored for unsupported extension: {filename}",
                        "storage_path": storage_path,
                        "operation": op,
                    },
                    202,
                )

            if not os.path.exists(storage_path):
                return {"error": f"File not found for upsert: {relative_path}"}, 404

            page_count = None
            if os.path.splitext(filename)[1].lower() == ".pdf":
                page_count = self._ingestion.pdf_page_count(storage_path)

            self._ingestion.delete_document_by_storage_path(
                storage_path=storage_path,
                upload_folder=self._limits.upload_folder,
            )
            ingestion_result = self._ingestion.ingest_file(
                file_path=storage_path,
                original_filename=filename,
                size_bytes=self._ingestion.file_size_bytes(storage_path),
                page_count=page_count,
            )
            refresh_result = self.refresh_search_index(allow_empty=False)
            return (
                {
                    "message": refresh_result["message"],
                    "operation": op,
                    "storage_path": storage_path,
                    "ingestion": ingestion_result,
                },
                200 if refresh_result["ok"] else refresh_result["status_code"],
            )

        deleted = self._ingestion.delete_document_by_storage_path(
            storage_path=storage_path,
            upload_folder=self._limits.upload_folder,
        )
        refresh_result = self.refresh_search_index(allow_empty=True)
        return (
            {
                "message": refresh_result["message"],
                "operation": op,
                "storage_path": storage_path,
                "deleted_count": deleted.get("deleted_count", 0),
            },
            200 if refresh_result["ok"] else refresh_result["status_code"],
        )


class QueryService:
    def __init__(self, retrieval: RetrievalPort, default_top_k: int) -> None:
        self._retrieval = retrieval
        self._default_top_k = default_top_k

    def execute(self, query_text: str, requested_k: int | None = None) -> dict[str, Any]:
        if not query_text:
            raise ValueError("Query cannot be empty.")
        results = self._retrieval.query_context(query_text, k=requested_k or self._default_top_k)
        return {"query": query_text, "results": results}


class ChatService:
    def __init__(self, chat: ChatPort) -> None:
        self._chat = chat

    def execute(self, user_query: str, conversation_context: str = "") -> dict[str, Any]:
        if not user_query:
            raise ValueError("Query cannot be empty.")
        return self._chat.process_chat_query(user_query, conversation_context=conversation_context)


class McpService:
    def __init__(self, mcp: McpPort) -> None:
        self._mcp = mcp

    def status(self) -> dict[str, Any]:
        return self._mcp.get_mcp_status()

    def list_tools(self) -> list[dict[str, Any]]:
        return self._mcp.list_tools()

    def execute_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._mcp.execute_tool(tool_name, arguments=arguments)


class HealthService:
    def __init__(
        self,
        db_health: DatabaseHealthPort,
        retrieval: RetrievalPort,
        chat: ChatPort,
        mcp: McpPort | None = None,
    ) -> None:
        self._db_health = db_health
        self._retrieval = retrieval
        self._chat = chat
        self._mcp = mcp

    def execute(self) -> tuple[dict[str, Any], int]:
        try:
            self._db_health.check_connection()
            db_ok = True
            db_error = None
        except Exception as exc:  # noqa: BLE001
            db_ok = False
            db_error = str(exc)

        retrieval_status = self._retrieval.get_retrieval_status()
        chat_status = self._chat.get_chat_status()
        if self._mcp is None:
            mcp_status = {
                "enabled": False,
                "reachable": False,
                "error": "MCP adapter not configured.",
            }
        else:
            try:
                mcp_status = self._mcp.get_mcp_status()
            except Exception as exc:  # noqa: BLE001
                mcp_status = {
                    "enabled": True,
                    "reachable": False,
                    "error": str(exc),
                }
        chat_ready = chat_status["token_present"] and (
            chat_status["dns"]["api_inference"]["ok"] or chat_status["dns"]["router"]["ok"]
        )
        ready = db_ok and retrieval_status["embeddings_initialized"] and chat_ready
        status_code = 200 if ready else 503

        return (
            {
                "status": "ok" if ready else "degraded",
                "database": {"ok": db_ok, "error": db_error},
                "retrieval": retrieval_status,
                "chat": chat_status,
                "mcp": mcp_status,
            },
            status_code,
        )


class StartupService:
    def __init__(
        self,
        db_health: DatabaseHealthPort,
        ingestion: IngestionPort,
        retrieval: RetrievalPort,
        chat: ChatPort,
        context: ContextService,
        embedding_model_id: str,
        logger: Logger,
        mcp: McpPort | None = None,
    ) -> None:
        self._db_health = db_health
        self._ingestion = ingestion
        self._retrieval = retrieval
        self._chat = chat
        self._context = context
        self._embedding_model_id = embedding_model_id
        self._logger = logger
        self._mcp = mcp

    def run(self) -> None:
        self._logger.info("Starting app readiness checks.")
        self._logger.info("Checking PostgreSQL connectivity.")
        self._db_health.check_connection()
        self._logger.info("PostgreSQL connection is healthy.")

        self._logger.info("Ensuring documents/chunks schema exists.")
        self._ingestion.create_schema()
        self._logger.info("Schema check completed.")

        self._logger.info("Initializing embedding model '%s'.", self._embedding_model_id)
        self._retrieval.initialize_embeddings()
        self._logger.info("Embedding model is ready.")
        self._logger.info("Initializing chat inference client and DNS checks.")
        self._chat.startup_check_chat_client()
        self._logger.info("Chat inference startup checks completed.")
        if self._mcp is not None:
            self._logger.info("Checking MCP server connectivity.")
            self._mcp.startup_check_mcp_client()
            self._logger.info("MCP startup checks completed.")

        state = self._context.current_state()
        self._logger.info(
            "Current corpus state: documents=%s total_size_bytes=%s blocked=%s",
            state["usage"]["document_count"],
            state["usage"]["total_size_bytes"],
            state["is_upload_blocked"],
        )
        self._logger.info("Readiness checks completed successfully.")
