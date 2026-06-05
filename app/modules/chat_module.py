from __future__ import annotations

import inspect
import logging
import os
import socket
import threading
import time

import requests
from huggingface_hub import InferenceClient

from infrastructure.mcp import McpHttpClient, McpServerSettings
from infrastructure.prompt_store import FilePromptStore
from infrastructure.rag_workflow import (
    FilesystemMcpTool,
    FilesystemStep,
    RagWorkflowOrchestrator,
    ResponderStep,
    RetrieverStep,
    RetrievingTool,
    RouterStep,
)
from infrastructure.runtime import get_default_retrieval_service

from .config import (
    HF_API_TOKEN,
    HF_MODEL_ID,
    HF_PROVIDER,
    HF_TIMEOUT,
    MAX_DOCUMENTS,
    MAX_FILE_SIZE_BYTES,
    MAX_PDF_PAGES,
    MAX_TOTAL_SIZE_BYTES,
    MCP_SERVER_ENABLED,
    MCP_SERVER_URL,
    MCP_TIMEOUT,
    PROMPT_STORE_PATH,
    SUPPORTED_EXTENSIONS,
    UPLOAD_FOLDER,
)

HF_CLIENTS: dict[str, InferenceClient] = {}
RAG_ORCHESTRATOR: RagWorkflowOrchestrator | None = None
logger = logging.getLogger(__name__)
_FILESYSTEM_CONTEXT_LOCK = threading.Lock()


class MCPFilesystemToolClient:
    def __init__(self) -> None:
        self._client = McpHttpClient(
            McpServerSettings(
                enabled=MCP_SERVER_ENABLED,
                base_url=MCP_SERVER_URL,
                timeout_seconds=MCP_TIMEOUT,
            )
        )

    def list_directory(self, path: str = ".") -> list[dict]:
        safe_path = self._sanitize_relative_path(path, default_path=".")
        result = self._client.execute_tool("filesystem.list_directory", arguments={"path": safe_path})
        data = self._extract_data_object(result)

        structured = data.get("structuredContent")
        if isinstance(structured, dict):
            rows = self._extract_entries(structured)
            if rows:
                return rows

        rows = self._extract_entries(data)
        if rows:
            return rows

        content_rows = self._extract_entries_from_content(data.get("content"))
        if content_rows:
            return content_rows
        return []

    def read_file(self, path: str) -> str:
        safe_path = self._sanitize_relative_path(path, default_path=".")
        result = self._client.execute_tool("filesystem.read_file", arguments={"path": safe_path})
        data = self._extract_data_object(result)
        text = self._extract_text(data)
        return text.strip()

    def write_file(self, path: str, content: str) -> str:
        safe_relative_path = self._sanitize_relative_path(path, default_path="generated.txt")
        local_storage_path = self._resolve_local_storage_path(safe_relative_path)
        content_bytes = self._content_size_bytes(content)

        with _FILESYSTEM_CONTEXT_LOCK:
            write_plan = self._plan_write_quota(local_storage_path, content_bytes)
            result = self._client.execute_tool(
                "filesystem.write_file",
                arguments={"path": safe_relative_path, "content": content},
            )
            self._sync_written_file_to_context(
                local_storage_path=local_storage_path,
                size_bytes=content_bytes,
                existing_document_id=write_plan.get("existing_document_id"),
            )

        data = self._extract_data_object(result)
        text = self._extract_text(data)
        return text.strip() or f"File written: {path}"

    def delete_file(self, path: str) -> str:
        safe_relative_path = self._sanitize_relative_path(path, default_path="")
        if not safe_relative_path or safe_relative_path in {".", "/"}:
            raise ValueError("A valid file path is required for delete_file.")
        local_storage_path = self._resolve_local_storage_path(safe_relative_path)

        with _FILESYSTEM_CONTEXT_LOCK:
            result = self._client.execute_tool("filesystem.delete_file", arguments={"path": safe_relative_path})
            existing_document = self._find_document_by_storage_path(local_storage_path)
            if existing_document and existing_document.get("id") is not None:
                self._delete_document_record(int(existing_document["id"]))
            retrieval_service = get_default_retrieval_service()
            try:
                retrieval_service.refresh_vectorstore_cache()
            except ValueError as exc:
                if "No chunks found in database." not in str(exc):
                    raise

        data = self._extract_data_object(result)
        text = self._extract_text(data)
        return text.strip() or f"File deleted: {path}"

    @staticmethod
    def _sanitize_relative_path(path: str, *, default_path: str) -> str:
        candidate = str(path or "").strip().replace("\\", "/")
        candidate = os.path.normpath(candidate).replace("\\", "/")
        if not candidate or candidate in {".", "./"}:
            return default_path
        if candidate.startswith("/") or candidate.startswith("../") or "/../" in candidate or candidate == "..":
            raise ValueError("Invalid path. Only relative paths inside context root are allowed.")
        return candidate

    @staticmethod
    def _content_size_bytes(content: str) -> int:
        size = len((content or "").encode("utf-8"))
        if size <= 0:
            raise ValueError("File content cannot be empty.")
        return size

    @staticmethod
    def _resolve_local_storage_path(relative_path: str) -> str:
        upload_root = os.path.abspath(UPLOAD_FOLDER)
        target = os.path.abspath(os.path.join(upload_root, relative_path))
        if target != upload_root and not target.startswith(f"{upload_root}{os.sep}"):
            raise ValueError("Resolved path is outside upload folder.")
        return target

    def _plan_write_quota(self, local_storage_path: str, content_bytes: int) -> dict:
        from modules.ingestion import load_context_state

        extension = os.path.splitext(local_storage_path)[1].lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported file format for context sync: {extension}. Supported: {supported}")
        if content_bytes > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File exceeds limit of {MAX_FILE_SIZE_BYTES} bytes. Current content size: {content_bytes} bytes."
            )

        state = load_context_state(
            max_documents=MAX_DOCUMENTS,
            max_file_size_bytes=MAX_FILE_SIZE_BYTES,
            max_total_size_bytes=MAX_TOTAL_SIZE_BYTES,
            max_pdf_pages=MAX_PDF_PAGES,
        )
        current_docs = int(state["usage"]["document_count"])
        current_total_size = int(state["usage"]["total_size_bytes"])
        existing_document = self._find_document_by_storage_path(local_storage_path)
        existing_document_id = existing_document.get("id") if existing_document else None
        existing_size = int(existing_document.get("size_bytes", 0)) if existing_document else 0

        if existing_document is None and current_docs >= MAX_DOCUMENTS:
            raise ValueError(
                f"Document limit reached ({MAX_DOCUMENTS}/{MAX_DOCUMENTS}). Remove one document before writing."
            )

        projected_total = current_total_size - existing_size + content_bytes
        if projected_total > MAX_TOTAL_SIZE_BYTES:
            raise ValueError(
                f"Write exceeds total size limit of {MAX_TOTAL_SIZE_BYTES} bytes. Projected total: {projected_total} bytes."
            )

        return {
            "existing_document_id": existing_document_id,
            "projected_total_size_bytes": projected_total,
        }

    @staticmethod
    def _find_document_by_storage_path(local_storage_path: str) -> dict | None:
        from infrastructure.runtime import get_db_connection
        from modules.ingestion import create_schema

        normalized_target = os.path.normcase(os.path.abspath(local_storage_path))
        create_schema()
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, size_bytes, storage_path
                    FROM documents
                    WHERE storage_path IS NOT NULL;
                    """
                )
                rows = cursor.fetchall()
                for row in rows:
                    row_id, row_size, row_path = row
                    if not row_path:
                        continue
                    normalized_row = os.path.normcase(os.path.abspath(str(row_path)))
                    if normalized_row == normalized_target:
                        return {"id": row_id, "size_bytes": row_size, "storage_path": row_path}
        finally:
            conn.close()
        return None

    @staticmethod
    def _delete_document_record(document_id: int) -> None:
        from infrastructure.runtime import get_db_connection

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM documents WHERE id = %s;", (document_id,))
            conn.commit()
        finally:
            conn.close()

    def _sync_written_file_to_context(
        self,
        *,
        local_storage_path: str,
        size_bytes: int,
        existing_document_id: int | None,
    ) -> None:
        from modules.ingestion import ingest_file

        os.makedirs(os.path.dirname(local_storage_path), exist_ok=True)
        if existing_document_id is not None:
            self._delete_document_record(int(existing_document_id))

        ingest_file(
            file_path=local_storage_path,
            original_filename=os.path.basename(local_storage_path),
            size_bytes=size_bytes,
            page_count=None,
        )
        retrieval_service = get_default_retrieval_service()
        try:
            retrieval_service.refresh_vectorstore_cache()
        except ValueError as exc:
            if "No chunks found in database." not in str(exc):
                raise

    @staticmethod
    def _extract_data_object(result: dict) -> dict:
        data = result.get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected MCP response format.")
        return data

    @staticmethod
    def _extract_entries(container: dict) -> list[dict]:
        entries = container.get("entries")
        if not isinstance(entries, list):
            return []
        parsed: list[dict] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            parsed.append({"name": name, "is_directory": bool(item.get("is_directory", False))})
        return parsed

    @staticmethod
    def _extract_entries_from_content(raw_content: object) -> list[dict]:
        if not isinstance(raw_content, list):
            return []
        lines: list[str] = []
        for block in raw_content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    lines.extend(text.splitlines())
        parsed: list[dict] = []
        for line in lines:
            entry = line.strip().lstrip("-").strip()
            if not entry:
                continue
            is_directory = entry.endswith("/")
            parsed.append(
                {
                    "name": entry[:-1] if is_directory else entry,
                    "is_directory": is_directory,
                }
            )
        return parsed

    @staticmethod
    def _extract_text(data: dict) -> str:
        structured = data.get("structuredContent")
        if isinstance(structured, dict):
            for key in ("content", "text", "result", "message"):
                value = structured.get(key)
                if isinstance(value, str) and value.strip():
                    return value

        for key in ("content", "text", "result", "message"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value

        blocks = data.get("content")
        if isinstance(blocks, list):
            parts = []
            for item in blocks:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            if parts:
                return "\n".join(parts)

        return ""


def _build_hf_client(provider: str):
    if not HF_API_TOKEN:
        raise ValueError("Missing required environment variable: HF_API_TOKEN")

    init_params = inspect.signature(InferenceClient.__init__).parameters
    kwargs = {"timeout": HF_TIMEOUT}

    if "api_key" in init_params:
        kwargs["api_key"] = HF_API_TOKEN
    elif "token" in init_params:
        kwargs["token"] = HF_API_TOKEN
    else:
        raise RuntimeError(
            "Unsupported huggingface_hub.InferenceClient signature: expected 'api_key' or 'token'."
        )

    provider_value = (provider or "auto").strip()
    if provider_value.lower() != "auto":
        if "provider" in init_params:
            kwargs["provider"] = provider_value
        else:
            logger.warning(
                "HF_PROVIDER='%s' ignored: current InferenceClient version does not support provider argument.",
                provider_value,
            )

    client = InferenceClient(**kwargs)
    logger.info(
        "Hugging Face InferenceClient initialized (provider=%s, args=%s).",
        provider_value,
        sorted(kwargs.keys()),
    )
    return client


def _get_hf_client(provider: str | None = None, force_recreate: bool = False):
    resolved_provider = (provider or HF_PROVIDER or "auto").strip()
    cache_key = resolved_provider.lower()
    if force_recreate or cache_key not in HF_CLIENTS:
        HF_CLIENTS[cache_key] = _build_hf_client(resolved_provider)
    return HF_CLIENTS[cache_key]


class HFTextGenerator:
    def __init__(self, retries: int = 2, delay_seconds: float = 3.0) -> None:
        self._retries = retries
        self._delay_seconds = delay_seconds

    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        last_exception = None
        for attempt in range(1, self._retries + 1):
            try:
                return self._generate_via_router(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            except Exception as router_exc:  # noqa: BLE001
                last_exception = router_exc
                logger.error(
                    "Router text generation attempt %s/%s failed (model=%s): %s",
                    attempt,
                    self._retries,
                    HF_MODEL_ID,
                    router_exc,
                )
            try:
                return self._generate_via_client(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            except Exception as client_exc:  # noqa: BLE001
                last_exception = client_exc
                logger.error(
                    "InferenceClient text generation attempt %s/%s failed (provider=%s, model=%s): %s",
                    attempt,
                    self._retries,
                    HF_PROVIDER,
                    HF_MODEL_ID,
                    client_exc,
                )

            if attempt < self._retries:
                time.sleep(self._delay_seconds)

        raise RuntimeError(
            f"Failed to generate text after {self._retries} attempts "
            f"(provider={HF_PROVIDER}, model={HF_MODEL_ID}): {last_exception}"
        ) from last_exception

    @staticmethod
    def _generate_via_router(prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": HF_MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT)
        if response.status_code >= 400:
            raise RuntimeError(f"Router generation failed ({response.status_code}): {response.text[:300]}")

        body = response.json()
        if isinstance(body, dict):
            choices = body.get("choices") or []
            if choices:
                first = choices[0] or {}
                message = first.get("message") or {}
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        raise RuntimeError(f"Unexpected router response format: {type(body).__name__}")

    @staticmethod
    def _generate_via_client(prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        client = _get_hf_client()
        try:
            return client.text_generation(
                prompt=prompt,
                model=HF_MODEL_ID,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except ValueError as exc:
            if "Supported task: conversational" not in str(exc):
                raise
            completion = client.chat_completion(
                model=HF_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return completion.choices[0].message.content


def _build_rag_orchestrator() -> RagWorkflowOrchestrator:
    prompt_store = FilePromptStore(PROMPT_STORE_PATH)
    text_llm = HFTextGenerator(retries=2, delay_seconds=3)
    retrieval_service = get_default_retrieval_service()
    retrieving_tool = RetrievingTool(retrieval_fn=retrieval_service.query_context)
    filesystem_client = MCPFilesystemToolClient()
    filesystem_tool = FilesystemMcpTool(
        list_directory_fn=filesystem_client.list_directory,
        read_file_fn=filesystem_client.read_file,
        write_file_fn=filesystem_client.write_file,
        delete_file_fn=filesystem_client.delete_file,
    )

    return RagWorkflowOrchestrator(
        router=RouterStep(prompt_store=prompt_store, llm=text_llm),
        retriever=RetrieverStep(prompt_store=prompt_store, llm=text_llm, tool=retrieving_tool),
        filesystem=FilesystemStep(tool=filesystem_tool),
        responder=ResponderStep(prompt_store=prompt_store, llm=text_llm),
        prefer_langgraph=True,
        logger_instance=logger,
    )


def _get_rag_orchestrator() -> RagWorkflowOrchestrator:
    global RAG_ORCHESTRATOR
    if RAG_ORCHESTRATOR is None:
        RAG_ORCHESTRATOR = _build_rag_orchestrator()
    return RAG_ORCHESTRATOR


def _resolve_host(hostname: str):
    try:
        resolved = socket.getaddrinfo(hostname, 443)
        return {"ok": True, "addresses": sorted({item[4][0] for item in resolved})}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "addresses": []}


def get_chat_status():
    status = {
        "token_present": bool(HF_API_TOKEN),
        "provider": HF_PROVIDER,
        "model": HF_MODEL_ID,
        "timeout_seconds": HF_TIMEOUT,
        "prompt_store_path": PROMPT_STORE_PATH,
        "dns": {
            "api_inference": _resolve_host("api-inference.huggingface.co"),
            "router": _resolve_host("router.huggingface.co"),
        },
    }
    return status


def startup_check_chat_client():
    if not HF_API_TOKEN:
        logger.error("HF_API_TOKEN is missing. Chat requests will fail.")
        return

    try:
        _get_hf_client()
        _get_rag_orchestrator()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to initialize chat components at startup.")
        return

    status = get_chat_status()
    logger.info(
        "Chat startup status: provider=%s model=%s token_present=%s prompt_store=%s api_dns_ok=%s router_dns_ok=%s",
        status["provider"],
        status["model"],
        status["token_present"],
        status["prompt_store_path"],
        status["dns"]["api_inference"]["ok"],
        status["dns"]["router"]["ok"],
    )


def identify_intent(query: str):
    use_retrieval = _get_rag_orchestrator().route_only(user_input=query, conversation_context="")
    return "requires_retrieval" if use_retrieval else "direct_response"


def query_hf_api(payload, retries=2, delay=3):
    del retries
    del delay
    prompt = payload.get("inputs", "")
    parameters = payload.get("parameters", {})
    max_new_tokens = int(parameters.get("max_length", 200))
    temperature = float(parameters.get("temperature", 0.2))
    llm = HFTextGenerator(retries=2, delay_seconds=3)
    generated_text = llm.generate(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
    return [{"generated_text": f"{prompt}{generated_text}"}]


def process_chat_query(user_query: str, conversation_context: str = ""):
    workflow_state = _get_rag_orchestrator().run(
        user_input=user_query,
        conversation_context=conversation_context or "",
    )
    return {"query": user_query, "response": workflow_state.get("response", "")}
