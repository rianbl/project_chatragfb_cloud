from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Protocol, TypedDict

try:
    import langchain

    # Enable LangChain debug mode to provide additional visibility into graph execution if supported
    langchain.debug = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

_SUPPORTED_ROUTER_TOOLS = {"retrieval", "filesystem"}


class RetrievedDocument(TypedDict):
    content: str
    source: str


class FilesystemEntry(TypedDict):
    name: str
    is_directory: bool


class RagWorkflowState(TypedDict, total=False):
    user_input: str
    conversation_context: str
    tools: list[str]
    tool_inputs: dict[str, dict[str, str]]
    use_retrieval: bool
    use_filesystem: bool
    search_query: str
    retrieved_documents: list[RetrievedDocument]
    filesystem_path: str
    filesystem_operation: str
    filesystem_entries: list[FilesystemEntry]
    filesystem_content: str
    filesystem_result: str
    tool_errors: list[str]
    response: str


class PromptStorePort(Protocol):
    def render(self, prompt_name: str, **variables) -> str:
        ...


class TextGeneratorPort(Protocol):
    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        ...


class RetrievingToolPort(Protocol):
    def retrieve(self, search_query: str) -> list[RetrievedDocument]:
        ...


class FilesystemToolPort(Protocol):
    def list_directory(self, path: str = ".") -> list[FilesystemEntry]:
        ...

    def read_file(self, path: str) -> str:
        ...

    def write_file(self, path: str, content: str) -> str:
        ...

    def delete_file(self, path: str) -> str:
        ...


class RouterDecisionPolicyPort(Protocol):
    def decide_tools(
        self,
        *,
        user_input: str,
        conversation_context: str,
        model_tools: list[str],
        model_tool_inputs: dict[str, dict[str, str]],
    ) -> list[str]:
        ...


class ConservativeRouterDecisionPolicy:
    """Biases routing toward retrieval and enables filesystem when file-system intent is explicit."""

    _OBVIOUS_DIRECT_RESPONSE_PATTERNS = (
        r"^\s*(oi|ola|olá|hello|hi|hey)\b",
        r"^\s*(obrigad[oa]|thanks|thank you)\b",
        r"^\s*(bom dia|boa tarde|boa noite)\b",
        r"\bqual\s+a\s+capital\b",
        r"\bqual\s+e\s+a\s+capital\b",
        r"\bqual\s+é\s+a\s+capital\b",
        r"\bwhat\s+is\s+the\s+capital\b",
        r"^\s*\d+(\s*[\+\-\*/]\s*\d+)+\s*\??\s*$",
        r"^\s*quanto\s+e\s+\d+(\s*[\+\-\*/]\s*\d+)+\s*\??\s*$",
    )
    _FILESYSTEM_HINT_PATTERNS = (
        r"\barquivos?\b",
        r"\bpasta\b",
        r"\bdiret[oó]rio\b",
        r"\bdirectory\b",
        r"\bfolder\b",
        r"\bfile\s*system\b",
        r"\bfilesystem\b",
        r"\blist(ar|e)?\s+os?\s+arquivos?\b",
        r"\bdelete\b",
        r"\bdeletar\b",
        r"\bapagar\b",
        r"\bremover\b",
        r"\bexcluir\b",
    )
    _DOCUMENT_HINT_PATTERNS = (
        r"\bmanual\b",
        r"\bpdf\b",
        r"\bdocumento\b",
        r"\bdocument\b",
        r"\brelat[oó]rio\b",
        r"\bcontrato\b",
        r"\bpol[ií]tica\b",
    )
    _AFFIRMATIVE_INPUTS = {"yes", "sim", "ok", "okay", "pode", "prossiga", "continue", "confirmo", "confirma"}

    def decide_tools(
        self,
        *,
        user_input: str,
        conversation_context: str,
        model_tools: list[str],
        model_tool_inputs: dict[str, dict[str, str]],
    ) -> list[str]:
        tools = self._normalize_tools(model_tools)
        normalized_input = " ".join((user_input or "").lower().split())
        normalized_context = " ".join((conversation_context or "").lower().split())
        filesystem_inputs = model_tool_inputs.get("filesystem", {}) if isinstance(model_tool_inputs, dict) else {}
        filesystem_operation = str(filesystem_inputs.get("operation", "")).strip().lower()
        is_affirmative_followup = (
            normalized_input in self._AFFIRMATIVE_INPUTS
            and self._looks_like_filesystem_request(normalized_context)
        )
        pure_filesystem_action = filesystem_operation in {
            "write_file",
            "read_file",
            "delete_file",
            "list_directory",
            "write",
            "read",
            "delete",
            "list",
        } or self._looks_like_filesystem_request(normalized_input) or is_affirmative_followup

        if pure_filesystem_action and "filesystem" not in tools:
            tools.append("filesystem")

        if (
            "filesystem" in tools
            and "retrieval" in tools
            and pure_filesystem_action
            and not self._looks_like_document_request(normalized_input)
            and not (conversation_context or "").strip()
        ):
            tools = [tool_name for tool_name in tools if tool_name != "retrieval"]

        if "retrieval" not in tools:
            if (
                "filesystem" in tools
                and not self._looks_like_document_request(normalized_input)
                and not (conversation_context or "").strip()
            ):
                return tools
            if (conversation_context or "").strip():
                logger.info("RouterDecisionPolicy override: retrieval forced by available conversation context.")
                tools.append("retrieval")
            elif not normalized_input:
                tools.append("retrieval")
            elif not self._is_obvious_direct_question(normalized_input):
                logger.info("RouterDecisionPolicy override: conservative retrieval applied.")
                tools.append("retrieval")

        return tools

    @staticmethod
    def _normalize_tools(model_tools: list[str]) -> list[str]:
        ordered: list[str] = []
        for item in model_tools:
            normalized = str(item).strip().lower()
            if normalized in _SUPPORTED_ROUTER_TOOLS and normalized not in ordered:
                ordered.append(normalized)
        return ordered

    def _is_obvious_direct_question(self, normalized_input: str) -> bool:
        for pattern in self._OBVIOUS_DIRECT_RESPONSE_PATTERNS:
            if re.search(pattern, normalized_input):
                return True
        return False

    def _looks_like_filesystem_request(self, normalized_input: str) -> bool:
        for pattern in self._FILESYSTEM_HINT_PATTERNS:
            if re.search(pattern, normalized_input):
                return True
        return False

    def _looks_like_document_request(self, normalized_input: str) -> bool:
        for pattern in self._DOCUMENT_HINT_PATTERNS:
            if re.search(pattern, normalized_input):
                return True
        return False


def _extract_router_json_object(raw_output: str) -> dict | None:
    text = (raw_output or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def _parse_router_output(raw_output: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    payload = _extract_router_json_object(raw_output)
    if payload is None:
        lowered = (raw_output or "").lower()
        if '"use_retrieval": false' in lowered:
            return ([], {})
        if '"use_retrieval": true' in lowered:
            return (["retrieval"], {})
        return ([], {})

    if isinstance(payload.get("tools"), list):
        tools = [str(item).strip().lower() for item in payload["tools"] if isinstance(item, str)]
        tool_inputs = _sanitize_tool_inputs(payload.get("tool_inputs"))
        return (tools, tool_inputs)

    if isinstance(payload.get("use_retrieval"), bool):
        if payload["use_retrieval"]:
            return (["retrieval"], {})
        return ([], {})

    return ([], {})


def _sanitize_tool_inputs(raw_tool_inputs: object) -> dict[str, dict[str, str]]:
    if not isinstance(raw_tool_inputs, dict):
        return {}
    sanitized: dict[str, dict[str, str]] = {}
    for tool_name, tool_value in raw_tool_inputs.items():
        normalized_name = str(tool_name).strip().lower()
        if normalized_name not in _SUPPORTED_ROUTER_TOOLS:
            continue
        if not isinstance(tool_value, dict):
            continue
        tool_payload: dict[str, str] = {}
        for key, value in tool_value.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                tool_payload[str(key)] = str(value)
        sanitized[normalized_name] = tool_payload
    return sanitized


@dataclass
class RouterStep:
    prompt_store: PromptStorePort
    llm: TextGeneratorPort
    decision_policy: RouterDecisionPolicyPort = field(default_factory=ConservativeRouterDecisionPolicy)

    def execute(self, state: RagWorkflowState) -> RagWorkflowState:
        logger.info("RouterStep: selecting tools for current user input.")
        prompt = self.prompt_store.render(
            "router",
            conversation_context=state.get("conversation_context", ""),
            user_input=state["user_input"],
        )
        raw_output = self.llm.generate(prompt, temperature=0.0, max_new_tokens=128)
        model_tools, tool_inputs = _parse_router_output(raw_output)
        selected_tools = self.decision_policy.decide_tools(
            user_input=state["user_input"],
            conversation_context=state.get("conversation_context", ""),
            model_tools=model_tools,
            model_tool_inputs=tool_inputs,
        )
        filtered_tool_inputs = {
            tool_name: tool_inputs.get(tool_name, {}) for tool_name in selected_tools if tool_name in _SUPPORTED_ROUTER_TOOLS
        }
        logger.info("RouterStep decision: tools=%s", selected_tools)
        return {
            "tools": selected_tools,
            "tool_inputs": filtered_tool_inputs,
            "use_retrieval": "retrieval" in selected_tools,
            "use_filesystem": "filesystem" in selected_tools,
        }


@dataclass
class RetrieverStep:
    prompt_store: PromptStorePort
    llm: TextGeneratorPort
    tool: RetrievingToolPort

    def execute(self, state: RagWorkflowState) -> RagWorkflowState:
        logger.info("RetrieverStep: generating search query and retrieving documents.")
        prompt = self.prompt_store.render(
            "retriever",
            conversation_context=state.get("conversation_context", ""),
            user_input=state["user_input"],
        )
        generated_query = self.llm.generate(prompt, temperature=0.0, max_new_tokens=128).strip()
        search_query = generated_query.splitlines()[0].strip() if generated_query else ""
        if not search_query:
            search_query = state["user_input"].strip()
            logger.warning("RetrieverStep: empty query generated. Falling back to user input.")
        logger.info("RetrieverStep: search query generated: '%s'", search_query)
        try:
            retrieved_documents = self.tool.retrieve(search_query)
        except ValueError as exc:
            if "No chunks found in database" not in str(exc):
                raise
            logger.info("RetrieverStep: no chunks available in database, continuing with empty retrieval context.")
            retrieved_documents = []
        logger.info("RetrieverStep: retrieved %d documents.", len(retrieved_documents))
        return {
            "search_query": search_query,
            "retrieved_documents": retrieved_documents,
        }


@dataclass
class FilesystemStep:
    tool: FilesystemToolPort

    def execute(self, state: RagWorkflowState) -> RagWorkflowState:
        tool_inputs = state.get("tool_inputs", {}) or {}
        fs_inputs = tool_inputs.get("filesystem", {}) if isinstance(tool_inputs, dict) else {}
        user_input = state.get("user_input", "")
        conversation_context = state.get("conversation_context", "")
        operation = str(fs_inputs.get("operation", "")).strip().lower()
        if not operation:
            operation = self._infer_operation(user_input, conversation_context)
        path = str(fs_inputs.get("path", "")).strip()
        if not path:
            path = self._infer_path(user_input, operation, state.get("conversation_context", ""))
        if not path:
            path = "."
        content = str(fs_inputs.get("content", ""))
        if operation in {"write", "write_file", "create", "touch"} and not content:
            content = self._infer_content(user_input)

        if operation in {"write", "write_file", "create", "touch"}:
            logger.info("FilesystemStep: writing file path='%s'.", path)
            write_result = self.tool.write_file(path=path, content=content)
            return {
                "filesystem_operation": "write_file",
                "filesystem_path": path,
                "filesystem_result": write_result,
                "filesystem_entries": [],
                "filesystem_content": "",
            }

        if operation in {"read", "read_file", "cat"}:
            logger.info("FilesystemStep: reading file path='%s'.", path)
            file_content = self.tool.read_file(path=path)
            return {
                "filesystem_operation": "read_file",
                "filesystem_path": path,
                "filesystem_content": file_content,
                "filesystem_entries": [],
            }

        if operation in {"delete", "delete_file", "remove", "rm"}:
            logger.info("FilesystemStep: deleting file path='%s'.", path)
            delete_result = self.tool.delete_file(path=path)
            return {
                "filesystem_operation": "delete_file",
                "filesystem_path": path,
                "filesystem_result": delete_result,
                "filesystem_entries": [],
                "filesystem_content": "",
            }

        logger.info("FilesystemStep: listing directory path='%s'.", path)
        entries = self.tool.list_directory(path=path)
        logger.info("FilesystemStep: listed %d entries for path='%s'.", len(entries), path)
        return {
            "filesystem_operation": "list_directory",
            "filesystem_path": path,
            "filesystem_entries": entries,
            "filesystem_content": "",
        }

    @staticmethod
    def _infer_operation(user_input: str, conversation_context: str = "") -> str:
        text = (user_input or "").lower()
        if any(token in text for token in ("delete", "deletar", "apagar", "remover", "excluir", "rm ")):
            return "delete_file"
        if any(token in text for token in ("crie", "criar", "create", "write", "escreva", "salve")):
            return "write_file"
        if any(token in text for token in ("leia", "read", "mostrar conteudo", "mostre conteudo", "cat ")):
            return "read_file"
        if (
            text.strip() in {"yes", "sim", "ok", "okay", "pode", "prossiga", "continue", "confirmo", "confirma"}
            and any(
                token in (conversation_context or "").lower()
                for token in ("delete", "deletar", "apagar", "remover", "excluir", "arquivo")
            )
        ):
            return "delete_file"
        return "list_directory"

    @staticmethod
    def _infer_path(user_input: str, operation: str, conversation_context: str = "") -> str:
        text = (user_input or "")
        path_match = re.search(r"([a-zA-Z0-9_\-./]+?\.[a-zA-Z0-9]{1,8})", text)
        if path_match:
            return path_match.group(1)
        context_match = re.search(r"([a-zA-Z0-9_\-./]+?\.[a-zA-Z0-9]{1,8})", conversation_context or "")
        if context_match:
            return context_match.group(1)
        if operation in {"write", "write_file", "create", "touch"}:
            return "generated.txt"
        return "."

    @staticmethod
    def _infer_content(user_input: str) -> str:
        text = user_input or ""
        quoted_match = re.search(r'"([^"]+)"', text)
        if quoted_match:
            return quoted_match.group(1)
        single_match = re.search(r"'([^']+)'", text)
        if single_match:
            return single_match.group(1)
        return ""


@dataclass
class ResponderStep:
    prompt_store: PromptStorePort
    llm: TextGeneratorPort

    def execute(self, state: RagWorkflowState) -> RagWorkflowState:
        docs = state.get("retrieved_documents", [])
        fs_entries = state.get("filesystem_entries", [])
        logger.info(
            "ResponderStep: generating final response using docs=%d filesystem_entries=%d.",
            len(docs),
            len(fs_entries),
        )
        docs_text = self._format_docs(docs)
        filesystem_text = self._format_filesystem(
            fs_entries,
            path=state.get("filesystem_path", "."),
            operation=state.get("filesystem_operation", ""),
            filesystem_content=state.get("filesystem_content", ""),
            filesystem_result=state.get("filesystem_result", ""),
        )
        errors_text = self._format_tool_errors(state.get("tool_errors", []))
        prompt = self.prompt_store.render(
            "responder",
            conversation_context=state.get("conversation_context", ""),
            retrieved_documents=docs_text,
            filesystem_context=filesystem_text,
            tool_errors=errors_text,
            user_input=state["user_input"],
        )
        response = self.llm.generate(prompt, temperature=0.2, max_new_tokens=320).strip()
        logger.info("ResponderStep: response generated (length: %d chars).", len(response))
        return {"response": response}

    @staticmethod
    def _format_docs(documents: list[RetrievedDocument]) -> str:
        if not documents:
            return "[]"
        lines = []
        for item in documents:
            lines.append(f"- source: {item.get('source', 'unknown')}\n  content: {item.get('content', '')}")
        return "\n".join(lines)

    @staticmethod
    def _format_filesystem(
        entries: list[FilesystemEntry],
        path: str,
        operation: str,
        filesystem_content: str,
        filesystem_result: str,
    ) -> str:
        lines = []
        if operation:
            lines.append(f"operation: {operation}")
        lines.append(f"path: {path}")
        if filesystem_result:
            lines.append(f"result: {filesystem_result}")
        if filesystem_content:
            lines.append("content:")
            lines.append(filesystem_content)
        if not entries:
            if len(lines) == 1 and lines[0] == f"path: {path}":
                return "[]"
            return "\n".join(lines)
        for entry in entries:
            entry_type = "dir" if entry.get("is_directory") else "file"
            lines.append(f"- [{entry_type}] {entry.get('name', '')}")
        return "\n".join(lines)

    @staticmethod
    def _format_tool_errors(errors: list[str]) -> str:
        if not errors:
            return "[]"
        return "\n".join(f"- {error}" for error in errors)


class RetrievingTool:
    def __init__(self, retrieval_fn: Callable[[str], list[dict]]) -> None:
        self._retrieval_fn = retrieval_fn

    def retrieve(self, search_query: str) -> list[RetrievedDocument]:
        rows = self._retrieval_fn(search_query) or []
        docs: list[RetrievedDocument] = []
        for row in rows:
            metadata = row.get("metadata") or {}
            docs.append(
                {
                    "content": str(row.get("content", "")),
                    "source": str(metadata.get("filename") or metadata.get("document_id") or "unknown"),
                }
            )
        return docs


class FilesystemMcpTool:
    def __init__(
        self,
        *,
        list_directory_fn: Callable[[str], list[dict]],
        read_file_fn: Callable[[str], str],
        write_file_fn: Callable[[str, str], str],
        delete_file_fn: Callable[[str], str],
    ) -> None:
        self._list_directory_fn = list_directory_fn
        self._read_file_fn = read_file_fn
        self._write_file_fn = write_file_fn
        self._delete_file_fn = delete_file_fn

    def list_directory(self, path: str = ".") -> list[FilesystemEntry]:
        rows = self._list_directory_fn(path) or []
        entries: list[FilesystemEntry] = []
        for row in rows:
            entries.append(
                {
                    "name": str(row.get("name", "")),
                    "is_directory": bool(row.get("is_directory", False)),
                }
            )
        return entries

    def read_file(self, path: str) -> str:
        return str(self._read_file_fn(path))

    def write_file(self, path: str, content: str) -> str:
        return str(self._write_file_fn(path, content))

    def delete_file(self, path: str) -> str:
        return str(self._delete_file_fn(path))


class RagWorkflowOrchestrator:
    def __init__(
        self,
        *,
        router: RouterStep,
        retriever: RetrieverStep,
        responder: ResponderStep,
        filesystem: FilesystemStep | None = None,
        prefer_langgraph: bool = True,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        self._router = router
        self._retriever = retriever
        self._filesystem = filesystem
        self._responder = responder
        self._logger = logger_instance or logger
        self._compiled_graph = self._build_graph() if prefer_langgraph else None

    def run(self, *, user_input: str, conversation_context: str = "") -> RagWorkflowState:
        self._logger.info(
            "Starting RagWorkflow run for user input: '%s'",
            user_input[:50] + "..." if len(user_input) > 50 else user_input,
        )
        initial_state: RagWorkflowState = {
            "user_input": user_input,
            "conversation_context": conversation_context,
            "retrieved_documents": [],
            "filesystem_entries": [],
            "tool_errors": [],
        }
        if self._compiled_graph is not None:
            self._logger.info("Executing workflow using LangGraph.")
            return self._compiled_graph.invoke(initial_state)
        self._logger.info("Executing workflow using local fallback executor.")
        return self._run_without_graph(initial_state)

    def route_only(self, *, user_input: str, conversation_context: str = "") -> bool:
        self._logger.info("Starting route_only check.")
        state: RagWorkflowState = {
            "user_input": user_input,
            "conversation_context": conversation_context,
        }
        output = self._router.execute(state)
        return "retrieval" in output.get("tools", [])

    def _run_without_graph(self, state: RagWorkflowState) -> RagWorkflowState:
        router_state = self._router.execute(state)
        state.update(router_state)
        tool_state = self._execute_selected_tools(state)
        state.update(tool_state)
        response_state = self._responder.execute(state)
        state.update(response_state)
        return state

    def _execute_selected_tools(self, state: RagWorkflowState) -> RagWorkflowState:
        tools = state.get("tools", []) or []
        futures = {}
        combined_state: RagWorkflowState = {}
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=max(1, len(tools) or 1)) as executor:
            if "retrieval" in tools:
                futures["retrieval"] = executor.submit(self._retriever.execute, state)
            if "filesystem" in tools:
                if self._filesystem is None:
                    errors.append("filesystem tool requested but not configured.")
                else:
                    futures["filesystem"] = executor.submit(self._filesystem.execute, state)

            for task_name, future in futures.items():
                try:
                    task_state = future.result()
                    combined_state.update(task_state)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Tool execution failed: %s", task_name)
                    errors.append(f"{task_name}: {exc}")

        if errors:
            combined_state["tool_errors"] = errors
        return combined_state

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph

            self._logger.info("Successfully imported LangGraph. Building StateGraph.")
        except Exception:  # noqa: BLE001
            self._logger.warning("LangGraph is not available. Falling back to local workflow executor.")
            return None

        graph = StateGraph(RagWorkflowState)
        graph.add_node("router", self._router.execute)
        graph.add_node("tool_executor", self._execute_selected_tools)
        graph.add_node("responder", self._responder.execute)

        graph.set_entry_point("router")
        graph.add_edge("router", "tool_executor")
        graph.add_edge("tool_executor", "responder")
        graph.set_finish_point("responder")
        self._logger.info("LangGraph workflow compiled successfully.")
        return graph.compile()
