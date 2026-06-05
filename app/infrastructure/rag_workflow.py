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

_SUPPORTED_ROUTER_TOOLS = {"retrieval", "filesystem", "memory"}


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
    use_memory: bool
    search_query: str
    retrieved_documents: list[RetrievedDocument]
    filesystem_path: str
    filesystem_operation: str
    filesystem_entries: list[FilesystemEntry]
    filesystem_content: str
    filesystem_result: str
    memory_query: str
    memory_context: str
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


class MemoryToolPort(Protocol):
    def search_nodes(self, query: str) -> dict:
        ...

    def open_nodes(self, names: list[str]) -> dict:
        ...

    def add_observations(self, entity_name: str, contents: list[str]) -> dict:
        ...

    def create_entities(self, entities: list[dict]) -> dict:
        ...

    def create_relations(self, relations: list[dict]) -> dict:
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
    _MEMORY_HINT_PATTERNS = (
        r"\blembra\b",
        r"\blembre\b",
        r"\brmember\b",
        r"\bremember\b",
        r"\bmemor(y|ize|ized|izing)\b",
        r"\bmemoriza(r|do|da|cao|ção)?\b",
        r"\bmem[oó]ria\b",
        r"\bmemory\b",
        r"\bprefer[eê]ncia\b",
        r"\bpreference\b",
        r"\bgosto\b",
        r"\blike\b",
        r"\bperfil\b",
        r"\bprofile\b",
    )
    _AFFIRMATIVE_INPUTS = {"yes", "sim", "ok", "okay", "pode", "prossiga", "continue", "confirmo", "confirma"}

    def __init__(self, *, memory_enabled: bool = True) -> None:
        self._memory_enabled = memory_enabled

    def decide_tools(
        self,
        *,
        user_input: str,
        conversation_context: str,
        model_tools: list[str],
        model_tool_inputs: dict[str, dict[str, str]],
    ) -> list[str]:
        tools = self._normalize_tools(model_tools)
        if not self._memory_enabled:
            tools = [tool for tool in tools if tool != "memory"]
        normalized_input = " ".join((user_input or "").lower().split())
        normalized_context = " ".join((conversation_context or "").lower().split())
        filesystem_inputs = model_tool_inputs.get("filesystem", {}) if isinstance(model_tool_inputs, dict) else {}
        memory_inputs = model_tool_inputs.get("memory", {}) if isinstance(model_tool_inputs, dict) else {}
        filesystem_operation = str(filesystem_inputs.get("operation", "")).strip().lower()
        memory_operation = str(memory_inputs.get("operation", "")).strip().lower()
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
        pure_memory_action = (
            memory_operation in {"search_nodes", "open_nodes", "read_graph", "add_observations", "store_graph", "search", "open", "read", "memorize"}
            or self._looks_like_memory_request(normalized_input)
        )

        if pure_filesystem_action and "filesystem" not in tools:
            tools.append("filesystem")
        if self._memory_enabled and pure_memory_action and "memory" not in tools:
            tools.append("memory")

        if (
            "filesystem" in tools
            and "retrieval" in tools
            and pure_filesystem_action
            and not self._looks_like_document_request(normalized_input)
            and not (conversation_context or "").strip()
        ):
            tools = [tool_name for tool_name in tools if tool_name != "retrieval"]
        if (
            "memory" in tools
            and "retrieval" in tools
            and pure_memory_action
            and not self._looks_like_document_request(normalized_input)
            and not (conversation_context or "").strip()
        ):
            tools = [tool_name for tool_name in tools if tool_name != "retrieval"]

        if "retrieval" not in tools:
            input_looks_like_document_request = self._looks_like_document_request(normalized_input)
            context_looks_like_document_request = self._looks_like_document_request(normalized_context)
            if ("filesystem" in tools or "memory" in tools) and not input_looks_like_document_request and not context_looks_like_document_request:
                return tools
            if context_looks_like_document_request:
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

    def _looks_like_memory_request(self, normalized_input: str) -> bool:
        for pattern in self._MEMORY_HINT_PATTERNS:
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
            "use_memory": "memory" in selected_tools,
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
class MemoryStep:
    tool: MemoryToolPort
    operation_planner: Callable[[str, str], dict] | None = None
    graph_builder: Callable[[str, str], dict] | None = None
    max_entities: int = 8
    max_observations_per_entity: int = 3
    max_text_chars: int = 2000

    def execute(self, state: RagWorkflowState) -> RagWorkflowState:
        tool_inputs = state.get("tool_inputs", {}) or {}
        memory_inputs = tool_inputs.get("memory", {}) if isinstance(tool_inputs, dict) else {}
        user_input = state.get("user_input", "")
        conversation_context = state.get("conversation_context", "")
        planned = self._build_operation_plan(
            user_input=user_input,
            conversation_context=conversation_context,
            memory_inputs=memory_inputs if isinstance(memory_inputs, dict) else {},
        )
        operation = planned.get("operation", "search_nodes")
        query = planned.get("query", "")
        names = planned.get("names", [])

        if operation in {"add_observations", "memorize", "store_graph"}:
            content = planned.get("content", "")
            if not content:
                return {
                    "memory_query": "",
                    "memory_context": "memory write skipped: empty content.",
                }
            entity_name = planned.get("entity_name", "session_memory")
            graph_summary = ""
            if operation in {"memorize", "store_graph"} and self.graph_builder is not None:
                logger.info("MemoryStep: extracting graph from memory content.")
                candidate_graph = self.graph_builder(content, conversation_context)
                entities = self._sanitize_candidate_entities(candidate_graph.get("entities"))
                relations = self._sanitize_candidate_relations(candidate_graph.get("relations"))
                relations = self._filter_relations_by_existing_entities(relations, entities)
                if entities:
                    self.tool.create_entities(entities)
                    if relations:
                        self.tool.create_relations(relations)
                    graph_summary = (
                        f"memory graph updated: entities={len(entities)} relations={len(relations)}."
                    )

            if graph_summary:
                return {
                    "memory_query": "",
                    "memory_context": graph_summary,
                }

            logger.info("MemoryStep: storing memory observation for entity='%s'.", entity_name)
            payload = self.tool.add_observations(entity_name=entity_name, contents=[content])
            return {
                "memory_query": "",
                "memory_context": self._format_memory_write_payload(payload, entity_name=entity_name),
            }

        if operation == "open_nodes":
            if not names:
                return {
                    "memory_query": "",
                    "memory_context": "[]",
                }
            logger.info("MemoryStep: opening memory nodes names=%s.", names)
            payload = self.tool.open_nodes(names=names)
            memory_context = self._format_memory_payload(payload)
            return {
                "memory_query": ", ".join(names),
                "memory_context": memory_context,
            }

        if not query:
            return {
                "memory_query": "",
                "memory_context": "[]",
            }

        logger.info("MemoryStep: searching memory nodes with query='%s'.", query)
        payload = self.tool.search_nodes(query=query)
        memory_context = self._format_memory_payload(payload)
        logger.info("MemoryStep: memory context generated (length=%d).", len(memory_context))
        return {
            "memory_query": query,
            "memory_context": memory_context,
        }

    def _build_operation_plan(self, *, user_input: str, conversation_context: str, memory_inputs: dict[str, str]) -> dict:
        planned: dict = {}
        if self.operation_planner is not None:
            try:
                planner_output = self.operation_planner(user_input, conversation_context)
                if isinstance(planner_output, dict):
                    planned = planner_output
            except Exception:  # noqa: BLE001
                logger.exception("MemoryStep: operation planner failed, using fallback logic.")

        allowed_operations = {"search_nodes", "open_nodes", "add_observations", "memorize", "store_graph"}
        raw_operation = str(planned.get("operation") or "").strip().lower()
        if raw_operation not in allowed_operations:
            raw_operation = str(memory_inputs.get("operation") or "").strip().lower()
        if raw_operation not in allowed_operations:
            raw_operation = self._infer_operation(user_input=user_input, conversation_context=conversation_context)
        operation = raw_operation if raw_operation in allowed_operations else "search_nodes"

        query = str(planned.get("query") or memory_inputs.get("query") or user_input).strip()
        content = str(planned.get("content") or memory_inputs.get("content") or self._infer_memory_content(user_input)).strip()
        entity_name = str(planned.get("entity_name") or memory_inputs.get("entity_name") or "session_memory").strip() or "session_memory"

        raw_names = planned.get("names")
        if not isinstance(raw_names, list):
            raw_names = memory_inputs.get("names", [])
        names = [str(item).strip() for item in raw_names if str(item).strip()] if isinstance(raw_names, list) else []
        if operation == "open_nodes" and not names:
            names = self._infer_names(user_input)

        return {
            "operation": operation,
            "query": query,
            "content": content,
            "entity_name": entity_name,
            "names": names,
        }

    def _format_memory_payload(self, payload: dict) -> str:
        if not isinstance(payload, dict) or not payload:
            return "[]"

        structured = payload.get("structuredContent")
        if isinstance(structured, dict):
            entities = self._format_entities(structured.get("entities"))
            relations = self._format_relations(structured.get("relations"))
            if entities or relations:
                return self._compose_memory_text(entities, relations)

        entities = self._format_entities(payload.get("entities"))
        relations = self._format_relations(payload.get("relations"))
        if entities or relations:
            return self._compose_memory_text(entities, relations)

        text = self._extract_text(payload)
        if text:
            return text[: self.max_text_chars]
        return "[]"

    @staticmethod
    def _infer_memory_content(user_input: str) -> str:
        text = (user_input or "").strip()
        quoted_match = re.search(r'"([^"]+)"', text)
        if quoted_match:
            return quoted_match.group(1).strip()
        single_quote_match = re.search(r"'([^']+)'", text)
        if single_quote_match:
            return single_quote_match.group(1).strip()
        colon_match = re.search(r":\s*(.+)$", text)
        if colon_match:
            return colon_match.group(1).strip()
        return text

    @staticmethod
    def _infer_operation(user_input: str, conversation_context: str) -> str:
        text = f"{user_input or ''} {conversation_context or ''}".lower()
        if any(token in text for token in ("remember", "rmember", "memorize", "memorizar", "memoriza")):
            return "store_graph"
        if any(token in text for token in ("open memory", "open nodes", "abrir memoria", "abrir nodos")):
            return "open_nodes"
        if any(token in text for token in ("save memory", "store memory", "salve na memoria")):
            return "add_observations"
        return "search_nodes"

    @staticmethod
    def _infer_names(user_input: str) -> list[str]:
        if not user_input:
            return []
        quoted = re.findall(r'"([^"]+)"', user_input)
        if quoted:
            return [item.strip() for item in quoted if item.strip()]
        single_quoted = re.findall(r"'([^']+)'", user_input)
        if single_quoted:
            return [item.strip() for item in single_quoted if item.strip()]
        return []

    def _format_memory_write_payload(self, payload: dict, *, entity_name: str) -> str:
        if not isinstance(payload, dict):
            return f"memory write completed for entity '{entity_name}'."
        text = self._extract_text(payload)
        if text:
            return text[: self.max_text_chars]
        structured = payload.get("structuredContent")
        if isinstance(structured, dict):
            added = structured.get("added")
            if added is not None:
                return f"memory write completed: {added} observations added to '{entity_name}'."
        return f"memory write completed for entity '{entity_name}'."

    @staticmethod
    def _sanitize_candidate_entities(raw_entities: object) -> list[dict]:
        if not isinstance(raw_entities, list):
            return []
        entities: list[dict] = []
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            entity_type = str(item.get("entityType", "")).strip() or "unknown"
            if not name:
                continue
            raw_observations = item.get("observations")
            observations = (
                [str(obs).strip() for obs in raw_observations if str(obs).strip()]
                if isinstance(raw_observations, list)
                else []
            )
            entities.append(
                {
                    "name": name,
                    "entityType": entity_type,
                    "observations": observations,
                }
            )
        deduped: list[dict] = []
        seen_names: set[str] = set()
        for entity in entities:
            normalized = entity["name"].lower()
            if normalized in seen_names:
                continue
            seen_names.add(normalized)
            deduped.append(entity)
        return deduped

    @staticmethod
    def _sanitize_candidate_relations(raw_relations: object) -> list[dict]:
        if not isinstance(raw_relations, list):
            return []
        relations: list[dict] = []
        for item in raw_relations:
            if not isinstance(item, dict):
                continue
            source = str(item.get("from", "")).strip()
            target = str(item.get("to", "")).strip()
            relation_type = str(item.get("relationType", "")).strip()
            if not source or not target or not relation_type:
                continue
            relations.append({"from": source, "to": target, "relationType": relation_type})
        deduped: list[dict] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for relation in relations:
            edge_key = (
                relation["from"].lower(),
                relation["to"].lower(),
                relation["relationType"].lower(),
            )
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            deduped.append(relation)
        return deduped

    @staticmethod
    def _filter_relations_by_existing_entities(relations: list[dict], entities: list[dict]) -> list[dict]:
        if not relations or not entities:
            return []
        known = {str(entity.get("name", "")).strip().lower() for entity in entities if str(entity.get("name", "")).strip()}
        if not known:
            return []
        return [
            relation
            for relation in relations
            if str(relation.get("from", "")).strip().lower() in known and str(relation.get("to", "")).strip().lower() in known
        ]

    def _format_entities(self, raw_entities: object) -> list[str]:
        if not isinstance(raw_entities, list):
            return []
        lines: list[str] = []
        for entity in raw_entities[: self.max_entities]:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            entity_type = str(entity.get("entityType", "")).strip() or "unknown"
            raw_observations = entity.get("observations")
            observations = (
                [str(item).strip() for item in raw_observations if str(item).strip()]
                if isinstance(raw_observations, list)
                else []
            )
            if observations:
                top_observations = "; ".join(observations[: self.max_observations_per_entity])
                lines.append(f"- entity: {name} ({entity_type}) | observations: {top_observations}")
            else:
                lines.append(f"- entity: {name} ({entity_type})")
        return lines

    @staticmethod
    def _format_relations(raw_relations: object) -> list[str]:
        if not isinstance(raw_relations, list):
            return []
        lines: list[str] = []
        for relation in raw_relations:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("from", "")).strip()
            target = str(relation.get("to", "")).strip()
            relation_type = str(relation.get("relationType", "")).strip()
            if not source or not target or not relation_type:
                continue
            lines.append(f"- relation: {source} -[{relation_type}]-> {target}")
        return lines

    @staticmethod
    def _extract_text(payload: dict) -> str:
        for key in ("text", "result", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        raw_content = payload.get("content")
        if not isinstance(raw_content, list):
            return ""
        texts: list[str] = []
        for item in raw_content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        return "\n".join(texts).strip()

    def _compose_memory_text(self, entities: list[str], relations: list[str]) -> str:
        lines: list[str] = []
        if entities:
            lines.append("entities:")
            lines.extend(entities)
        if relations:
            lines.append("relations:")
            lines.extend(relations[: self.max_entities])
        if not lines:
            return "[]"
        return "\n".join(lines)


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
        memory_text = state.get("memory_context", "[]")
        errors_text = self._format_tool_errors(state.get("tool_errors", []))
        prompt = self.prompt_store.render(
            "responder",
            conversation_context=state.get("conversation_context", ""),
            retrieved_documents=docs_text,
            filesystem_context=filesystem_text,
            memory_context=memory_text,
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


class MemoryMcpTool:
    def __init__(
        self,
        *,
        search_nodes_fn: Callable[[str], dict],
        open_nodes_fn: Callable[[list[str]], dict] | None = None,
        add_observations_fn: Callable[[str, list[str]], dict],
        create_entities_fn: Callable[[list[dict]], dict] | None = None,
        create_relations_fn: Callable[[list[dict]], dict] | None = None,
    ) -> None:
        self._search_nodes_fn = search_nodes_fn
        self._open_nodes_fn = open_nodes_fn
        self._add_observations_fn = add_observations_fn
        self._create_entities_fn = create_entities_fn
        self._create_relations_fn = create_relations_fn

    def search_nodes(self, query: str) -> dict:
        payload = self._search_nodes_fn(query)
        if isinstance(payload, dict):
            return payload
        return {}

    def open_nodes(self, names: list[str]) -> dict:
        if self._open_nodes_fn is None:
            return {}
        payload = self._open_nodes_fn(names)
        if isinstance(payload, dict):
            return payload
        return {}

    def add_observations(self, entity_name: str, contents: list[str]) -> dict:
        payload = self._add_observations_fn(entity_name, contents)
        if isinstance(payload, dict):
            return payload
        return {}

    def create_entities(self, entities: list[dict]) -> dict:
        if self._create_entities_fn is None:
            return {}
        payload = self._create_entities_fn(entities)
        if isinstance(payload, dict):
            return payload
        return {}

    def create_relations(self, relations: list[dict]) -> dict:
        if self._create_relations_fn is None:
            return {}
        payload = self._create_relations_fn(relations)
        if isinstance(payload, dict):
            return payload
        return {}


class RagWorkflowOrchestrator:
    def __init__(
        self,
        *,
        router: RouterStep,
        retriever: RetrieverStep,
        responder: ResponderStep,
        filesystem: FilesystemStep | None = None,
        memory: MemoryStep | None = None,
        prefer_langgraph: bool = True,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        self._router = router
        self._retriever = retriever
        self._filesystem = filesystem
        self._memory = memory
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
            "memory_context": "[]",
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
            if "memory" in tools:
                if self._memory is None:
                    errors.append("memory tool requested but not configured.")
                else:
                    futures["memory"] = executor.submit(self._memory.execute, state)

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
