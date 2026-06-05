from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypedDict

logger = logging.getLogger(__name__)

_RETRIEVAL_TOOL_MANIFEST: dict[str, Any] = {
    "name": "retrieval",
    "description": "Search inside uploaded documents/chunks in the local RAG context.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (defaults to user question)."}
        },
        "additionalProperties": False,
    },
}


class RetrievedDocument(TypedDict):
    content: str
    source: str


class RagWorkflowState(TypedDict, total=False):
    user_input: str
    conversation_context: str
    available_tools: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]          # [{"name": str, "arguments": dict}]
    retrieved_documents: list[RetrievedDocument]
    tool_results: dict[str, Any]              # tool_name -> raw result
    tool_errors: list[str]
    response: str


class PromptStorePort(Protocol):
    def render(self, prompt_name: str, **variables) -> str: ...


class TextGeneratorPort(Protocol):
    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str: ...


class RetrievingToolPort(Protocol):
    def retrieve(self, search_query: str) -> list[RetrievedDocument]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_manifests(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raw = []
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append({
            "name": name,
            "description": str(item.get("description", "")).strip(),
            "inputSchema": item.get("inputSchema") if isinstance(item.get("inputSchema"), dict) else {},
        })
    # always include retrieval
    if not any(m["name"] == "retrieval" for m in result):
        result.insert(0, dict(_RETRIEVAL_TOOL_MANIFEST))
    return result


def _render_manifest(manifests: list[dict[str, Any]]) -> str:
    return json.dumps(
        [{"name": m["name"], "description": m["description"], "inputSchema": m["inputSchema"]} for m in manifests],
        ensure_ascii=False,
        indent=2,
    )


def _parse_tool_calls(raw_output: str, *, available_names: set[str]) -> list[dict[str, Any]]:
    """Parse LLM output into a list of {name, arguments} dicts."""
    text = (raw_output or "").strip()
    payload: dict | None = None

    # Strip markdown code fences if present
    code_fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_fence:
        text = code_fence.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                payload = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if isinstance(payload, dict):
        calls: list[dict[str, Any]] = []
        seen: set[str] = set()

        # preferred: {"tool_calls": [{"name": ..., "arguments": {...}}]}
        if isinstance(payload.get("tool_calls"), list):
            for item in payload["tool_calls"]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip().lower()
                if name not in available_names or name in seen:
                    continue
                seen.add(name)
                args = item.get("arguments") or {}
                calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
            return calls

        # fallback: {"tools": [...], "tool_inputs": {...}}
        if isinstance(payload.get("tools"), list):
            tool_inputs: dict = payload.get("tool_inputs") or {}
            for name in payload["tools"]:
                name = str(name).strip().lower()
                if name not in available_names or name in seen:
                    continue
                seen.add(name)
                args = tool_inputs.get(name) or {}
                calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
            return calls

        # single tool object: {"name": "...", "arguments": {...}}
        if isinstance(payload.get("name"), str):
            name = str(payload["name"]).strip().lower()
            if name in available_names:
                args = payload.get("arguments") or payload.get("parameters") or {}
                return [{"name": name, "arguments": args if isinstance(args, dict) else {}}]

    # Last resort: scan raw text for any available tool name mentioned
    found: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for name in sorted(available_names):  # sorted for determinism
        if name in seen_names:
            continue
        # match whole tool name as word boundary in the raw text
        if re.search(re.escape(name), (raw_output or ""), re.IGNORECASE):
            found.append({"name": name, "arguments": {}})
            seen_names.add(name)
    # Only use text-scan fallback if exactly one tool was found (ambiguous = skip)
    if len(found) == 1:
        return found

    return []


def _is_surgical_read_request(user_input: str) -> bool:
    lowered = (user_input or "").lower()
    filename_pattern = re.compile(r"\b[\w./-]+\.[a-z0-9]{1,10}\b")
    read_markers = (
        "read_file",
        "leia",
        "conteudo exato",
        "conteúdo exato",
        "literal",
        "header",
        "cabecalho",
        "cabeçalho",
        "csv",
        "linha",
        "linhas",
        "debug",
        "transform",
        "regex",
    )
    return bool(filename_pattern.search(lowered)) and any(marker in lowered for marker in read_markers)


def _looks_semantic_context_query(user_input: str) -> bool:
    lowered = (user_input or "").lower()
    semantic_markers = (
        "resuma",
        "summary",
        "summarize",
        "compare",
        "compare com",
        "analise",
        "analyze",
        "o que diz",
        "what does",
        "documento",
        "arquivo",
        "manual",
        "contrato",
        "contexto",
        "upload",
    )
    operational_markers = (
        "liste",
        "list",
        "crie",
        "create file",
        "delete",
        "apague",
        "escreva",
        "write file",
        "renomeie",
        "rename",
    )
    if any(marker in lowered for marker in operational_markers) and not any(
        marker in lowered for marker in semantic_markers
    ):
        return False
    return any(marker in lowered for marker in semantic_markers)


def _apply_routing_policy(
    tool_calls: list[dict[str, Any]],
    *,
    user_input: str,
    available_names: set[str],
) -> list[dict[str, Any]]:
    if "retrieval" not in available_names:
        return tool_calls

    names = [str(call.get("name", "")).strip().lower() for call in tool_calls]
    has_memory = any(name.startswith("memory.") for name in names)
    has_filesystem_non_read = any(name.startswith("filesystem.") and name != "filesystem.read_file" for name in names)
    has_retrieval = "retrieval" in names
    has_read = "filesystem.read_file" in names
    surgical_read = _is_surgical_read_request(user_input)

    if has_retrieval and has_read and not surgical_read:
        tool_calls = [call for call in tool_calls if call.get("name") != "filesystem.read_file"]
        has_read = False

    if has_read and not has_retrieval and not surgical_read:
        tool_calls = [call for call in tool_calls if call.get("name") != "filesystem.read_file"]
        has_retrieval = False

    if (
        not has_retrieval
        and not has_memory
        and not has_filesystem_non_read
        and _looks_semantic_context_query(user_input)
    ):
        tool_calls.insert(0, {"name": "retrieval", "arguments": {"query": user_input}})

    return tool_calls


def _format_docs(documents: list[RetrievedDocument]) -> str:
    if not documents:
        return "[]"
    return "\n".join(
        f"- source: {d.get('source', 'unknown')}\n  content: {d.get('content', '')}"
        for d in documents
    )


def _format_tool_results(results: dict[str, Any]) -> str:
    if not results:
        return "[]"
    return json.dumps(results, ensure_ascii=False)[:4000]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

@dataclass
class RouterStep:
    prompt_store: PromptStorePort
    llm: TextGeneratorPort

    def execute(self, state: RagWorkflowState) -> RagWorkflowState:
        manifests = _normalize_manifests(state.get("available_tools", []))
        available_names = {m["name"] for m in manifests}
        prompt = self.prompt_store.render(
            "router",
            conversation_context=state.get("conversation_context", ""),
            user_input=state["user_input"],
            available_tools_manifest=_render_manifest(manifests),
        )
        raw = self.llm.generate(prompt, temperature=0.0, max_new_tokens=512)
        logger.info("RouterStep: raw_output=%r", raw[:400] if raw else "")
        tool_calls = _parse_tool_calls(raw, available_names=available_names)
        tool_calls = _apply_routing_policy(
            tool_calls,
            user_input=state.get("user_input", ""),
            available_names=available_names,
        )
        logger.info("RouterStep: tool_calls=%s", [c["name"] for c in tool_calls])
        return {"tool_calls": tool_calls}


@dataclass
class RetrieverStep:
    prompt_store: PromptStorePort
    llm: TextGeneratorPort
    tool: RetrievingToolPort

    def execute(self, state: RagWorkflowState, arguments: dict[str, Any]) -> list[RetrievedDocument]:
        query = str(arguments.get("query", "")).strip() or state["user_input"]
        prompt = self.prompt_store.render(
            "retriever",
            conversation_context=state.get("conversation_context", ""),
            user_input=query,
        )
        generated = self.llm.generate(prompt, temperature=0.0, max_new_tokens=128).strip()
        search_query = generated.splitlines()[0].strip() if generated else query
        logger.info("RetrieverStep: query='%s'", search_query)
        try:
            docs = self.tool.retrieve(search_query)
        except ValueError as exc:
            if "No chunks found in database" not in str(exc):
                raise
            docs = []
        return docs


@dataclass
class ResponderStep:
    prompt_store: PromptStorePort
    llm: TextGeneratorPort

    def execute(self, state: RagWorkflowState) -> RagWorkflowState:
        docs_text = _format_docs(state.get("retrieved_documents", []))
        tool_results_text = _format_tool_results(state.get("tool_results", {}))
        errors_text = "\n".join(f"- {e}" for e in (state.get("tool_errors") or [])) or "[]"
        prompt = self.prompt_store.render(
            "responder",
            conversation_context=state.get("conversation_context", ""),
            retrieved_documents=docs_text,
            tool_results=tool_results_text,
            tool_errors=errors_text,
            user_input=state["user_input"],
        )
        response = self.llm.generate(prompt, temperature=0.2, max_new_tokens=320).strip()
        return {"response": response}


# ---------------------------------------------------------------------------
# Concrete tool adapters (kept thin — just bridge to MCP callables)
# ---------------------------------------------------------------------------

class RetrievingTool:
    def __init__(self, retrieval_fn: Callable[[str], list[dict]]) -> None:
        self._fn = retrieval_fn

    def retrieve(self, search_query: str) -> list[RetrievedDocument]:
        rows = self._fn(search_query) or []
        return [
            {
                "content": str(r.get("content", "")),
                "source": str((r.get("metadata") or {}).get("filename") or (r.get("metadata") or {}).get("document_id") or "unknown"),
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class RagWorkflowOrchestrator:
    def __init__(
        self,
        *,
        router: RouterStep,
        retriever: RetrieverStep,
        responder: ResponderStep,
        available_tools_provider: Callable[[], list[dict[str, Any]]] | None = None,
        mcp_tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        graph_builder_llm: TextGeneratorPort | None = None,
        prompt_store: PromptStorePort | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        self._router = router
        self._retriever = retriever
        self._responder = responder
        self._available_tools_provider = available_tools_provider
        self._mcp_tool_executor = mcp_tool_executor
        self._graph_builder_llm = graph_builder_llm
        self._prompt_store = prompt_store
        self._logger = logger_instance or logger
        self._compiled_graph = self._build_graph()

    def run(self, *, user_input: str, conversation_context: str = "") -> RagWorkflowState:
        available_tools = self._fetch_tools()
        state: RagWorkflowState = {
            "user_input": user_input,
            "conversation_context": conversation_context,
            "available_tools": available_tools,
            "retrieved_documents": [],
            "tool_results": {},
            "tool_errors": [],
        }
        if self._compiled_graph is not None:
            self._logger.info("Executing workflow using LangGraph.")
            return self._compiled_graph.invoke(state)
        return self._run_linear(state)

    def route_only(self, *, user_input: str, conversation_context: str = "") -> bool:
        state: RagWorkflowState = {
            "user_input": user_input,
            "conversation_context": conversation_context,
            "available_tools": self._fetch_tools(),
        }
        out = self._router.execute(state)
        return any(c["name"] == "retrieval" for c in out.get("tool_calls", []))

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_tool_args(tool_name: str, args: dict[str, Any], state: RagWorkflowState) -> dict[str, Any]:
        """Ensure tool arguments are valid before dispatch. Fixes incomplete LLM-generated args."""
        if tool_name == "memory.add_observations":
            observations = args.get("observations")
            if not isinstance(observations, list) or not observations:
                user_input = state.get("user_input", "").strip()
                return {
                    "observations": [
                        {"entityName": "session_memory", "contents": [user_input]}
                    ]
                }
            # Validate each item has entityName (string) and contents (non-empty list of strings)
            clean: list[dict[str, Any]] = []
            for item in observations:
                if not isinstance(item, dict):
                    continue
                entity = str(item.get("entityName") or item.get("entity_name") or "session_memory").strip()
                contents = item.get("contents")
                if not isinstance(contents, list) or not contents:
                    contents = [state.get("user_input", "").strip()]
                clean_contents = [str(c).strip() for c in contents if str(c).strip()]
                if clean_contents:
                    clean.append({"entityName": entity, "contents": clean_contents})
            if not clean:
                user_input = state.get("user_input", "").strip()
                clean = [{"entityName": "session_memory", "contents": [user_input]}]
            return {"observations": clean}
        if tool_name == "memory.search_nodes":
            query = str(args.get("query") or state.get("user_input") or "").strip()
            return {"query": query}
        if tool_name == "memory.open_nodes":
            names = args.get("names")
            if not isinstance(names, list) or not names:
                return {"names": ["session_memory"]}
            return {"names": [str(n).strip() for n in names if str(n).strip()]}
        return args

    def _handle_memory_write(self, state: RagWorkflowState) -> dict[str, Any]:
        """Extract graph from user_input via LLM, then create_entities + create_relations + add_observations."""
        user_input = state.get("user_input", "").strip()
        graph = self._build_memory_graph(user_input)
        results: dict[str, Any] = {}

        entities = graph.get("entities") or []
        relations = graph.get("relations") or []

        if entities and self._mcp_tool_executor:
            try:
                results["create_entities"] = self._mcp_tool_executor(
                    "memory.create_entities", {"entities": entities}
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("memory.create_entities failed: %s", exc)

        if relations and self._mcp_tool_executor:
            try:
                results["create_relations"] = self._mcp_tool_executor(
                    "memory.create_relations", {"relations": relations}
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("memory.create_relations failed: %s", exc)

        # always store raw text as observation so it's searchable
        if self._mcp_tool_executor:
            try:
                results["add_observations"] = self._mcp_tool_executor(
                    "memory.add_observations",
                    {"observations": [{"entityName": "session_memory", "contents": [user_input]}]},
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("memory.add_observations failed: %s", exc)

        self._logger.info(
            "_handle_memory_write: entities=%d relations=%d",
            len(entities), len(relations),
        )
        return results

    def _build_memory_graph(self, text: str) -> dict[str, Any]:
        """Use LLM to extract entities and relations from text."""
        if self._graph_builder_llm is None or self._prompt_store is None:
            return {"entities": [], "relations": []}
        try:
            prompt = self._prompt_store.render("memory_graph_builder", text=text)
            raw = self._graph_builder_llm.generate(prompt, temperature=0.0, max_new_tokens=800)
            # strip code fences
            raw = raw.strip()
            fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if fence:
                raw = fence.group(1).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    return parsed
        except Exception:  # noqa: BLE001
            self._logger.exception("_build_memory_graph failed, returning empty graph.")
        return {"entities": [], "relations": []}

    def _fetch_tools(self) -> list[dict[str, Any]]:
        if self._available_tools_provider is None:
            return [dict(_RETRIEVAL_TOOL_MANIFEST)]
        try:
            tools = self._available_tools_provider() or []
            return tools if tools else [dict(_RETRIEVAL_TOOL_MANIFEST)]
        except Exception:
            self._logger.exception("Failed to fetch MCP tool list. Using retrieval-only fallback.")
            return [dict(_RETRIEVAL_TOOL_MANIFEST)]

    def _run_linear(self, state: RagWorkflowState) -> RagWorkflowState:
        state.update(self._router.execute(state))
        state.update(self._execute_tools(state))
        state.update(self._responder.execute(state))
        return state

    def _execute_tools(self, state: RagWorkflowState) -> RagWorkflowState:
        tool_calls: list[dict[str, Any]] = state.get("tool_calls") or []
        retrieved_documents: list[RetrievedDocument] = []
        tool_results: dict[str, Any] = {}
        errors: list[str] = []

        def run_call(call: dict[str, Any]) -> tuple[str, Any]:
            name: str = call["name"]
            args: dict[str, Any] = call.get("arguments") or {}
            if name == "retrieval":
                docs = self._retriever.execute(state, args)
                return ("retrieval", docs)
            if self._mcp_tool_executor is None:
                raise RuntimeError(f"No MCP executor configured for tool '{name}'.")
            # memory write intent: extract graph then persist
            if name == "memory.add_observations":
                return (name, self._handle_memory_write(state))
            args = self._normalize_tool_args(name, args, state)
            return (name, self._mcp_tool_executor(name, args))

        with ThreadPoolExecutor(max_workers=max(1, len(tool_calls))) as pool:
            futures = {pool.submit(run_call, call): call["name"] for call in tool_calls}
            for future, name in futures.items():
                try:
                    result_name, result_value = future.result()
                    if result_name == "retrieval":
                        retrieved_documents.extend(result_value)
                    else:
                        tool_results[result_name] = result_value
                except Exception as exc:
                    self._logger.exception("Tool execution failed: %s", name)
                    errors.append(f"{name}: {exc}")

        out: RagWorkflowState = {}
        if retrieved_documents:
            out["retrieved_documents"] = retrieved_documents
        if tool_results:
            out["tool_results"] = tool_results
        if errors:
            out["tool_errors"] = errors
        return out

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph
        except Exception:
            self._logger.warning("LangGraph not available. Using linear executor.")
            return None

        graph = StateGraph(RagWorkflowState)
        graph.add_node("router", self._router.execute)
        graph.add_node("tool_executor", self._execute_tools)
        graph.add_node("responder", self._responder.execute)
        graph.set_entry_point("router")
        graph.add_edge("router", "tool_executor")
        graph.add_edge("tool_executor", "responder")
        graph.set_finish_point("responder")
        self._logger.info("LangGraph workflow compiled.")
        return graph.compile()
