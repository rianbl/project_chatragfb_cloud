from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

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

_MEMORY_GRAPH_WRITE_TOOL_NAMES: set[str] = {
    "memory.create_entities",
    "memory.create_relations",
    "memory.add_observations",
}

_MEMORY_GRAPH_UPSERT_MANIFEST: dict[str, Any] = {
    "name": "memory.graph_upsert",
    "description": (
        "Update long-term memory graph from user-provided facts. "
        "Use when the user asks to remember/store knowledge."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "Optional raw text to ingest into memory graph. "
                    "Defaults to current user question."
                ),
            }
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
    tool_calls: list[dict[str, Any]]  # [{"name": str, "arguments": dict}]
    retrieved_documents: list[RetrievedDocument]
    tool_results: dict[str, Any]  # tool_name -> raw result
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
        result.append(
            {
                "name": name,
                "description": str(item.get("description", "")).strip(),
                "inputSchema": (
                    item.get("inputSchema") if isinstance(item.get("inputSchema"), dict) else {}
                ),
            }
        )
    # always include retrieval
    if not any(m["name"] == "retrieval" for m in result):
        result.insert(0, dict(_RETRIEVAL_TOOL_MANIFEST))
    return result


def _render_manifest(manifests: list[dict[str, Any]]) -> str:
    rendered: list[dict[str, Any]] = []
    for manifest in manifests:
        input_schema = manifest.get("inputSchema")
        properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
        required = (
            input_schema.get("required", [])
            if isinstance(input_schema, dict) and isinstance(input_schema.get("required"), list)
            else []
        )
        required_names = {str(item) for item in required}
        argument_docs: list[dict[str, Any]] = []
        if isinstance(properties, dict):
            for arg_name, arg_schema in properties.items():
                if not isinstance(arg_schema, dict):
                    continue
                argument_docs.append(
                    {
                        "name": str(arg_name),
                        "required": str(arg_name) in required_names,
                        "type": str(arg_schema.get("type", "any")),
                        "description": str(arg_schema.get("description", "")).strip(),
                    }
                )
        rendered.append(
            {
                "name": manifest["name"],
                "description": manifest["description"],
                "arguments": argument_docs,
                "inputSchema": manifest["inputSchema"],
            }
        )
    return json.dumps(
        rendered,
        ensure_ascii=False,
        indent=2,
    )


def _build_router_manifests(raw_tools: object) -> list[dict[str, Any]]:
    manifests = _normalize_manifests(raw_tools)
    has_memory_graph_write_tools = any(
        manifest["name"] in _MEMORY_GRAPH_WRITE_TOOL_NAMES for manifest in manifests
    )
    filtered = [
        manifest for manifest in manifests if manifest["name"] not in _MEMORY_GRAPH_WRITE_TOOL_NAMES
    ]
    if has_memory_graph_write_tools and not any(
        manifest["name"] == _MEMORY_GRAPH_UPSERT_MANIFEST["name"] for manifest in filtered
    ):
        filtered.append(dict(_MEMORY_GRAPH_UPSERT_MANIFEST))
    return filtered


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

        def append_call(name: str, args: object) -> None:
            if name not in available_names:
                return
            safe_args = args if isinstance(args, dict) else {}
            signature = json.dumps(
                {"name": name, "arguments": safe_args},
                ensure_ascii=False,
                sort_keys=True,
            )
            if signature in seen:
                return
            seen.add(signature)
            calls.append({"name": name, "arguments": safe_args})

        # preferred: {"tool_calls": [{"name": ..., "arguments": {...}}]}
        if isinstance(payload.get("tool_calls"), list):
            for item in payload["tool_calls"]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip().lower()
                args = item.get("arguments") or {}
                append_call(name, args)
            return calls

        # fallback: {"tools": [...], "tool_inputs": {...}}
        if isinstance(payload.get("tools"), list):
            tool_inputs: dict = payload.get("tool_inputs") or {}
            for name in payload["tools"]:
                name = str(name).strip().lower()
                args = tool_inputs.get(name) or {}
                append_call(name, args)
            return calls

        # single tool object: {"name": "...", "arguments": {...}}
        if isinstance(payload.get("name"), str):
            name = str(payload["name"]).strip().lower()
            args = payload.get("arguments") or payload.get("parameters") or {}
            append_call(name, args)
            return calls

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
    return bool(filename_pattern.search(lowered)) and any(
        marker in lowered for marker in read_markers
    )


def _looks_semantic_context_query(user_input: str) -> bool:
    lowered = (user_input or "").lower()
    analysis_markers = (
        "resuma",
        "summary",
        "summarize",
        "compare",
        "compare com",
        "analise",
        "analyze",
        "o que diz",
        "what does",
    )
    contextual_markers = (
        "documento",
        "manual",
        "contrato",
        "contexto",
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
        "abra",
        "open",
        "mostre",
        "show",
        "leia",
        "read",
    )
    has_operational = any(marker in lowered for marker in operational_markers)
    has_analysis = any(marker in lowered for marker in analysis_markers)
    if has_operational and not has_analysis:
        return False
    if has_analysis:
        return True
    return any(marker in lowered for marker in contextual_markers)


def _apply_routing_policy(
    tool_calls: list[dict[str, Any]],
    *,
    user_input: str,
    available_names: set[str],
) -> list[dict[str, Any]]:
    if "retrieval" not in available_names:
        return tool_calls

    semantic_query = _looks_semantic_context_query(user_input)
    names = [str(call.get("name", "")).strip().lower() for call in tool_calls]
    has_memory = any(name.startswith("memory.") for name in names)
    has_filesystem_any = any(name.startswith("filesystem.") for name in names)
    has_filesystem_non_read = any(
        name.startswith("filesystem.") and name != "filesystem.read_file" for name in names
    )
    has_retrieval = "retrieval" in names
    has_read = "filesystem.read_file" in names
    surgical_read = _is_surgical_read_request(user_input)

    if has_retrieval and (has_memory or has_filesystem_any) and not semantic_query:
        tool_calls = [call for call in tool_calls if call.get("name") != "retrieval"]
        has_retrieval = False

    if has_read and not surgical_read:
        if has_retrieval and semantic_query:
            tool_calls = [call for call in tool_calls if call.get("name") != "filesystem.read_file"]
            has_read = False
        elif has_retrieval and not semantic_query:
            tool_calls = [call for call in tool_calls if call.get("name") != "retrieval"]
            has_retrieval = False
        elif not has_retrieval and semantic_query:
            tool_calls = [call for call in tool_calls if call.get("name") != "filesystem.read_file"]
            has_read = False
            tool_calls.insert(0, {"name": "retrieval", "arguments": {"query": user_input}})
            has_retrieval = True

    if (
        not has_retrieval
        and not has_memory
        and not has_filesystem_any
        and semantic_query
    ):
        tool_calls.insert(0, {"name": "retrieval", "arguments": {"query": user_input}})

    if has_filesystem_non_read and has_retrieval and not semantic_query:
        tool_calls = [call for call in tool_calls if call.get("name") != "retrieval"]

    return tool_calls


def _format_docs(documents: list[RetrievedDocument]) -> str:
    if not documents:
        return "[]"

    def clipped(value: str, max_chars: int = 1000) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}...[truncated]"

    return "\n".join(
        f"- source: {d.get('source', 'unknown')}\n  content: {clipped(str(d.get('content', '')))}"
        for d in documents[:6]
    )


def _format_tool_results(results: dict[str, Any]) -> str:
    if not results:
        return "[]"
    return json.dumps(results, ensure_ascii=False)[:4000]


def _split_text_chunks(text: str, *, max_chars: int = 2200) -> list[str]:
    source = str(text or "").strip()
    if not source:
        return []
    if len(source) <= max_chars:
        return [source]

    chunks: list[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", source) if part.strip()]
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        words = paragraph.split()
        sentence = ""
        for word in words:
            next_sentence = f"{sentence} {word}".strip()
            if len(next_sentence) <= max_chars:
                sentence = next_sentence
                continue
            if sentence:
                chunks.append(sentence)
            sentence = word
        if sentence:
            current = sentence

    if current:
        chunks.append(current)
    return chunks


def _extract_json_dict_candidates(raw_text: str) -> list[dict[str, Any]]:
    text = str(raw_text or "").strip()
    if not text:
        return []

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()

    candidates: list[dict[str, Any]] = []
    seen_payloads: set[str] = set()

    def add_candidate(payload: object) -> None:
        if not isinstance(payload, dict):
            return
        signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if signature in seen_payloads:
            return
        seen_payloads.add(signature)
        candidates.append(payload)

    try:
        add_candidate(json.loads(text))
    except Exception:  # noqa: BLE001
        pass

    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        for end in range(start, len(text)):
            char = text[end]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    fragment = text[start : end + 1]
                    try:
                        add_candidate(json.loads(fragment))
                    except Exception:  # noqa: BLE001
                        pass
                    break
        if len(candidates) >= 8:
            break

    return candidates


def _sanitize_memory_graph(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"entities": [], "relations": []}

    raw_entities = payload.get("entities", [])
    raw_relations = payload.get("relations", [])

    entities: list[dict[str, Any]] = []
    for item in raw_entities if isinstance(raw_entities, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        entity_type = str(item.get("entityType", "")).strip() or "unknown"
        observations_raw = item.get("observations", [])
        observations = (
            [str(obs).strip() for obs in observations_raw if str(obs).strip()]
            if isinstance(observations_raw, list)
            else []
        )
        if not name:
            continue
        entities.append({"name": name, "entityType": entity_type, "observations": observations})

    entity_names = {entity["name"] for entity in entities}
    relations: list[dict[str, Any]] = []
    for item in raw_relations if isinstance(raw_relations, list) else []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("from", "")).strip()
        target = str(item.get("to", "")).strip()
        relation_type = str(item.get("relationType", "")).strip()
        if not source or not target or not relation_type:
            continue
        if source not in entity_names or target not in entity_names:
            continue
        relations.append({"from": source, "to": target, "relationType": relation_type})

    return {"entities": entities, "relations": relations}


def _merge_memory_graphs(parts: list[dict[str, Any]]) -> dict[str, Any]:
    merged_entities: dict[str, dict[str, Any]] = {}
    merged_relations: set[tuple[str, str, str]] = set()

    for part in parts:
        for entity in part.get("entities", []):
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            entity_type = str(entity.get("entityType", "")).strip() or "unknown"
            observations = entity.get("observations", [])
            normalized_observations = (
                [str(obs).strip() for obs in observations if str(obs).strip()]
                if isinstance(observations, list)
                else []
            )
            existing = merged_entities.get(name)
            if existing is None:
                merged_entities[name] = {
                    "name": name,
                    "entityType": entity_type,
                    "observations": normalized_observations,
                }
                continue
            if existing.get("entityType") == "unknown" and entity_type != "unknown":
                existing["entityType"] = entity_type
            existing_obs = set(existing.get("observations", []))
            for obs in normalized_observations:
                if obs not in existing_obs:
                    existing.setdefault("observations", []).append(obs)
                    existing_obs.add(obs)

        for relation in part.get("relations", []):
            source = str(relation.get("from", "")).strip()
            target = str(relation.get("to", "")).strip()
            relation_type = str(relation.get("relationType", "")).strip()
            if not source or not target or not relation_type:
                continue
            merged_relations.add((source, target, relation_type))

    entity_name_set = set(merged_entities.keys())
    relations = [
        {"from": source, "to": target, "relationType": relation_type}
        for source, target, relation_type in sorted(merged_relations)
        if source in entity_name_set and target in entity_name_set
    ]

    return {"entities": list(merged_entities.values()), "relations": relations}


def _as_memory_graph(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        return {"entities": [], "relations": []}

    if isinstance(raw.get("entities"), list) and isinstance(raw.get("relations"), list):
        entities = [item for item in raw.get("entities", []) if isinstance(item, dict)]
        relations = [item for item in raw.get("relations", []) if isinstance(item, dict)]
        return {"entities": entities, "relations": relations}

    structured = raw.get("structuredContent")
    if isinstance(structured, dict):
        return _as_memory_graph(structured)

    content = raw.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except Exception:  # noqa: BLE001
                continue
            graph = _as_memory_graph(parsed)
            if graph["entities"] or graph["relations"]:
                return graph

    return {"entities": [], "relations": []}


def _tool_data_payload(result: Any) -> Any:
    if isinstance(result, dict) and "data" in result:
        return result.get("data")
    return result


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@dataclass
class RouterStep:
    prompt_store: PromptStorePort
    llm: TextGeneratorPort

    def execute(self, state: RagWorkflowState) -> RagWorkflowState:
        manifests = _build_router_manifests(state.get("available_tools", []))
        available_names = {m["name"] for m in manifests}
        logger.info("RouterStep: available_tools=%s", sorted(available_names))
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

    def execute(
        self, state: RagWorkflowState, arguments: dict[str, Any]
    ) -> list[RetrievedDocument]:
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
                "source": str(
                    (r.get("metadata") or {}).get("filename")
                    or (r.get("metadata") or {}).get("document_id")
                    or "unknown"
                ),
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
    def _normalize_tool_args(
        tool_name: str, args: dict[str, Any], state: RagWorkflowState
    ) -> dict[str, Any]:
        """Ensure tool arguments are valid before dispatch. Fixes incomplete LLM-generated args."""
        if tool_name == "memory.graph_upsert":
            text_value = args.get("text")
            if isinstance(text_value, str) and text_value.strip():
                return {"text": text_value.strip()}
            return {"text": state.get("user_input", "").strip()}
        if tool_name == "memory.add_observations":
            observations = args.get("observations")
            if not isinstance(observations, list) or not observations:
                user_input = state.get("user_input", "").strip()
                return {
                    "observations": [{"entityName": "session_memory", "contents": [user_input]}]
                }
            # Validate each item has entityName (string) and contents (non-empty list of strings)
            clean: list[dict[str, Any]] = []
            for item in observations:
                if not isinstance(item, dict):
                    continue
                entity = str(
                    item.get("entityName") or item.get("entity_name") or "session_memory"
                ).strip()
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

    def _handle_memory_write(self, state: RagWorkflowState, *, source_text: str | None = None) -> dict[str, Any]:
        """Extract graph from user_input via LLM, then create_entities + create_relations + add_observations."""
        raw_memory_text = str(source_text or state.get("user_input") or "").strip()
        memory_text = self._normalize_memory_text(raw_memory_text)
        graph = self._build_memory_graph(memory_text)
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

        should_store_session_memory = not entities and not relations
        if should_store_session_memory and self._mcp_tool_executor and memory_text:
            try:
                results["add_observations"] = self._mcp_tool_executor(
                    "memory.add_observations",
                    {"observations": [{"entityName": "session_memory", "contents": [memory_text]}]},
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("memory.add_observations failed: %s", exc)

        if entities and self._mcp_tool_executor:
            try:
                read_graph_result = self._mcp_tool_executor("memory.read_graph", {})
                graph_after = _as_memory_graph(_tool_data_payload(read_graph_result))
                non_internal_entities = [
                    item
                    for item in graph_after.get("entities", [])
                    if str(item.get("name", "")).strip().lower() != "session_memory"
                ]
                persisted_count = len(non_internal_entities)
                expected_count = len(entities)
                if persisted_count == 0 and expected_count > 0:
                    self._logger.warning(
                        "Memory graph write appears incomplete (expected_entities=%d, persisted_entities=%d). "
                        "Running resilience add_observations per entity.",
                        expected_count,
                        persisted_count,
                    )
                    resilience_observations: list[dict[str, Any]] = []
                    for entity in entities[:50]:
                        if not isinstance(entity, dict):
                            continue
                        entity_name = str(entity.get("name", "")).strip()
                        if not entity_name:
                            continue
                        raw_obs = entity.get("observations", [])
                        normalized_obs = (
                            [str(item).strip() for item in raw_obs if str(item).strip()]
                            if isinstance(raw_obs, list)
                            else []
                        )
                        if not normalized_obs:
                            entity_type = str(entity.get("entityType", "")).strip() or "unknown"
                            normalized_obs = [f"Entity type: {entity_type}"]
                        resilience_observations.append(
                            {"entityName": entity_name, "contents": normalized_obs[:3]}
                        )
                    if resilience_observations:
                        results["resilience_add_observations"] = self._mcp_tool_executor(
                            "memory.add_observations",
                            {"observations": resilience_observations},
                        )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("memory persistence verification failed: %s", exc)

        self._logger.info(
            "_handle_memory_write: entities=%d relations=%d text_len=%d",
            len(entities),
            len(relations),
            len(memory_text),
        )
        return results

    def _normalize_memory_text(self, text: str) -> str:
        raw_text = str(text or "").strip()
        if not raw_text:
            return ""
        if self._graph_builder_llm is None or self._prompt_store is None:
            return raw_text
        try:
            prompt = self._prompt_store.render("memory_text_normalizer", text=raw_text)
            normalized = self._graph_builder_llm.generate(
                prompt,
                temperature=0.0,
                max_new_tokens=1200,
            ).strip()
            if not normalized:
                return raw_text
            normalized = normalized.strip().strip('"').strip("'").strip()
            if not normalized:
                return raw_text
            if len(raw_text) > 320 and len(normalized) < int(len(raw_text) * 0.75):
                self._logger.warning(
                    "memory_text_normalizer output looks truncated (raw_len=%d normalized_len=%d). "
                    "Using raw text.",
                    len(raw_text),
                    len(normalized),
                )
                return raw_text
            return normalized
        except Exception:  # noqa: BLE001
            self._logger.exception("_normalize_memory_text failed, using raw text.")
            return raw_text

    def _build_memory_graph(self, text: str) -> dict[str, Any]:
        """Use LLM to extract entities and relations from text."""
        if self._graph_builder_llm is None or self._prompt_store is None:
            return {"entities": [], "relations": []}
        source_text = str(text or "").strip()
        if not source_text:
            return {"entities": [], "relations": []}

        chunks = _split_text_chunks(source_text, max_chars=2200)
        graphs: list[dict[str, Any]] = []
        for chunk in chunks[:8]:
            try:
                prompt = self._prompt_store.render("memory_graph_builder", text=chunk)
                raw = self._graph_builder_llm.generate(prompt, temperature=0.0, max_new_tokens=1400)
            except Exception:  # noqa: BLE001
                self._logger.exception("_build_memory_graph chunk generation failed.")
                continue

            parsed_graph = {"entities": [], "relations": []}
            for candidate in _extract_json_dict_candidates(raw):
                sanitized = _sanitize_memory_graph(candidate)
                if sanitized["entities"] or sanitized["relations"]:
                    parsed_graph = sanitized
                    break
            if parsed_graph["entities"] or parsed_graph["relations"]:
                graphs.append(parsed_graph)

        merged = _merge_memory_graphs(graphs)
        self._logger.info(
            "_build_memory_graph: chunks=%d extracted_entities=%d extracted_relations=%d",
            len(chunks[:8]),
            len(merged.get("entities", [])),
            len(merged.get("relations", [])),
        )
        return merged

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
            args = self._normalize_tool_args(name, args, state)
            if name == "memory.graph_upsert":
                return (name, self._handle_memory_write(state, source_text=str(args.get("text", "")).strip()))
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
