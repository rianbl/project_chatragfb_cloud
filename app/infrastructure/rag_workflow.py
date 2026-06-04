from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol, TypedDict

try:
    import langchain
    # Enable LangChain debug mode to provide additional visibility into graph execution if supported
    langchain.debug = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class RetrievedDocument(TypedDict):
    content: str
    source: str


class RagWorkflowState(TypedDict, total=False):
    user_input: str
    conversation_context: str
    use_retrieval: bool
    search_query: str
    retrieved_documents: list[RetrievedDocument]
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


class RouterDecisionPolicyPort(Protocol):
    def decide(self, *, user_input: str, conversation_context: str, model_decision: bool) -> bool:
        ...


class ConservativeRouterDecisionPolicy:
    """Biases routing toward retrieval unless a direct answer is clearly safe."""

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

    def decide(self, *, user_input: str, conversation_context: str, model_decision: bool) -> bool:
        if model_decision:
            return True

        normalized_input = " ".join((user_input or "").lower().split())
        if not normalized_input:
            return True

        for pattern in self._OBVIOUS_DIRECT_RESPONSE_PATTERNS:
            if re.search(pattern, normalized_input):
                return False

        if (conversation_context or "").strip():
            logger.info("RouterDecisionPolicy override: retrieval forced by available conversation context.")
            return True

        logger.info("RouterDecisionPolicy override: conservative retrieval applied.")
        return True


def _parse_router_output(raw_output: str) -> bool:
    text = (raw_output or "").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("use_retrieval"), bool):
            return parsed["use_retrieval"]
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict) and isinstance(parsed.get("use_retrieval"), bool):
                return parsed["use_retrieval"]
        except json.JSONDecodeError:
            pass

    lowered = text.lower()
    if '"use_retrieval": false' in lowered:
        return False
    if '"use_retrieval": true' in lowered:
        return True

    return True


@dataclass
class RouterStep:
    prompt_store: PromptStorePort
    llm: TextGeneratorPort
    decision_policy: RouterDecisionPolicyPort = field(default_factory=ConservativeRouterDecisionPolicy)

    def execute(self, state: RagWorkflowState) -> RagWorkflowState:
        logger.info("RouterStep: analyzing user input to decide if retrieval is needed.")
        prompt = self.prompt_store.render(
            "router",
            conversation_context=state.get("conversation_context", ""),
            user_input=state["user_input"],
        )
        raw_output = self.llm.generate(prompt, temperature=0.0, max_new_tokens=64)
        model_decision = _parse_router_output(raw_output)
        use_retrieval = self.decision_policy.decide(
            user_input=state["user_input"],
            conversation_context=state.get("conversation_context", ""),
            model_decision=model_decision,
        )
        if use_retrieval != model_decision:
            logger.info(
                "RouterStep decision overridden by policy: model=%s final=%s",
                model_decision,
                use_retrieval,
            )
        logger.info("RouterStep decision: use_retrieval=%s", use_retrieval)
        return {"use_retrieval": use_retrieval}


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
        retrieved_documents = self.tool.retrieve(search_query)
        logger.info("RetrieverStep: retrieved %d documents.", len(retrieved_documents))
        return {
            "search_query": search_query,
            "retrieved_documents": retrieved_documents,
        }


@dataclass
class ResponderStep:
    prompt_store: PromptStorePort
    llm: TextGeneratorPort

    def execute(self, state: RagWorkflowState) -> RagWorkflowState:
        docs = state.get("retrieved_documents", [])
        logger.info("ResponderStep: generating final response using %d documents.", len(docs))
        docs_text = self._format_docs(docs)
        prompt = self.prompt_store.render(
            "responder",
            conversation_context=state.get("conversation_context", ""),
            retrieved_documents=docs_text,
            user_input=state["user_input"],
        )
        response = self.llm.generate(prompt, temperature=0.2, max_new_tokens=300).strip()
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


class RagWorkflowOrchestrator:
    def __init__(
        self,
        *,
        router: RouterStep,
        retriever: RetrieverStep,
        responder: ResponderStep,
        prefer_langgraph: bool = True,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        self._router = router
        self._retriever = retriever
        self._responder = responder
        self._logger = logger_instance or logger
        self._compiled_graph = self._build_graph() if prefer_langgraph else None

    def run(self, *, user_input: str, conversation_context: str = "") -> RagWorkflowState:
        self._logger.info("Starting RagWorkflow run for user input: '%s'", user_input[:50] + "..." if len(user_input) > 50 else user_input)
        initial_state: RagWorkflowState = {
            "user_input": user_input,
            "conversation_context": conversation_context,
            "retrieved_documents": [],
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
        return bool(output["use_retrieval"])

    def _run_without_graph(self, state: RagWorkflowState) -> RagWorkflowState:
        router_state = self._router.execute(state)
        state.update(router_state)
        if state.get("use_retrieval"):
            retrieval_state = self._retriever.execute(state)
            state.update(retrieval_state)
        response_state = self._responder.execute(state)
        state.update(response_state)
        return state

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph
            self._logger.info("Successfully imported LangGraph. Building StateGraph.")
        except Exception:  # noqa: BLE001
            self._logger.warning("LangGraph is not available. Falling back to local workflow executor.")
            return None

        graph = StateGraph(RagWorkflowState)
        graph.add_node("router", self._router.execute)
        graph.add_node("retriever", self._retriever.execute)
        graph.add_node("responder", self._responder.execute)

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "retriever": "retriever",
                "responder": "responder",
            },
        )
        graph.add_edge("retriever", "responder")
        graph.set_finish_point("responder")
        self._logger.info("LangGraph workflow compiled successfully.")
        return graph.compile()

    @staticmethod
    def _route_after_router(state: RagWorkflowState) -> str:
        if state.get("use_retrieval"):
            logger.info("Graph routing: decision 'retriever' based on use_retrieval=True")
            return "retriever"
        logger.info("Graph routing: decision 'responder' based on use_retrieval=False")
        return "responder"
