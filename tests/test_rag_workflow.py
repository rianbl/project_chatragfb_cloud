import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from infrastructure.rag_workflow import RagWorkflowOrchestrator, ResponderStep, RetrieverStep, RouterStep


class _PromptStore:
    def render(self, prompt_name: str, **variables) -> str:
        if prompt_name == "router":
            return f"ROUTER::{variables['user_input']}"
        if prompt_name == "retriever":
            return f"RETRIEVER::{variables['user_input']}"
        if prompt_name == "responder":
            return "RESPONDER"
        raise KeyError(prompt_name)


class _LLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        del prompt
        del temperature
        del max_new_tokens
        if not self._outputs:
            raise RuntimeError("No mock output available")
        return self._outputs.pop(0)


class _RetrieveTool:
    def __init__(self, docs):
        self._docs = docs
        self.calls = []

    def retrieve(self, search_query: str):
        self.calls.append(search_query)
        return list(self._docs)


class RagWorkflowTests(unittest.TestCase):
    def test_router_forces_retrieval_for_semantic_context_question(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(['{"tool_calls":[]}']),
        )
        state = router.execute(
            {
                "user_input": "resuma o documento wikipedia.txt",
                "available_tools": [{"name": "retrieval", "description": "", "inputSchema": {}}],
            }
        )

        self.assertEqual(state["tool_calls"][0]["name"], "retrieval")

    def test_router_drops_read_file_when_not_surgical(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(['{"tool_calls":[{"name":"retrieval","arguments":{}},{"name":"filesystem.read_file","arguments":{"path":"wikipedia.txt"}}]}']),
        )
        state = router.execute(
            {
                "user_input": "compare o conteúdo com o contexto",
                "available_tools": [
                    {"name": "retrieval", "description": "", "inputSchema": {}},
                    {"name": "filesystem.read_file", "description": "", "inputSchema": {}},
                ],
            }
        )

        names = [call["name"] for call in state["tool_calls"]]
        self.assertIn("retrieval", names)
        self.assertNotIn("filesystem.read_file", names)

    def test_orchestrator_executes_retrieval_and_tool_call(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(['{"tool_calls":[{"name":"retrieval","arguments":{"query":"wikipedia"}},{"name":"filesystem.list_directory","arguments":{"path":"."}}]}']),
        )
        retriever_tool = _RetrieveTool([{"content": "x", "source": "wikipedia.txt"}])
        retriever = RetrieverStep(prompt_store=_PromptStore(), llm=_LLM(["query-normalized"]), tool=retriever_tool)
        responder = ResponderStep(prompt_store=_PromptStore(), llm=_LLM(["ok"]))

        executed = []

        def _mcp_executor(tool_name, arguments):
            executed.append((tool_name, arguments))
            return {"ok": True}

        orchestrator = RagWorkflowOrchestrator(
            router=router,
            retriever=retriever,
            responder=responder,
            available_tools_provider=lambda: [
                {"name": "retrieval", "description": "", "inputSchema": {}},
                {"name": "filesystem.list_directory", "description": "", "inputSchema": {}},
            ],
            mcp_tool_executor=_mcp_executor,
        )

        state = orchestrator.run(user_input="liste e compare", conversation_context="")

        self.assertEqual(state["response"], "ok")
        self.assertEqual(retriever_tool.calls[0], "query-normalized")
        self.assertEqual(executed[0][0], "filesystem.list_directory")


if __name__ == "__main__":
    unittest.main()
