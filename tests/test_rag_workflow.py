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
            return (
                f"RETRIEVER::{variables['conversation_context']}::"
                f"{variables['user_input']}"
            )
        if prompt_name == "responder":
            return (
                f"RESPONDER::{variables['conversation_context']}::"
                f"{variables['retrieved_documents']}::{variables['user_input']}"
            )
        raise KeyError(prompt_name)


class _LLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.prompts = []

    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        del temperature
        del max_new_tokens
        self.prompts.append(prompt)
        if not self._outputs:
            raise RuntimeError("No outputs left")
        return self._outputs.pop(0)


class _RetrieveTool:
    def __init__(self, docs):
        self._docs = docs
        self.last_query = None
        self.calls = 0

    def retrieve(self, search_query: str):
        self.calls += 1
        self.last_query = search_query
        return list(self._docs)


class RagWorkflowTests(unittest.TestCase):
    def test_router_step_parses_json_output(self):
        step = RouterStep(prompt_store=_PromptStore(), llm=_LLM(['{"use_retrieval": false}']))

        payload = step.execute({"user_input": "oi"})

        self.assertFalse(payload["use_retrieval"])

    def test_router_step_defaults_to_retrieval_on_invalid_output(self):
        step = RouterStep(prompt_store=_PromptStore(), llm=_LLM(["not-json"]))

        payload = step.execute({"user_input": "oi"})

        self.assertTrue(payload["use_retrieval"])

    def test_router_step_overrides_false_for_document_like_question(self):
        step = RouterStep(prompt_store=_PromptStore(), llm=_LLM(['{"use_retrieval": false}']))

        payload = step.execute({"user_input": "quem comprou a tv?", "conversation_context": ""})

        self.assertTrue(payload["use_retrieval"])

    def test_retriever_step_generates_query_and_calls_tool(self):
        llm = _LLM(["contrato fornecimento 2024"])
        tool = _RetrieveTool([{"content": "doc1", "source": "file1.pdf"}])
        step = RetrieverStep(prompt_store=_PromptStore(), llm=llm, tool=tool)

        state = step.execute({"user_input": "o que diz o contrato?", "conversation_context": "empresa abc"})

        self.assertEqual(state["search_query"], "contrato fornecimento 2024")
        self.assertEqual(tool.last_query, "contrato fornecimento 2024")
        self.assertEqual(state["retrieved_documents"][0]["source"], "file1.pdf")

    def test_retriever_step_falls_back_to_user_input_when_query_is_empty(self):
        llm = _LLM(["   "])
        tool = _RetrieveTool([{"content": "doc1", "source": "file1.pdf"}])
        step = RetrieverStep(prompt_store=_PromptStore(), llm=llm, tool=tool)

        state = step.execute({"user_input": "quem comprou a tv?", "conversation_context": ""})

        self.assertEqual(state["search_query"], "quem comprou a tv?")
        self.assertEqual(tool.last_query, "quem comprou a tv?")

    def test_orchestrator_skips_retriever_when_router_false(self):
        router = RouterStep(prompt_store=_PromptStore(), llm=_LLM(['{"use_retrieval": false}']))
        retriever_tool = _RetrieveTool([{"content": "doc", "source": "s"}])
        retriever = RetrieverStep(prompt_store=_PromptStore(), llm=_LLM(["query"]), tool=retriever_tool)
        responder = ResponderStep(prompt_store=_PromptStore(), llm=_LLM(["resposta final"]))
        orchestrator = RagWorkflowOrchestrator(
            router=router,
            retriever=retriever,
            responder=responder,
            prefer_langgraph=False,
        )

        state = orchestrator.run(user_input="qual a capital da frança?", conversation_context="")

        self.assertFalse(state["use_retrieval"])
        self.assertEqual(retriever_tool.calls, 0)
        self.assertEqual(state["response"], "resposta final")

    def test_orchestrator_runs_retriever_when_router_true(self):
        router = RouterStep(prompt_store=_PromptStore(), llm=_LLM(['{"use_retrieval": true}']))
        retriever_tool = _RetrieveTool([{"content": "doc", "source": "s"}])
        retriever = RetrieverStep(prompt_store=_PromptStore(), llm=_LLM(["query-doc"]), tool=retriever_tool)
        responder = ResponderStep(prompt_store=_PromptStore(), llm=_LLM(["resposta com docs"]))
        orchestrator = RagWorkflowOrchestrator(
            router=router,
            retriever=retriever,
            responder=responder,
            prefer_langgraph=False,
        )

        state = orchestrator.run(user_input="o que diz o pdf?", conversation_context="arquivo contrato")

        self.assertTrue(state["use_retrieval"])
        self.assertEqual(retriever_tool.calls, 1)
        self.assertEqual(state["search_query"], "query-doc")
        self.assertEqual(state["response"], "resposta com docs")


if __name__ == "__main__":
    unittest.main()
