import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from infrastructure.rag_workflow import FilesystemStep, RagWorkflowOrchestrator, ResponderStep, RetrieverStep, RouterStep


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
                f"{variables['retrieved_documents']}::"
                f"{variables['filesystem_context']}::"
                f"{variables['tool_errors']}::"
                f"{variables['user_input']}"
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


class _FilesystemTool:
    def __init__(self, entries):
        self._entries = entries
        self.calls = 0
        self.last_path = None
        self.write_calls = []
        self.read_calls = []
        self.delete_calls = []

    def list_directory(self, path="."):
        self.calls += 1
        self.last_path = path
        return list(self._entries)

    def read_file(self, path):
        self.read_calls.append(path)
        return "file content"

    def write_file(self, path, content):
        self.write_calls.append((path, content))
        return f"File written: {path}"

    def delete_file(self, path):
        self.delete_calls.append(path)
        return f"File deleted: {path}"


class RagWorkflowTests(unittest.TestCase):
    def test_router_step_parses_tools_contract(self):
        step = RouterStep(prompt_store=_PromptStore(), llm=_LLM(['{"tools":["retrieval","filesystem"]}']))

        payload = step.execute({"user_input": "liste arquivos e compare com manual"})

        self.assertTrue(payload["use_retrieval"])
        self.assertTrue(payload["use_filesystem"])
        self.assertEqual(payload["tools"], ["retrieval", "filesystem"])

    def test_router_step_parses_legacy_contract(self):
        step = RouterStep(prompt_store=_PromptStore(), llm=_LLM(['{"use_retrieval": false}']))

        payload = step.execute({"user_input": "oi"})

        self.assertFalse(payload["use_retrieval"])
        self.assertEqual(payload["tools"], [])

    def test_router_step_defaults_to_retrieval_on_invalid_output(self):
        step = RouterStep(prompt_store=_PromptStore(), llm=_LLM(["not-json"]))

        payload = step.execute({"user_input": "o que diz o contrato?"})

        self.assertTrue(payload["use_retrieval"])

    def test_router_step_keeps_filesystem_only_for_create_file_request(self):
        step = RouterStep(prompt_store=_PromptStore(), llm=_LLM(['{"tools":["filesystem"],"tool_inputs":{"filesystem":{"operation":"write_file","path":"hello_world.csv","content":"hello world"}}}']))

        payload = step.execute({"user_input": "crie arquivo hello_world.csv com hello world"})

        self.assertEqual(payload["tools"], ["filesystem"])
        self.assertFalse(payload["use_retrieval"])

    def test_retriever_step_generates_query_and_calls_tool(self):
        llm = _LLM(["contrato fornecimento 2024"])
        tool = _RetrieveTool([{"content": "doc1", "source": "file1.pdf"}])
        step = RetrieverStep(prompt_store=_PromptStore(), llm=llm, tool=tool)

        state = step.execute({"user_input": "o que diz o contrato?", "conversation_context": "empresa abc"})

        self.assertEqual(state["search_query"], "contrato fornecimento 2024")
        self.assertEqual(tool.last_query, "contrato fornecimento 2024")
        self.assertEqual(state["retrieved_documents"][0]["source"], "file1.pdf")

    def test_filesystem_step_uses_default_path_when_missing(self):
        tool = _FilesystemTool([{"name": "manual.pdf", "is_directory": False}])
        step = FilesystemStep(tool=tool)

        state = step.execute({"user_input": "liste arquivos", "tool_inputs": {}})

        self.assertEqual(state["filesystem_path"], ".")
        self.assertEqual(tool.last_path, ".")
        self.assertEqual(state["filesystem_entries"][0]["name"], "manual.pdf")

    def test_filesystem_step_writes_file_when_operation_is_write_file(self):
        tool = _FilesystemTool([])
        step = FilesystemStep(tool=tool)

        state = step.execute(
            {
                "user_input": "crie arquivo",
                "tool_inputs": {"filesystem": {"operation": "write_file", "path": "hello_world.csv", "content": "hello world"}},
            }
        )

        self.assertEqual(tool.write_calls[0], ("hello_world.csv", "hello world"))
        self.assertEqual(state["filesystem_operation"], "write_file")
        self.assertIn("File written", state["filesystem_result"])

    def test_filesystem_step_infers_path_and_content_from_user_input(self):
        tool = _FilesystemTool([])
        step = FilesystemStep(tool=tool)

        state = step.execute(
            {
                "user_input": 'crie um arquivo chamado hello_world.csv com o texto "hello world"',
                "tool_inputs": {"filesystem": {}},
            }
        )

        self.assertEqual(tool.write_calls[0], ("hello_world.csv", "hello world"))
        self.assertEqual(state["filesystem_path"], "hello_world.csv")

    def test_filesystem_step_deletes_file_when_operation_is_delete_file(self):
        tool = _FilesystemTool([])
        step = FilesystemStep(tool=tool)

        state = step.execute(
            {
                "user_input": "delete o arquivo hello_world.csv",
                "tool_inputs": {"filesystem": {"operation": "delete_file", "path": "hello_world.csv"}},
            }
        )

        self.assertEqual(tool.delete_calls[0], "hello_world.csv")
        self.assertEqual(state["filesystem_operation"], "delete_file")
        self.assertIn("File deleted", state["filesystem_result"])

    def test_filesystem_step_yes_followup_uses_context_to_delete(self):
        tool = _FilesystemTool([])
        step = FilesystemStep(tool=tool)

        state = step.execute(
            {
                "user_input": "yes",
                "conversation_context": "assistant: It seems you want to delete the file hello_world.csv.",
                "tool_inputs": {"filesystem": {}},
            }
        )

        self.assertEqual(tool.delete_calls[0], "hello_world.csv")
        self.assertEqual(state["filesystem_operation"], "delete_file")

    def test_orchestrator_runs_retrieval_and_filesystem_tools(self):
        router = RouterStep(prompt_store=_PromptStore(), llm=_LLM(['{"tools":["retrieval","filesystem"]}']))
        retriever_tool = _RetrieveTool([{"content": "doc", "source": "manual.pdf"}])
        fs_tool = _FilesystemTool([{"name": "manual.pdf", "is_directory": False}])
        retriever = RetrieverStep(prompt_store=_PromptStore(), llm=_LLM(["query-doc"]), tool=retriever_tool)
        filesystem = FilesystemStep(tool=fs_tool)
        responder = ResponderStep(prompt_store=_PromptStore(), llm=_LLM(["resposta com docs e arquivos"]))
        orchestrator = RagWorkflowOrchestrator(
            router=router,
            retriever=retriever,
            filesystem=filesystem,
            responder=responder,
            prefer_langgraph=False,
        )

        state = orchestrator.run(user_input="liste os arquivos e compare com manual", conversation_context="")

        self.assertTrue(state["use_retrieval"])
        self.assertEqual(retriever_tool.calls, 1)
        self.assertEqual(fs_tool.calls, 1)
        self.assertEqual(state["search_query"], "query-doc")
        self.assertEqual(state["filesystem_entries"][0]["name"], "manual.pdf")
        self.assertEqual(state["response"], "resposta com docs e arquivos")

    def test_orchestrator_skips_tools_when_router_returns_empty(self):
        router = RouterStep(prompt_store=_PromptStore(), llm=_LLM(['{"tools":[]}']))
        retriever_tool = _RetrieveTool([{"content": "doc", "source": "s"}])
        fs_tool = _FilesystemTool([{"name": "a.txt", "is_directory": False}])
        retriever = RetrieverStep(prompt_store=_PromptStore(), llm=_LLM(["query"]), tool=retriever_tool)
        filesystem = FilesystemStep(tool=fs_tool)
        responder = ResponderStep(prompt_store=_PromptStore(), llm=_LLM(["resposta final"]))
        orchestrator = RagWorkflowOrchestrator(
            router=router,
            retriever=retriever,
            filesystem=filesystem,
            responder=responder,
            prefer_langgraph=False,
        )

        state = orchestrator.run(user_input="qual a capital da frança?", conversation_context="")

        self.assertFalse(state["use_retrieval"])
        self.assertEqual(retriever_tool.calls, 0)
        self.assertEqual(fs_tool.calls, 0)
        self.assertEqual(state["response"], "resposta final")

    def test_orchestrator_supports_langgraph_mode(self):
        router = RouterStep(prompt_store=_PromptStore(), llm=_LLM(['{"tools":["retrieval"]}']))
        retriever_tool = _RetrieveTool([{"content": "doc", "source": "manual.pdf"}])
        retriever = RetrieverStep(prompt_store=_PromptStore(), llm=_LLM(["query-doc"]), tool=retriever_tool)
        responder = ResponderStep(prompt_store=_PromptStore(), llm=_LLM(["resposta com grafo"]))
        orchestrator = RagWorkflowOrchestrator(
            router=router,
            retriever=retriever,
            responder=responder,
            prefer_langgraph=True,
        )

        state = orchestrator.run(user_input="o que diz o manual?", conversation_context="")

        self.assertEqual(state["response"], "resposta com grafo")
        self.assertEqual(retriever_tool.calls, 1)


if __name__ == "__main__":
    unittest.main()
