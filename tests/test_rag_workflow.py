import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from infrastructure.rag_workflow import MemoryStep, FilesystemStep, RagWorkflowOrchestrator, ResponderStep, RetrieverStep, RouterStep


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
                f"{variables['memory_context']}::"
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


class _MemoryTool:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.open_calls = []
        self.write_calls = []
        self.created_entities = []
        self.created_relations = []

    def search_nodes(self, query: str):
        self.calls.append(query)
        return dict(self.payload)

    def open_nodes(self, names: list[str]):
        self.open_calls.append(list(names))
        return dict(self.payload)

    def add_observations(self, entity_name: str, contents: list[str]):
        self.write_calls.append((entity_name, list(contents)))
        return {
            "content": [{"type": "text", "text": f"Added {len(contents)} observations."}],
            "structuredContent": {"added": len(contents)},
        }

    def create_entities(self, entities: list[dict]):
        self.created_entities.append(list(entities))
        return {"structuredContent": {"created": len(entities), "entities": entities}}

    def create_relations(self, relations: list[dict]):
        self.created_relations.append(list(relations))
        return {"structuredContent": {"created": len(relations), "relations": relations}}


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

    def test_router_step_keeps_memory_without_forcing_retrieval_when_context_exists(self):
        step = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(['{"tools":["memory"],"tool_inputs":{"memory":{"operation":"search_nodes","query":"captain elara"}}}']),
        )

        payload = step.execute(
            {
                "user_input": "remember and memorize captain elara details",
                "conversation_context": "user: remember and memorize captain elara details",
            }
        )

        self.assertEqual(payload["tools"], ["memory"])
        self.assertFalse(payload["use_retrieval"])
        self.assertTrue(payload["use_memory"])

    def test_router_step_detects_memory_intent_even_with_rmember_typo(self):
        step = RouterStep(prompt_store=_PromptStore(), llm=_LLM(['{"tools":[]}']))

        payload = step.execute({"user_input": "Rmember and memorize this lore for later"})

        self.assertIn("memory", payload["tools"])
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

    def test_memory_step_builds_context_from_entities(self):
        step = MemoryStep(tool=_MemoryTool({"entities": [{"name": "Alice", "entityType": "person", "observations": ["prefers tea"]}]}))

        state = step.execute({"user_input": "o que voce sabe sobre alice?", "tool_inputs": {"memory": {"query": "alice"}}})

        self.assertEqual(state["memory_query"], "alice")
        self.assertIn("entity: Alice (person)", state["memory_context"])

    def test_orchestrator_runs_memory_tool(self):
        router = RouterStep(prompt_store=_PromptStore(), llm=_LLM(['{"tools":["memory"],"tool_inputs":{"memory":{"operation":"search_nodes","query":"alice"}}}']))
        retriever_tool = _RetrieveTool([{"content": "doc", "source": "manual.pdf"}])
        retriever = RetrieverStep(prompt_store=_PromptStore(), llm=_LLM(["query-doc"]), tool=retriever_tool)
        responder = ResponderStep(prompt_store=_PromptStore(), llm=_LLM(["resposta com memoria"]))
        memory_tool = _MemoryTool({"entities": [{"name": "Alice", "entityType": "person", "observations": ["likes coffee"]}]})
        memory = MemoryStep(tool=memory_tool)
        orchestrator = RagWorkflowOrchestrator(
            router=router,
            retriever=retriever,
            responder=responder,
            memory=memory,
            prefer_langgraph=False,
        )

        state = orchestrator.run(user_input="o que voce lembra da alice?", conversation_context="")

        self.assertEqual(memory_tool.calls[0], "alice")
        self.assertEqual(state["response"], "resposta com memoria")
        self.assertIn("Alice", state["memory_context"])

    def test_memory_step_adds_observations_when_operation_is_add_observations(self):
        memory_tool = _MemoryTool({})
        step = MemoryStep(tool=memory_tool)

        state = step.execute(
            {
                "user_input": 'Remember and memorize: "Captain Elara leads Silver Hawks"',
                "tool_inputs": {
                    "memory": {
                        "operation": "add_observations",
                        "entity_name": "session_memory",
                        "content": "Captain Elara leads Silver Hawks",
                    }
                },
            }
        )

        self.assertEqual(memory_tool.write_calls[0], ("session_memory", ["Captain Elara leads Silver Hawks"]))
        self.assertIn("Added 1 observations.", state["memory_context"])

    def test_memory_step_builds_graph_entities_and_relations_when_extractor_returns_structured_data(self):
        memory_tool = _MemoryTool({})
        step = MemoryStep(
            tool=memory_tool,
            operation_planner=lambda user_input, conversation_context: {
                "operation": "store_graph",
                "content": "Captain Elara leads Silver Hawks",
            },
            graph_builder=lambda content, context: {
                "entities": [
                    {"name": "Captain Elara", "entityType": "person", "observations": ["Leads Silver Hawks"]},
                    {"name": "Silver Hawks", "entityType": "organization", "observations": ["Elite scouting company"]},
                ],
                "relations": [{"from": "Captain Elara", "to": "Silver Hawks", "relationType": "leads"}],
            },
        )

        state = step.execute(
            {
                "user_input": 'Remember and memorize: "Captain Elara leads Silver Hawks"',
                "conversation_context": "ctx",
                "tool_inputs": {
                    "memory": {
                        "operation": "add_observations",
                        "content": "Captain Elara leads Silver Hawks",
                    }
                },
            }
        )

        self.assertEqual(len(memory_tool.created_entities), 1)
        self.assertEqual(len(memory_tool.created_entities[0]), 2)
        self.assertEqual(memory_tool.created_relations[0][0]["relationType"], "leads")
        self.assertEqual(memory_tool.write_calls, [])
        self.assertIn("memory graph updated: entities=2 relations=1.", state["memory_context"])

    def test_memory_step_filters_relations_without_known_entities(self):
        memory_tool = _MemoryTool({})
        step = MemoryStep(
            tool=memory_tool,
            operation_planner=lambda user_input, conversation_context: {
                "operation": "store_graph",
                "content": "Captain Elara leads Silver Hawks",
            },
            graph_builder=lambda content, context: {
                "entities": [
                    {"name": "Captain Elara", "entityType": "person", "observations": ["Leads Silver Hawks"]},
                ],
                "relations": [
                    {"from": "Captain Elara", "to": "Silver Hawks", "relationType": "leads"},
                ],
            },
        )

        state = step.execute({"user_input": "remember this", "conversation_context": "ctx"})

        self.assertEqual(len(memory_tool.created_entities), 1)
        self.assertEqual(memory_tool.created_relations, [])
        self.assertIn("memory graph updated: entities=1 relations=0.", state["memory_context"])

    def test_memory_step_falls_back_to_inferred_operation_when_planner_is_invalid(self):
        memory_tool = _MemoryTool({})
        step = MemoryStep(
            tool=memory_tool,
            operation_planner=lambda user_input, conversation_context: {
                "operation": "invalid_operation",
            },
        )

        state = step.execute({"user_input": "remember this fact please", "conversation_context": ""})

        self.assertEqual(memory_tool.calls, [])
        self.assertEqual(memory_tool.created_entities, [])
        self.assertEqual(memory_tool.write_calls[0][0], "session_memory")
        self.assertIn("Added 1 observations.", state["memory_context"])

    def test_memory_step_uses_operation_planner_for_open_nodes(self):
        memory_tool = _MemoryTool({"entities": [{"name": "Captain Elara", "entityType": "person", "observations": []}]})
        step = MemoryStep(
            tool=memory_tool,
            operation_planner=lambda user_input, conversation_context: {
                "operation": "open_nodes",
                "names": ["Captain Elara"],
            },
        )

        state = step.execute({"user_input": "open memory for Elara", "conversation_context": "ctx"})

        self.assertEqual(memory_tool.open_calls[0], ["Captain Elara"])
        self.assertIn("Captain Elara", state["memory_context"])

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
