import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from infrastructure.rag_workflow import (
    RagWorkflowOrchestrator,
    ResponderStep,
    RetrieverStep,
    RouterStep,
    _merge_memory_graphs,
    _sanitize_memory_graph,
    _split_text_chunks,
    _parse_tool_calls,
)


class _PromptStore:
    def render(self, prompt_name: str, **variables) -> str:
        if prompt_name == "router":
            return f"ROUTER::{variables['user_input']}"
        if prompt_name == "retriever":
            return f"RETRIEVER::{variables['user_input']}"
        if prompt_name == "responder":
            return "RESPONDER"
        if prompt_name == "memory_graph_builder":
            return f"MEMORY::{variables['text']}"
        if prompt_name == "memory_text_normalizer":
            return f"NORMALIZE::{variables['text']}"
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
    def test_split_text_chunks_splits_long_text(self):
        text = ("alpha " * 900).strip()
        chunks = _split_text_chunks(text, max_chars=500)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))

    def test_sanitize_memory_graph_keeps_only_valid_items(self):
        payload = {
            "entities": [
                {"name": "Alice", "entityType": "person", "observations": ["likes tea"]},
                {"name": "", "entityType": "person", "observations": ["invalid"]},
            ],
            "relations": [
                {"from": "Alice", "to": "Bob", "relationType": "knows"},
                {"from": "Alice", "to": "Alice", "relationType": "self"},
            ],
        }
        sanitized = _sanitize_memory_graph(payload)
        self.assertEqual(len(sanitized["entities"]), 1)
        self.assertEqual(len(sanitized["relations"]), 1)

    def test_merge_memory_graphs_deduplicates_entities_and_relations(self):
        merged = _merge_memory_graphs(
            [
                {
                    "entities": [{"name": "Alice", "entityType": "unknown", "observations": ["obs1"]}],
                    "relations": [],
                },
                {
                    "entities": [{"name": "Alice", "entityType": "person", "observations": ["obs2"]}],
                    "relations": [{"from": "Alice", "to": "Alice", "relationType": "self"}],
                },
            ]
        )
        self.assertEqual(len(merged["entities"]), 1)
        self.assertEqual(merged["entities"][0]["entityType"], "person")
        self.assertIn("obs1", merged["entities"][0]["observations"])
        self.assertIn("obs2", merged["entities"][0]["observations"])
        self.assertEqual(len(merged["relations"]), 1)

    def test_parse_tool_calls_does_not_infer_from_plain_text_mentions(self):
        parsed = _parse_tool_calls(
            "No tool is needed. retrieval would be optional.",
            available_names={"retrieval", "filesystem.read_file"},
        )
        self.assertEqual(parsed, [])

    def test_parse_tool_calls_allows_same_tool_with_distinct_arguments(self):
        raw = (
            '{"tool_calls":['
            '{"name":"filesystem.read_file","arguments":{"path":"a.txt"}},'
            '{"name":"filesystem.read_file","arguments":{"path":"b.txt"}}'
            "]}"
        )
        parsed = _parse_tool_calls(raw, available_names={"filesystem.read_file", "retrieval"})

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["name"], "filesystem.read_file")
        self.assertEqual(parsed[0]["arguments"]["path"], "a.txt")
        self.assertEqual(parsed[1]["arguments"]["path"], "b.txt")

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
            llm=_LLM(
                [
                    '{"tool_calls":[{"name":"retrieval","arguments":{}},{"name":"filesystem.read_file","arguments":{"path":"wikipedia.txt"}}]}'
                ]
            ),
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

    def test_router_keeps_read_file_for_direct_file_open_request(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(
                ['{"tool_calls":[{"name":"filesystem.read_file","arguments":{"path":"README.md"}}]}']
            ),
        )
        state = router.execute(
            {
                "user_input": "abra o arquivo README.md",
                "available_tools": [
                    {"name": "retrieval", "description": "", "inputSchema": {}},
                    {"name": "filesystem.read_file", "description": "", "inputSchema": {}},
                ],
            }
        )

        names = [call["name"] for call in state["tool_calls"]]
        self.assertIn("filesystem.read_file", names)
        self.assertNotIn("retrieval", names)

    def test_router_replaces_non_surgical_read_with_retrieval_for_semantic_query(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(
                ['{"tool_calls":[{"name":"filesystem.read_file","arguments":{"path":"README.md"}}]}']
            ),
        )
        state = router.execute(
            {
                "user_input": "compare o conteúdo do README.md com o contexto",
                "available_tools": [
                    {"name": "retrieval", "description": "", "inputSchema": {}},
                    {"name": "filesystem.read_file", "description": "", "inputSchema": {}},
                ],
            }
        )

        names = [call["name"] for call in state["tool_calls"]]
        self.assertIn("retrieval", names)
        self.assertNotIn("filesystem.read_file", names)

    def test_router_removes_retrieval_for_operational_list_directory_request(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(
                [
                    '{"tool_calls":[{"name":"filesystem.list_directory","arguments":{"path":"uploads"}},{"name":"retrieval","arguments":{}}]}'
                ]
            ),
        )
        state = router.execute(
            {
                "user_input": "liste os arquivos em uploads",
                "available_tools": [
                    {"name": "retrieval", "description": "", "inputSchema": {}},
                    {"name": "filesystem.list_directory", "description": "", "inputSchema": {}},
                ],
            }
        )

        names = [call["name"] for call in state["tool_calls"]]
        self.assertIn("filesystem.list_directory", names)
        self.assertNotIn("retrieval", names)

    def test_router_does_not_force_memory_tool_when_llm_returns_empty_calls(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(['{"tool_calls":[]}']),
        )
        state = router.execute(
            {
                "user_input": 'Remember this: "The Rozvi Empire was a Shona state."',
                "available_tools": [
                    {"name": "retrieval", "description": "", "inputSchema": {}},
                    {"name": "memory.add_observations", "description": "", "inputSchema": {}},
                ],
            }
        )

        self.assertEqual(state["tool_calls"], [])

    def test_router_hides_memory_graph_write_tools_from_manifest(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(['{"tool_calls":[{"name":"memory.add_observations","arguments":{}}]}']),
        )
        state = router.execute(
            {
                "user_input": "remember this fact",
                "available_tools": [
                    {"name": "retrieval", "description": "", "inputSchema": {}},
                    {"name": "memory.add_observations", "description": "", "inputSchema": {}},
                    {"name": "memory.create_entities", "description": "", "inputSchema": {}},
                ],
            }
        )

        self.assertEqual(state["tool_calls"], [])

    def test_router_accepts_memory_graph_upsert_tool(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(['{"tool_calls":[{"name":"memory.graph_upsert","arguments":{"text":"x"}}]}']),
        )
        state = router.execute(
            {
                "user_input": "remember this fact",
                "available_tools": [
                    {"name": "retrieval", "description": "", "inputSchema": {}},
                    {"name": "memory.add_observations", "description": "", "inputSchema": {}},
                ],
            }
        )

        self.assertEqual(state["tool_calls"][0]["name"], "memory.graph_upsert")
        self.assertEqual(state["tool_calls"][0]["arguments"]["text"], "x")

    def test_orchestrator_executes_retrieval_and_tool_call(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(
                [
                    '{"tool_calls":[{"name":"retrieval","arguments":{"query":"wikipedia"}},{"name":"filesystem.list_directory","arguments":{"path":"."}}]}'
                ]
            ),
        )
        retriever_tool = _RetrieveTool([{"content": "x", "source": "wikipedia.txt"}])
        retriever = RetrieverStep(
            prompt_store=_PromptStore(), llm=_LLM(["query-normalized"]), tool=retriever_tool
        )
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

    def test_orchestrator_executes_memory_graph_upsert_via_graph_builder(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(
                [
                    '{"tool_calls":[{"name":"memory.graph_upsert","arguments":{"text":"Alice likes tea"}}]}'
                ]
            ),
        )
        retriever = RetrieverStep(
            prompt_store=_PromptStore(), llm=_LLM(["unused"]), tool=_RetrieveTool([])
        )
        responder = ResponderStep(prompt_store=_PromptStore(), llm=_LLM(["ok"]))

        executed = []

        def _mcp_executor(tool_name, arguments):
            executed.append((tool_name, arguments))
            if tool_name == "memory.read_graph":
                return {
                    "ok": True,
                    "data": {
                        "entities": [
                            {
                                "name": "alice",
                                "entityType": "person",
                                "observations": ["likes tea"],
                            }
                        ],
                        "relations": [],
                    },
                }
            return {"ok": True}

        orchestrator = RagWorkflowOrchestrator(
            router=router,
            retriever=retriever,
            responder=responder,
            available_tools_provider=lambda: [
                {"name": "retrieval", "description": "", "inputSchema": {}},
                {"name": "memory.add_observations", "description": "", "inputSchema": {}},
                {"name": "memory.create_entities", "description": "", "inputSchema": {}},
                {"name": "memory.create_relations", "description": "", "inputSchema": {}},
            ],
            mcp_tool_executor=_mcp_executor,
            graph_builder_llm=_LLM(
                [
                    "Alice likes tea",
                    '{"entities":[{"name":"alice","entityType":"person","observations":["likes tea"]}],"relations":[]}'
                ]
            ),
            prompt_store=_PromptStore(),
        )

        state = orchestrator.run(user_input="salve isso na memoria", conversation_context="")

        self.assertEqual(state["response"], "ok")
        called_names = [name for name, _args in executed]
        self.assertNotIn("memory.graph_upsert", called_names)
        self.assertIn("memory.create_entities", called_names)
        self.assertIn("memory.read_graph", called_names)
        self.assertNotIn("memory.add_observations", called_names)

    def test_memory_upsert_uses_raw_text_when_normalizer_output_looks_truncated(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(['{"tool_calls":[{"name":"memory.graph_upsert","arguments":{}}]}']),
        )
        retriever = RetrieverStep(
            prompt_store=_PromptStore(), llm=_LLM(["unused"]), tool=_RetrieveTool([])
        )
        responder = ResponderStep(prompt_store=_PromptStore(), llm=_LLM(["ok"]))

        executed = []

        def _mcp_executor(tool_name, arguments):
            executed.append((tool_name, arguments))
            if tool_name == "memory.read_graph":
                return {"ok": True, "data": {"entities": [], "relations": []}}
            return {"ok": True}

        long_text = "Memorize this: " + ("The Rozvi Empire was powerful. " * 40)

        orchestrator = RagWorkflowOrchestrator(
            router=router,
            retriever=retriever,
            responder=responder,
            available_tools_provider=lambda: [
                {"name": "retrieval", "description": "", "inputSchema": {}},
                {"name": "memory.add_observations", "description": "", "inputSchema": {}},
                {"name": "memory.create_entities", "description": "", "inputSchema": {}},
                {"name": "memory.create_relations", "description": "", "inputSchema": {}},
                {"name": "memory.read_graph", "description": "", "inputSchema": {}},
            ],
            mcp_tool_executor=_mcp_executor,
            graph_builder_llm=_LLM(
                [
                    "The Rozvi Empire was powerful.",
                    '{"entities":[],"relations":[]}',
                ]
            ),
            prompt_store=_PromptStore(),
        )

        orchestrator.run(user_input=long_text, conversation_context="")

        session_obs_call = next(args for name, args in executed if name == "memory.add_observations")
        stored_text = session_obs_call["observations"][0]["contents"][0]
        self.assertEqual(stored_text, long_text.strip())

    def test_memory_upsert_runs_resilience_observations_when_graph_not_persisted(self):
        router = RouterStep(
            prompt_store=_PromptStore(),
            llm=_LLM(['{"tool_calls":[{"name":"memory.graph_upsert","arguments":{"text":"Alice founded Rozvi"}}]}']),
        )
        retriever = RetrieverStep(
            prompt_store=_PromptStore(), llm=_LLM(["unused"]), tool=_RetrieveTool([])
        )
        responder = ResponderStep(prompt_store=_PromptStore(), llm=_LLM(["ok"]))

        executed = []

        def _mcp_executor(tool_name, arguments):
            executed.append((tool_name, arguments))
            if tool_name == "memory.read_graph":
                return {"ok": True, "data": {"entities": [], "relations": []}}
            return {"ok": True}

        orchestrator = RagWorkflowOrchestrator(
            router=router,
            retriever=retriever,
            responder=responder,
            available_tools_provider=lambda: [
                {"name": "retrieval", "description": "", "inputSchema": {}},
                {"name": "memory.add_observations", "description": "", "inputSchema": {}},
                {"name": "memory.create_entities", "description": "", "inputSchema": {}},
                {"name": "memory.create_relations", "description": "", "inputSchema": {}},
                {"name": "memory.read_graph", "description": "", "inputSchema": {}},
            ],
            mcp_tool_executor=_mcp_executor,
            graph_builder_llm=_LLM(
                [
                    "Alice founded Rozvi",
                    '{"entities":[{"name":"Alice","entityType":"person","observations":["founded Rozvi"]}],"relations":[]}',
                ]
            ),
            prompt_store=_PromptStore(),
        )

        orchestrator.run(user_input="remember", conversation_context="")

        resilience_call = None
        for name, args in executed:
            if name != "memory.add_observations":
                continue
            entity_name = args.get("observations", [{}])[0].get("entityName")
            if entity_name == "Alice":
                resilience_call = args
                break

        self.assertIsNotNone(resilience_call)


if __name__ == "__main__":
    unittest.main()
