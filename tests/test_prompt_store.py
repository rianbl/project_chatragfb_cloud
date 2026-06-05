import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from infrastructure.prompt_store import FilePromptStore
from modules.config import PROMPT_STORE_PATH


class PromptStoreTests(unittest.TestCase):
    def test_loads_prompts_from_single_store_file(self):
        store = FilePromptStore(PROMPT_STORE_PATH)

        router_prompt = store.get_prompt("router")
        retriever_prompt = store.get_prompt("retriever")
        responder_prompt = store.get_prompt("responder")

        self.assertIn("You are a router for a tool-augmented Retrieval-Augmented Generation", router_prompt)
        self.assertIn("You are a retrieval specialist for a RAG system.", retriever_prompt)
        self.assertIn("You are a helpful AI assistant.", responder_prompt)

    def test_render_replaces_template_variables(self):
        store = FilePromptStore(PROMPT_STORE_PATH)

        rendered = store.render(
            "retriever",
            conversation_context="historico",
            user_input="qual contrato?",
        )

        self.assertIn("historico", rendered)
        self.assertIn("qual contrato?", rendered)

    def test_render_keeps_literal_json_braces_in_router_prompt(self):
        store = FilePromptStore(PROMPT_STORE_PATH)

        rendered = store.render(
            "router",
            conversation_context="historico xyz",
            user_input="Onde esta no PDF?",
        )

        self.assertIn(
            '{"tools":["retrieval","filesystem"],"tool_inputs":{"filesystem":{"operation":"list_directory","path":"."}}}',
            rendered,
        )
        self.assertIn('tool_inputs.filesystem.operation in {"list_directory","read_file","write_file","delete_file"}', rendered)
        self.assertIn('{"tools":[]}', rendered)
        self.assertIn("historico xyz", rendered)
        self.assertIn("Onde esta no PDF?", rendered)

    def test_render_raises_when_required_variable_is_missing(self):
        store = FilePromptStore(PROMPT_STORE_PATH)

        with self.assertRaises(KeyError):
            store.render("retriever", user_input="pergunta sem contexto")


if __name__ == "__main__":
    unittest.main()
