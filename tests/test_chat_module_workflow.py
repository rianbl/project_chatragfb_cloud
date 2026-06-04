import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from modules import chat_module


class _FakeOrchestrator:
    def __init__(self, use_retrieval=True):
        self.use_retrieval = use_retrieval
        self.last_run = None
        self.last_route = None

    def run(self, *, user_input: str, conversation_context: str = ""):
        self.last_run = (user_input, conversation_context)
        return {"response": "resposta mock"}

    def route_only(self, *, user_input: str, conversation_context: str = ""):
        self.last_route = (user_input, conversation_context)
        return self.use_retrieval


class ChatModuleWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.original = chat_module.RAG_ORCHESTRATOR

    def tearDown(self):
        chat_module.RAG_ORCHESTRATOR = self.original

    def test_process_chat_query_uses_orchestrator_run(self):
        fake = _FakeOrchestrator()
        chat_module.RAG_ORCHESTRATOR = fake

        payload = chat_module.process_chat_query("o que diz o arquivo?", conversation_context="historico x")

        self.assertEqual(payload["query"], "o que diz o arquivo?")
        self.assertEqual(payload["response"], "resposta mock")
        self.assertEqual(fake.last_run, ("o que diz o arquivo?", "historico x"))

    def test_identify_intent_maps_router_true(self):
        fake = _FakeOrchestrator(use_retrieval=True)
        chat_module.RAG_ORCHESTRATOR = fake

        label = chat_module.identify_intent("consulte o contrato")

        self.assertEqual(label, "requires_retrieval")
        self.assertEqual(fake.last_route, ("consulte o contrato", ""))

    def test_identify_intent_maps_router_false(self):
        fake = _FakeOrchestrator(use_retrieval=False)
        chat_module.RAG_ORCHESTRATOR = fake

        label = chat_module.identify_intent("quanto e 2+2?")

        self.assertEqual(label, "direct_response")
        self.assertEqual(fake.last_route, ("quanto e 2+2?", ""))


if __name__ == "__main__":
    unittest.main()
