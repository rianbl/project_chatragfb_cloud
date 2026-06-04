import pathlib
import sys
import unittest

APP_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from modules.chat_core import (
    ChatPromptBuilder,
    FailoverChatGateway,
    IntentClassifier,
    RagChatProcessor,
    RetryPolicy,
)


class _Provider:
    def __init__(self, responses=None, error=None):
        self.responses = responses or []
        self.error = error
        self.calls = 0

    def generate(self, **kwargs):
        del kwargs
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return "fallback-response"


class ChatCoreTests(unittest.TestCase):
    def test_intent_classifier(self):
        classifier = IntentClassifier()
        self.assertEqual(classifier.identify("Oi, tudo bem?"), "greeting")
        self.assertEqual(classifier.identify("quem e voce"), "small_talk")
        self.assertEqual(classifier.identify("qual foi a receita"), "data_query")

    def test_failover_uses_secondary_when_primary_fails(self):
        primary = _Provider(error=RuntimeError("router down"))
        secondary = _Provider(responses=["ok-from-secondary"])
        gateway = FailoverChatGateway(
            primary=primary,
            secondary=secondary,
            retry_policy=RetryPolicy(retries=2, delay_seconds=0),
            model_id="model-x",
            provider="auto",
            sleep_fn=lambda _: None,
        )

        result = gateway.generate(
            prompt="p",
            system_message="",
            user_query="q",
            context="c",
            max_new_tokens=10,
            temperature=0.2,
        )

        self.assertEqual(result, "ok-from-secondary")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(secondary.calls, 1)

    def test_failover_raises_after_max_retries(self):
        primary = _Provider(error=RuntimeError("router down"))
        secondary = _Provider(error=RuntimeError("client down"))
        gateway = FailoverChatGateway(
            primary=primary,
            secondary=secondary,
            retry_policy=RetryPolicy(retries=3, delay_seconds=0),
            model_id="model-x",
            provider="auto",
            sleep_fn=lambda _: None,
        )

        with self.assertRaises(RuntimeError):
            gateway.generate(
                prompt="p",
                system_message="",
                user_query="q",
                context="c",
                max_new_tokens=10,
                temperature=0.2,
            )

        self.assertEqual(primary.calls, 3)
        self.assertEqual(secondary.calls, 3)

    def test_rag_processor_returns_no_context_message(self):
        processor = RagChatProcessor(
            intent_classifier=IntentClassifier(),
            retrieval_fn=lambda _: [],
            generation_gateway=_Provider(responses=["unused"]),
            prompt_builder=ChatPromptBuilder(),
        )

        payload = processor.process_query("me traga os dados")

        self.assertIn("Nao encontrei informacoes", payload["response"])

    def test_rag_processor_parses_assistant_response(self):
        gateway = _Provider(responses=[" resposta final "])
        processor = RagChatProcessor(
            intent_classifier=IntentClassifier(),
            retrieval_fn=lambda _: [{"content": "ctx 1"}],
            generation_gateway=gateway,
            prompt_builder=ChatPromptBuilder(),
        )

        payload = processor.process_query("pergunta de dados")

        self.assertEqual(payload["query"], "pergunta de dados")
        self.assertEqual(payload["response"], "resposta final")
        self.assertEqual(gateway.calls, 1)


if __name__ == "__main__":
    unittest.main()
