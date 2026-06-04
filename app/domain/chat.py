from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class ChatGenerator(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        system_message: str,
        user_query: str,
        context: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        ...


class IntentClassifier:
    def __init__(
        self,
        greeting_keywords: list[str] | None = None,
        small_talk_keywords: list[str] | None = None,
    ) -> None:
        self._greeting_keywords = greeting_keywords or ["ola", "oi", "bom dia", "boa tarde", "boa noite"]
        self._small_talk_keywords = small_talk_keywords or [
            "tudo bem",
            "como voce esta",
            "quem e voce",
            "o que voce faz",
        ]

    def identify(self, query: str) -> str:
        query_lower = (query or "").lower()

        if any(re.search(rf"\b{re.escape(word)}\b", query_lower) for word in self._greeting_keywords):
            return "greeting"

        if any(re.search(rf"\b{re.escape(word)}\b", query_lower) for word in self._small_talk_keywords):
            return "small_talk"

        return "data_query"


class ChatPromptBuilder:
    @staticmethod
    def build_inputs(system_message: str, user_query: str, context: str) -> str:
        return (
            f"System: {system_message}\n"
            f"User: Using the context below, {user_query}\n"
            f"Context: {context}\n"
            "Assistant:"
        )

    @staticmethod
    def build_messages(system_message: str, user_query: str, context: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_message or "You are a helpful assistant."},
            {"role": "user", "content": f"Using the context below, answer: {user_query}\n\nContext:\n{context}"},
        ]

    @staticmethod
    def extract_assistant_response(generated_text: str) -> str:
        if "Assistant:" in generated_text:
            return generated_text.split("Assistant:", 1)[1].strip()
        return generated_text.strip()


@dataclass(frozen=True)
class RetryPolicy:
    retries: int = 2
    delay_seconds: float = 3.0


class FailoverChatGateway:
    def __init__(
        self,
        primary: ChatGenerator,
        secondary: ChatGenerator,
        *,
        retry_policy: RetryPolicy,
        model_id: str,
        provider: str,
        logger_instance: logging.Logger | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._retry_policy = retry_policy
        self._model_id = model_id
        self._provider = provider
        self._logger = logger_instance or logger
        self._sleep_fn = sleep_fn or time.sleep

    def generate(
        self,
        *,
        prompt: str,
        system_message: str,
        user_query: str,
        context: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        last_exception: Exception | None = None

        for attempt in range(1, self._retry_policy.retries + 1):
            try:
                return self._primary.generate(
                    prompt=prompt,
                    system_message=system_message,
                    user_query=user_query,
                    context=context,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as primary_exc:  # noqa: BLE001
                last_exception = primary_exc
                self._logger.error(
                    "Primary router call attempt %s/%s failed (model=%s): %s",
                    attempt,
                    self._retry_policy.retries,
                    self._model_id,
                    primary_exc,
                )

            try:
                return self._secondary.generate(
                    prompt=prompt,
                    system_message=system_message,
                    user_query=user_query,
                    context=context,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as secondary_exc:  # noqa: BLE001
                last_exception = secondary_exc
                self._logger.error(
                    "Secondary InferenceClient call attempt %s/%s failed (provider=%s, model=%s): %s",
                    attempt,
                    self._retry_policy.retries,
                    self._provider,
                    self._model_id,
                    secondary_exc,
                )

            if attempt < self._retry_policy.retries:
                self._sleep_fn(self._retry_policy.delay_seconds)

        raise RuntimeError(
            f"Failed to call chat inference after {self._retry_policy.retries} attempts "
            f"(router_primary=True, provider={self._provider}, model={self._model_id}): {last_exception}"
        ) from last_exception


class RagChatProcessor:
    def __init__(
        self,
        *,
        intent_classifier: IntentClassifier,
        retrieval_fn: Callable[[str], list[dict]],
        generation_gateway: ChatGenerator,
        prompt_builder: ChatPromptBuilder,
        system_message: str = "",
        temperature: float = 0.2,
        max_new_tokens: int = 200,
    ) -> None:
        self._intent_classifier = intent_classifier
        self._retrieval_fn = retrieval_fn
        self._generation_gateway = generation_gateway
        self._prompt_builder = prompt_builder
        self._system_message = system_message
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens

    def process_query(self, user_query: str) -> dict[str, str]:
        intent = self._intent_classifier.identify(user_query)

        if intent == "greeting":
            return {"query": user_query, "response": "Ola! Como posso ajudar com seus dados?"}

        if intent == "small_talk":
            return {
                "query": user_query,
                "response": "Sou um assistente focado em responder com base nos dados carregados.",
            }

        search_results = self._retrieval_fn(user_query)
        if not search_results:
            return {
                "query": user_query,
                "response": "Nao encontrei informacoes relacionadas a sua pergunta nos dados disponiveis.",
            }

        retrieved_content = "\n\n".join(item["content"] for item in search_results)
        prompt = self._prompt_builder.build_inputs(self._system_message, user_query, retrieved_content)
        generated_text = self._generation_gateway.generate(
            prompt=prompt,
            system_message=self._system_message,
            user_query=user_query,
            context=retrieved_content,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )
        assistant_response = self._prompt_builder.extract_assistant_response(f"{prompt}{generated_text}")

        return {"query": user_query, "response": assistant_response}

