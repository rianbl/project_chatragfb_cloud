from __future__ import annotations

import inspect
import logging
import socket
import time

import requests
from huggingface_hub import InferenceClient

from infrastructure.prompt_store import FilePromptStore
from infrastructure.rag_workflow import RagWorkflowOrchestrator, ResponderStep, RetrieverStep, RetrievingTool, RouterStep
from infrastructure.runtime import get_default_retrieval_service

from .config import HF_API_TOKEN, HF_MODEL_ID, HF_PROVIDER, HF_TIMEOUT, PROMPT_STORE_PATH

HF_CLIENTS: dict[str, InferenceClient] = {}
RAG_ORCHESTRATOR: RagWorkflowOrchestrator | None = None
logger = logging.getLogger(__name__)


def _build_hf_client(provider: str):
    if not HF_API_TOKEN:
        raise ValueError("Missing required environment variable: HF_API_TOKEN")

    init_params = inspect.signature(InferenceClient.__init__).parameters
    kwargs = {"timeout": HF_TIMEOUT}

    if "api_key" in init_params:
        kwargs["api_key"] = HF_API_TOKEN
    elif "token" in init_params:
        kwargs["token"] = HF_API_TOKEN
    else:
        raise RuntimeError(
            "Unsupported huggingface_hub.InferenceClient signature: expected 'api_key' or 'token'."
        )

    provider_value = (provider or "auto").strip()
    if provider_value.lower() != "auto":
        if "provider" in init_params:
            kwargs["provider"] = provider_value
        else:
            logger.warning(
                "HF_PROVIDER='%s' ignored: current InferenceClient version does not support provider argument.",
                provider_value,
            )

    client = InferenceClient(**kwargs)
    logger.info(
        "Hugging Face InferenceClient initialized (provider=%s, args=%s).",
        provider_value,
        sorted(kwargs.keys()),
    )
    return client


def _get_hf_client(provider: str | None = None, force_recreate: bool = False):
    resolved_provider = (provider or HF_PROVIDER or "auto").strip()
    cache_key = resolved_provider.lower()
    if force_recreate or cache_key not in HF_CLIENTS:
        HF_CLIENTS[cache_key] = _build_hf_client(resolved_provider)
    return HF_CLIENTS[cache_key]


class HFTextGenerator:
    def __init__(self, retries: int = 2, delay_seconds: float = 3.0) -> None:
        self._retries = retries
        self._delay_seconds = delay_seconds

    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        last_exception = None
        for attempt in range(1, self._retries + 1):
            try:
                return self._generate_via_router(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            except Exception as router_exc:  # noqa: BLE001
                last_exception = router_exc
                logger.error(
                    "Router text generation attempt %s/%s failed (model=%s): %s",
                    attempt,
                    self._retries,
                    HF_MODEL_ID,
                    router_exc,
                )
            try:
                return self._generate_via_client(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            except Exception as client_exc:  # noqa: BLE001
                last_exception = client_exc
                logger.error(
                    "InferenceClient text generation attempt %s/%s failed (provider=%s, model=%s): %s",
                    attempt,
                    self._retries,
                    HF_PROVIDER,
                    HF_MODEL_ID,
                    client_exc,
                )

            if attempt < self._retries:
                time.sleep(self._delay_seconds)

        raise RuntimeError(
            f"Failed to generate text after {self._retries} attempts "
            f"(provider={HF_PROVIDER}, model={HF_MODEL_ID}): {last_exception}"
        ) from last_exception

    @staticmethod
    def _generate_via_router(prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": HF_MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT)
        if response.status_code >= 400:
            raise RuntimeError(f"Router generation failed ({response.status_code}): {response.text[:300]}")

        body = response.json()
        if isinstance(body, dict):
            choices = body.get("choices") or []
            if choices:
                first = choices[0] or {}
                message = first.get("message") or {}
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        raise RuntimeError(f"Unexpected router response format: {type(body).__name__}")

    @staticmethod
    def _generate_via_client(prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        client = _get_hf_client()
        try:
            return client.text_generation(
                prompt=prompt,
                model=HF_MODEL_ID,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except ValueError as exc:
            if "Supported task: conversational" not in str(exc):
                raise
            completion = client.chat_completion(
                model=HF_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return completion.choices[0].message.content


def _build_rag_orchestrator() -> RagWorkflowOrchestrator:
    prompt_store = FilePromptStore(PROMPT_STORE_PATH)
    text_llm = HFTextGenerator(retries=2, delay_seconds=3)
    retrieval_service = get_default_retrieval_service()
    retrieving_tool = RetrievingTool(retrieval_fn=retrieval_service.query_context)

    return RagWorkflowOrchestrator(
        router=RouterStep(prompt_store=prompt_store, llm=text_llm),
        retriever=RetrieverStep(prompt_store=prompt_store, llm=text_llm, tool=retrieving_tool),
        responder=ResponderStep(prompt_store=prompt_store, llm=text_llm),
        prefer_langgraph=True,
        logger_instance=logger,
    )


def _get_rag_orchestrator() -> RagWorkflowOrchestrator:
    global RAG_ORCHESTRATOR
    if RAG_ORCHESTRATOR is None:
        RAG_ORCHESTRATOR = _build_rag_orchestrator()
    return RAG_ORCHESTRATOR


def _resolve_host(hostname: str):
    try:
        resolved = socket.getaddrinfo(hostname, 443)
        return {"ok": True, "addresses": sorted({item[4][0] for item in resolved})}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "addresses": []}


def get_chat_status():
    status = {
        "token_present": bool(HF_API_TOKEN),
        "provider": HF_PROVIDER,
        "model": HF_MODEL_ID,
        "timeout_seconds": HF_TIMEOUT,
        "prompt_store_path": PROMPT_STORE_PATH,
        "dns": {
            "api_inference": _resolve_host("api-inference.huggingface.co"),
            "router": _resolve_host("router.huggingface.co"),
        },
    }
    return status


def startup_check_chat_client():
    if not HF_API_TOKEN:
        logger.error("HF_API_TOKEN is missing. Chat requests will fail.")
        return

    try:
        _get_hf_client()
        _get_rag_orchestrator()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to initialize chat components at startup.")
        return

    status = get_chat_status()
    logger.info(
        "Chat startup status: provider=%s model=%s token_present=%s prompt_store=%s api_dns_ok=%s router_dns_ok=%s",
        status["provider"],
        status["model"],
        status["token_present"],
        status["prompt_store_path"],
        status["dns"]["api_inference"]["ok"],
        status["dns"]["router"]["ok"],
    )


def identify_intent(query: str):
    use_retrieval = _get_rag_orchestrator().route_only(user_input=query, conversation_context="")
    return "requires_retrieval" if use_retrieval else "direct_response"


def query_hf_api(payload, retries=2, delay=3):
    del retries
    del delay
    prompt = payload.get("inputs", "")
    parameters = payload.get("parameters", {})
    max_new_tokens = int(parameters.get("max_length", 200))
    temperature = float(parameters.get("temperature", 0.2))
    llm = HFTextGenerator(retries=2, delay_seconds=3)
    generated_text = llm.generate(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
    return [{"generated_text": f"{prompt}{generated_text}"}]


def process_chat_query(user_query: str, conversation_context: str = ""):
    workflow_state = _get_rag_orchestrator().run(
        user_input=user_query,
        conversation_context=conversation_context or "",
    )
    return {"query": user_query, "response": workflow_state.get("response", "")}

