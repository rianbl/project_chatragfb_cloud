import inspect
import logging
import socket

import requests
from huggingface_hub import InferenceClient

from .chat_core import (
    ChatPromptBuilder,
    FailoverChatGateway,
    IntentClassifier,
    RagChatProcessor,
    RetryPolicy,
)
from .config import HF_API_TOKEN, HF_MODEL_ID, HF_PROVIDER, HF_TIMEOUT
from .retrieval import query_context

HF_CLIENTS = {}
CHAT_PROCESSOR = None
logger = logging.getLogger(__name__)


def _build_hf_client(provider):
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


def _get_hf_client(provider=None, force_recreate=False):
    resolved_provider = (provider or HF_PROVIDER or "auto").strip()
    cache_key = resolved_provider.lower()
    if force_recreate or cache_key not in HF_CLIENTS:
        HF_CLIENTS[cache_key] = _build_hf_client(resolved_provider)
    return HF_CLIENTS[cache_key]


class HFRouterProvider:
    def generate(
        self,
        *,
        prompt,
        system_message,
        user_query,
        context,
        max_new_tokens,
        temperature,
    ):
        del prompt
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        }
        messages = ChatPromptBuilder.build_messages(system_message, user_query, context)
        payload = {
            "model": HF_MODEL_ID,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Router fallback failed ({response.status_code}): {response.text[:300]}"
            )

        body = response.json()
        if isinstance(body, dict):
            choices = body.get("choices") or []
            if choices:
                first = choices[0] or {}
                message = first.get("message") or {}
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

        raise RuntimeError(f"Unexpected router fallback response format: {type(body).__name__}")


class HFInferenceClientProvider:
    def __init__(self, client_getter):
        self._client_getter = client_getter

    def generate(
        self,
        *,
        prompt,
        system_message,
        user_query,
        context,
        max_new_tokens,
        temperature,
    ):
        client = self._client_getter()
        try:
            generated_text = client.text_generation(
                prompt=prompt,
                model=HF_MODEL_ID,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except ValueError as exc:
            if "Supported task: conversational" not in str(exc):
                raise
            messages = ChatPromptBuilder.build_messages(system_message, user_query, context)
            completion = client.chat_completion(
                model=HF_MODEL_ID,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            generated_text = completion.choices[0].message.content
        return generated_text


def _build_gateway(retries=2, delay=3):
    return FailoverChatGateway(
        primary=HFRouterProvider(),
        secondary=HFInferenceClientProvider(_get_hf_client),
        retry_policy=RetryPolicy(retries=retries, delay_seconds=delay),
        model_id=HF_MODEL_ID,
        provider=HF_PROVIDER,
        logger_instance=logger,
    )


def _get_chat_processor():
    global CHAT_PROCESSOR
    if CHAT_PROCESSOR is None:
        CHAT_PROCESSOR = RagChatProcessor(
            intent_classifier=IntentClassifier(),
            retrieval_fn=query_context,
            generation_gateway=_build_gateway(retries=2, delay=3),
            prompt_builder=ChatPromptBuilder(),
            system_message="",
            temperature=0.2,
            max_new_tokens=200,
        )
    return CHAT_PROCESSOR


def _resolve_host(hostname):
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
    except Exception:  # noqa: BLE001
        logger.exception("Failed to initialize Hugging Face InferenceClient at startup.")
        return

    status = get_chat_status()
    logger.info(
        "Chat startup status: provider=%s model=%s token_present=%s router_primary=%s api_dns_ok=%s router_dns_ok=%s",
        status["provider"],
        status["model"],
        status["token_present"],
        True,
        status["dns"]["api_inference"]["ok"],
        status["dns"]["router"]["ok"],
    )


def identify_intent(query):
    classifier = IntentClassifier()
    return classifier.identify(query)


def query_hf_api(payload, retries=2, delay=3):
    prompt = payload.get("inputs", "")
    system_message = payload.get("system_message", "")
    user_query = payload.get("user_query", "")
    context = payload.get("context", "")
    parameters = payload.get("parameters", {})
    max_new_tokens = int(parameters.get("max_length", 200))
    temperature = float(parameters.get("temperature", 0.2))

    gateway = _build_gateway(retries=retries, delay=delay)
    generated_text = gateway.generate(
        prompt=prompt,
        system_message=system_message,
        user_query=user_query,
        context=context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return [{"generated_text": f"{prompt}{generated_text}"}]


def process_chat_query(user_query):
    return _get_chat_processor().process_query(user_query)
