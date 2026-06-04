import inspect
import logging
import re
import socket
import time

import requests
from huggingface_hub import InferenceClient

from .config import HF_API_TOKEN, HF_MODEL_ID, HF_PROVIDER, HF_TIMEOUT
from .retrieval import query_context

HF_CLIENTS = {}
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


def _query_hf_router_fallback(system_message, user_query, context, max_new_tokens, temperature):
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "system", "content": system_message or "You are a helpful assistant."},
        {"role": "user", "content": f"Using the context below, answer: {user_query}\n\nContext:\n{context}"},
    ]
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
    greeting_keywords = ["ola", "oi", "bom dia", "boa tarde", "boa noite"]
    small_talk_keywords = ["tudo bem", "como voce esta", "quem e voce", "o que voce faz"]

    query_lower = query.lower()

    if any(re.search(rf"\b{re.escape(word)}\b", query_lower) for word in greeting_keywords):
        return "greeting"

    if any(re.search(rf"\b{re.escape(word)}\b", query_lower) for word in small_talk_keywords):
        return "small_talk"

    return "data_query"


def _call_hf_client(client, prompt, system_message, user_query, context, max_new_tokens, temperature):
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
        messages = [
            {"role": "system", "content": system_message or "You are a helpful assistant."},
            {"role": "user", "content": f"Using the context below, answer: {user_query}\n\nContext:\n{context}"},
        ]
        completion = client.chat_completion(
            model=HF_MODEL_ID,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        generated_text = completion.choices[0].message.content
    return [{"generated_text": f"{prompt}{generated_text}"}]


def query_hf_api(payload, retries=2, delay=3):
    prompt = payload.get("inputs", "")
    system_message = payload.get("system_message", "")
    user_query = payload.get("user_query", "")
    context = payload.get("context", "")
    parameters = payload.get("parameters", {})
    max_new_tokens = int(parameters.get("max_length", 200))
    temperature = float(parameters.get("temperature", 0.2))

    last_exception = None

    for attempt in range(1, retries + 1):
        try:
            generated_text = _query_hf_router_fallback(
                system_message,
                user_query,
                context,
                max_new_tokens,
                temperature,
            )
            return [{"generated_text": f"{prompt}{generated_text}"}]
        except Exception as router_exc:  # noqa: BLE001
            last_exception = router_exc
            logger.error(
                "Primary router call attempt %s/%s failed (model=%s): %s",
                attempt,
                retries,
                HF_MODEL_ID,
                router_exc,
            )
            try:
                client = _get_hf_client()
                return _call_hf_client(
                    client,
                    prompt,
                    system_message,
                    user_query,
                    context,
                    max_new_tokens,
                    temperature,
                )
            except Exception as hf_client_exc:  # noqa: BLE001
                last_exception = hf_client_exc
                logger.error(
                    "Secondary InferenceClient call attempt %s/%s failed (provider=%s, model=%s): %s",
                    attempt,
                    retries,
                    HF_PROVIDER,
                    HF_MODEL_ID,
                    hf_client_exc,
                )

            if attempt < retries:
                time.sleep(delay)

    raise RuntimeError(
        f"Failed to call chat inference after {retries} attempts "
        f"(router_primary=True, provider={HF_PROVIDER}, model={HF_MODEL_ID}): {last_exception}"
    ) from last_exception


def process_chat_query(user_query):
    intent = identify_intent(user_query)

    if intent == "greeting":
        return {"query": user_query, "response": "Ola! Como posso ajudar com seus dados?"}

    if intent == "small_talk":
        return {
            "query": user_query,
            "response": "Sou um assistente focado em responder com base nos dados carregados.",
        }

    search_results = query_context(user_query)
    if not search_results:
        return {
            "query": user_query,
            "response": "Nao encontrei informacoes relacionadas a sua pergunta nos dados disponiveis.",
        }

    retrieved_content = "\n\n".join(item["content"] for item in search_results)
    system_message = ""
    inputs = (
        f"System: {system_message}\n"
        f"User: Using the context below, {user_query}\n"
        f"Context: {retrieved_content}\n"
        "Assistant:"
    )

    response = query_hf_api(
        {
            "inputs": inputs,
            "system_message": system_message,
            "user_query": user_query,
            "context": retrieved_content,
            "parameters": {"temperature": 0.2, "max_length": 200},
        }
    )

    generated_text = response[0].get("generated_text", "")
    assistant_response = (
        generated_text.split("Assistant:", 1)[1].strip()
        if "Assistant:" in generated_text
        else generated_text.strip()
    )

    return {"query": user_query, "response": assistant_response}
