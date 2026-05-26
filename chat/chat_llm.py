import re
import time
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from huggingface_hub import InferenceClient

# Flask and CORS setup
app = Flask(__name__)
CORS(app, origins=["http://localhost:8080"])
app.debug = True

# Set up logger
app.logger.setLevel(logging.INFO)

SEARCH_API_URL = "http://search:5000/query"
SYSTEM_MESSAGE = (
    ""
)
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto").strip()
HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "60"))

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()
if not HF_API_TOKEN:
    raise RuntimeError("Missing required environment variable: HF_API_TOKEN")

if HF_PROVIDER.lower() == "auto":
    HF_CLIENT = InferenceClient(api_key=HF_API_TOKEN, timeout=HF_TIMEOUT)
else:
    HF_CLIENT = InferenceClient(api_key=HF_API_TOKEN, provider=HF_PROVIDER, timeout=HF_TIMEOUT)

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,  # Change to logging.INFO for less verbosity
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]  # Logs to stdout (visible in container logs)
)

# Helper to call Hugging Face API
def query_hf_api(payload, retries=2, delay=5):
    prompt = payload.get("inputs", "")
    system_message = payload.get("system_message", "")
    user_query = payload.get("user_query", "")
    context = payload.get("context", "")
    parameters = payload.get("parameters", {})
    max_new_tokens = int(parameters.get("max_length", 200))
    temperature = float(parameters.get("temperature", 0.2))
    last_exception = None

    for attempt in range(retries):
        try:
            try:
                generated_text = HF_CLIENT.text_generation(
                    prompt=prompt,
                    model=HF_MODEL_ID,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
            except ValueError as e:
                # Some providers expose this model only for conversational task.
                if "Supported task: conversational" not in str(e):
                    raise
                messages = [
                    {"role": "system", "content": system_message or "Você é um assistente útil e objetivo."},
                    {
                        "role": "user",
                        "content": f"Usando o contexto seguinte, responda: {user_query}\n\nContexto:\n{context}"
                    }
                ]
                completion = HF_CLIENT.chat_completion(
                    model=HF_MODEL_ID,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens
                )
                generated_text = completion.choices[0].message.content

            return [{"generated_text": f"{prompt}{generated_text}"}]
        except Exception as e:
            last_exception = e
            logging.error(f"HF call attempt {attempt+1}/{retries} failed: {type(e).__name__}: {e}")
        time.sleep(delay) 

    raise RuntimeError(
        f"Failed to call Hugging Face Inference API after {retries} attempts "
        f"(provider={HF_PROVIDER}, model={HF_MODEL_ID}): {last_exception}"
    ) from last_exception

# Intent identification function
def identify_intent(query):
    greeting_keywords = ["olá", "oi", "bom dia", "boa tarde", "boa noite"]
    small_talk_keywords = ["tudo bem", "como você está", "quem é você", "o que você faz"]
    
    query_lower = query.lower()
    
    # Check for greetings
    if any(re.search(rf"\b{re.escape(word)}\b", query_lower) for word in greeting_keywords):
        return "greeting"
    
    # Check for small talk
    if any(re.search(rf"\b{re.escape(word)}\b", query_lower) for word in small_talk_keywords):
        return "small_talk"
    
    return "data_query"

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_query = request.json.get("query", "").strip()
        if not user_query:
            return jsonify({"error": "Query cannot be empty."}), 400

        # Identify intent
        intent = identify_intent(user_query)
        logging.debug(f"User query: {user_query}")
        logging.debug(f"Identified intent: {intent}")

        # Handle greeting intent
        if intent == "greeting":
            return jsonify({
                "query": user_query,
                "response": "Olá! Como posso ajudar com suas dúvidas sobre os dados?"
            })

        # Handle small talk intent
        if intent == "small_talk":
            return jsonify({
                "query": user_query,
                "response": "Desculpe, mas sou um assistente focado em esclarecer dúvidas com base nos dados fornecidos. Como posso ajudar você hoje?"
            })

        # Data query intent
        search_response = requests.post(
            SEARCH_API_URL, json={"query": user_query}
        )
        if search_response.status_code != 200:
            logging.error("Search API returned an error.")
            return jsonify({"error": "Failed to retrieve search results."}), 500

        search_results = search_response.json().get("results", [])
        if not search_results:
            logging.info("No search results found for the query.")
            return jsonify({
                "query": user_query,
                "response": "Desculpe, não encontrei informações relacionadas à sua pergunta nos dados disponíveis."
            })

        # Prepare context for HF model
        retrieved_content = "\n\n".join([res["content"] for res in search_results])

        # HF API payload
        system_message = SYSTEM_MESSAGE
        inputs = f"System: {system_message}\nUser: Usando o contexto seguinte, {user_query}\nContext: {retrieved_content}\nAssistant:"
        app.logger.info(f"Model call full inputs: {inputs}")
        response = query_hf_api({
            "inputs": inputs,
            "system_message": system_message,
            "user_query": user_query,
            "context": retrieved_content,
            "parameters": {
                "temperature": 0.2,
                "max_length": 200
            }
        })

        # Extract generated response
        generated_text = response[0].get("generated_text", "")
        assistant_response = (
            generated_text.split("Assistant:", 1)[1].strip()
            if "Assistant:" in generated_text
            else generated_text.strip()
        )

        logging.info(f"HF API response: {assistant_response}")
        return jsonify({"query": user_query, "response": assistant_response})
    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
