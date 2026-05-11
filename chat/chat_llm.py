import re
import time
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

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

HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"

# Read Hugging Face API token securely from the secret file
HF_API_TOKEN_FILE = os.getenv("HF_API_TOKEN_FILE", "/run/secrets/hf_api_token")

def get_hf_api_token():
    """Retrieve Hugging Face API token from the secret file."""
    try:
        with open(HF_API_TOKEN_FILE, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.error("HF API token file not found.")
        raise
    except Exception as e:
        logging.error(f"Error reading HF API token file: {e}")
        raise

HF_API_TOKEN = get_hf_api_token()

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,  # Change to logging.INFO for less verbosity
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]  # Logs to stdout (visible in container logs)
)

# Helper to call Hugging Face API
def query_hf_api(payload, retries=2, delay=5):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    for attempt in range(retries):
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()
        
        print(f"Tentativa {attempt+1}/{retries} falhou: {response.text}")
        time.sleep(delay) 
    
    response.raise_for_status()

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
