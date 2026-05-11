import psycopg2
from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
import logging
import os

app = Flask(__name__)
app.debug = True

# Set up logger
app.logger.setLevel(logging.INFO)

# Function to fetch the database password from secrets
def get_db_password():
    db_password_file = os.getenv("DATABASE_PASSWORD_FILE")
    if db_password_file and os.path.exists(db_password_file):
        with open(db_password_file, 'r') as file:
            return file.read().strip()
    return os.getenv("DATABASE_PASSWORD", "admin")  # Default to "admin" if not provided

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DATABASE_HOST", "postgres"),
    "port": os.getenv("DATABASE_PORT", "5432"),
    "dbname": os.getenv("DATABASE_NAME", "llm_data"),
    "user": os.getenv("DATABASE_USER", "admin"),
    "password": get_db_password()
}

# Global variables for caching
VECTORSTORE_CACHE = None
EMBEDDINGS = None  # Global variable for embeddings

def initialize_embeddings():
    """Initialize embeddings model."""
    global EMBEDDINGS
    app.logger.info("Initializing embeddings model...")
    EMBEDDINGS = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def fetch_data_from_db():
    """Busca dados do banco PostgreSQL"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM source;")  # Consulta ajustável
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        return rows, columns
    except Exception as e:
        app.logger.error(f"Erro ao conectar ao banco de dados: {e}")
        return [], []

def build_vectorstore(data, columns):
    """Cria um Vectorstore FAISS com os dados"""
    documents = [
        Document(
            page_content=" ".join([f"{col}: {row[i]}" for i, col in enumerate(columns)]),
            metadata={"id": idx}
        )
        for idx, row in enumerate(data)
    ]
    vectorstore = FAISS.from_documents(documents, EMBEDDINGS)
    return vectorstore

def get_vectorstore():
    global VECTORSTORE_CACHE
    if VECTORSTORE_CACHE is None:
        app.logger.info("Building vector store from database...")
        data, columns = fetch_data_from_db()
        if not data:
            raise Exception("Nenhum dado encontrado no banco de dados.")
        VECTORSTORE_CACHE = build_vectorstore(data, columns)
    else:
        app.logger.info("Reusing cached vector store.")
    return VECTORSTORE_CACHE

@app.route("/query", methods=["POST"])
def query_vectorstore():
    """Consulta o Vectorstore via API"""
    try:
        query = request.json.get("query", "")
        if not query:
            return jsonify({"error": "A query não pode estar vazia."}), 400
        
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        docs = retriever.get_relevant_documents(query)
        
        results = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
        app.logger.info(f"Search full results: {results}")
        return jsonify({"query": query, "results": results})
    except Exception as e:
        app.logger.error(f"Erro na consulta do Vectorstore: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/refresh", methods=["POST"])
def refresh_vectorstore():
    """Atualiza o vectorstore"""
    try:
        global VECTORSTORE_CACHE
        data, columns = fetch_data_from_db()
        if not data:
            return jsonify({"error": "Nenhum dado encontrado no banco de dados."}), 404
        
        VECTORSTORE_CACHE = build_vectorstore(data, columns)
        app.logger.info("Vectorstore atualizado com sucesso.")
        return jsonify({"message": "Vectorstore atualizado com sucesso."}), 200
    except Exception as e:
        app.logger.error(f"Erro ao atualizar o Vectorstore: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    initialize_embeddings()  # Initialize embeddings at startup
    app.run(host="0.0.0.0", port=5000)
