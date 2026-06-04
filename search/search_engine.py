import logging
import os

import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

app = Flask(__name__)
app.debug = True

# Set up logger
app.logger.setLevel(logging.INFO)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DATABASE_HOST", "postgres"),
    "port": os.getenv("DATABASE_PORT", "5432"),
    "dbname": os.getenv("DATABASE_NAME", "llm_data"),
    "user": os.getenv("DATABASE_USER", "admin"),
    "password": os.getenv("DATABASE_PASSWORD", os.getenv("POSTGRES_PASSWORD", "admin")),
}

VECTORSTORE_CACHE = None
EMBEDDINGS = None
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "all-MiniLM-L6-v2")


def initialize_embeddings():
    """Initialize embeddings model."""
    global EMBEDDINGS
    app.logger.info("Initializing embeddings model...")
    EMBEDDINGS = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_ID)


def fetch_chunks_from_db():
    """Fetch chunk corpus from PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    c.content,
                    c.chunk_index,
                    c.metadata,
                    d.id AS document_id,
                    d.filename,
                    d.file_type
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY d.id, c.chunk_index;
                """
            )
            return cursor.fetchall()
    finally:
        conn.close()


def build_vectorstore(chunks):
    """Build an in-memory FAISS index from chunk rows."""
    documents = []
    for row in chunks:
        metadata = {
            "chunk_id": row["chunk_id"],
            "chunk_index": row["chunk_index"],
            "document_id": row["document_id"],
            "filename": row["filename"],
            "file_type": row["file_type"],
        }
        extra_metadata = row.get("metadata")
        if isinstance(extra_metadata, dict):
            metadata.update(extra_metadata)

        documents.append(Document(page_content=row["content"], metadata=metadata))

    if not documents:
        raise ValueError("Nenhum chunk disponível para indexação.")

    return FAISS.from_documents(documents, EMBEDDINGS)


def refresh_vectorstore_cache():
    global VECTORSTORE_CACHE
    chunks = fetch_chunks_from_db()
    if not chunks:
        VECTORSTORE_CACHE = None
        raise ValueError("Nenhum chunk encontrado no banco de dados.")
    VECTORSTORE_CACHE = build_vectorstore(chunks)


def get_vectorstore():
    global VECTORSTORE_CACHE
    if VECTORSTORE_CACHE is None:
        app.logger.info("Building vector store from database...")
        refresh_vectorstore_cache()
    else:
        app.logger.info("Reusing cached vector store.")
    return VECTORSTORE_CACHE


@app.route("/query", methods=["POST"])
def query_vectorstore():
    """Query FAISS retriever using top-k chunk retrieval."""
    try:
        payload = request.json or {}
        query = payload.get("query", "").strip()
        if not query:
            return jsonify({"error": "A query não pode estar vazia."}), 400

        try:
            k_value = int(payload.get("k", DEFAULT_TOP_K))
        except (TypeError, ValueError):
            k_value = DEFAULT_TOP_K
        k_value = max(1, min(k_value, 20))

        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
        docs = retriever.invoke(query)

        results = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        app.logger.info(f"Search full results: {results}")
        return jsonify({"query": query, "results": results})
    except ValueError as e:
        app.logger.warning(f"Query sem dados: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        app.logger.error(f"Erro na consulta do Vectorstore: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/refresh", methods=["POST"])
def refresh_vectorstore():
    """Refresh in-memory FAISS index from chunk corpus."""
    try:
        refresh_vectorstore_cache()
        app.logger.info("Vectorstore atualizado com sucesso.")
        return jsonify({"message": "Vectorstore atualizado com sucesso."}), 200
    except ValueError as e:
        app.logger.warning(f"Refresh sem dados: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        app.logger.error(f"Erro ao atualizar o Vectorstore: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    initialize_embeddings()
    app.run(host="0.0.0.0", port=5000)
