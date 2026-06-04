import logging

from psycopg2.extras import RealDictCursor
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import EMBEDDING_MODEL_ID, RETRIEVAL_TOP_K
from .db import get_db_connection

VECTORSTORE_CACHE = None
EMBEDDINGS = None
logger = logging.getLogger(__name__)


def initialize_embeddings():
    global EMBEDDINGS
    if EMBEDDINGS is None:
        import sentence_transformers
        import torch
        import transformers

        logger.info(
            "Embedding runtime versions: sentence-transformers=%s transformers=%s torch=%s",
            sentence_transformers.__version__,
            transformers.__version__,
            torch.__version__,
        )
        logger.info(
            "Initializing embeddings model '%s' (this may download model files on first run).",
            EMBEDDING_MODEL_ID,
        )
        EMBEDDINGS = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_ID)
        logger.info("Embeddings model '%s' initialized successfully.", EMBEDDING_MODEL_ID)
    return EMBEDDINGS


def fetch_chunks_from_db():
    conn = get_db_connection()
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
            rows = cursor.fetchall()
            logger.info("Loaded %s chunks from PostgreSQL for indexing/querying.", len(rows))
            return rows
    finally:
        conn.close()


def build_vectorstore(chunks):
    initialize_embeddings()

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
        raise ValueError("No chunks available for indexing.")

    logger.info("Building FAISS index from %s chunk documents.", len(documents))
    vectorstore = FAISS.from_documents(documents, EMBEDDINGS)
    logger.info("FAISS index build completed.")
    return vectorstore


def refresh_vectorstore_cache():
    global VECTORSTORE_CACHE

    chunks = fetch_chunks_from_db()
    if not chunks:
        VECTORSTORE_CACHE = None
        raise ValueError("No chunks found in database.")

    VECTORSTORE_CACHE = build_vectorstore(chunks)
    logger.info("Vectorstore cache refreshed successfully.")


def get_vectorstore():
    global VECTORSTORE_CACHE

    if VECTORSTORE_CACHE is None:
        logger.info("Vectorstore cache is empty. Refreshing from database.")
        refresh_vectorstore_cache()
    else:
        logger.info("Reusing cached vectorstore.")

    return VECTORSTORE_CACHE


def query_context(query, k=None):
    if not query:
        raise ValueError("Query cannot be empty.")

    if k is None:
        k_value = RETRIEVAL_TOP_K
    else:
        k_value = int(k)

    k_value = max(1, min(k_value, 20))
    logger.info("Running retrieval query with k=%s. Query preview='%s'.", k_value, query[:80])
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": k_value})
    docs = retriever.invoke(query)

    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]


def get_retrieval_status():
    return {
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "embeddings_initialized": EMBEDDINGS is not None,
        "vectorstore_cached": VECTORSTORE_CACHE is not None,
    }
