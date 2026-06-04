# ChatRAG App (Dockerized)

A containerized Retrieval-Augmented Generation (RAG) application that ingests CSV, TXT, and PDF files, chunks them, indexes chunks in FAISS, and answers questions grounded in the loaded corpus.

## Runtime Architecture

The deployment now uses **two services only**:

- `app`: unified application service (frontend + upload/ingestion + retrieval + chat + feedback)
- `postgres`: database service (persistent storage for documents/chunks)

```text
Browser (localhost:8080)
  -> app service (Flask)
       -> /upload, /documents, /query, /chat, /feedback
       -> Chat workflow: Router -> (Retriever?) -> Responder
       -> in-memory FAISS index (built from PostgreSQL chunks)
       -> Hugging Face Inference API for final answer generation
  -> postgres (localhost:5432)
       -> documents + chunks tables
```

## Repository Layout

```text
.
|- app/
|  |- domain/              # regras de negocio puras (entidades, chat core)
|  |- application/         # casos de uso + contratos (ports)
|  |- infrastructure/      # adapters concretos para DB/retrieval/chat/ingestion
|  |- interfaces/          # camada de entrada (HTTP/Flask)
|  |- bootstrap/           # composition root / dependency wiring
|  |- modules/             # modulos de dominio tecnico (chat, ingestion, config)
|  |- static/      # frontend assets (HTML/CSS/JS)
|  |- server.py    # entrypoint (create_app + startup)
|  |- Dockerfile
|  |- requirements.txt
|- docker-compose.yaml
|- tests/          # unit tests
|- .env.example
```

## Features

- Unified app service for cloud-friendly container deployment
- Multi-format ingestion: CSV, TXT, PDF
- Chunk-based retrieval pipeline with configurable chunk size/overlap
- FAISS vector index in-memory (refreshed from DB corpus)
- Context limits and safety checks:
  - max documents
  - max file size
  - max total corpus size
  - max PDF pages
- Context sidebar with per-file delete and live usage tracking

## Quick Start

1. Create your env file:

```bash
cp .env.example .env
```

2. Set required values in `.env`:

```bash
HF_API_TOKEN=YOUR_HF_TOKEN
POSTGRES_PASSWORD=admin
```

3. Start the stack:

```bash
docker compose up --build
```

4. Open the app:

```text
http://localhost:8080
```

## Services and Ports

| Service | Container Port | Host Port | Purpose |
|---|---:|---:|---|
| `app` | 8080 | 8080 | Frontend + APIs |
| `postgres` | 5432 | 5432 | PostgreSQL |

## API Reference

All endpoints are served by `app` on `http://localhost:8080`.

### Chat

`POST /chat`

```bash
curl -X POST "http://localhost:8080/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query":"What did Patricia buy?",
    "conversation_context":"User asked about purchases in the uploaded report."
  }'
```

`conversation_context` is optional (string or list of strings).

### Retrieval Query

`POST /query`

```bash
curl -X POST "http://localhost:8080/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"Alice bought","k":4}'
```

### Upload

`POST /upload` (`multipart/form-data`)

```bash
curl -X POST "http://localhost:8080/upload" \
  -F "file=@your_file.csv"
```

### Documents State

`GET /documents`

```bash
curl -X GET "http://localhost:8080/documents"
```

### Delete Document

`DELETE /documents/{id}`

```bash
curl -X DELETE "http://localhost:8080/documents/1"
```

### Feedback

`POST /feedback`

```bash
curl -X POST "http://localhost:8080/feedback" \
  -H "Content-Type: application/json" \
  -d '{"feedback_type":"thumbsUp","message":"Great answer"}'
```

## Configuration

Main variables:

- `HF_API_TOKEN`
- `POSTGRES_PASSWORD`

Optional model/runtime vars:

- `HF_MODEL_ID`
- `HF_PROVIDER`
- `HF_TIMEOUT`
- `PROMPT_STORE_PATH` (default: `app/config/prompt_store.yaml`)
- `EMBEDDING_MODEL_ID`
- `RETRIEVAL_TOP_K`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `MAX_DOCUMENTS`
- `MAX_FILE_SIZE_BYTES`
- `MAX_TOTAL_SIZE_BYTES`
- `MAX_PDF_PAGES`

## Notes

- FAISS index is in-memory and rebuilt from DB chunks via `/refresh` or during retrieval when needed.
- PostgreSQL is persistent via `postgres_data` volume.
