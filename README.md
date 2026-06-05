# ChatRAG App (Dockerized)

A containerized Retrieval-Augmented Generation (RAG) application that ingests CSV, TXT, and PDF files, chunks them, indexes chunks in FAISS, and answers questions grounded in the loaded corpus.

## Runtime Architecture

The deployment now uses **three services**:

- `app`: unified application service (frontend + upload/ingestion + retrieval + chat + feedback)
- `postgres`: database service (persistent storage for documents/chunks)
- `mcp-server`: MCP-capable tool service focused on filesystem tools

```text
Browser (localhost:8080)
  -> app service (Flask)
       -> /upload, /documents, /query, /chat, /feedback
       -> MCP client module (HTTP) for /mcp/* and future tool-aware orchestration
       -> Chat workflow: Router -> (Retriever?/Filesystem?/Memory?) -> Responder
       -> in-memory FAISS index (built from PostgreSQL chunks)
       -> Hugging Face Inference API for final answer generation
  -> mcp-server (localhost:8090)
       -> tool registry namespaces: filesystem.* and memory.*
       -> filesystem tools proxied to official prebuilt filesystem MCP server package
       -> memory tools proxied to official prebuilt memory MCP server package (with local fallback store)
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
| `mcp-server` | 8090 | 8090 | MCP HTTP transport + tool registry |
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

### MCP Integration (App API)

`GET /mcp/health` - MCP connectivity and status

`GET /mcp/tools` - list registered namespaced tools

`POST /mcp/tools/{tool_name}` - execute one tool

Example:

```bash
curl -X POST "http://localhost:8080/mcp/tools/filesystem.list_directory" \
  -H "Content-Type: application/json" \
  -d '{"arguments":{"path":"."}}'
```

## Configuration

Main variables:

- `HF_API_TOKEN`
- `POSTGRES_PASSWORD`

Optional model/runtime vars:

- `HF_MODEL_ID`
- `HF_PROVIDER`
- `HF_TIMEOUT`
- `MCP_SERVER_ENABLED`
- `MCP_SERVER_URL`
- `MCP_TIMEOUT`
- `MCP_MEMORY_ENABLED`
- `MEMORY_TOP_K`
- `MEMORY_MAX_OBSERVATIONS`
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
