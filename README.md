# ChatRAG Files (Dockerized)

A containerized Retrieval-Augmented Generation (RAG) application that lets you upload CSV, TXT, or PDF files and chat with their content through a web interface.

The stack includes:
- `webapp`: static frontend (NGINX)
- `chatllm`: chat API that orchestrates retrieval + LLM generation
- `search`: semantic retrieval service (FAISS + sentence-transformers)
- `postgres`: database + multi-format ingestion API + feedback API

## Features

- Upload CSV, TXT, or PDF files from the browser (drag-and-drop or file picker)
- Parse file content into chunked document units
- Persist documents/chunks in PostgreSQL
- Build/refresh a FAISS vector store from chunked content
- File context controls with maximum of 3 active files
- Upload guardrails: max file size, max total corpus size, max PDF pages
- Ask questions in natural language and get answers grounded in your data
- Capture thumbs up/down feedback events

## Architecture

```text
Browser (localhost:8080)
  -> Upload file (CSV/TXT/PDF) -> Data Loader API (postgres:5001)
       -> Parse + chunk + persist documents/chunks in PostgreSQL (postgres:5432)
       -> Trigger vector refresh (search:5000/refresh)
  -> Ask question -> Chat API (chatllm:8081/chat)
       -> Retrieve context (search:5000/query)
       -> Generate answer (Hugging Face Inference API)
```

## Repository Layout

```text
.
|- chat/        # Chat API (Flask) + Hugging Face inference call
|- search/      # Retrieval API (Flask + LangChain + FAISS)
|- database/    # PostgreSQL image + file ingestion/chunking + feedback API
|- webapp/      # Static frontend (HTML/CSS/JS via NGINX)
|- .env.example # Environment template for local secrets/config
|- docker-compose.yaml
```

## Prerequisites

- Docker + Docker Compose v2
- Hugging Face access token (for inference API)
- (Windows only) Docker Desktop with WSL2 integration recommended

## Quick Start

1. Create your environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and set at least:

```bash
HF_API_TOKEN=YOUR_HF_TOKEN
POSTGRES_PASSWORD=admin
# Optional
HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
HF_PROVIDER=auto
HF_TIMEOUT=60
```

3. Start the stack:

```bash
docker compose up --build
```

4. Open the app:

```text
http://localhost:8080
```

5. Upload a CSV, TXT, or PDF file in the UI, then ask questions in the chat panel.

## Services and Ports

| Service | Container Port | Host Port | Purpose |
|---|---:|---:|---|
| `webapp` | 80 | 8080 | Frontend UI |
| `chatllm` | 8081 | 8081 | Chat endpoint |
| `search` | 5000 | 5000 | Retrieval/query endpoint |
| `postgres` | 5432 | 5432 | PostgreSQL |
| `postgres` | 5001 | 5001 | File upload/ingestion API |
| `postgres` | 5002 | 5002 | Feedback API |

## API Reference

### Chat

`POST /chat` on `http://localhost:8081/chat`

```bash
curl -X POST "http://localhost:8081/chat" \
  -H "Content-Type: application/json" \
  -d '{"query":"What did Patricia buy?"}'
```

### Retrieval Query

`POST /query` on `http://localhost:5000/query`

```bash
curl -X POST "http://localhost:5000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"Alice bought"}'
```

### File Upload (CSV/TXT/PDF)

`POST /upload` on `http://localhost:5001/upload` (`multipart/form-data`)

```bash
curl -X POST "http://localhost:5001/upload" \
  -F "file=@customer_data_0.csv"
```

### List Context Files

`GET /documents` on `http://localhost:5001/documents`

```bash
curl -X GET "http://localhost:5001/documents"
```

### Delete Context File

`DELETE /documents/{id}` on `http://localhost:5001/documents/{id}`

```bash
curl -X DELETE "http://localhost:5001/documents/1"
```

### Feedback

`POST /feedback` on `http://localhost:5002/feedback`

```bash
curl -X POST "http://localhost:5002/feedback" \
  -H "Content-Type: application/json" \
  -d '{"feedback_type":"thumbsUp","message":"Great answer"}'
```

## Operational Notes

- Secrets/config are read from environment variables (`.env` with Docker Compose).
- Main required variables:
  - `HF_API_TOKEN`
  - `POSTGRES_PASSWORD`
- Optional:
  - `HF_MODEL_ID` (defaults to `Qwen/Qwen2.5-7B-Instruct`)
  - `HF_PROVIDER` (defaults to `auto`; use `hf-inference` only if the model is supported there)
  - `HF_TIMEOUT` (seconds, defaults to `60`)
  - `MAX_DOCUMENTS` (defaults to `3`)
  - `MAX_FILE_SIZE_BYTES` (defaults to `10485760` = 10MB)
  - `MAX_TOTAL_SIZE_BYTES` (defaults to `31457280` = 30MB)
  - `MAX_PDF_PAGES` (defaults to `150`)
- Retrieval cache is built in-memory and can be refreshed via `/refresh`.
- If no file has been uploaded yet, chat responses may indicate missing relevant data.

## Troubleshooting

- `500` from chat API:
  - Check whether the Hugging Face token is valid.
  - Confirm the `search` service is healthy.
- Upload succeeds but retrieval is empty:
  - Inspect `postgres` logs for schema/population errors.
  - Trigger `POST /refresh` manually on the search service.
- CORS/browser errors:
  - Ensure UI is accessed from `http://localhost:8080`.

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Add or update tests where possible
4. Open a pull request with a clear description and reproduction steps

## Security

- Never commit real credentials or private tokens.
- Rotate any token immediately if it was exposed.

## License

No open-source license file is currently included in this repository.  
For open-source distribution, add a `LICENSE` file (for example MIT, Apache-2.0, or GPL-3.0).

## Author

- Rian Lopes
- GitHub: https://github.com/rianbl
- YouTube: https://www.youtube.com/@datarvw
