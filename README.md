# ChatRAG CSV (Dockerized)

A containerized Retrieval-Augmented Generation (RAG) application that lets you upload a CSV file and chat with its content through a web interface.

The stack includes:
- `webapp`: static frontend (NGINX)
- `chatllm`: chat API that orchestrates retrieval + LLM generation
- `search`: semantic retrieval service (FAISS + sentence-transformers)
- `postgres`: database + CSV ingestion API + feedback API

## Features

- Upload a CSV file from the browser (drag-and-drop or file picker)
- Auto-create database schema from CSV columns
- Auto-populate PostgreSQL table with uploaded data
- Build/refresh a FAISS vector store from the database
- Ask questions in natural language and get answers grounded in your data
- Capture thumbs up/down feedback events

## Architecture

```text
Browser (localhost:8080)
  -> Upload CSV -> Data Loader API (postgres:5001)
       -> Create table + insert rows in PostgreSQL (postgres:5432)
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
|- database/    # PostgreSQL image + CSV ingestion + feedback API
|- webapp/      # Static frontend (HTML/CSS/JS via NGINX)
|- secrets/     # Runtime secret files (HF token, Postgres password)
|- docker-compose.yaml
```

## Prerequisites

- Docker + Docker Compose v2
- Hugging Face access token (for inference API)
- (Windows only) Docker Desktop with WSL2 integration recommended

## Quick Start

1. Create required secret files:

```bash
mkdir -p secrets
echo "YOUR_HF_TOKEN" > secrets/hf_api_token.txt
echo "admin" > secrets/postgres_password.txt
```

2. Start the stack:

```bash
docker compose up --build
```

3. Open the app:

```text
http://localhost:8080
```

4. Upload a CSV file in the UI, then ask questions in the chat panel.

## Services and Ports

| Service | Container Port | Host Port | Purpose |
|---|---:|---:|---|
| `webapp` | 80 | 8080 | Frontend UI |
| `chatllm` | 8081 | 8081 | Chat endpoint |
| `search` | 5000 | 5000 | Retrieval/query endpoint |
| `postgres` | 5432 | 5432 | PostgreSQL |
| `postgres` | 5001 | 5001 | CSV upload/data loader API |
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

### CSV Upload

`POST /upload` on `http://localhost:5001/upload` (`multipart/form-data`)

```bash
curl -X POST "http://localhost:5001/upload" \
  -F "file=@customer_data_0.csv"
```

### Feedback

`POST /feedback` on `http://localhost:5002/feedback`

```bash
curl -X POST "http://localhost:5002/feedback" \
  -H "Content-Type: application/json" \
  -d '{"feedback_type":"thumbsUp","message":"Great answer"}'
```

## Operational Notes

- Secrets are read from mounted files:
  - `secrets/hf_api_token.txt`
  - `secrets/postgres_password.txt`
- Current ingestion scripts in `database/` expect PostgreSQL password `admin` (hardcoded).
- Retrieval cache is built in-memory and can be refreshed via `/refresh`.
- If no CSV has been uploaded yet, chat responses may indicate missing relevant data.

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
