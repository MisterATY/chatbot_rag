## FastAPI RAG Service Documentation

This project exposes a FastAPI service that:

- Answers questions using **RAG** (Qdrant + embeddings + reranker + your LLM server)
- Ingests `.docx` and `.txt` files into Qdrant over HTTP
- Ingests **question-answer (Q&A) pairs** from JSON into the same collection

Backend app file: `app/main.py`

---

## 1. Prerequisites

- Python virtualenv activated
- Dependencies installed:

```bash
pip install -r requirements.txt
```

Qdrant must be running and reachable at the host/port configured in `app/config.py`:

- `QDRANT_HOST`
- `QDRANT_PORT`

Your LLM server must also be reachable:

- `LLM_SERVER_URL`
- `LLM_MODEL`

Configured in `app/config.py`.

---

## 2. Starting the FastAPI server

The FastAPI app is defined in `app/main.py` as `app`.

You can run it in two ways from the project root (`client` folder):

### Option A – via `python -m app.main`

`app/main.py` reads `API_HOST` and `API_PORT` from `app/config.py`:

- `API_HOST = "0.0.0.0"`
- `API_PORT = 8080`

Run:

```bash
python -m app.main
```

The service will listen on:

```text
http://0.0.0.0:8080
```

On the same machine you can use:

```text
http://localhost:8080
```

### Option B – via `uvicorn` directly

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

You can change the port if needed:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

---

## 3. API Overview

Base URL (assuming default config):

```text
http://localhost:8080
```

### 3.1 Health check

**GET `/`**

Simple status:

```bash
curl "http://localhost:8080/"
```

**GET `/health`**

```bash
curl "http://localhost:8080/health"
```

---

### 3.2 Ask a question (RAG)

Endpoint uses:

- Qdrant retrieval (`app/retriever.py`)
- Reranker (`app/reranker.py`)
- Context extraction
- Your LLM server (`LLM_SERVER_URL`) for final answer

#### GET `/ask`

Query parameters:

- `query` (**required**): the question
- `top_k` (optional, default 5): how many chunks to retrieve from Qdrant
- `max_context_sentences` (optional, default 3): how many sentences per context chunk

Example (simple):

```bash
curl "http://localhost:8080/ask?query=savol"
```

Example (with extra params):

```bash
curl "http://localhost:8080/ask?query=tovarlarni%20eksport%20qilish&top_k=5&max_context_sentences=3"
```

Response schema:

```json
{
  "answer": "LLM-generated answer text",
  "context": ["context chunk 1", "context chunk 2", "..."],
  "query": "original query string"
}
```

#### POST `/ask`

JSON body:

```json
{
  "query": "savol",
  "top_k": 5,
  "max_context_sentences": 3
}
```

Example (Windows PowerShell / CMD):

```bash
curl -X POST "http://localhost:8080/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"savol\", \"top_k\": 5, \"max_context_sentences\": 3}"
```

Example (Linux/macOS):

```bash
curl -X POST "http://localhost:8080/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "savol", "top_k": 5, "max_context_sentences": 3}'
```

---

## 4. Ingest files over HTTP

The service can ingest `.docx` and `.txt` files into Qdrant.

Implementation: `app/main.py` → `/ingest` uses `upload_file()` from `app/ingestion_docx.py`.

### 4.1 Ingest a single file

**POST `/ingest`**

Multipart form-data:

- Field name: `file`
- File: `.docx` or `.txt`

Example (Windows PowerShell / CMD):

```bash
curl -X POST "http://localhost:8080/ingest" ^
  -F "file=@C:\path\to\document.docx"
```

Example (Linux/macOS):

```bash
curl -X POST "http://localhost:8080/ingest" \
  -F "file=@/path/to/document.docx"
```

Sample response:

```json
{
  "ok": true,
  "message": "Ingested document.docx",
  "filename": "document.docx",
  "chunks": 42
}
```

### 4.2 Ingest multiple files

**POST `/ingest/files`**

Multipart form-data:

- Field name: `files`
- Multiple files: `.docx` and/or `.txt`

Example (Windows PowerShell / CMD):

```bash
curl -X POST "http://localhost:8080/ingest/files" ^
  -F "files=@C:\path\to\doc1.docx" ^
  -F "files=@C:\path\to\doc2.txt"
```

Example (Linux/macOS):

```bash
curl -X POST "http://localhost:8080/ingest/files" \
  -F "files=@/path/to/doc1.docx" \
  -F "files=@/path/to/doc2.txt"
```

Sample response:

```json
{
  "results": [
    {
      "filename": "doc1.docx",
      "ok": true,
      "message": "Ingested doc1.docx",
      "chunks": 33
    },
    {
      "filename": "doc2.txt",
      "ok": false,
      "message": "File too short or empty"
    }
  ]
}
```

### 4.3 Ingest question-answer pairs (same collection)

You can add Q&A data to the same Qdrant collection so RAG retrieves both documents and Q&A.

**JSON format** (see `data/qa_example.json`):

```json
{
  "source": "my_qa.json",
  "pairs": [
    { "question": "Savol matni?", "answer": "Javob matni." }
  ]
}
```

- `source`: optional name for this set of pairs.
- `pairs`: array of `{ "question": "...", "answer": "..." }`.

**POST `/ingest/qa`** (JSON body)

```bash
curl -X POST "http://localhost:8080/ingest/qa" \
  -H "Content-Type: application/json" \
  -d '{"pairs":[{"question":"Savol?","answer":"Javob."}],"source":"my_qa"}'
```

**POST `/ingest/qa/file`** (upload JSON file)

```bash
curl -X POST "http://localhost:8080/ingest/qa/file" \
  -F "file=@data/qa_example.json"
```

From disk: put your JSON file in the `data/` folder and run:

```bash
python -m app.ingestion
```

The script ingests all `.docx`, `.txt`, and `.json` (Q&A) files from `data/`.

---

## 5. Interactive API docs

FastAPI automatically exposes docs:

- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

You can:

- Try `/ask`, `/ingest`, `/ingest/files`, `/ingest/qa`, `/ingest/qa/file` from the browser
- Inspect request/response schemas

---

## 6. Typical workflow

1. **Start Qdrant** (e.g. Docker, local binary).
2. **Ensure LLM server is running** at `LLM_SERVER_URL`.
3. **Start FastAPI server**:

   ```bash
   python -m app.main
   ```

4. **Ingest documents and Q&A**:
   - Docs: `/ingest` or `/ingest/files`
   - Q&A: `/ingest/qa` (JSON body) or `/ingest/qa/file` (upload JSON)
   - From disk: put files in `data/` and run `python -m app.ingestion` (processes `.docx`, `.txt`, and `.json` Q&A files)

5. **Ask questions** via `/ask` (GET or POST).

