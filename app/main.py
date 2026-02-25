"""
FastAPI server: ask questions (RAG) and ingest files via HTTP.
"""
import json
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.ingestion import upload_file as ingest_upload_file, upload_qa_from_data, upload_qa_file
from app.retriever import retrieve_and_answer

# -------------------------------
# APP
# -------------------------------
app = FastAPI(
    title="Qdrant RAG API",
    description="Ask questions (RAG) and ingest DOCX/TXT files via HTTP.",
    version="1.0.0",
)


# -------------------------------
# REQUEST / RESPONSE MODELS
# -------------------------------
class AskRequest(BaseModel):
    """Body for POST /ask."""

    query: str = Field(..., description="Question to answer using RAG")
    top_k: int = Field(12, ge=1, le=50, description="Number of chunks to retrieve from Qdrant")
    max_context_tokens: int = Field(3500, ge=500, le=8000, description="Max tokens in assembled context for LLM")
    lang: Optional[str] = Field("oz", description="Language code: oz (default), uz, ru, en")


class AskResponse(BaseModel):
    """Response for /ask."""

    answer: str
    context: list[str]
    query: str


class IngestResponse(BaseModel):
    """Response for /ingest."""

    ok: bool
    message: str
    filename: Optional[str] = None
    chunks: Optional[int] = None


class QAPair(BaseModel):
    """One question-answer pair."""

    question: str = Field(..., description="Question text")
    answer: str = Field(..., description="Answer text")


class IngestQABody(BaseModel):
    """Body for POST /ingest/qa."""

    pairs: list[QAPair] = Field(..., description="List of question-answer pairs")
    source: Optional[str] = Field("qa", description="Source name for these pairs")


# -------------------------------
# ROUTES
# -------------------------------
@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "service": "Qdrant RAG API"}


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/ask", response_model=AskResponse)
def ask_get(
    query: str = Query(..., description="Question to answer"),
    top_k: int = Query(12, ge=1, le=50),
    max_context_tokens: int = Query(3500, ge=500, le=8000),
    lang: str = Query("oz", description="Language code: oz (default), uz, ru, en"),
):
    """Get an answer to a question using RAG (GET)."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    result = retrieve_and_answer(
        query=query.strip(),
        top_k=top_k,
        max_context_tokens=max_context_tokens,
        lang=lang,
    )
    return AskResponse(
        answer=result["answer"],
        context=result["context"],
        query=query.strip(),
    )


@app.post("/ask", response_model=AskResponse)
def ask_post(body: AskRequest):
    """Get an answer to a question using RAG (POST)."""
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    result = retrieve_and_answer(
        query=body.query.strip(),
        top_k=body.top_k,
        max_context_tokens=body.max_context_tokens,
        lang=(body.lang or "oz"),
    )
    return AskResponse(
        answer=result["answer"],
        context=result["context"],
        query=body.query.strip(),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Ingest a single file (DOCX or TXT)."""
    filename = file.filename or "unknown"
    if not filename.lower().endswith((".docx", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only .docx and .txt files are supported",
        )

    suffix = ".docx" if filename.lower().endswith(".docx") else ".txt"
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    # Save to a temporary file and ingest
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingest_upload_file(tmp_path)
        return IngestResponse(
            ok=result.get("ok", False),
            message=result.get("message", ""),
            filename=result.get("filename"),
            chunks=result.get("chunks"),
        )
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/ingest/files")
async def ingest_multiple(files: list[UploadFile] = File(...)):
    """Ingest multiple files (DOCX or TXT)."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    results = []
    tmp_paths = []

    for file in files:
        filename = file.filename or "unknown"
        if not filename.lower().endswith((".docx", ".txt")):
            results.append(
                {
                    "filename": filename,
                    "ok": False,
                    "message": "Only .docx and .txt are supported",
                }
            )
            continue

        suffix = ".docx" if filename.lower().endswith(".docx") else ".txt"
        try:
            content = await file.read()
        except Exception as e:
            results.append({"filename": filename, "ok": False, "message": str(e)})
            continue

        if not content:
            results.append({"filename": filename, "ok": False, "message": "File is empty"})
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            tmp_paths.append(tmp_path)

        result = ingest_upload_file(tmp_path)
        results.append(
            {
                "filename": filename,
                "ok": result.get("ok", False),
                "message": result.get("message", ""),
                "chunks": result.get("chunks"),
            }
        )

    for path in tmp_paths:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except Exception:
                pass

    return {"results": results}


@app.post("/ingest/qa")
def ingest_qa(body: IngestQABody):
    """Ingest question-answer pairs into the same collection (JSON body)."""
    pairs = [{"question": p.question, "answer": p.answer} for p in body.pairs]
    result = upload_qa_from_data(pairs=pairs, source=body.source or "qa")
    return {
        "ok": result.get("ok", False),
        "message": result.get("message", ""),
        "count": result.get("count", 0),
        "source": result.get("source"),
    }


@app.post("/ingest/qa/file")
async def ingest_qa_file(file: UploadFile = File(...)):
    """Ingest question-answer pairs from a JSON file (same format as data/qa_example.json)."""
    filename = file.filename or "unknown"
    if not filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are supported")
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    if isinstance(data, list):
        pairs = data
        source = filename
    else:
        pairs = data.get("pairs") or data.get("qa") or []
        source = data.get("source") or filename
    result = upload_qa_from_data(pairs=pairs, source=source)
    return {
        "ok": result.get("ok", False),
        "message": result.get("message", ""),
        "count": result.get("count", 0),
        "source": result.get("source"),
    }


if __name__ == "__main__":
    import uvicorn

    from app.config import API_HOST, API_PORT

    uvicorn.run(app, host=API_HOST, port=API_PORT)
