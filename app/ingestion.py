import json
import os
import sys
import traceback
import uuid
import re
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from app.qdrant_db import client
from app.embeddings import embedding_model
from app.config import COLLECTION_NAME, VECTOR_SIZE
from app.tokenizer_util import count_tokens, get_tokenizer, MAX_MODEL_TOKENS
from qdrant_client.models import PointStruct, VectorParams, Distance

# -------------------------------
# CONFIG
# -------------------------------
DATA_FOLDER = "data"  # folder containing your .docx and .txt files

# Article-based chunking (legislation): chunk size in tokens
ARTICLE_CHUNK_SIZE_TOKENS = 700
ARTICLE_CHUNK_OVERLAP_TOKENS = 50
ARTICLE_MAX_SINGLE_POINT_TOKENS = 500  # articles <= this stored as one point
# Avoid MemoryError when doc has many tiny "sentences" (e.g. few sentence boundaries)
MAX_SENTENCES_PER_CHUNK = 2000

# Set True to print per-article and per-step details (find infinite loops / repeating steps)
DEBUG_INGESTION = True

# Legacy word-based chunking (kept for non-article flow if needed)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_CHUNK_TOKENS = 1024

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def read_txt(path: str) -> str:
    """Read a TXT file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_docx(path: str) -> str:
    """
    Read DOCX file including paragraphs, tables, and preserve structure.
    """
    doc = Document(path)
    text_parts = []
    
    # Process all elements in document order
    for element in doc.element.body:
        if isinstance(element, CT_P):
            # Paragraph
            para = Paragraph(element, doc)
            text = para.text.strip()
            if text:
                text_parts.append(text)
        elif isinstance(element, CT_Tbl):
            # Table
            table = Table(element, doc)
            table_text = []
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    table_text.append(row_text)
            if table_text:
                text_parts.append(" | ".join(table_text))
    
    return "\n".join(text_parts)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences (handles Uzbek text)."""
    # Split by sentence endings, preserving the punctuation
    sentences = re.split(r'([.!?]+(?:\s+|$))', text)
    # Recombine sentences with their punctuation
    result = []
    i = 0
    while i < len(sentences):
        if sentences[i].strip():
            # Combine sentence with its punctuation if exists
            sentence = sentences[i].strip()
            if i + 1 < len(sentences) and re.match(r'^[.!?]+', sentences[i + 1]):
                sentence += sentences[i + 1].strip()
                i += 1
            if len(sentence) > 5:  # Filter very short fragments
                result.append(sentence)
        i += 1
    return result


# -------------------------------
# ARTICLE EXTRACTION (legislation)
# -------------------------------
# Pattern to detect article boundaries: Modda N, Article N, § N (Uzbek/English/common legal)
_ARTICLE_HEADER_PATTERN = re.compile(
    r"(?m)^(Modda\s+\d+[\.\-]?|Article\s+\d+[\.\-]?|§\s*\d+[\.\-]?)\s*",
    re.IGNORECASE,
)


def extract_articles(text: str) -> list[dict]:
    """
    Split document into articles by standard legislation headers.
    Returns list of {"title": str, "text": str}. If no headers found, whole doc is one article.
    """
    text = text.strip()
    if not text:
        return []

    parts = _ARTICLE_HEADER_PATTERN.split(text)
    # parts[0] = content before first header (preamble or empty); then (header, content), (header, content)...
    articles = []
    if parts[0].strip():
        articles.append({"title": "Kirish" if "kirish" in text[:100].lower() else "Preamble", "text": parts[0].strip()})
    for i in range(1, len(parts) - 1, 2):
        title = (parts[i] or "").strip()
        content = (parts[i + 1] or "").strip()
        if not title:
            continue
        if content:
            articles.append({"title": title, "text": content})

    if not articles:
        return [{"title": "Document", "text": text}]
    return articles


def chunk_article_by_tokens(
    text: str,
    chunk_size: int = ARTICLE_CHUNK_SIZE_TOKENS,
    overlap: int = ARTICLE_CHUNK_OVERLAP_TOKENS,
    debug_label: str = "",
) -> list[str]:
    """
    Split article text into chunks by token count. Respects sentence boundaries
    and never splits mid-word. Uses overlap between consecutive chunks.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    if DEBUG_INGESTION and debug_label:
        print(f"  [chunk_article] {debug_label} | num_sentences={len(sentences)}")

    tokenizer = get_tokenizer()
    sent_tokens = [
        len(tokenizer.encode(s, add_special_tokens=False, max_length=MAX_MODEL_TOKENS, truncation=True))
        for s in sentences
    ]
    chunks = []
    current_chunk_sentences = []
    current_tokens = 0
    i = 0
    _last_log = None
    _repeat_count = 0

    while i < len(sentences):
        sent = sentences[i]
        tok = sent_tokens[i]

        # Single sentence longer than chunk_size: split by words to avoid mid-word, then by token windows
        if tok > chunk_size:
            if DEBUG_INGESTION and debug_label:
                print(f"  [chunk_article] {debug_label} | i={i} LONG_SENT tok={tok} (split by words)")
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_tokens = 0
            words = sent.split()
            segment_tokens = 0
            segment_words = []
            for w in words:
                w_tok = len(tokenizer.encode(w, add_special_tokens=False, max_length=MAX_MODEL_TOKENS, truncation=True))
                if segment_tokens + w_tok > chunk_size and segment_words:
                    chunks.append(" ".join(segment_words))
                    # Overlap: keep last words that fit in overlap
                    overlap_tok = 0
                    overlap_words = []
                    for j in range(len(segment_words) - 1, -1, -1):
                        ow = segment_words[j]
                        ot = len(tokenizer.encode(ow, add_special_tokens=False, max_length=MAX_MODEL_TOKENS, truncation=True))
                        if overlap_tok + ot <= overlap:
                            overlap_words.insert(0, ow)
                            overlap_tok += ot
                        else:
                            break
                    segment_words = overlap_words
                    segment_tokens = overlap_tok
                segment_words.append(w)
                segment_tokens += w_tok
            if segment_words:
                chunks.append(" ".join(segment_words))
            i += 1
            continue

        # Flush when over token limit or when too many sentences (avoid MemoryError on join)
        force_flush = len(current_chunk_sentences) >= MAX_SENTENCES_PER_CHUNK
        if (current_tokens + tok > chunk_size or force_flush) and current_chunk_sentences:
            if DEBUG_INGESTION and debug_label:
                log = f"i={i} FLUSH n_sents={len(current_chunk_sentences)} current_tokens={current_tokens} tok={tok} force_flush={force_flush}"
                if _last_log == log:
                    _repeat_count += 1
                    print(f"  [chunk_article] {debug_label} | *** REPEAT #{_repeat_count} *** {log}")
                else:
                    _repeat_count = 0
                    _last_log = log
                    print(f"  [chunk_article] {debug_label} | {log}")
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(chunk_text)
            # Overlap: take last sentences that fit in overlap tokens (cap overlap list size)
            start_idx = i - len(current_chunk_sentences)
            overlap_tok = 0
            overlap_sents = []
            for j in range(len(current_chunk_sentences) - 1, -1, -1):
                if len(overlap_sents) >= MAX_SENTENCES_PER_CHUNK // 2:
                    break
                t = sent_tokens[start_idx + j]
                if overlap_tok + t <= overlap:
                    overlap_sents.insert(0, current_chunk_sentences[j])
                    overlap_tok += t
                else:
                    break
            current_chunk_sentences = overlap_sents
            current_tokens = overlap_tok
            # If current sentence still doesn't fit with overlap, consume it (split by words) to avoid infinite loop
            if overlap_tok + tok > chunk_size:
                if DEBUG_INGESTION and debug_label:
                    print(f"  [chunk_article] {debug_label} | i={i} WORD_SPLIT (sentence doesn't fit with overlap) overlap_tok={overlap_tok} tok={tok}")
                words = sent.split()
                segment_tokens = 0
                segment_words = []
                for w in words:
                    w_tok = len(tokenizer.encode(w, add_special_tokens=False, max_length=MAX_MODEL_TOKENS, truncation=True))
                    if segment_tokens + w_tok > chunk_size and segment_words:
                        chunks.append(" ".join(segment_words))
                        overlap_tok_w = 0
                        overlap_words = []
                        for jj in range(len(segment_words) - 1, -1, -1):
                            ow = segment_words[jj]
                            ot = len(tokenizer.encode(ow, add_special_tokens=False, max_length=MAX_MODEL_TOKENS, truncation=True))
                            if overlap_tok_w + ot <= overlap:
                                overlap_words.insert(0, ow)
                                overlap_tok_w += ot
                            else:
                                break
                        segment_words = overlap_words
                        segment_tokens = overlap_tok_w
                    segment_words.append(w)
                    segment_tokens += w_tok
                if segment_words:
                    chunks.append(" ".join(segment_words))
                i += 1
        else:
            if DEBUG_INGESTION and debug_label and (i < 5 or i % 500 == 0 or i == len(sentences) - 1):
                log = f"i={i} ADD n_sents={len(current_chunk_sentences)+1} current_tokens={current_tokens}+{tok} sent_preview={sent[:40]!r}..."
                print(f"  [chunk_article] {debug_label} | {log}")
            current_chunk_sentences.append(sent)
            current_tokens += tok
            i += 1

    if current_chunk_sentences:
        # Join in parts if somehow over cap (safety)
        if len(current_chunk_sentences) <= MAX_SENTENCES_PER_CHUNK:
            chunks.append(" ".join(current_chunk_sentences))
        else:
            for start in range(0, len(current_chunk_sentences), MAX_SENTENCES_PER_CHUNK):
                part = current_chunk_sentences[start : start + MAX_SENTENCES_PER_CHUNK]
                chunks.append(" ".join(part))

    if DEBUG_INGESTION and debug_label:
        print(f"  [chunk_article] {debug_label} | DONE total_chunks={len(chunks)}")
    return [c for c in chunks if c.strip()] if chunks else [text] if text.strip() else []


def chunk_text_smart(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Smart chunking that respects sentence boundaries and uses token-aware sizing.
    Better for long context and semantic coherence.
    """
    # First split into sentences
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_words = len(sentence.split())
        # print([current_word_count, sentence_words, chunk_size, current_chunk])

        # If a single sentence is longer than the target chunk size,
        # handle it specially to avoid infinite loops and RAM blow-up.
        if sentence_words > chunk_size:
            # Split this long sentence into fixed-size pieces directly.
            words = sentence.split()
            for start in range(0, len(words), chunk_size):
                sub_sentence = " ".join(words[start:start + chunk_size])
                if sub_sentence.strip():
                    chunks.append(sub_sentence)

            # Reset current chunk state and move on to the next sentence.
            current_chunk = []
            current_word_count = 0
            i += 1
            continue

        # If adding this sentence would exceed chunk size
        if current_word_count + sentence_words > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # Start new chunk with overlap
            # Go back to include overlap sentences
            overlap_words = 0
            overlap_sentences = []
            j = len(current_chunk) - 1
            while j >= 0 and overlap_words < overlap:
                sent = current_chunk[j]
                sent_words = len(sent.split())
                if overlap_words + sent_words <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_words += sent_words
                    j -= 1
                else:
                    break

            current_chunk = overlap_sentences
            current_word_count = overlap_words
            # print([current_word_count, sentence_words, chunk_size, current_chunk])
            # print([overlap_words, overlap_sentences, chunk_text])
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_words
            i += 1
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Filter out very short chunks (likely artifacts)
    chunks = [chunk for chunk in chunks if len(chunk.split()) >= 20]
    
    return chunks


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of tokens (1 token ≈ 0.75 words for Uzbek).
    More accurate than word count for embedding models.
    """
    words = len(text.split())
    return int(words * 1.33)  # Conservative estimate


# -------------------------------
# ARTICLE-BASED INGESTION
# -------------------------------
def document_to_article_payloads(doc_id: str, full_text: str, source_label: str = "") -> list[dict]:
    """
    Convert a document (full text) into a list of payloads for Qdrant.
    Each payload has: doc_id, article_id, article_title, chunk_index, total_chunks, text.
    Articles <= 500 tokens become one point; longer articles are split with 500/50 token chunking.
    """
    articles = extract_articles(full_text)
    if DEBUG_INGESTION:
        print(f"[document_to_article_payloads] total_articles={len(articles)}")
    payloads = []
    for art_idx, art in enumerate(articles):
        title = art["title"]
        text = art["text"].strip()
        if not text:
            if DEBUG_INGESTION:
                print(f"  [article {art_idx}] SKIP empty title={title!r}")
            continue
        article_id = f"{doc_id}#art_{art_idx}"
        tokens = count_tokens(text)
        debug_label = f"art_{art_idx} {title!r}"

        if tokens <= ARTICLE_MAX_SINGLE_POINT_TOKENS:
            if DEBUG_INGESTION:
                print(f"  [article {art_idx}] title={title!r} tokens={tokens} -> SINGLE_POINT")
            payloads.append({
                "doc_id": doc_id,
                "article_id": article_id,
                "article_title": title,
                "chunk_index": 0,
                "total_chunks": 1,
                "text": text,
                "source": source_label,
            })
        else:
            if DEBUG_INGESTION:
                print(f"  [article {art_idx}] title={title!r} tokens={tokens} -> CHUNKING")
            chunks = chunk_article_by_tokens(
                text,
                chunk_size=ARTICLE_CHUNK_SIZE_TOKENS,
                overlap=ARTICLE_CHUNK_OVERLAP_TOKENS,
                debug_label=debug_label,
            )
            for chunk_idx, chunk_text in enumerate(chunks):
                payloads.append({
                    "doc_id": doc_id,
                    "article_id": article_id,
                    "article_title": title,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "text": chunk_text,
                    "source": source_label,
                })
    return payloads


def ingest_articles_from_text(full_text: str, doc_id: str, source_label: str = "") -> list[PointStruct]:
    """
    Build Qdrant points (with vectors) from document text using article-based chunking.
    Returns list of PointStruct ready for upsert.
    """
    if DEBUG_INGESTION:
        print(f"[ingest_articles_from_text] doc_id={doc_id!r} building payloads...")
    payloads = document_to_article_payloads(doc_id, full_text, source_label)

    if DEBUG_INGESTION:
        print(f"[ingest_articles_from_text] payloads={len(payloads)}")
    if not payloads:
        return []

    # Truncate each text to model max length (SentenceTransformer.encode doesn't accept max_length/truncation)
    if DEBUG_INGESTION:
        print(f"[ingest_articles_from_text] truncating {len(payloads)} texts...")
    tokenizer = get_tokenizer()
    texts = []
    for i, p in enumerate(payloads):
        t = p["text"]
        ids = tokenizer.encode(t, add_special_tokens=False, max_length=MAX_MODEL_TOKENS, truncation=True)
        texts.append(tokenizer.decode(ids, skip_special_tokens=True) if ids else t)
        if DEBUG_INGESTION and (i + 1) % 50 == 0:
            print(f"[ingest_articles_from_text]   truncation progress: {i + 1}/{len(payloads)}")
    if DEBUG_INGESTION:
        print(f"[ingest_articles_from_text] truncation done. Calling embedding_model.encode({len(texts)} texts)...")
    vectors = embedding_model.encode(texts, normalize_embeddings=True)
    if DEBUG_INGESTION:
        print(f"[ingest_articles_from_text] encode() returned. Converting to list...")
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    else:
        vectors = [v.tolist() for v in vectors]

    if DEBUG_INGESTION:
        print(f"[ingest_articles_from_text] building {len(payloads)} points...")
    points = []
    for i, (payload, vec) in enumerate(zip(payloads, vectors)):
        # Payload for Qdrant: only stored fields (no 'source' required by spec but useful)
        p = {
            "doc_id": payload["doc_id"],
            "article_id": payload["article_id"],
            "article_title": payload["article_title"],
            "chunk_index": payload["chunk_index"],
            "total_chunks": payload["total_chunks"],
            "text": payload["text"],
        }
        if payload.get("source"):
            p["source"] = payload["source"]
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=p,
            )
        )
        if DEBUG_INGESTION and (i + 1) % 50 == 0:
            print(f"[ingest_articles_from_text]   points progress: {i + 1}/{len(payloads)}")
    if DEBUG_INGESTION:
        print(f"[ingest_articles_from_text] done. points={len(points)}")
    return points


# -------------------------------
# INGEST DOCX FILE
# -------------------------------
def ensure_collection():
    """Create Qdrant collection if it does not exist (cosine, vector size from config)."""
    try:
        collections = client.get_collections().collections
        if COLLECTION_NAME not in [c.name for c in collections]:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"  ✅ Collection '{COLLECTION_NAME}' created!")
    except Exception as e:
        print(f"  ⚠️  Collection check failed: {e}")


def upload_docx(file_path: str) -> dict:
    """
    Read DOCX, treat as articles, chunk by tokens (500/50), embed with BGE-M3, upload to Qdrant.
    Returns {"ok": bool, "message": str, "filename": str, "chunks": int}.
    """
    filename = os.path.basename(file_path)
    if filename.startswith("~$") or filename.startswith("~"):
        print(f"  ⏭️  Skipping temporary file: {filename}")
        return {"ok": False, "message": "Skipping temporary file", "filename": filename, "chunks": 0}

    print(f"Processing: {file_path}")
    try:
        text = read_docx(file_path)
    except Exception as e:
        print(f"  ❌ Error reading {file_path}: {e}")
        return {"ok": False, "message": str(e), "filename": filename, "chunks": 0}

    if not text or len(text.strip()) < 50:
        print(f"  ⚠️  Skipping {file_path}: document too short or empty")
        return {"ok": False, "message": "Document too short or empty", "filename": filename, "chunks": 0}

    doc_id = str(os.path.abspath(file_path))
    source_label = filename
    try:
        points = ingest_articles_from_text(text, doc_id=doc_id, source_label=source_label)
    except Exception as e:
        print(f"  ❌ Error building points from {file_path}: {e}", file=sys.stderr)
        traceback.print_exc()
        return {"ok": False, "message": str(e), "filename": filename, "chunks": 0}

    if not points:
        print(f"  ⚠️  No article points from {file_path}")
        return {"ok": False, "message": "No article points created", "filename": filename, "chunks": 0}

    ensure_collection()
    batch_size = 50
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            print(f"  ✅ Uploaded batch of {len(batch)} points")
        except Exception as e:
            print(f"  ❌ Error uploading batch: {e}")
            return {"ok": False, "message": str(e), "filename": filename, "chunks": 0}
    print(f"  ✅ Ingested {len(points)} points from {file_path}\n")
    return {"ok": True, "message": f"Ingested {filename}", "filename": filename, "chunks": len(points)}


def upload_file(file_path: str) -> dict:
    """
    Ingest a single file (DOCX or TXT). Returns result dict with status and message.
    """
    filename = os.path.basename(file_path)
    if filename.startswith("~$") or filename.startswith("~"):
        return {"ok": False, "message": "Skipping temporary file", "chunks": 0}

    if not os.path.exists(file_path):
        return {"ok": False, "message": "File not found", "chunks": 0}

    if filename.endswith(".docx"):
        result = upload_docx(file_path)
        return result
    elif filename.endswith(".txt"):
        try:
            text = read_txt(file_path)
            if not text or len(text.strip()) < 50:
                return {"ok": False, "message": "File too short or empty", "filename": filename}
            doc_id = str(os.path.abspath(file_path))
            points = ingest_articles_from_text(text, doc_id=doc_id, source_label=filename)
            if not points:
                return {"ok": False, "message": "No article points created", "filename": filename}
            ensure_collection()
            if points:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
            return {"ok": True, "message": f"Ingested {filename}", "filename": filename, "chunks": len(points)}
        except Exception as e:
            return {"ok": False, "message": str(e), "filename": filename}
    else:
        return {"ok": False, "message": "Only .docx and .txt are supported", "filename": filename}


# -------------------------------
# Q&A INGESTION (same collection)
# -------------------------------
def upload_qa_from_data(
    pairs: list[dict],
    source: str = "qa",
) -> dict:
    """
    Ingest question-answer pairs.
    """
    if not pairs:
        return {"ok": False, "message": "No pairs provided", "count": 0}
    try:
        collections = client.get_collections().collections
        if COLLECTION_NAME not in [c.name for c in collections]:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
    except Exception:
        pass
    points = []
    for i, item in enumerate(pairs):
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if not q and not a:
            continue
        # One searchable text so similar questions retrieve this QA
        text = f"Savol: {q}\nJavob: {a}"
        if len(text) > 8000:
            text = text[:8000]
        try:
            vector = embedding_model.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            continue
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": text,
                    "question": q,
                    "answer": a,
                    "source": source,
                    "type": "qa",
                    "language": "uzbek",
                },
            )
        )
    if not points:
        return {"ok": False, "message": "No valid pairs to ingest", "count": 0}
    try:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        return {"ok": True, "message": f"Ingested {len(points)} Q&A pairs", "count": len(points), "source": source}
    except Exception as e:
        return {"ok": False, "message": str(e), "count": 0}


def upload_qa_file(file_path: str) -> dict:
    """Load Q&A JSON from file and ingest into the same collection."""
    if not os.path.exists(file_path):
        return {"ok": False, "message": "File not found", "count": 0}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {"ok": False, "message": f"Invalid JSON: {e}", "count": 0}
    except Exception as e:
        return {"ok": False, "message": str(e), "count": 0}
    # Support both { "pairs": [...] } and [ { "question": "...", "answer": "..." }, ... ]
    if isinstance(data, list):
        pairs = data
        source = os.path.basename(file_path)
    else:
        pairs = data.get("pairs") or data.get("qa") or []
        source = data.get("source") or os.path.basename(file_path)
    return upload_qa_from_data(pairs=pairs, source=source)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: Data folder '{DATA_FOLDER}' not found!")
        exit(1)
    
    # Filter: .docx, .txt, and .json (Q&A)
    all_files = os.listdir(DATA_FOLDER)
    doc_files = [
        f for f in all_files
        if (f.endswith(".docx") or f.endswith(".txt"))
        and not f.startswith("~$")
        and not f.startswith("~")
    ]
    qa_files = [f for f in all_files if f.endswith(".json") and not f.startswith("~")]
    
    if not doc_files and not qa_files:
        print(f"⚠️  No .docx, .txt or .json (Q&A) files found in '{DATA_FOLDER}'")
        exit(0)
    
    print(f"📚 Found {len(doc_files)} doc(s) and {len(qa_files)} Q&A JSON(s)\n")
    print("=" * 60)

    failed = []
    for filename in doc_files:
        file_path = os.path.join(DATA_FOLDER, filename)
        result = upload_file(file_path)
        status = "✅" if result.get("ok") else "❌"
        chunks = result.get("chunks", 0)
        print(f"{status} {filename}: {result.get('message', '')}" + (f" ({chunks} chunks)" if chunks else ""))
        if not result.get("ok"):
            failed.append((filename, result.get("message", "Unknown error")))

    for filename in qa_files:
        file_path = os.path.join(DATA_FOLDER, filename)
        result = upload_qa_file(file_path)
        status = "✅" if result.get("ok") else "❌"
        print(f"{status} Q&A {filename}: {result.get('message', '')} (count: {result.get('count', 0)})")
        if not result.get("ok"):
            failed.append((filename, result.get("message", "Unknown error")))

    print("=" * 60)
    if failed:
        print("❌ Some files failed:")
        for name, msg in failed:
            print(f"   - {name}: {msg}")
        exit(1)
    print("✅ All files processed successfully!")
