import json
import os
import sys
import traceback
import uuid
import re
import numpy as np
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from app.qdrant_db import client
from app.embeddings import embedding_model
from app.config import COLLECTION_NAME, VECTOR_SIZE, DEFAULT_LANG, LANGUAGES
from app.tokenizer_util import count_tokens, get_tokenizer, MAX_MODEL_TOKENS
from app.file_registry import compute_file_hash, get_doc_id_by_hash, register_file
from qdrant_client.models import PointStruct, VectorParams, Distance

# -------------------------------
# CONFIG
# -------------------------------
DATA_FOLDER = "data"
# Legislation: articles (1-modda, 2-modda) = one point each; normal docs = chunked by tokens
DATA_ARTICLES_FOLDER = "data/articles"   # each article (modda) stored as single chunk
DATA_DOCUMENTS_FOLDER = "data/documents"  # normal text/table, chunked by token size

# Article-based chunking (legislation): chunk size in tokens
ARTICLE_CHUNK_SIZE_TOKENS = 500
ARTICLE_CHUNK_OVERLAP_TOKENS = 50
ARTICLE_MAX_SINGLE_POINT_TOKENS = 500  # articles <= this stored as one point
# Avoid MemoryError when doc has many tiny "sentences" (e.g. few sentence boundaries)
MAX_SENTENCES_PER_CHUNK = 2000
# Trailing fragment from word-split below this (tokens) is merged into previous chunk to avoid tiny chunks
MIN_CHUNK_TOKENS = 300
# Normal documents: paragraph below this (tokens) = one chunk with group_id=-1; above = split with same group_id
DOCUMENT_PARAGRAPH_CHUNK_SIZE = 500
# Table storage: "json" or "markdown"; chunked by token size (DOCUMENT_PARAGRAPH_CHUNK_SIZE), never exceed chunk size
TABLE_STORE_FORMAT = "json"
# Section headers for document paragraphs: 1-bob, 2-bob, 3-bo'lim, etc. (case-insensitive)
DOCUMENT_SECTION_HEADER_PATTERN = re.compile(
    r"(?m)^\s*\d+[\-\.]?\s*(?:bob|bo'lim)[\s\.\-]",
    re.IGNORECASE,
)

# Set True to print per-article and per-step details (find infinite loops / repeating steps)
DEBUG_INGESTION = True
# Encode embeddings in batches to avoid RAM crash on large docs; batch size when True
EMBED_BY_CHUNK = False
EMBED_BATCH_SIZE = 50

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


def read_docx_elements(path: str) -> list[dict]:
    """
    Read DOCX and return elements in order: paragraphs and tables.
    Each element is {"type": "paragraph", "text": str} or {"type": "table", "data": list[list[str]]} (rows of cell values).
    """
    doc = Document(path)
    elements = []
    for element in doc.element.body:
        if isinstance(element, CT_P):
            para = Paragraph(element, doc)
            text = para.text.strip()
            if text:
                elements.append({"type": "paragraph", "text": text})
        elif isinstance(element, CT_Tbl):
            table = Table(element, doc)
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    rows.append(cells)
            if rows:
                elements.append({"type": "table", "data": rows})
    return elements


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
# Pattern to detect article boundaries: 1-modda, 2-modda, Modda N, Article N, § N (Uzbek/English/common legal)
_ARTICLE_HEADER_PATTERN = re.compile(
    r"(?m)^(\d+[\-\.]?\s*modda[\s\.\-]*|Modda\s+\d+[\.\-]?|Article\s+\d+[\.\-]?|§\s*\d+[\.\-]?)\s*",
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
    chunk_size = max(1, chunk_size)
    overlap = min(overlap, chunk_size - 1) if chunk_size > 1 else 0
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
    max_iter = max(len(sentences) * 3, 10000)

    while i < len(sentences):
        if i >= max_iter:
            if DEBUG_INGESTION and debug_label:
                print(f"  [chunk_article] {debug_label} | SAFETY: flushing remainder at i={i} (max_iter={max_iter})")
            current_chunk_sentences.extend(sentences[i:])
            i = len(sentences)
            break
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
                tail_text = " ".join(segment_words)
                if segment_tokens < MIN_CHUNK_TOKENS and chunks:
                    chunks[-1] = chunks[-1] + " " + tail_text
                else:
                    chunks.append(tail_text)
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
                    tail_text = " ".join(segment_words)
                    if segment_tokens < MIN_CHUNK_TOKENS and chunks:
                        chunks[-1] = chunks[-1] + " " + tail_text
                    else:
                        chunks.append(tail_text)
                i += 1
            else:
                # Sentence fits with overlap; add it and advance to avoid infinite loop
                current_chunk_sentences.append(sent)
                current_tokens += tok
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
# NORMAL DOCUMENTS (section-based paragraphs: bob, bo'lim; group_id = first chunk index or -1)
# -------------------------------
def split_document_into_section_paragraphs(full_text: str) -> list[str]:
    """
    Split document into paragraphs by section headers: 1-bob, 2-bob, 3-bo'lim, etc.
    Each paragraph starts at a header or at the start of the document (intro).
    """
    full_text = full_text.strip()
    if not full_text:
        return []
    matches = list(DOCUMENT_SECTION_HEADER_PATTERN.finditer(full_text))
    if not matches:
        return [full_text]
    paragraphs = []
    # Intro before first header
    if matches[0].start() > 0:
        intro = full_text[: matches[0].start()].strip()
        if intro:
            paragraphs.append(intro)
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        part = full_text[start:end].strip()
        if part:
            paragraphs.append(part)
    return paragraphs


def _dedupe_row_cells(row: list[str]) -> list[str]:
    """
    Return unique non-empty cell values in order (first occurrence kept).
    Used for layout/org-chart tables where merged cells repeat the same value many times.
    """
    seen = set()
    out = []
    for c in row:
        s = str(c).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def table_data_to_string(data: list[list[str]], fmt: str = "json") -> str:
    """
    Convert table rows to a single string for storage. fmt: "json" or "markdown".
    Rows are deduplicated (unique non-empty values per row) to avoid storing repeated
    values from merged cells / unstructured tables (e.g. org charts).
    """
    if not data:
        return "" if fmt == "markdown" else "[]"
    deduped = [_dedupe_row_cells(row) for row in data]
    deduped = [row for row in deduped if row]
    if not deduped:
        return "" if fmt == "markdown" else "[]"
    if fmt == "markdown":
        lines = []
        for i, row in enumerate(deduped):
            cells = [c.replace("|", "\\|").replace("\n", " ") for c in row]
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("| " + " | ".join("---" for _ in row) + " |")
        return "\n".join(lines)
    return json.dumps(deduped, ensure_ascii=False)


def elements_to_blocks(elements: list[dict]) -> list[dict]:
    """
    Convert DOCX elements (paragraphs + tables) into blocks: section text blocks and table blocks.
    Section paragraphs are grouped by bob/bo'lim headers; each table is its own block with separate group_id.
    Headings above a table (last 1–2 paragraphs before the table) are stored with the table in "heading".
    Returns list of {"type": "section", "text": str} or {"type": "table", "data": ..., "heading": str}.
    """
    blocks = []
    section_lines = []
    for el in elements:
        if el["type"] == "paragraph":
            line = el["text"]
            if not line.strip():
                continue
            if section_lines and DOCUMENT_SECTION_HEADER_PATTERN.match(line):
                blocks.append({"type": "section", "text": "\n\n".join(section_lines)})
                section_lines = [line]
            else:
                section_lines.append(line)
        elif el["type"] == "table":
            heading_lines = []
            if section_lines:
                heading_lines = section_lines[-2:] if len(section_lines) >= 2 else section_lines[-1:]
                section_lines = section_lines[: -len(heading_lines)]
            if section_lines:
                blocks.append({"type": "section", "text": "\n\n".join(section_lines)})
                section_lines = []
            heading = "\n\n".join(heading_lines).strip() if heading_lines else ""
            blocks.append({"type": "table", "data": el["data"], "heading": heading})
    if section_lines:
        blocks.append({"type": "section", "text": "\n\n".join(section_lines)})
    return blocks


def document_to_document_payloads(
    doc_id: str,
    full_text: str | None = None,
    source_label: str = "",
    elements: list[dict] | None = None,
) -> list[dict]:
    """
    For normal documents (data/documents).
    If elements is provided (from DOCX): sections by bob/bo'lim + tables as JSON; tables get separate group_id.
    If full_text is provided (from TXT): split by section headers only.
    - Single-chunk block: group_id=-1. Multi-chunk block: group_id = first chunk's global index.
    """
    if elements is not None:
        blocks = elements_to_blocks(elements)
        if DEBUG_INGESTION:
            print(f"[document_to_document_payloads] blocks from elements: {len(blocks)}")
    else:
        if full_text is None:
            full_text = ""
        paragraphs = split_document_into_section_paragraphs(full_text)
        if DEBUG_INGESTION:
            print(f"[document_to_document_payloads] section paragraphs={len(paragraphs)}")
        blocks = [{"type": "section", "text": p} for p in paragraphs]

    payloads = []
    table_group_counter = 0  # separate group_id space for tables (negative: -2, -3, ...)
    for block in blocks:
        if block["type"] == "section":
            para = block["text"]
            tokens = count_tokens(para)
            if tokens <= DOCUMENT_PARAGRAPH_CHUNK_SIZE:
                payloads.append({
                    "doc_id": doc_id,
                    "type": "document",
                    "segment_id": 0,
                    "title": "",
                    "chunk_index": len(payloads),
                    "total_chunks": 0,
                    "group_id": -1,
                    "text": para,
                    "source": source_label,
                })
            else:
                chunks = chunk_article_by_tokens(
                    para,
                    chunk_size=DOCUMENT_PARAGRAPH_CHUNK_SIZE,
                    overlap=ARTICLE_CHUNK_OVERLAP_TOKENS,
                    debug_label="doc_section",
                )
                first_chunk_index = len(payloads)
                for chunk_text in chunks:
                    payloads.append({
                        "doc_id": doc_id,
                        "type": "document",
                        "segment_id": 0,
                        "title": "",
                        "chunk_index": len(payloads),
                        "total_chunks": 0,
                        "group_id": first_chunk_index,
                        "text": chunk_text,
                        "source": source_label,
                    })
        else:
            data = block["data"]
            deduped_data = [row for row in (_dedupe_row_cells(r) for r in data) if row]
            heading = block.get("heading", "").strip()
            table_body = table_data_to_string(deduped_data, TABLE_STORE_FORMAT)
            table_str = (heading + "\n\n" + table_body) if heading else table_body
            tokens = count_tokens(table_str)
            num_rows = len(deduped_data)
            if DEBUG_INGESTION:
                print(f"[document_to_document_payloads] [TABLE] rows={num_rows} (deduped) tokens={tokens} format={TABLE_STORE_FORMAT} str_len={len(table_str)}")
            if tokens <= DOCUMENT_PARAGRAPH_CHUNK_SIZE:
                if DEBUG_INGESTION:
                    print(f"[document_to_document_payloads]   -> single chunk group_id=-1 payload_idx={len(payloads)} (within chunk size)")
                payloads.append({
                    "doc_id": doc_id,
                    "type": "document",
                    "segment_id": 0,
                    "title": "",
                    "chunk_index": len(payloads),
                    "total_chunks": 0,
                    "group_id": -1,
                    "text": table_str,
                    "table": True,
                    "table_data": deduped_data,
                    "source": source_label,
                })
            else:
                table_group_counter += 1
                table_group_id = -(table_group_counter + 1)
                chunks = chunk_article_by_tokens(
                    table_str,
                    chunk_size=DOCUMENT_PARAGRAPH_CHUNK_SIZE,
                    overlap=ARTICLE_CHUNK_OVERLAP_TOKENS,
                    debug_label="table",
                )
                if DEBUG_INGESTION:
                    print(f"[document_to_document_payloads]   -> multi-chunk group_id={table_group_id} num_chunks={len(chunks)} payload_idx_from={len(payloads)} (token-split, max chunk size)")
                for chunk_text in chunks:
                    if DEBUG_INGESTION:
                        print(f"[document_to_document_payloads]     table chunk payload_idx={len(payloads)} text_len={len(chunk_text)}")
                    payloads.append({
                        "doc_id": doc_id,
                        "type": "document",
                        "segment_id": 0,
                        "title": "",
                        "chunk_index": len(payloads),
                        "total_chunks": 0,
                        "group_id": table_group_id,
                        "text": chunk_text,
                        "table": True,
                        "source": source_label,
                    })
    n = len(payloads)
    for p in payloads:
        p["total_chunks"] = n
    return payloads


# -------------------------------
# ARTICLE-BASED INGESTION
# -------------------------------
def document_to_article_payloads(
    doc_id: str,
    full_text: str,
    source_label: str = "",
    one_article_per_point: bool = False,
) -> list[dict]:
    """
    Convert a document (full text) into a list of payloads for Qdrant.
    - one_article_per_point=True (article files): type="article", one point per modda; chunk_index 0..N-1, total_chunks=N.
    - one_article_per_point=False (normal docs): type="document", chunked by token size.
    """
    content_type = "article" if one_article_per_point else "document"
    articles = extract_articles(full_text)
    if DEBUG_INGESTION:
        print(f"[document_to_article_payloads] total_articles={len(articles)} one_article_per_point={one_article_per_point} type={content_type}")
    payloads = []
    for art_idx, art in enumerate(articles):
        title = art["title"]
        text = art["text"].strip()
        if not text:
            if DEBUG_INGESTION:
                print(f"  [article {art_idx}] SKIP empty title={title!r}")
            continue
        segment_id = art_idx
        tokens = count_tokens(text)
        debug_label = f"art_{art_idx} {title!r}"

        if one_article_per_point:
            if tokens <= ARTICLE_CHUNK_SIZE_TOKENS:
                if DEBUG_INGESTION:
                    print(f"  [article {art_idx}] title={title!r} tokens={tokens} -> SINGLE_CHUNK group_id=-1")
                payloads.append({
                    "doc_id": doc_id,
                    "type": content_type,
                    "segment_id": segment_id,
                    "title": title,
                    "chunk_index": len(payloads),
                    "total_chunks": 0,
                    "group_id": -1,
                    "text": text,
                    "source": source_label,
                })
            else:
                if DEBUG_INGESTION:
                    print(f"  [article {art_idx}] title={title!r} tokens={tokens} -> CHUNKING group_id={art_idx}")
                chunks = chunk_article_by_tokens(
                    text,
                    chunk_size=ARTICLE_CHUNK_SIZE_TOKENS,
                    overlap=ARTICLE_CHUNK_OVERLAP_TOKENS,
                    debug_label=debug_label,
                )
                for chunk_text in chunks:
                    payloads.append({
                        "doc_id": doc_id,
                        "type": content_type,
                        "segment_id": segment_id,
                        "title": title,
                        "chunk_index": len(payloads),
                        "total_chunks": 0,
                        "group_id": art_idx,
                        "text": chunk_text,
                        "source": source_label,
                    })
        elif tokens <= ARTICLE_MAX_SINGLE_POINT_TOKENS:
            if DEBUG_INGESTION:
                print(f"  [article {art_idx}] title={title!r} tokens={tokens} -> SINGLE_POINT")
            payloads.append({
                "doc_id": doc_id,
                "type": content_type,
                "segment_id": segment_id,
                "title": title,
                "chunk_index": len(payloads),
                "total_chunks": 0,
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
            for chunk_text in chunks:
                payloads.append({
                    "doc_id": doc_id,
                    "type": content_type,
                    "segment_id": segment_id,
                    "title": title,
                    "chunk_index": len(payloads),
                    "total_chunks": 0,
                    "text": chunk_text,
                    "source": source_label,
                })
    if one_article_per_point and payloads:
        n = len(payloads)
        for i, p in enumerate(payloads):
            p["chunk_index"] = i
            p["total_chunks"] = n
    return payloads


def ingest_articles_from_text(
    full_text: str | None = None,
    doc_id: str = "",
    source_label: str = "",
    one_article_per_point: bool = False,
    elements: list[dict] | None = None,
    lang: str = DEFAULT_LANG,
) -> list[PointStruct]:
    """
    Build Qdrant points (with vectors) from document text or DOCX elements.
    one_article_per_point=True: each article (1-modda, 2-modda) = one point even if long (truncated at encode).
    For normal docs from DOCX, pass elements=read_docx_elements(path); for TXT pass full_text.
    """
    if DEBUG_INGESTION:
        print(f"[ingest_articles_from_text] doc_id={doc_id!r} one_article_per_point={one_article_per_point} building payloads...")
    if one_article_per_point:
        payloads = document_to_article_payloads(doc_id, full_text or "", source_label, one_article_per_point=True)
    else:
        if elements is not None:
            payloads = document_to_document_payloads(doc_id, full_text=None, source_label=source_label, elements=elements)
        else:
            payloads = document_to_document_payloads(doc_id, full_text=full_text or "", source_label=source_label)

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
        if p.get("table"):
            t_len = len(t)
            preview = (t[:300] + "...") if t_len > 300 else t
            if DEBUG_INGESTION:
                print(f"[ingest_articles_from_text]   [TABLE] payload_idx={i} chunk_index={p.get('chunk_index')} group_id={p.get('group_id')} text_len={t_len} preview={preview!r}")
        ids = tokenizer.encode(t, add_special_tokens=False, max_length=MAX_MODEL_TOKENS, truncation=True)
        texts.append(tokenizer.decode(ids, skip_special_tokens=True) if ids else t)
        if DEBUG_INGESTION and not p.get("table") and (i + 1) % 50 == 0:
            print(f"[ingest_articles_from_text]   truncation progress: {i + 1}/{len(payloads)}")
    table_count = sum(1 for p in payloads if p.get("table"))
    if DEBUG_INGESTION:
        print(f"[ingest_articles_from_text] truncation done. {len(texts)} texts to encode ({table_count} table chunks). Calling embedding_model.encode...")
    if EMBED_BY_CHUNK and len(texts) > EMBED_BATCH_SIZE:
        batch_vectors = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            end = min(start + EMBED_BATCH_SIZE, len(texts))
            batch = texts[start:end]
            batch_num = start // EMBED_BATCH_SIZE + 1
            table_in_batch = [i for i in range(start, end) if i < len(payloads) and payloads[i].get("table")]
            if DEBUG_INGESTION and table_in_batch:
                print(f"[ingest_articles_from_text]   [encoding batch {batch_num}] payload indices {start}..{end-1}, TABLE count={len(table_in_batch)} at payload_idx: {table_in_batch}")
                for idx in table_in_batch:
                    t = texts[idx]
                    preview = (t[:200] + "...") if len(t) > 200 else t
                    print(f"[ingest_articles_from_text]     TABLE payload_idx={idx}: text_len={len(t)} preview={preview!r}")
            if DEBUG_INGESTION and not table_in_batch:
                print(f"[ingest_articles_from_text]   encoded batch {batch_num} ({len(batch)} texts, no tables)")
            v = embedding_model.encode(batch, normalize_embeddings=True)
            if hasattr(v, "numpy"):
                v = v.numpy()
            batch_vectors.append(v)
            if DEBUG_INGESTION and table_in_batch:
                print(f"[ingest_articles_from_text]   encoded batch {batch_num} done ({len(batch)} texts)")
        vectors = np.concatenate(batch_vectors, axis=0)
    else:
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
        # Payload for Qdrant: type, segment_id, chunk_index; group_id for document paragraphs/tables (join at retrieval)
        p = {
            "doc_id": payload["doc_id"],
            "type": payload.get("type", "article"),
            "segment_id": payload.get("segment_id", 0),
            "title": payload.get("title", ""),
            "chunk_index": payload["chunk_index"],
            "total_chunks": payload["total_chunks"],
            "text": payload["text"],
        }
        p["lang"] = lang or DEFAULT_LANG
        if "group_id" in payload:
            p["group_id"] = payload["group_id"]
        if payload.get("table"):
            p["table"] = True
            if "table_data" in payload and payload["table_data"] is not None:
                p["table_data"] = payload["table_data"]
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


def upload_docx(file_path: str, article_file: bool = False, lang: str = DEFAULT_LANG) -> dict:
    """
    Read DOCX and upload to Qdrant.
    article_file=True: from data/articles – each article (1-modda, 2-modda) = one point.
    article_file=False: from data/documents – normal chunking by token size.
    Returns {"ok": bool, "message": str, "filename": str, "chunks": int}.
    """
    filename = os.path.basename(file_path)
    if filename.startswith("~$") or filename.startswith("~"):
        print(f"  ⏭️  Skipping temporary file: {filename}")
        return {"ok": False, "message": "Skipping temporary file", "filename": filename, "chunks": 0}

    print(f"Processing: {file_path}")
    if article_file:
        try:
            text = read_docx(file_path)
        except Exception as e:
            print(f"  ❌ Error reading {file_path}: {e}")
            return {"ok": False, "message": str(e), "filename": filename, "chunks": 0}
        if not text or len(text.strip()) < 50:
            print(f"  ⚠️  Skipping {file_path}: document too short or empty")
            return {"ok": False, "message": "Document too short or empty", "filename": filename, "chunks": 0}
    else:
        try:
            elements = read_docx_elements(file_path)
        except Exception as e:
            print(f"  ❌ Error reading {file_path}: {e}")
            return {"ok": False, "message": str(e), "filename": filename, "chunks": 0}
        text_len = sum(len(el.get("text", "")) for el in elements if el.get("type") == "paragraph") + sum(
            len(json.dumps(el.get("data", []), ensure_ascii=False)) for el in elements if el.get("type") == "table"
        )
        if not elements or text_len < 50:
            print(f"  ⚠️  Skipping {file_path}: document too short or empty")
            return {"ok": False, "message": "Document too short or empty", "filename": filename, "chunks": 0}

    try:
        file_hash = compute_file_hash(file_path)
    except Exception as e:
        print(f"  ❌ Error hashing file: {e}")
        return {"ok": False, "message": str(e), "filename": filename, "chunks": 0}
    existing_doc_id = get_doc_id_by_hash(file_hash)
    if existing_doc_id is not None:
        print(f"  ⏭️  Already ingested (same content): {filename}")
        return {"ok": True, "skipped": True, "message": "Already ingested", "filename": filename, "chunks": 0}

    doc_id = uuid.uuid4().hex
    source_label = filename
    try:
        if article_file:
            points = ingest_articles_from_text(
                full_text=text, doc_id=doc_id, source_label=source_label, one_article_per_point=True, lang=lang
            )
        else:
            points = ingest_articles_from_text(
                doc_id=doc_id, source_label=source_label, one_article_per_point=False, elements=elements, lang=lang
            )
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
    try:
        register_file(doc_id=doc_id, file_path=file_path, file_hash=file_hash)
    except Exception as e:
        print(f"  ⚠️  Registry update failed: {e}")
    print(f"  ✅ Ingested {len(points)} points from {file_path}\n")
    return {"ok": True, "message": f"Ingested {filename}", "filename": filename, "chunks": len(points)}


def upload_file(file_path: str, article_file: bool = False, lang: str = DEFAULT_LANG) -> dict:
    """
    Ingest a single file (DOCX or TXT).
    article_file=True: from data/articles – each article (1-modda, 2-modda) = one point.
    article_file=False: from data/documents – normal chunking.
    """
    filename = os.path.basename(file_path)
    if filename.startswith("~$") or filename.startswith("~"):
        return {"ok": False, "message": "Skipping temporary file", "chunks": 0}

    if not os.path.exists(file_path):
        return {"ok": False, "message": "File not found", "chunks": 0}

    if filename.endswith(".docx"):
        return upload_docx(file_path, article_file=article_file, lang=lang)
    elif filename.endswith(".txt"):
        try:
            text = read_txt(file_path)
            if not text or len(text.strip()) < 50:
                return {"ok": False, "message": "File too short or empty", "filename": filename}
            file_hash = compute_file_hash(file_path)
            if get_doc_id_by_hash(file_hash) is not None:
                return {"ok": True, "skipped": True, "message": "Already ingested", "filename": filename, "chunks": 0}
            doc_id = uuid.uuid4().hex
            points = ingest_articles_from_text(
                text, doc_id=doc_id, source_label=filename, one_article_per_point=article_file, lang=lang
            )
            if not points:
                return {"ok": False, "message": "No article points created", "filename": filename}
            ensure_collection()
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            register_file(doc_id=doc_id, file_path=file_path, file_hash=file_hash)
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
    lang: str = DEFAULT_LANG,
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
                    "lang": lang,
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


def upload_qa_file(file_path: str, lang: str = DEFAULT_LANG) -> dict:
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
    return upload_qa_from_data(pairs=pairs, source=source, lang=lang)


# -------------------------------
# MAIN
# -------------------------------
def _list_doc_files(folder: str) -> list[str]:
    """Return list of .docx and .txt paths in folder (non-recursive)."""
    if not os.path.isdir(folder):
        return []
    paths = []
    for name in os.listdir(folder):
        if name.startswith("~$") or name.startswith("~"):
            continue
        if name.endswith(".docx") or name.endswith(".txt"):
            paths.append(os.path.join(folder, name))
    return paths


if __name__ == "__main__":
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: Data folder '{DATA_FOLDER}' not found!")
        exit(1)

    # Detect per-language subfolders: data/<lang>/{articles,documents,*.json}
    lang_dirs = [lang for lang in LANGUAGES if os.path.isdir(os.path.join(DATA_FOLDER, lang))]

    # Fallback to legacy single-language layout if no lang subfolders exist
    if not lang_dirs:
        lang_dirs = [DEFAULT_LANG]
        base_articles = DATA_ARTICLES_FOLDER
        base_documents = DATA_DOCUMENTS_FOLDER
        base_qa_root = DATA_FOLDER
        lang_roots = {DEFAULT_LANG: (base_articles, base_documents, base_qa_root)}
    else:
        lang_roots = {}
        for lang in lang_dirs:
            root = os.path.join(DATA_FOLDER, lang)
            lang_roots[lang] = (
                os.path.join(root, "articles"),
                os.path.join(root, "documents"),
                root,
            )

    failed = []

    for lang in lang_dirs:
        articles_dir, documents_dir, qa_root = lang_roots[lang]
        article_files = _list_doc_files(articles_dir)
        document_files = _list_doc_files(documents_dir)
        qa_files = []
        if os.path.isdir(qa_root):
            for name in os.listdir(qa_root):
                if name.endswith(".json") and not name.startswith("~"):
                    qa_files.append(os.path.join(qa_root, name))

        if not article_files and not document_files and not qa_files:
            print(f"⚠️  No files for lang='{lang}' in '{articles_dir}', '{documents_dir}' or Q&A in '{qa_root}'")
            continue

        print(
            f"\n=== Language: {lang} | Articles: {len(article_files)} | "
            f"Documents: {len(document_files)} | Q&A: {len(qa_files)} ==="
        )

        for file_path in article_files:
            result = upload_file(file_path, article_file=True, lang=lang)
            status = "✅" if result.get("ok") else "❌"
            chunks = result.get("chunks", 0)
            name = os.path.basename(file_path)
            print(f"{status} [articles] {name}: {result.get('message', '')}" + (f" ({chunks} chunks)" if chunks else ""))
            if not result.get("ok"):
                failed.append((lang, name, result.get("message", "Unknown error")))

        for file_path in document_files:
            result = upload_file(file_path, article_file=False, lang=lang)
            status = "✅" if result.get("ok") else "❌"
            chunks = result.get("chunks", 0)
            name = os.path.basename(file_path)
            print(f"{status} [documents] {name}: {result.get('message', '')}" + (f" ({chunks} chunks)" if chunks else ""))
            if not result.get("ok"):
                failed.append((lang, name, result.get("message", "Unknown error")))

        for file_path in qa_files:
            result = upload_qa_file(file_path, lang=lang)
            status = "✅" if result.get("ok") else "❌"
            name = os.path.basename(file_path)
            print(f"{status} Q&A {name}: {result.get('message', '')} (count: {result.get('count', 0)})")
            if not result.get("ok"):
                failed.append((lang, name, result.get("message", "Unknown error")))

    print("\n" + "=" * 60)
    if failed:
        print("❌ Some files failed:")
        for lang, name, msg in failed:
            print(f"   - [{lang}] {name}: {msg}")
        exit(1)
    print("✅ All files processed successfully!")
