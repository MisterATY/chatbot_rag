import re
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from app.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, LLM_SERVER_URL, LLM_MODEL
from app.embeddings import embedding_model
from app.tokenizer_util import count_tokens, get_tokenizer
from app.reranker import rerank
import numpy as np

# -------------------------------
# SETUP
# -------------------------------
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Retrieval constants (production RAG)
RETRIEVAL_TOP_K = 12
MAX_CONTEXT_TOKENS = 25000
NEIGHBOR_CHUNKS_BEFORE = 2
NEIGHBOR_CHUNKS_AFTER = 2


# -------------------------------
# PAYLOAD HELPERS
# -------------------------------
def _payload_from_hit(hit) -> dict:
    """Extract payload dict from a Qdrant hit (ScoredPoint or similar)."""
    if hasattr(hit, "payload"):
        return hit.payload or {}
    if isinstance(hit, tuple) and len(hit) >= 3:
        return hit[2] or {}
    if isinstance(hit, dict):
        return hit.get("payload", {})
    return {}


def _is_article_chunk(payload: dict) -> bool:
    """True if payload has article_id and chunk_index (article-based ingestion)."""
    return isinstance(payload, dict) and "article_id" in payload and "chunk_index" in payload

def _is_question_answerable(payload: dict) -> bool:
    """True if payload has article_id and chunk_index (article-based ingestion)."""
    return isinstance(payload, dict) and "type" in payload and payload["type"] == "qa"


# -------------------------------
# NEIGHBOR EXPANSION
# -------------------------------
# def get_chunks_by_article_id(article_id: str) -> list[dict]:
#     """
#     Fetch all chunks for the given article_id from Qdrant (same article, any chunk_index).
#     Returns list of payload dicts sorted by chunk_index. Deterministic.
#     """
#     filter_ = Filter(
#         must=[FieldCondition(key="article_id", match=MatchValue(value=article_id))]
#     )
#     points, _ = client.scroll(
#         collection_name=COLLECTION_NAME,
#         scroll_filter=filter_,
#         with_payload=True,
#         with_vectors=False,
#         limit=500,
#     )
#     payloads = [_payload_from_hit(p) for p in points]
#     payloads = [p for p in payloads if _is_article_chunk(p)]
#     payloads.sort(key=lambda p: (p.get("chunk_index", 0), p.get("text", "")))
#     return payloads


# def expand_neighbors(
#     best_payload: dict,
#     all_article_chunks: list[dict],
#     before: int = NEIGHBOR_CHUNKS_BEFORE,
#     after: int = NEIGHBOR_CHUNKS_AFTER,
# ) -> list[dict]:
#     """
#     From the best-matching chunk and the full list of chunks for that article,
#     return a contiguous window: up to `before` chunks before and `after` chunks after
#     the best chunk (by list position). Same article_id, logical order.
#     """
#     if not all_article_chunks:
#         return []
#     best_idx = best_payload.get("chunk_index", 0)
#     pos = next(
#         (i for i, p in enumerate(all_article_chunks) if p.get("chunk_index") == best_idx),
#         0,
#     )
#     n = len(all_article_chunks)
#     start = max(0, pos - before)
#     end = min(n, pos + after + 1)
#     return all_article_chunks[start:end]


# -------------------------------
# CONTEXT ASSEMBLY
# -------------------------------
def _format_single_block(payload: dict, index: int) -> str:
    """Format one chunk payload as [Article ... – Part x/y]\\n{text}."""
    # article_id = payload.get("article_id", "")
    # article_title = payload.get("article_title", "")
    # chunk_index = payload.get("chunk_index", 0)
    # total_chunks = payload.get("total_chunks", 1)
    text = payload.get("text", "")
    # part_label = f"Part {chunk_index + 1}/{total_chunks}"
    return f"Context {index}: \n{text}"


# def assemble_context_for_llm(
#     chunks: list[dict],
#     max_tokens: int = MAX_CONTEXT_TOKENS,
#     center_index: int | None = None,
# ) -> list[str]:
#     """
#     Build the final context for the LLM from a list of chunk payloads.
#     Format: [Article {article_id} – {article_title} – Part {chunk_index + 1}/{total_chunks}]\n{text}
#     Chunks must be in logical order (by chunk_index). Trim to max_tokens total.
#     If center_index is set, trimming keeps contiguous chunks centered on that index.
#     """
#     if not chunks:
#         return []

#     formatted = []
#     for p in chunks:
#         article_id = p.get("article_id", "")
#         article_title = p.get("article_title", "")
#         chunk_index = p.get("chunk_index", 0)
#         total_chunks = p.get("total_chunks", 1)
#         text = p.get("text", "")
#         part_label = f"Part {chunk_index + 1}/{total_chunks}"
#         block = f"[Article {article_id} – {article_title} – {part_label}]\n{text}"
#         formatted.append(block)

#     total = sum(count_tokens(block) for block in formatted)
#     if total <= max_tokens:
#         return formatted

#     # Trim: keep contiguous chunks centered on the best (most relevant) chunk
#     n = len(formatted)
#     c = center_index if center_index is not None and 0 <= center_index < n else 0
#     start, end = c, c + 1
#     acc = count_tokens(formatted[c])
#     while start > 0 or end < n:
#         add_left = count_tokens(formatted[start - 1]) if start > 0 else float("inf")
#         add_right = count_tokens(formatted[end]) if end < n else float("inf")
#         if add_left <= add_right and start > 0 and acc + add_left <= max_tokens:
#             start -= 1
#             acc += add_left
#         elif end < n and acc + add_right <= max_tokens:
#             end += 1
#             acc += add_right
#         else:
#             break
#     return formatted[start:end]


# -------------------------------
# UTILITY FUNCTIONS (legacy / optional)
# -------------------------------
# def split_into_sentences(text: str) -> list[str]:
#     """Split text into sentences (handles Uzbek text)."""
#     # Split by common sentence endings
#     sentences = re.split(r'[.!?]\s+', text)
#     # Filter out empty sentences and very short ones
#     sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
#     return sentences


# def extract_relevant_sentences(query: str, chunk: str, max_sentences: int = 3) -> str:
#     """
#     Extract the most relevant sentences from a chunk based on query similarity.
#     Returns a smaller, more focused context while maintaining coherence.
#     """
#     sentences = split_into_sentences(chunk)
    
#     if len(sentences) <= max_sentences:
#         return chunk
    
#     # Embed query and all sentences
#     query_embedding = embedding_model.encode(query, normalize_embeddings=True)
#     sentence_embeddings = embedding_model.encode(sentences, normalize_embeddings=True)
    
#     # Calculate cosine similarity
#     similarities = np.dot(sentence_embeddings, query_embedding)
    
#     # Find the most relevant sentence
#     most_relevant_idx = np.argmax(similarities)
    
#     # Extract a window around the most relevant sentence
#     # This maintains context while keeping it small
#     start_idx = max(0, most_relevant_idx - 1)
#     end_idx = min(len(sentences), most_relevant_idx + max_sentences)
    
#     # If we have room, try to include other highly relevant sentences
#     if end_idx - start_idx < max_sentences:
#         # Get top sentences and merge with window
#         top_indices = set(range(start_idx, end_idx))
#         other_top = np.argsort(similarities)[-max_sentences:][::-1]
        
#         for idx in other_top:
#             if len(top_indices) >= max_sentences:
#                 break
#             top_indices.add(idx)
        
#         top_indices = sorted(list(top_indices))
#     else:
#         top_indices = list(range(start_idx, end_idx))
    
#     # Extract and join sentences in original order
#     relevant_sentences = [sentences[i] for i in top_indices]
#     return ". ".join(relevant_sentences) + "."


# -------------------------------
# RETRIEVAL FUNCTION (article-based RAG)
# -------------------------------
def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> list[str]:
    """
    Retrieve context for RAG.
    """
    query_vector = embedding_model.encode(query, normalize_embeddings=True).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    points = results.points if hasattr(results, "points") else list(results)
    if not points:
        return []

    # Rerank found points before proceeding
    texts = [_payload_from_hit(p).get("text", "") for p in points]
    reranked_texts = rerank(query, texts, top_n=7)
    # Reorder points to match reranked order (best first)
    pairs = list(zip(points, texts))
    points_reranked = []
    for t in reranked_texts:
        for j, (pt, tx) in enumerate(pairs):
            if tx == t:
                points_reranked.append(pt)
                del pairs[j]
                break
    points = points_reranked

    best = points[0]
    best_payload = _payload_from_hit(best)

    if not _is_article_chunk(best_payload):
        # Legacy: single chunk
        if _is_question_answerable(best_payload):
            text = best_payload.get("text", "")
            if not text:
                return []
            block = f"Question&Answer Context: \n{text}"
            if count_tokens(block) > max_context_tokens:
                tokenizer = get_tokenizer()
                ids = tokenizer.encode(text, add_special_tokens=False)[:max_context_tokens]
                text = tokenizer.decode(ids, skip_special_tokens=True)
                block = f"Question&Answer Context: \n{text}"
            return [block]
        text = best_payload.get("text", "")
        if not text:
            return []
        block = f"[Article – – Part 1/1]\n{text}"
        if count_tokens(block) > max_context_tokens:
            tokenizer = get_tokenizer()
            ids = tokenizer.encode(text, add_special_tokens=False)[:max_context_tokens]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            block = f"[Article – – Part 1/1]\n{text}"
        return [block]

    article_id = best_payload.get("article_id", "")
    article_title = best_payload.get("article_title", "")
    total_chunks = best_payload.get("total_chunks", 1)
    center = best_payload.get("chunk_index", 0)

    # Neighbor chunks: single filter query (article_id + chunk_index in [center-3, center+3])
    if total_chunks <= 1:
        neighbor_chunks = [best_payload]
    else:
        neighbor_filter = Filter(
            must=[
                FieldCondition(key="article_id", match=MatchValue(value=article_id)),
                FieldCondition(
                    key="chunk_index",
                    range=Range(gte=center - NEIGHBOR_CHUNKS_BEFORE, lte=center + NEIGHBOR_CHUNKS_AFTER),
                ),
            ]
        )
        points_neighbors, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=neighbor_filter,
            with_payload=True,
            with_vectors=False,
            limit=NEIGHBOR_CHUNKS_BEFORE + 1 + NEIGHBOR_CHUNKS_AFTER,
        )
        neighbor_chunks = [_payload_from_hit(p) for p in points_neighbors]
        neighbor_chunks = [p for p in neighbor_chunks if _is_article_chunk(p)]
        neighbor_chunks.sort(key=lambda p: p.get("chunk_index", 0))

    # Merged set: (article_id, chunk_index) for everything in the first block
    merged_set = {(p.get("article_id"), p.get("chunk_index")) for p in neighbor_chunks}

    # First block: merge neighbor chunks into one block
    neighbor_texts = [p.get("text", "") for p in neighbor_chunks]
    ci_min = min(p.get("chunk_index", 0) for p in neighbor_chunks)
    ci_max = max(p.get("chunk_index", 0) for p in neighbor_chunks)
    merged_block = "\n\n".join(neighbor_texts)

    # Remaining blocks: other top_k chunks not in merged set (each as separate block)
    remaining_blocks = []
    index = 2
    for hit in points[1:]:
        pl = _payload_from_hit(hit)
        key = (pl.get("article_id"), pl.get("chunk_index"))
        if key in merged_set:
            continue
        remaining_blocks.append(_format_single_block(pl, index))
        index += 1

    context_blocks = [f"Context 1: \n{merged_block}"] + remaining_blocks

    # Trim to max_context_tokens (drop blocks from the end)
    total = sum(count_tokens(b) for b in context_blocks)
    while total > max_context_tokens and len(context_blocks) > 1:
        total -= count_tokens(context_blocks[-1])
        context_blocks.pop()

    

    return context_blocks


# -------------------------------
# LLM INTEGRATION
# -------------------------------
def query_llm(query: str, context: list[str]) -> str:
    """
    Send query and context to LLM server and get response.
    context: list of formatted blocks from retrieve() (Article – title – Part x/y + text).
    """
    context_text = "\n\n".join(context) if context else ""

    # Format the prompt with context
    user_prompt = f"""Quyidagi ma'lumotlarga asoslanib, savolga javob bering.

Ma'lumotlar:
{context_text}

Savol: {query}

Javob:"""
    
    # Prepare the request payload
    payload = {
        "model": LLM_MODEL,
        "reasoning": False,
        "guided_decoding_backend": "none",
        "messages": [
            {
                "role": "system",
                "content": f"""You are an AI assistant specialized in customs and customs legislation.

Rules:
1. You answer ONLY questions related to customs and customs legislation.
2. When answering, use ONLY the provided “Information” (context).
3. If the context is in Question & Answer format, answer ONLY based on the answer part of that context.
4. If the question is NOT related to customs, customs authorities, or customs legislation, return the following response:
   "Kechirasiz, siz bergan savol bojxona yoki bojxona qonunchiligiga oid emas!"
5. If the question IS related to customs or customs legislation, but the provided “Information” does not contain sufficient data to answer it, return the following response:
   "Sizning savolingizga javob berish uchun yetarlicha bilimga ega emasman!"
6. Never make assumptions or fabricate information.
7. Write the answer in as much detail as possible, clearly, thoroughly, and in a formal style.
8. The answer must be written ONLY in the Uzbek language.

Your task is to provide a reliable, context-based answer to the given question."""
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "max_tokens": 4096,
        "stream": False,
        "model": "bojxona",
        "temperature": 0.7
    }
    # 6. Javobni aniq, qisqa va rasmiy-uslubda yozing.
    try:
        response = requests.post(
            LLM_SERVER_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=1200  # 20 minute timeout
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the message content from the response
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "LLM javob olishda xatolik yuz berdi."
            
    except requests.exceptions.RequestException as e:
        return f"LLM serverga ulanishda xatolik: {str(e)}"
    except Exception as e:
        return f"Xatolik: {str(e)}"


def retrieve_and_answer(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> dict:
    """
    Complete RAG pipeline: retrieve with neighbor expansion, assemble context, get LLM answer.
    Returns dict with 'context' (list of formatted blocks) and 'answer'.
    """
    context = retrieve(query, top_k=top_k, max_context_tokens=max_context_tokens)
    answer = query_llm(query, context)
    return {"context": context, "answer": answer}


# -------------------------------
# TEST
# -------------------------------
if __name__ == "__main__":
    query = input("Enter your query: ")
    result = retrieve_and_answer(query)
    print("\n" + "=" * 80)
    print("Context (article-based, neighbor-expanded):")
    print("=" * 80)
    for i, block in enumerate(result["context"], 1):
        print(f"\n--- Block {i} ---\n{block}\n")
    print("=" * 80)
    print("LLM Answer:")
    print("=" * 80)
    print(f"\n{result['answer']}\n")
