from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -------------------------------
# LOAD RERANKER MODEL (once)
# -------------------------------
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
model.eval()

# -------------------------------
# RERANK FUNCTION
# -------------------------------
def rerank(query: str, chunks: list[str], top_n: int = 5):
    """
    Rerank retrieved chunks and return top_n most relevant.
    """
    pairs = [(query, chunk) for chunk in chunks]

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        # max_length=512
    )

    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)

    scored_chunks = list(zip(chunks, scores.tolist()))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks[:top_n]]
