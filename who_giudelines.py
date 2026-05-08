# ===============================
# WHO GUIDELINES — Lightweight Version
# Uses TF-IDF instead of sentence_transformers
# No torch, no heavy models, works on low RAM PCs
# ===============================
import fitz   # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# LOAD PDF AND SPLIT INTO CHUNKS
# ===============================
def load_pdf_chunks(pdf_path: str, chunk_size: int = 300) -> list:
    """Extract text from PDF and split into chunks."""
    doc    = fitz.open(pdf_path)
    text   = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    # Split into chunks by words
    words  = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# ===============================
# CREATE TF-IDF INDEX
# ===============================
def create_index(chunks: list):
    """Build a TF-IDF matrix from chunks."""
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix     = vectorizer.fit_transform(chunks)
    return vectorizer, matrix   # returns (vectorizer, matrix) — replaces (index, embeddings)

# ===============================
# SEARCH — top relevant chunk
# ===============================
def search(query: str, chunks: list, vectorizer, top_k: int = 3) -> str:
    """Find most relevant chunks for a query using TF-IDF cosine similarity."""
    try:
        query_vec   = vectorizer.transform([query])
        # vectorizer here is actually (vectorizer, matrix) tuple passed from create_index
        # Handle both calling styles
        if isinstance(vectorizer, tuple):
            vec, matrix = vectorizer
            query_vec   = vec.transform([query])
        else:
            matrix = vectorizer  # fallback

        scores      = cosine_similarity(query_vec, matrix).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        results     = [chunks[i] for i in top_indices if scores[i] > 0]
        return "\n\n".join(results) if results else "No relevant guidelines found."
    except Exception:
        return "No relevant guidelines found."