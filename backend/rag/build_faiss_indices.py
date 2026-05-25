import os
import pickle
import faiss
import numpy as np

import fitz

# from sentence_transformers import (
#     SentenceTransformer
# )
from fastembed import TextEmbedding
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from tqdm import tqdm

# =========================================================
# Paths
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BOOKS_DIR = os.path.join(
    BASE_DIR,
    "pdfs"
)

VECTOR_STORE_DIR = os.path.join(
    BASE_DIR,
    "vector_store"
)

os.makedirs(
    VECTOR_STORE_DIR,
    exist_ok=True
)

# =========================================================
# Embedding Model
# =========================================================

# embedding_model = SentenceTransformer(
#     "sentence-transformers/all-MiniLM-L6-v2"
# )

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# =========================================================
# Chunking
# =========================================================

text_splitter = (
    RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=200
    )
)

# ===================================
# Filtering
# ===================================
bad_patterns = [
    "summary", "references", "bibliography", "recommended", "q&a",
    "question and answer", "forums", "websites", "next chapter", 
    "follow along", "tutorial", "amazon", "http", "www.", 
    "table of contents", "contents"
]
CORE_ML_KEYWORDS = {
    # Core Concepts
    "machine learning", "deep learning", "artificial intelligence", "neural network",
    "supervised", "unsupervised", "reinforcement learning", "transfer learning",
    
    # Math & Stats foundational to ML
    "gradient descent", "loss function", "backpropagation", "optimization",
    "probability distribution", "eigenvector", "eigenvalue", "matrix", "calculus",
    
    # Architecture & Algorithms
    "linear regression", "logistic regression", "decision tree", "random forest",
    "svm", "xgboost", "cnn", "rnn", "lstm", "transformer", "bert", "gpt", "llm",
    "autoencoder", "gan", "diffusion model", "attention mechanism",
    
    # Evaluation & Data
    "overfitting", "underfitting", "regularization", "cross validation", "bias",
    "variance", "precision", "recall", "f1 score", "auc", "roc", "confusion matrix",
    "feature engineering", "normalization", "embeddings", "vector space"
}


# =========================================================
# Storage
# =========================================================
import os, pickle, numpy as np, fitz, faiss

book_data = {}
pdf_files = [f for f in os.listdir(BOOKS_DIR) if f.endswith(".pdf")]

# --- Process PDFs ---
for pdf_file in pdf_files:
    print(f"\nProcessing: {pdf_file}")
    doc = fitz.open(os.path.join(BOOKS_DIR, pdf_file))
    book_chunks, book_metadata = [], []

    for page_num in range(len(doc)):
        text = doc.load_page(page_num).get_text()
        if not text.strip(): continue

        for chunk in text_splitter.split_text(text):
            clean_chunk = " ".join(chunk.split())
            lower_chunk = clean_chunk.lower()

            if (
                any(p in lower_chunk for p in bad_patterns)
                or len(clean_chunk.split()) < 40
                or sum(c.isalpha() for c in clean_chunk) / max(len(clean_chunk), 1) < 0.55
                or not any(kw in lower_chunk for kw in CORE_ML_KEYWORDS)
            ):
                continue

            book_chunks.append(clean_chunk)
            
            book_metadata.append({
                "source_book": pdf_file,
                "page": page_num + 1
            })

    if not book_chunks:
        print(f"No valid chunks in {pdf_file}")
        continue

    book_data[pdf_file] = {"chunks": book_chunks, "metadata": book_metadata}

# --- Build Per-Book & Global Indices ---
global_chunks, global_metadata, global_embeddings = [], [], []

for book_name, data in book_data.items():
    print(f"\nBuilding index for {book_name}")
    chunks, metadata = data["chunks"], data["metadata"]
    
    # embeddings = np.array(embedding_model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)).astype("float32")

    embeddings = np.array(
        list(
            tqdm(
                embedding_model.embed(chunks, batch_size=64),
                total=len(chunks),
                desc="Embedding chunks"
            )
        ),
        dtype=np.float32
    )
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save Per-Book
    book_dir = os.path.join(VECTOR_STORE_DIR, "per_book", os.path.splitext(book_name)[0])
    os.makedirs(book_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(book_dir, "faiss.index"))
    
    with open(os.path.join(book_dir, "chunks.pkl"), "wb") as f: pickle.dump(chunks, f)
    with open(os.path.join(book_dir, "metadata.pkl"), "wb") as f: pickle.dump(metadata, f)

    global_chunks.extend(chunks)
    global_metadata.extend(metadata)
    global_embeddings.append(embeddings)

# --- Save Global Index ---
global_embeddings = np.vstack(global_embeddings)
global_index = faiss.IndexFlatIP(global_embeddings.shape[1])
global_index.add(global_embeddings)

global_dir = os.path.join(VECTOR_STORE_DIR, "global")
os.makedirs(global_dir, exist_ok=True)
faiss.write_index(global_index, os.path.join(global_dir, "faiss.index"))

with open(os.path.join(global_dir, "chunks.pkl"), "wb") as f: pickle.dump(global_chunks, f)
with open(os.path.join(global_dir, "metadata.pkl"), "wb") as f: pickle.dump(global_metadata, f)

print("\nFinished building all indices.")

