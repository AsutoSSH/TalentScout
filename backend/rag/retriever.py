import os
import pickle
import numpy as np
import faiss
from rapidfuzz import process, fuzz
import random
# from sentence_transformers import (
#     SentenceTransformer
# )
from fastembed import TextEmbedding
from collections import defaultdict

from backend.rag.book_registry import (
    ROLE_BOOK_MAPPING
)

# =========================================================
# Paths
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VECTOR_STORE_DIR = os.path.join(
    BASE_DIR,
    "vector_store"
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
# Load FAISS Index
# =========================================================

BOOK_INDICES = {}
PER_BOOK_DIR = os.path.join(VECTOR_STORE_DIR, "per_book")

for book_name in os.listdir(PER_BOOK_DIR):
    book_dir = os.path.join(PER_BOOK_DIR, book_name)
    
    with open(os.path.join(book_dir, "chunks.pkl"), "rb") as fc, open(os.path.join(book_dir, "metadata.pkl"), "rb") as fm:
        BOOK_INDICES[book_name + ".pdf"] = {
            "index": faiss.read_index(os.path.join(book_dir, "faiss.index")),
            "chunks": pickle.load(fc),
            "metadata": pickle.load(fm)
        }


def get_closest_role(
    input_role: str,
    threshold: int = 70
):
    input_lower = input_role.lower()
    input_words = set(input_lower.split())

    original_roles = list(ROLE_BOOK_MAPPING.keys())
    role_book_maps_lower = [role.lower() for role in ROLE_BOOK_MAPPING.keys()]

    match = process.extractOne(
        input_lower,
        role_book_maps_lower,
        scorer=fuzz.WRatio
    )

    if match is None:
        return None

    matched_role, score, index = match

    if len(input_words) < 2 and score < 95:
        return None
    
    # Additional check: at least 1 word from input appears in matched role
    matched_words = set(matched_role.split())
    if not input_words.intersection(matched_words):
        print(f"[Role Match] No word overlap: '{input_role}' vs '{matched_role}'")
        return None

    print(f"[Role Match] {matched_role} (score={score})")

    if score < threshold:
        return None

    original_matched_role = original_roles[index]

    return original_matched_role


def retrieve_relevant_chunks(
    query: str,
    k: int = 5,
    matched_role: str = None
):

    # ==========================================
    # Find closest mapped role
    # ==========================================


    if matched_role is None:

        return {

            "use_rag": False,
            "matched_role": None,
            "retrieved_chunks": []
        }

    # ==========================================
    # Allowed books
    # ==========================================

    allowed_books = ROLE_BOOK_MAPPING[
        matched_role
    ]
    print(f"Allowed books: {allowed_books}")

    # ==========================================
    # Embed query
    # ==========================================

    # query_embedding = embedding_model.encode(

    #     query,

    #     normalize_embeddings=True,
    #     convert_to_numpy=True

    # ).astype(np.float32).reshape(1, -1)

    query_embedding = np.array(

        list(
            embedding_model.embed([query])
        ),
        dtype=np.float32)

    faiss.normalize_L2(query_embedding)

    all_candidates = []

    # --- Search ONLY allowed books ---
    for book in allowed_books:
        if book not in BOOK_INDICES: continue
        store = BOOK_INDICES[book]
        distances, indices = store["index"].search(query_embedding, max(k * 3, 10))

        for score, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            all_candidates.append({
                "text": store["chunks"][idx],
                "source_book": store["metadata"][idx]["source_book"],
                "page": store["metadata"][idx]["page"],
                "score": float(score)
            })

    # --- Sort and Filter ---
    # 1. Group and sort books internally (Descending by score)
    book_groups = defaultdict(list)
    for c in all_candidates:
        book_groups[c["source_book"]].append(c)

    for book in book_groups:
        book_groups[book].sort(key=lambda x: x["score"], reverse=True)

    # 2. Round-robin retrieval
    retrieved_chunks = []
    book_counts = defaultdict(int)
    max_per_book = int(k / len(allowed_books))

    while len(retrieved_chunks) < k:
        added = False
        
        for book in allowed_books:
            # Stop early if we hit our global target 'k'
            if len(retrieved_chunks) >= k:
                break
                
            # Skip if book is empty or has reached its individual max cap
            if not book_groups[book] or book_counts[book] >= max_per_book:
                continue
                
            retrieved_chunks.append(book_groups[book].pop(0))
            book_counts[book] += 1
            added = True

        if not added:  # Break out if no books have items left or all caps are hit
            break

    use_rag = len(retrieved_chunks) > 0
    print(f"Retrieved Chunks: {retrieved_chunks}")
    return {
        "use_rag": use_rag,
        "matched_role": matched_role,
        "retrieved_chunks": retrieved_chunks if use_rag else []
    }