"""
ingest.py — Task 1
Load all 8 policy documents, chunk them, embed with all-MiniLM-L6-v2,
and store in a ChromaDB persistent collection.

Run ONCE before starting the API server:
    python ingest.py
"""

import os
import glob
from sentence_transformers import SentenceTransformer
import chromadb

# ── Paths & constants ──────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR        = os.path.join(BASE_DIR, "docs")
CHROMA_DIR      = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "zepto_policies"
MODEL_NAME      = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 500   # characters
CHUNK_OVERLAP   = 50    # characters


# ── Step 1: Load documents ─────────────────────────────────────────────────

def load_documents(docs_dir: str) -> list:
    """Read every .txt file in docs_dir and return a list of doc dicts."""
    docs = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        docs.append({"id": doc_id, "content": content, "path": path})
        print(f"  Loaded: {doc_id}  ({len(content)} chars)")
    return docs


# ── Step 2: Chunk documents ────────────────────────────────────────────────

def chunk_document(doc: dict, chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list:
    """
    Fixed-size character chunking with overlap.
    Given the short length of each policy doc (~400–600 chars), most docs
    will produce a single chunk — this handles the general case cleanly.
    """
    content = doc["content"]
    chunks  = []
    start   = 0
    idx     = 0

    while start < len(content):
        end        = min(start + chunk_size, len(content))
        chunk_text = content[start:end]
        chunk_id   = f"{doc['id']}_chunk_{idx}"

        chunks.append({
            "id":       chunk_id,
            "content":  chunk_text,
            "doc_id":   doc["id"],
            "source":   doc["path"],
        })

        if end == len(content):
            break
        start = end - overlap
        idx  += 1

    return chunks


# ── Step 3: Embed & store ──────────────────────────────────────────────────

def ingest():
    print("=" * 55)
    print("STEP 1 — Loading documents")
    print("=" * 55)
    docs = load_documents(DOCS_DIR)
    print(f"Total documents: {len(docs)}\n")

    print("=" * 55)
    print("STEP 2 — Chunking")
    print("=" * 55)
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['id']}: {len(chunks)} chunk(s)")
    print(f"Total chunks: {len(all_chunks)}\n")

    print("=" * 55)
    print(f"STEP 3 — Embedding with '{MODEL_NAME}'")
    print("=" * 55)
    model      = SentenceTransformer(MODEL_NAME)
    texts      = [c["content"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    print(f"Embeddings shape: {embeddings.shape}\n")

    print("=" * 55)
    print(f"STEP 4 — Storing in ChromaDB at '{CHROMA_DIR}'")
    print("=" * 55)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop and recreate for idempotent re-runs
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Cleared existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids        = [c["id"]      for c in all_chunks],
        embeddings = embeddings.tolist(),
        documents  = texts,
        metadatas  = [{"doc_id": c["doc_id"], "source": c["source"]}
                      for c in all_chunks],
    )

    stored = collection.count()
    print(f"  ✓ Stored {stored} chunks in collection '{COLLECTION_NAME}'.")
    print("\nIngestion complete. You can now start the API:")
    print("  uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    ingest()
