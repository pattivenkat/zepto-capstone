# Module 3 — Zepto Support Assistant

An offline-capable, retrieval-augmented customer support chatbot built with
LangGraph, sentence-transformers, ChromaDB, and FastAPI.

---

## Architecture

```
User query (POST /ask)
        │
        ▼
  ┌─────────────┐
  │ classify_   │  Node 1 — keyword heuristic (MOCK) or LLM call (real)
  │  intent     │  Output:  intent = "policy_question" | "general_question"
  └──────┬──────┘
         │ conditional edge
   ┌─────┴──────────────────┐
   │                        │
   ▼                        ▼
┌──────────────────┐   ┌────────────────┐
│ retrieve_and_    │   │ direct_answer  │  Node 3 — no retrieval needed
│  answer          │   │                │  → canned response (MOCK)
│  Node 2          │   └───────┬────────┘
│  • embed query   │           │
│  • ChromaDB top3 │           │
│  • build answer  │           │
└────────┬─────────┘           │
         │                     │
         └──────────┬──────────┘
                    ▼
            PolicyResponse
          { answer, sources, confidence }
                    │
                    ▼
           FastAPI POST /ask
```

### Component Breakdown

| Component | File | Description |
|---|---|---|
| Policy corpus | `docs/doc_01.txt` – `doc_08.txt` | 8 plain-text Zepto policy documents |
| Ingestion pipeline | `ingest.py` | Load → chunk (500 char / 50 overlap) → embed → ChromaDB |
| Embedding model | `all-MiniLM-L6-v2` | Local sentence-transformers model, no API key |
| Vector store | ChromaDB (`chroma_db/`) | Persistent, cosine similarity, collection: `zepto_policies` |
| LangGraph graph | `graph.py` | StateGraph with 3 nodes + conditional edge |
| Prompt template | `graph.py::PROMPT_TEMPLATE` | role · context · task · format · length + negative constraint + few-shot |
| Output schema | `graph.py::PolicyResponse` | Pydantic v2: `answer` (str), `sources` (List[str]), `confidence` (float 0–1) |
| API server | `main.py` | FastAPI POST /ask + GET /health |

### MOCK_LLM Toggle

| `MOCK_LLM` | Mode | LLM call | Graded? |
|---|---|---|---|
| `1` (default) | Mock / offline | None | ✅ Yes — graded baseline |
| `0` | Real LLM | Plug in your client | Optional extension |

In mock mode, all generation steps use deterministic offline logic — no API
keys, no network calls, fully reproducible.

---

## Setup

### 1. Install dependencies

```bash
cd support_assistant
pip install -r requirements.txt
```

### 2. Ingest policy documents

Run once to embed docs and populate ChromaDB:

```bash
python ingest.py
```

Expected output:
```
STEP 1 — Loading documents
  Loaded: doc_01  (... chars)
  ...
  Total documents: 8

STEP 2 — Chunking
  doc_01: 1 chunk(s)
  ...
  Total chunks: 8+

STEP 3 — Embedding with 'all-MiniLM-L6-v2'
  Embeddings shape: (N, 384)

STEP 4 — Storing in ChromaDB
  ✓ Stored N chunks in collection 'zepto_policies'.
```

### 3. Start the API server

```bash
uvicorn main:app --reload --port 8000
```

---

## Docker

### Build

```bash
docker build -t zepto-support .
```

### Run (mock mode — default)

```bash
docker run -p 8000:8000 zepto-support
```

### Run (real LLM mode — optional, ungraded)

```bash
docker run -p 8000:8000 -e MOCK_LLM=0 zepto-support
```

---

## API Reference

### `POST /ask`

Ask a question about Zepto policies.

**Request body:**
```json
{ "query": "What is the return policy?" }
```

**Response:**
```json
{
  "answer": "Based on the retrieved context: Returns must be initiated within 48 hours ...",
  "sources": ["doc_02_chunk_0", "doc_02_chunk_1", "doc_01_chunk_0"],
  "confidence": 1.0
}
```

**Field definitions:**

| Field | Type | Description |
|---|---|---|
| `answer` | string | Response text |
| `sources` | list of strings | Chunk IDs used; empty for general questions |
| `confidence` | float (0–1) | Confidence score |

### `GET /health`

```json
{ "status": "ok" }
```

---

## Example API Calls

### curl

```bash
# Policy question (triggers retrieval)
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the return policy?"}' | python3 -m json.tool

# Delivery question
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How long does delivery take?"}' | python3 -m json.tool

# Membership question
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Zepto membership benefits?"}' | python3 -m json.tool

# General question (no retrieval — direct_answer node)
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather today?"}' | python3 -m json.tool

# Health check
curl -s http://localhost:8000/health
```

### Python

```python
import requests

url = "http://localhost:8000/ask"

queries = [
    "What is the refund policy for cancelled orders?",
    "How do I track my order?",
    "When does support team respond?",
    "Tell me about gift card denominations",
]

for q in queries:
    r = requests.post(url, json={"query": q})
    data = r.json()
    print(f"\nQ: {q}")
    print(f"A: {data['answer'][:120]}...")
    print(f"Sources: {data['sources']}")
    print(f"Confidence: {data['confidence']}")
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| `all-MiniLM-L6-v2` for embeddings | Fast, lightweight, runs locally — no API key, no cost |
| ChromaDB with cosine similarity | Semantic search matches meaning, not just keywords |
| Fixed-size character chunking (500/50) | Policy docs are short; avoids over-splitting while preserving overlap |
| MOCK_LLM toggle | Graded baseline needs zero API keys; real LLM is an optional extension |
| LangGraph StateGraph | Makes intent routing explicit and extensible (add nodes without refactoring) |
| Pydantic v2 response schema | Enforces structured output; `confidence` field signals answer quality |
| Idempotent ingest | `delete_collection` before recreating ensures clean re-runs |
