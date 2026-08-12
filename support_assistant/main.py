"""
main.py — Task 5
FastAPI application exposing POST /ask endpoint.

Run:
    uvicorn main:app --reload --port 8000

Docker:
    docker build -t zepto-support .
    docker run -p 8000:8000 zepto-support
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph import run_query, PolicyResponse

app = FastAPI(
    title="Zepto Support Assistant",
    description="AI-powered customer support assistant for Zepto policies.",
    version="1.0.0",
)


# ── Request schema ─────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str


# ── Health check ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Main endpoint ──────────────────────────────────────────────────────────

@app.post("/ask", response_model=PolicyResponse)
def ask(request: AskRequest):
    """
    Ask a question about Zepto policies.

    - **query**: The customer's question (plain text)

    Returns a JSON object with:
    - **answer**: Response text
    - **sources**: Chunk IDs used for retrieval (empty for general questions)
    - **confidence**: Float 0–1 indicating answer confidence
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    result: PolicyResponse = run_query(request.query)
    return result
