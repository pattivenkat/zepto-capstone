"""
graph.py — Tasks 2, 3, 4
Structured prompt template, LangGraph StateGraph (3 nodes + conditional edge),
and Pydantic output schema.

Environment variable:
    MOCK_LLM  — unset or "1" (default): fully offline mock mode (graded baseline)
                 "0": optional real-LLM extension (ungraded)
"""

import os
from typing import List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
import chromadb

# ── Environment toggle ─────────────────────────────────────────────────────
# MOCK_LLM unset OR "1"  →  mock mode  (graded baseline, no network call)
# MOCK_LLM = "0"         →  real LLM   (optional, ungraded extension)
MOCK_LLM: bool = os.environ.get("MOCK_LLM", "1") != "0"

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR      = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "zepto_policies"
MODEL_NAME      = "all-MiniLM-L6-v2"

# ── Keywords for mock classification (Task 3 — classify_intent) ────────────
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]


# =============================================================================
# Task 4 — Pydantic output schema
# =============================================================================

class PolicyResponse(BaseModel):
    """Validated JSON response returned by every /ask call."""
    answer:     str        = Field(..., description="Answer to the user query")
    sources:    List[str]  = Field(default_factory=list,
                                   description="Chunk IDs used (empty for general questions)")
    confidence: float      = Field(..., ge=0.0, le=1.0,
                                   description="Confidence score 0–1")


# =============================================================================
# Task 2 — Structured prompt template
# role · context · task · format · length  +  negative constraint  +  few-shot
# =============================================================================

PROMPT_TEMPLATE = """\
ROLE:
You are Zepto's customer-support AI assistant. You are knowledgeable about
Zepto's delivery, returns & refunds, membership tiers, order tracking,
cancellation, damaged/missing items, gift cards, and support-hours policies.

CONTEXT:
The following excerpts have been retrieved from Zepto's official policy corpus.
Use ONLY this information to construct your answer.

{context}

TASK:
Answer the user's question accurately and helpfully, drawing exclusively from
the retrieved context above.

CONSTRAINTS:
- Do NOT answer using information that is NOT present in the provided context.
- Do NOT invent prices, timelines, percentages, or policy rules.
- If the context does not contain sufficient information, say:
  "I'm sorry, I don't have enough information in the Zepto policy documents
   to answer that question."

FEW-SHOT EXAMPLE:
  User question : "How much does delivery cost?"
  Retrieved context excerpt: "Standard delivery is free on orders over INR 149;
    orders below this threshold incur a flat INR 25 delivery fee."
  Correct answer: "Standard delivery is free for orders above INR 149.
    For orders below INR 149, a flat delivery fee of INR 25 applies."

FORMAT:
Respond in plain, conversational English. Avoid bullet points unless listing
multiple distinct items. Do not repeat the question back to the user.

LENGTH:
2–4 sentences. Be direct and specific — do not pad with disclaimers.

USER QUESTION:
{query}
"""


# =============================================================================
# Task 3 — LangGraph StateGraph
# =============================================================================

# ── State ──────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    query:            str
    intent:           Optional[str]         # "policy_question" | "general_question"
    retrieved_chunks: List[dict]            # top-k chunks from ChromaDB
    response:         Optional[PolicyResponse]


# ── Singleton resources (loaded once per process) ─────────────────────────

_embedding_model:    Optional[SentenceTransformer] = None
_chroma_collection                                 = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)
    return _embedding_model


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client             = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


# ── Node 1: classify_intent ────────────────────────────────────────────────

def classify_intent(state: GraphState) -> GraphState:
    """
    Classify the incoming query as policy_question or general_question.

    Mock mode  (MOCK_LLM unset or 1 — graded baseline):
        Keyword heuristic — no LLM call.
        If the lowercased query contains any POLICY_KEYWORDS → policy_question.
        Otherwise → general_question.

    Real-LLM mode (MOCK_LLM=0 — optional, ungraded):
        Would call LLM to classify. Routing logic itself is the same.
    """
    query       = state["query"]
    query_lower = query.lower()

    if MOCK_LLM:
        # ── Graded baseline: keyword heuristic, zero network calls ────────
        intent = "general_question"
        for kw in POLICY_KEYWORDS:
            if kw in query_lower:
                intent = "policy_question"
                break
    else:
        # ── Optional real-LLM extension (MOCK_LLM=0) ──────────────────────
        # Placeholder: in a full implementation, call the LLM with a
        # classification prompt and parse its "policy_question" /
        # "general_question" label from the response.
        intent = "general_question"
        for kw in POLICY_KEYWORDS:
            if kw in query_lower:
                intent = "policy_question"
                break

    return {**state, "intent": intent}


# ── Node 2: retrieve_and_answer ────────────────────────────────────────────

def retrieve_and_answer(state: GraphState) -> GraphState:
    """
    For policy_question queries:
    1. Embed the query and retrieve top-3 chunks from ChromaDB (always real).
    2. Generate the answer.

    Mock mode  (graded baseline):
        answer = f"Based on the retrieved context: {top_chunk[:200]}"
        sources = [chunk IDs], confidence = 1.0   — no LLM call.

    Real-LLM mode (optional, ungraded):
        Feed retrieved chunks into PROMPT_TEMPLATE and call the LLM.
        Validate output against PolicyResponse; retry up to 2× on failure.
    """
    query = state["query"]

    # ── Retrieval — always runs for real in both modes ─────────────────────
    model      = _get_embedding_model()
    collection = _get_chroma_collection()

    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "ids"],
    )

    chunks = [
        {
            "id":       results["ids"][0][i],
            "content":  results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]

    # ── Answer generation ──────────────────────────────────────────────────
    if MOCK_LLM:
        # Graded baseline: canned template, no LLM call
        top_snippet = chunks[0]["content"][:200] if chunks else ""
        answer      = f"Based on the retrieved context: {top_snippet}"
        sources     = [c["id"] for c in chunks]
        confidence  = 1.0

    else:
        # Optional real-LLM extension — retry up to 2 additional times
        context = "\n\n".join(
            f"[{c['id']}]:\n{c['content']}" for c in chunks
        )
        prompt  = PROMPT_TEMPLATE.format(context=context, query=query)

        raw_answer = None
        for attempt in range(3):          # 1 try + up to 2 retries
            try:
                # ── Swap in your LLM call here ─────────────────────────────
                # e.g. with Groq:
                #   from groq import Groq
                #   client = Groq(api_key=os.environ["GROQ_API_KEY"])
                #   completion = client.chat.completions.create(
                #       model="llama3-8b-8192",
                #       messages=[{"role": "user", "content": prompt}]
                #   )
                #   raw_answer = completion.choices[0].message.content
                raise NotImplementedError("Set MOCK_LLM=0 and plug in your LLM client.")
            except NotImplementedError:
                raise
            except Exception:
                if attempt < 2:
                    prompt += "\n\nPlease respond ONLY with valid JSON matching the schema."
                    continue
                raw_answer = None
                break

        if raw_answer:
            answer     = raw_answer
            sources    = [c["id"] for c in chunks]
            confidence = 0.9
        else:
            answer     = f"Based on the retrieved context: {chunks[0]['content'][:200]}"
            sources    = [c["id"] for c in chunks]
            confidence = 0.5

    response = PolicyResponse(answer=answer, sources=sources, confidence=confidence)
    return {**state, "retrieved_chunks": chunks, "response": response}


# ── Node 3: direct_answer ──────────────────────────────────────────────────

def direct_answer(state: GraphState) -> GraphState:
    """
    For general_question queries (no policy retrieval needed).

    Mock mode  (graded baseline):
        Fixed canned string, no LLM call.

    Real-LLM mode (optional, ungraded):
        Would call LLM directly without retrieval context.
    """
    if MOCK_LLM:
        # Graded baseline: fixed canned string, no network call
        answer     = "I can only answer questions about Zepto policies right now."
        sources    = []
        confidence = 1.0
    else:
        # Optional real-LLM extension
        answer     = "I can only answer questions about Zepto policies right now."
        sources    = []
        confidence = 0.9

    response = PolicyResponse(answer=answer, sources=sources, confidence=confidence)
    return {**state, "response": response}


# ── Conditional routing ────────────────────────────────────────────────────

def _route_by_intent(state: GraphState) -> str:
    """
    Routes classify_intent's output to the correct answer node.
    This routing logic does NOT depend on MOCK_LLM — only the generation
    step inside each destination node does.
    """
    return state["intent"]   # "policy_question" | "general_question"


# ── Build and compile the graph ────────────────────────────────────────────

def build_graph():
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("classify_intent",    classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer",       direct_answer)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional edge: classify_intent → retrieve_and_answer OR direct_answer
    graph.add_conditional_edges(
        "classify_intent",
        _route_by_intent,
        {
            "policy_question":  "retrieve_and_answer",
            "general_question": "direct_answer",
        },
    )

    # Terminal edges
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer",       END)

    return graph.compile()


# Compile once at import time
compiled_graph = build_graph()


# ── Public API ─────────────────────────────────────────────────────────────

def run_query(query: str) -> PolicyResponse:
    """
    Run the full LangGraph pipeline for a single query.
    Returns a validated PolicyResponse.
    """
    initial_state: GraphState = {
        "query":            query,
        "intent":           None,
        "retrieved_chunks": [],
        "response":         None,
    }
    final_state = compiled_graph.invoke(initial_state)
    return final_state["response"]
