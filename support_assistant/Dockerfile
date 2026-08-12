# ── Stage: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# MOCK_LLM=1 is the graded baseline — no LLM API keys required.
# Override at runtime with:  docker run -e MOCK_LLM=0 ...
ENV MOCK_LLM=1

WORKDIR /app

# Install dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and policy documents
COPY . .

# Run the ingestion step so ChromaDB is populated before first request
RUN python ingest.py

# Expose FastAPI port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
