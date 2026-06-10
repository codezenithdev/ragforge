# Briefr

**AI research briefs with receipts.** Ask a question, optionally add your own PDFs/DOCX, and get a structured, sourced brief in about a minute — where **every section carries a faithfulness score** telling you how grounded it is in the retrieved evidence.

Unlike a chat answer, a Briefr brief:

- combines **your private documents** with corrective **web retrieval** in one context,
- **decomposes** the question into sub-questions and retrieves for each,
- ranks evidence with **hybrid retrieval + cross-encoder re-ranking**, not just vector similarity,
- **self-corrects**: if retrieval quality is poor, it searches the web before generating (CRAG),
- and **scores its own grounding** per section, flagging anything below 60% confidence.

## Architecture

```
                                ┌────────────────────────────────────────────────┐
                                │                LangGraph pipeline               │
   React + Vite (3000)          │                (Celery worker)                  │
  ┌──────────────────┐          │  decompose ──► HyDE ──► retrieve ──► CRAG ──┐  │
  │ upload PDFs/DOCX │  enqueue │   (haiku)    (haiku)   (hybrid+RRF)  check  │  │
  │ ask a question   │──────────│                                        │    │  │
  │ confidence bars  │  poll 2s │              ┌── corrective web search ◄┘    │  │
  └────────┬─────────┘          │              ▼        (Tavily)               │  │
           │                    │   cross-encoder rerank ──► generate brief    │  │
           ▼                    │   (ms-marco MiniLM)          (sonnet)        │  │
   FastAPI (8000)               │              ┌── score faithfulness ◄────────┘  │
   /documents /briefs /eval     │              ▼      (haiku judge)               │
           │                    └──────────────┼──────────────────────────────────┘
           │ Redis (broker)                    │ persist result + scores
   ┌───────┴────────┬──────────────────────────┴───────┐
   ▼                ▼                                   ▼
 Redis (6379)    ChromaDB (8001→8000)              PostgreSQL (5433→5432)
 job queue +     chunk corpus: embeddings,         documents, briefs,
 BM25 cache      text, metadata (cosine HNSW)      sub-queries (JSONB results)
```

Six services via Docker Compose: `postgres`, `redis`, `chroma`, `backend` (FastAPI), `worker` (Celery), `frontend` (nginx-served React build).

## The advanced-RAG techniques, and why each is here

| Technique | What it does | Why it matters |
|---|---|---|
| **Multi-query decomposition** | Claude Haiku splits the question into ~6 self-contained sub-questions; retrieval runs per sub-question | A single embedding of a broad question misses facets. Decomposition gives the retriever several precise targets and the brief comprehensive coverage. |
| **HyDE** (Hypothetical Document Embeddings) | For each sub-question, the LLM writes a fake "ideal answer" paragraph; *that* gets embedded for retrieval | Questions and answers live in different embedding neighborhoods. A hypothetical answer is shaped like the passage you want to find — much better recall for abstract/technical queries. |
| **Hybrid retrieval + RRF** | Dense (ChromaDB cosine) and sparse (BM25) retrieval run in parallel; results fuse via Reciprocal Rank Fusion `score = Σ 1/(60 + rank)` | Dense search nails paraphrases but misses exact terms/IDs; BM25 is the opposite. RRF fuses *rankings*, not scores, so no score-scale tuning is ever needed. |
| **Cross-encoder re-ranking** | `ms-marco-MiniLM-L-6-v2` jointly scores each (query, chunk) pair, keeping the top 10 | Bi-encoders score query and document independently; a cross-encoder reads them together — far more accurate for the final cut, and ~40 ms for 50 docs on CPU. |
| **CRAG** (Corrective RAG) | Before generating, the top chunk's cross-encoder relevance (sigmoid) is checked against a threshold; below it, the LLM writes 3 focused web queries and Tavily results join the context | Most RAG failures are retrieval failures. CRAG *evaluates before generating* instead of hoping — when your documents can't answer, it goes and finds something that can. |
| **Per-section faithfulness scoring** | An LLM judge (Haiku) scores what fraction of each section's claims its cited sources actually support; sections under 0.6 get a visual warning | Generation can sound confident while being ungrounded. Real per-section scores turn "trust me" into "verify me" — the UI shows green/amber/red bars per section. |
| **RAGAS evaluation** | On demand (`POST /eval/run`), a completed brief is scored with RAGAS faithfulness / answer relevancy / context precision | An independent, reproducible quality metric for benchmarking pipeline changes. |

## Quickstart

Prereqs: Docker Desktop + API keys for [Anthropic](https://console.anthropic.com), [OpenAI](https://platform.openai.com) (embeddings), and [Tavily](https://tavily.com) (web search).

```bash
cp .env.example .env        # then paste your three API keys into .env
docker compose up --build   # first build ~10 min (CPU torch + model downloads)
```

Then open **http://localhost:3000** — upload a PDF, ask a question, watch the pipeline stages, and read the brief with its confidence bars.

API docs (Swagger): **http://localhost:8000/docs**

```bash
# Or drive it over HTTP:
curl -F "file=@notes.pdf" http://localhost:8000/api/v1/documents/upload
curl -X POST http://localhost:8000/api/v1/briefs \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Anthropic's competitive position?", "document_ids": []}'
curl http://localhost:8000/api/v1/briefs/<brief_id>          # poll until "complete"
curl -X POST http://localhost:8000/api/v1/eval/run \
  -H "Content-Type: application/json" -d '{"brief_id": "<brief_id>"}'
```

## Project layout

```
backend/
  app/
    api/routes/          documents, briefs, eval endpoints
    core/                config (pydantic-settings), Postgres, ChromaDB, Celery
    models/              SQLAlchemy: documents, briefs, brief_sub_queries
    rag/
      ingestion/         PDF/DOCX/web loaders, semantic chunker, OpenAI embedder
      retrieval/         Chroma vector store, BM25, RRF hybrid, cross-encoder
      pipeline/          query decomposer + HyDE, CRAG, LangGraph graph
      generation/        instructor-typed brief generator + schemas
      evaluation/        faithfulness LLM-judge, RAGAS runner
    tasks.py             Celery task running the graph (persistent event loop)
  alembic/               migrations (schema source of truth)
  tests/                 18 fully-mocked tests — no keys or services needed
frontend/
  src/                   React 18 + TS + Tailwind + React Query
```

## Development

```bash
# Infra only, run API/worker/frontend on the host:
docker compose up -d postgres redis chroma
cd backend
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m alembic upgrade head
.venv/Scripts/python -m uvicorn app.main:app --port 8000
.venv/Scripts/python -m celery -A app.core.celery_app worker --pool=solo  # solo on Windows
cd ../frontend && npm install && npm run dev

# Tests (offline — all LLM/web/vector layers mocked):
cd backend && .venv/Scripts/python -m pytest -q
```

Every model name and threshold lives in `backend/app/core/config.py` and can be overridden via environment variables — nothing is hardcoded.

### Notes & decisions

- **Postgres is published on host `5433`** (container-internal 5432) to avoid colliding with a natively installed PostgreSQL on 5432. Chroma is published on `8001` (internal 8000) because the backend owns host 8000.
- **Semantic chunking** uses a per-document *percentile breakpoint* on adjacent-sentence embedding distances rather than a fixed cosine threshold — fixed thresholds over-split badly with MiniLM-class embedders.
- **Chroma runs in server mode** (not embedded) because the API and the Celery worker are separate processes that both need the vector store.
- **The Celery worker keeps one persistent asyncio event loop per process** — the async clients (Chroma, Anthropic, asyncpg) are loop-bound, so per-task `asyncio.run` would strand them.
- **Pinned 0.3-era LangChain stack**: `ragas 0.2.x` hard-imports modules removed in `langchain-community 0.4+`. Install from `requirements.txt`; don't upgrade these independently.
