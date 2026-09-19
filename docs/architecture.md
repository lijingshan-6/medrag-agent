> Historical development document. Some claims and defaults are superseded. For v0.1.0, use the repository README, docs/evaluation_report.md and docs/validation-2026-09-18.md.

# MedRAG-Agent — Architecture Reference

> Stack: LangGraph 0.2 · FastMCP 2.x · MiMo V2.5 / V2.5-Pro (API) · BGE-M3 · BGE-Reranker-v2-m3 · Qdrant · sentence_transformers

---

## 1. System Overview

MedRAG-Agent is a retrieval-augmented generation system for medical literature QA. It combines a vector database of PubMed/PMC abstracts with a LangGraph agentic loop that **retrieves, grades, rewrites, generates, and verifies** answers — terminating only when the answer is grounded in the retrieved evidence.

Two access paths:

```
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│  Browser (React + TypeScript)    │    │  Claude Desktop / Claude Code    │
│  http://localhost:5173           │    │  (MCP client)                    │
└──────────────────┬───────────────┘    └──────────────────┬───────────────┘
                   │  REST / WebSocket                      │  FastMCP 2.x
                   │  http://localhost:8000                 │  (stdio / SSE)
                   ▼                                        ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│  FastAPI Backend                 │    │  MedRAG MCP Server               │
│                                  │    │                                  │
│  /api/ask   WebSocket            │    │  Security Middleware (5 layers)  │
│  /api/search, /document, …       │    │  auth → rate_limit →             │
│  CORSMiddleware (all origins)    │    │  injection_guard → pii → audit   │
└──────────────────┬───────────────┘    └──────────────────┬───────────────┘
                   │                                        │
                   └──────────────┬─────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LangGraph Agentic Loop                            │
│                   (CompiledStateGraph + SqliteSaver)                │
│                                                                     │
│  START → route → retrieve → rerank → grade ──────► generate        │
│                    ▲           │               │         │          │
│                    │      (relevant)      (not rel,      │          │
│                    │           │           iter<1)       │          │
│                    │           ▼               │         ▼          │
│                    └───── rewrite ◄────────────┘      check        │
│                                                          │          │
│                                             (faithful) ──► END      │
│                                         (unfaithful,               │
│                                           smart gate) ──► END       │
│                                         (unfaithful,               │
│                                           regen<1)  ──► inc_regen  │
│                                                          │          │
│                                                     ─► generate    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Retrieval Pipelines                                │
│                                                                     │
│  P2 Hybrid:  BGE-M3 dense (1024-d) ──► RRF fusion                  │
│  P3 Reranker: P2 candidates ──► BGE-Reranker cross-encoder         │
│                                                                     │
│  Qdrant (localhost:6333) · collection: medrag_text                  │
│  ~186k chunks from PubMed abstracts + PMC full text                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. LangGraph Node Reference

| Node | Python function | LLM | Thinking | Responsibility |
|------|----------------|-----|----------|----------------|
| `route` | `route_query` | `llm_fast` | OFF | Classify query: factual / synthesis / multihop |
| `retrieve` | `hybrid_retrieve` | — | — | Dense RRF retrieval, returns top-20 candidates |
| `rerank` | `rerank_chunks` | — | — | BGE cross-encoder → shrink to top-5 |
| `grade` | `grade_relevance` | `llm_think` | ON | Score chunk relevance 0–1; set `rewrite_hint` |
| `rewrite` | `rewrite_query` | `llm_think` | ON | Rewrite failed query (MeSH synonyms, sub-questions) |
| `generate` | `generate_answer_node` | `llm_fast` | OFF | Structured JSON answer: {answer, citations, confidence} |
| `check` | `check_faithfulness` | `llm_think` | ON | Binary faithfulness audit with smart gate |
| `inc_regen` | `increment_regen` | — | — | Increment `regen_count` before re-generation |
| `append_history` | `append_history` | — | — | Persist completed Q&A turn to `state["history"]` |
| `summarize_gate` | lambda passthrough | — | — | Decide whether to compress history |
| `summarize` | `summarize_history` | `llm_fast` | OFF | Compress history to ≤200-word rolling summary |

### 2.1 Dual-LLM Strategy

```
llm_fast  (mimo-v2.5, thinking=disabled, temp=0.2)
  → route, generate, summarize
  → ~1–2 s per call, deterministic output

llm_think (mimo-v2.5-pro, thinking=disabled, temp=0.6)
  → grade, rewrite, check
  → ~2–4 s per call, higher accuracy than v2.5
```

Both tiers pass `extra_body={"thinking": {"type": "disabled"}}`. MiMo reasoning
models default to internal CoT which burns 1000–5000 reasoning tokens before
producing content (adds 15–27 s per call). Disabling it is required for
practical latency. The "think" label now refers to the Pro model tier, not
literal chain-of-thought.

Configured via env vars `MIMO_MODEL_FAST` and `MIMO_MODEL_THINK` (or `LLM_BACKEND=ollama`).

### 2.2 Conditional Routing

```
After grade:
  relevance_score ≥ threshold  →  generate
    thresholds: factual=0.5 · synthesis=0.6 · multihop=0.7
  score < threshold, iterations < MAX_REWRITES (1)  →  rewrite
  score < threshold, iterations ≥ MAX_REWRITES      →  generate (cap hit)

After check:
  faithful = True                                    →  append_history → END
  unfaithful, confidence ≥ 0.3 AND has citations    →  append_history → END (smart gate)
  unfaithful, regen_count < MAX_REGEN (1)            →  inc_regen → generate
  unfaithful, regen_count ≥ MAX_REGEN                →  append_history → END
```

Constants (`nodes.py`): `MAX_REWRITES=1`, `MAX_REGEN=1`, `GRADE_THRESHOLD=0.6` (base; overridden per query type), `REGEN_CONFIDENCE_SKIP=0.3`, `CANDIDATE_K=20`, `TOP_K=5`, `HISTORY_SUMMARIZE_EVERY=10`.

---

## 3. Two-Tier Memory Architecture

```
L1 — LangGraph SqliteSaver
  Storage:  data/checkpoints/agent.db (SQLite)
  Purpose:  crash recovery, multi-turn conversation continuity
  Scope:    full AgentState snapshot per step
  Key:      thread_id (set by client per user session)

L2 — Rolling Summarisation
  Trigger:  every 10 conversation turns (HISTORY_SUMMARIZE_EVERY=10)
  LLM:      llm_fast (thinking=OFF)
  Output:   ≤200-word summary in state["summary"]
  Purpose:  prevent context-window overflow in long sessions
```

---

## 4. Retrieval Pipeline

### 4.1 Embedding — sentence_transformers

The embedder uses `FlagEmbedding.inference.embedder.encoder_only.m3.M3Embedder` (submodule import, bypasses `FlagEmbedding.__init__`). The reranker uses `sentence_transformers.CrossEncoder`. Importing the FlagEmbedding top-level package (`from FlagEmbedding import ...`) triggers a `STATUS_ACCESS_VIOLATION` crash on Windows because `__init__` pulls in the decoder-only reranker (`modeling_minicpm_reranker.py`) whose C++ runtime conflicts with qdrant_client's gRPC runtime. The fix is to import only the encoder-only submodule directly.

```python
# embedder.py — uses FlagEmbedding submodule (not top-level package)
from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder
M3Embedder("BAAI/bge-m3", use_fp16=False, devices=["cpu"])
# Returns dense 1024-d float32 normalised vectors + sparse lexical weights.
# sparse keys are string token IDs castable to int for Qdrant SparseVector.

# reranker.py — sentence_transformers CrossEncoder (no FlagEmbedding dependency)
CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
# fp16 on CUDA, float32 on CPU
```

Device is auto-detected (`EMBEDDER_DEVICE` / `RERANKER_DEVICE` env vars, default `auto` → `cuda` if available, else `cpu`).

**Critical import order** (`app.py`): `import sentence_transformers` must appear before any `qdrant_client` import. On Windows, qdrant_client's gRPC C++ runtime conflicts with PyTorch if PyTorch loads after it. Pre-importing `sentence_transformers` at startup loads PyTorch first.

### 4.2 P2 Hybrid Retrieval

```
Query
  │
  └──► BGE-M3 encode (dense 1024-d, + sparse weights if available)
         │
         ├──► Qdrant dense search   →  top-20 by cosine similarity
         │
         ├──► Qdrant sparse search  →  top-20 (skipped when sparse_weights is empty)
         │
         └──► RRF fusion (k=60)    →  top-20 fused candidates
```

`HybridRetriever` calls `embedder.encode(return_sparse=True)` and checks `if sparse_weights:` before issuing the sparse Qdrant query. With the current `BGEM3Embedder` (sentence_transformers backend), sparse weights are always empty, so sparse search is skipped and the result is effectively dense-only RRF. If the embedder is replaced with one that produces sparse vectors, sparse search activates automatically with no changes to `hybrid.py`.

### 4.3 P3 Hybrid + Reranker

```
P2 output (top-20 candidates)
  │
  └──► BGE-Reranker-v2-m3 (cross-encoder, batch_size=8)
       Input:  [query, chunk_text] pairs
       Output: top-5 by reranker score
```

### 4.4 Evaluation Results (50-question golden dataset)

| Pipeline | R@5 | MRR@20 | Latency |
|----------|-----|--------|---------|
| P1 Dense | 98.0% | 0.963 | 0.48 s |
| P2 Hybrid | **100.0%** | **1.000** | 0.55 s |
| P3 Hybrid+Reranker | **100.0%** | **1.000** | ~65 s |
| P4 HyDE | 88.0% | 0.810 | 8.97 s |
| P5 Multi-Query | 96.0% | 0.936 | 8.09 s |

P3 is used for `/api/ask` (quality-critical). P2 is used for `/api/search` (speed-critical).

---

## 5. API Contract

### 5.1 REST — OpenAPI

FastAPI auto-generates `/openapi.json`. TypeScript types are generated from it:

```bash
python scripts/export_openapi.py      # write openapi.json
cd frontend && npm run generate-types  # write src/types/api.gen.ts
```

Key schemas: `ChunkOut`, `SearchResponse`, `DocumentResponse`, `ChunkContextResponse`, `CorpusStats`, `AnswerOut`.

### 5.2 WebSocket — AgentEvent (manual sync)

`/api/ask` is a WebSocket endpoint — not in the OpenAPI schema. Events are a Pydantic discriminated union (`models.py`) mirrored manually in `frontend/src/types/ws.ts`.

```
node_start        { event, node }
node_end          { event, node, data: NodeEndData }
chunk_retrieved   { event, node, data: ChunkRetrievedData }
done              { event: "done", node: null, data: AnswerOut }
error             { event: "error", node: null, data: { message } }
```

Answer arrives once in the `done` event (no incremental token streaming).

---

## 6. Corpus Statistics

| Metric | Value |
|--------|-------|
| Sources | PubMed abstracts + PMC full-text (Open Access) |
| Total chunks (evaluation run) | 44,768 (1,975 PubMed + 42,793 PMC) |
| Avg chunk length | ~300 tokens |
| Chunk overlap | 64 tokens |
| Embedding model | BAAI/bge-m3 (dense 1024-d, sentence_transformers) |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| Vector DB | Qdrant (single-node, localhost:6333) |
| Sparse vectors | Not produced by current embedder — sparse Qdrant search skipped at runtime |

---

## 7. Directory Structure

```
medrag-agent/
├── src/medrag/
│   ├── agent/
│   │   ├── graph.py        # StateGraph + SqliteSaver (graph assembly)
│   │   ├── nodes.py        # 11 node functions
│   │   ├── state.py        # AgentState TypedDict
│   │   ├── prompts.py      # LLM prompt templates
│   │   ├── llms.py         # Dual-LLM factory (mimo/ollama backends)
│   │   ├── utils.py        # strip_thinking(), build_answer_from_claims()
│   │   └── generator.py    # Week-1 baseline: single-shot answer (no agent loop)
│   ├── api/
│   │   ├── app.py          # FastAPI entry point (CORS, import order)
│   │   ├── models.py       # Pydantic models — single source of truth
│   │   ├── _helpers.py     # Shared utilities
│   │   └── routes/         # ask, search, document, chunk, history, corpus
│   ├── index/
│   │   ├── embedder.py     # BGEM3Embedder (sentence_transformers, dense 1024-d)
│   │   ├── indexer.py      # Qdrant upsert pipeline
│   │   └── qdrant_setup.py # Collection initialisation
│   ├── ingest/
│   │   ├── pubmed.py       # PubMed abstract fetcher
│   │   ├── pmc.py          # PMC OA full-text fetcher
│   │   └── chunker.py      # Sliding-window chunker (64-token overlap)
│   ├── retrieval/
│   │   ├── hybrid.py       # HybridRetriever (dense-only RRF)
│   │   ├── reranker.py     # BGEReranker (CrossEncoder)
│   │   ├── retriever.py    # DenseRetriever + RetrievedChunk
│   │   ├── hyde.py         # HyDERetriever
│   │   └── multi_query.py  # MultiQueryRetriever
│   └── mcp_server/
│       ├── server.py       # FastMCP server + 4 tools
│       └── security/       # auth, rate_limit, injection_guard, pii, audit
├── frontend/               # React + TypeScript (served separately)
│   ├── src/types/
│   │   ├── api.gen.ts      # Generated from openapi.json — do not edit
│   │   ├── index.ts        # Re-exports
│   │   └── ws.ts           # WebSocket types (manual mirror of models.py)
│   ├── .env.local          # VITE_API_URL=http://localhost:8000 (gitignored)
│   └── vite.config.ts      # No proxy — direct CORS to :8000
├── data/
│   ├── golden/             # Evaluation datasets (standard + hard set)
│   ├── eval/               # Evaluation outputs
│   └── checkpoints/        # LangGraph SqliteSaver state (runtime)
├── scripts/                # Numbered pipeline scripts (01–14)
├── openapi.json            # FastAPI OpenAPI schema
├── .env.example            # Required environment variables
├── start_dev.ps1           # Dev launcher (Windows)
├── start_setup.ps1         # First-run setup (Windows)
└── start_mcp.ps1           # MCP server launcher (Windows)
```
