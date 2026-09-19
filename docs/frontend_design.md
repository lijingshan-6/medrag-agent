> Historical development document. Some claims and defaults are superseded. For v0.1.0, use the repository README, docs/evaluation_report.md and docs/validation-2026-09-18.md.

# VeritasMed — Frontend Design & API Reference

---

## 1. Design Philosophy

VeritasMed's core value is **verifiability** — every answer is literature-backed, every reasoning step is traceable. The frontend exposes this chain rather than hiding it.

Three principles:

1. **Visible reasoning** — the agent loop (retrieve → grade → rewrite → generate → verify) streams in real time so users understand why the system gave a particular answer
2. **Transparent evidence** — every `[PMID:xxx]` citation in the answer is clickable, expanding to the raw chunk text with surrounding context
3. **Scored provenance** — each retrieved chunk shows its relevance score, source type (PubMed/PMC), and a direct link to the original publication

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (React + TypeScript)  · http://localhost:5173          │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  AnswerPage  │  │  ExplorerPage│  │  AgentTimeline       │  │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         └────────────────┴──────────────────────┘               │
│                          │                                       │
│     REST (Axios + VITE_API_URL)     WebSocket                    │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  FastAPI Backend  · http://localhost:8000                        │
│  CORS: allow all origins (configurable via CORS_ORIGINS env)    │
│                                                                 │
│  WS  /api/ask         streaming agent events                    │
│  GET /api/search      hybrid retrieval                          │
│  GET /api/document    document detail                           │
│  GET /api/chunk       chunk + context window                    │
│  GET /api/history     conversation history                      │
│  GET /api/corpus      corpus stats                              │
└─────────────────────────────────────────────────────────────────┘
```

The frontend and backend are **served separately**. The backend does not mount or serve the frontend. In production, use a reverse proxy (Nginx) or two containers.

---

## 3. Running in Development

**Terminal 1 — backend**
```powershell
.\start_dev.ps1         # Windows — Qdrant + backend + frontend
```

**Terminal 2 — frontend**
```bash
cd frontend
# Ensure frontend/.env.local contains: VITE_API_URL=http://localhost:8000
npm run dev             # Vite dev server at http://localhost:5173
```

`frontend/.env.local` is gitignored. Create it with one line:
```
VITE_API_URL=http://localhost:8000
```

---

## 4. Type System

### 4.1 REST types — generated, do not edit

REST types come from the FastAPI OpenAPI schema:

```bash
python scripts/export_openapi.py       # regenerate openapi.json at project root
cd frontend && npm run generate-types  # openapi-typescript → src/types/api.gen.ts
```

`src/types/index.ts` re-exports the generated types under friendly names:

```typescript
export type ChunkOut             = components['schemas']['ChunkOut']
export type SearchResponse       = components['schemas']['SearchResponse']
export type AnswerOut            // re-exported from ws.ts (WebSocket-only)
// …
```

### 4.2 WebSocket types — hand-maintained

`/api/ask` is a WebSocket endpoint and is not part of the OpenAPI schema. Types live in `src/types/ws.ts` as a manual mirror of `src/medrag/api/models.py`. **Keep them in sync whenever `models.py` changes.**

```typescript
// Discriminated union — mirror of AgentEvent in models.py
export type AgentEvent =
  | NodeStartEvent        // { event: 'node_start', node: string }
  | NodeEndEvent          // { event: 'node_end', node: string, data: NodeEndData }
  | ChunkRetrievedEvent   // { event: 'chunk_retrieved', node: string, data: ... }
  | DoneEvent             // { event: 'done', node: null, data: AnswerOut }
  | ErrorEvent            // { event: 'error', node: null, data: { message } }
```

`AnswerOut` is defined in `ws.ts` (not in `api.gen.ts`) because it only appears in the WebSocket `done` event.

---

## 5. WebSocket Protocol — `/api/ask`

### 5.1 Client → Server

```json
{
  "query": "What are the contraindications of warfarin in elderly patients?",
  "thread_id": "session-abc123",
  "pipeline": "p3"
}
```

`pipeline`: `"p2"` (fast, hybrid-only) or `"p3"` (slower, +reranker).

### 5.2 Server → Client event stream

```jsonc
// Node begins execution
{"event": "node_start", "node": "retrieve"}

// Chunk arrives during retrieval (one message per chunk)
{"event": "chunk_retrieved", "node": "retrieve", "data": {
  "chunk_id": "pubmed:12345:0",
  "citation": "PMID:12345",
  "title": "Warfarin therapy in elderly patients...",
  "score": 0.812,
  "text_snippet": "Absolute contraindications include active bleeding...",
  "source": "pubmed",
  "external_url": "https://pubmed.ncbi.nlm.nih.gov/12345/"
}}

// Node finishes — data fields vary by node
{"event": "node_end", "node": "grade", "data": {
  "relevance_score": 0.75,
  "relevant": true,
  "reason": "Chunks cover contraindications but lack bleeding-risk quantification",
  "rewrite_hint": ""
}}

// If grade is insufficient and rewrite fires:
{"event": "node_start", "node": "rewrite"}
{"event": "node_end", "node": "rewrite", "data": {
  "new_query": "warfarin absolute relative contraindications elderly bleeding risk"
}}

// Faithfulness check result
{"event": "node_end", "node": "check", "data": {
  "faithful": true,
  "issues": "",
  "confidence": 0.92
}}

// Final result — answer arrives once, no incremental token streaming
{"event": "done", "node": null, "data": {
  "answer": "Warfarin is contraindicated in... [PMID:12345] [PMC:doc88]",
  "citations": ["PMID:12345", "PMC:doc88"],
  "confidence": 0.92,
  "faithful": true,
  "faithfulness_issues": "",
  "iterations": 0,
  "regen_count": 0,
  "rewritten_queries": [],
  "chunks": [...],
  "thread_id": "session-abc123",
  "latency_ms": 4821.3
}}
```

`node_end` data fields per node:

| Node | Fields |
|------|--------|
| `retrieve` / `rerank` | `count` |
| `grade` | `relevance_score`, `relevant`, `reason`, `rewrite_hint` |
| `rewrite` | `new_query`, `rewritten_queries` |
| `check` | `faithful`, `issues`, `confidence` |
| `route` | `route` (query type) |

---

## 6. REST API Reference

### `GET /api/search`

```
GET /api/search?q=warfarin+elderly&k=10&pipeline=p2&highlight=true
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q` | string | required | query text |
| `k` | int | 5 | chunks to return (1–20) |
| `pipeline` | `p2`\|`p3` | `p2` | retrieval pipeline |
| `highlight` | bool | true | compute keyword highlight ranges |

Response: `SearchResponse` (see `api.gen.ts`).

---

### `GET /api/document/{citation}`

```
GET /api/document/PMID:12345
GET /api/document/PMC:doc196
```

Returns all chunks for the document plus metadata. Response: `DocumentResponse`.

---

### `GET /api/chunk/{chunk_id}`

```
GET /api/chunk/pubmed:12345:1?context_window=1
```

Returns the chunk plus `prev_chunk` and `next_chunk` within `context_window` steps. Response: `ChunkContextResponse`.

---

### `GET /api/history/{thread_id}`

Returns all Q&A turns and rolling summary for a conversation thread. Response: `HistoryResponse`.

---

### `GET /api/corpus/stats`

Returns total chunk count, per-source breakdown, collection name, embedding model. Response: `CorpusStats`.

---

### `GET /api/health`

```json
{"status": "ok", "qdrant": "connected", "llm": "connected"}
```

---

## 7. Frontend Component Reference

### 7.1 Pages

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | `AnswerPage` | Three-panel QA interface |
| `/explore` | `ExplorerPage` | Literature search browser |
| `/document/:citation` | `DocumentPage` | Full document with all chunks |

### 7.2 AnswerPage — three-panel layout

```
┌────────────────────────────────────────────────────────────────────┐
│  HEADER: VeritasMed  ·  Ask  ·  Explore                            │
├──────────────────┬────────────────────────┬────────────────────────┤
│                  │                        │                        │
│  AgentTimeline   │  AnswerPanel           │  EvidencePanel         │
│  (reasoning      │  (answer + faith-      │  (retrieved chunks     │
│   steps)         │   fulness badge)        │   with scores)         │
│  ~280px          │  flex-1                │  ~360px                │
│                  │                        │                        │
├──────────────────┴────────────────────────┴────────────────────────┤
│  QueryInput (query text + pipeline selector + send/cancel)         │
└────────────────────────────────────────────────────────────────────┘
```

### 7.3 AgentTimeline

Renders a vertical timeline of `TimelineNode` objects from the Zustand store. Each node shows:
- Status icon: waiting (circle) · running (spinner) · done (check) · rewrite (arrows) · error (x)
- Summary text (e.g. `"score: 0.75 · relevant"` for grade)
- For `grade` nodes: a mini score bar
- For `rewrite` nodes: old query vs. new query diff
- For `check` nodes: faithful / issues badge

### 7.4 AnswerPanel

Renders the final answer as annotated Markdown. `[PMID:xxx]` and `[PMC:xxx]` citation patterns are replaced with colored superscript buttons that scroll the EvidencePanel to the matching chunk. Includes a `FaithfulnessBadge` showing confidence, rewrite count, regen count, and latency.

### 7.5 EvidencePanel

Shows `liveChunks` (arriving during retrieval via `chunk_retrieved` events) or `result.chunks` (from the `done` event). Each `ChunkCard` has:
- Colored index badge (cycle through 8 hues matching AnswerPanel citation colors)
- Score bar
- Truncated text with keyword highlights (`highlight_ranges` from the search API)
- Context expander (`/api/chunk/{id}?context_window=1`)
- External link to PubMed / PMC

### 7.6 State Management (Zustand)

Key store fields:

| Field | Type | Purpose |
|-------|------|---------|
| `query` | `string` | current query input |
| `pipeline` | `'p2' \| 'p3'` | selected pipeline |
| `threadId` | `string` | conversation thread ID |
| `isStreaming` | `boolean` | WebSocket active |
| `timeline` | `TimelineNode[]` | agent reasoning steps |
| `liveChunks` | `ChunkOut[]` | chunks arriving during retrieval |
| `result` | `AnswerOut \| null` | final answer from `done` event |
| `selectedChunkId` | `string \| null` | highlighted chunk in EvidencePanel |
| `errorMessage` | `string \| null` | backend error to display |

### 7.7 useAgentStream hook

Manages the WebSocket lifecycle. On `send()`:
1. Closes any existing socket
2. Resets timeline, liveChunks, result, errorMessage
3. Opens new WebSocket to `wsAskUrl()` (derived from `VITE_API_URL`)
4. Dispatches each incoming `AgentEvent` to the store

```typescript
const { send, cancel } = useAgentStream()
```

---

## 8. Production Deployment

The backend and frontend are independent services — deploy them separately.

**Backend**
```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
  uvicorn medrag.api.app:app --host 0.0.0.0 --port 8000
```

Set `CORS_ORIGINS=https://your-frontend-domain.com` to restrict CORS in production.

**Frontend**
```bash
cd frontend
npm run build          # outputs to frontend/dist/
# Serve dist/ with Nginx, Caddy, or any static host
# Set VITE_API_URL to the backend's public URL before building
```

**Regenerate types after model changes**
```bash
python scripts/export_openapi.py
cd frontend && npm run generate-types
```
