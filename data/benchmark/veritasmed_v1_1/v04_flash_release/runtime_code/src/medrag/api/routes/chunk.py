"""
GET /api/chunk/{chunk_id} — single chunk + context window.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from medrag.api._helpers import external_url, scroll_by_chunk_ids
from medrag.api.models import ChunkContextResponse, ChunkSlim

router = APIRouter()


@router.get("/api/chunk/{chunk_id:path}", response_model=ChunkContextResponse)
async def get_chunk(
    chunk_id: str,
    context_window: int = Query(default=1, ge=0, le=3),
) -> ChunkContextResponse:
    """Return the chunk and its neighbouring chunks."""
    # chunk_id format: "pubmed:12345:1"  or  "pmc:doc196:3"
    parts = chunk_id.rsplit(":", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=422, detail=f"Invalid chunk_id: {chunk_id!r}")
    doc_key = parts[0]
    try:
        idx = int(parts[1])
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid chunk index in {chunk_id!r}")

    # Build requested range
    start = max(0, idx - context_window)
    target_ids = [f"{doc_key}:{i}" for i in range(start, idx + context_window + 1)]

    payloads = scroll_by_chunk_ids(target_ids)
    if not payloads:
        raise HTTPException(status_code=404, detail=f"Chunk {chunk_id!r} not found")

    # Index by chunk_id
    by_id: dict[str, dict] = {p["chunk_id"]: p for p in payloads if "chunk_id" in p}

    def _slim(cid: str) -> ChunkSlim | None:
        p = by_id.get(cid)
        if p is None:
            return None
        return ChunkSlim(chunk_id=cid, text=p.get("text", ""), score=None)

    main = _slim(chunk_id)
    if main is None:
        raise HTTPException(status_code=404, detail=f"Chunk {chunk_id!r} not found")

    # Document metadata
    first_payload = payloads[0]
    source = first_payload.get("source", "pubmed")
    doc_id = first_payload.get("doc_id", "")
    pmid = first_payload.get("pmid") or None
    title = first_payload.get("title", "")

    if source == "pubmed" and pmid:
        citation = f"PMID:{pmid}"
    else:
        citation = f"PMC:{doc_id}"

    prev_id = f"{doc_key}:{idx - 1}" if idx > 0 else None
    next_id = f"{doc_key}:{idx + 1}"

    return ChunkContextResponse(
        chunk=main,
        prev_chunk=_slim(prev_id) if prev_id else None,
        next_chunk=_slim(next_id),
        document={
            "title": title,
            "citation": citation,
            "external_url": external_url(source, doc_id, pmid),
        },
    )
