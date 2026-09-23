"""
GET /api/document/{citation} — full document with all chunks.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from medrag.api._helpers import external_url, scroll_by_doc_id
from medrag.api.models import DocumentChunkSlim, DocumentResponse

router = APIRouter()


def _parse_citation(citation: str) -> tuple[str, str]:
    """Return (source, doc_id) from 'PMID:12345' or 'PMC:doc196'."""
    upper = citation.upper()
    if upper.startswith("PMID:"):
        return "pubmed", citation[5:]
    if upper.startswith("PMC:"):
        return "pmc", citation[4:]
    raise ValueError(f"Unrecognised citation format: {citation!r}")


@router.get("/api/document/{citation:path}", response_model=DocumentResponse)
async def get_document(citation: str) -> DocumentResponse:
    try:
        source, doc_id = _parse_citation(citation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    payloads = scroll_by_doc_id(doc_id, source)
    if not payloads:
        raise HTTPException(status_code=404, detail=f"Document {citation!r} not found")

    # Metadata from first chunk
    first = payloads[0]
    title = first.get("title", "")
    pmid = first.get("pmid") or None
    ext_url = external_url(source, doc_id, pmid)

    chunks_slim = [
        DocumentChunkSlim(
            chunk_id=p.get("chunk_id", ""),
            chunk_idx=int(p.get("chunk_idx", i)),
            section=p.get("section") or None,
            text=p.get("text", ""),
        )
        for i, p in enumerate(payloads)
    ]

    return DocumentResponse(
        citation=citation,
        source=source,
        doc_id=doc_id,
        title=title,
        pmid=pmid,
        external_url=ext_url,
        total_chunks=len(chunks_slim),
        chunks=chunks_slim,
    )
