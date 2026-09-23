"""Text chunking for PubMed abstracts and PMC full texts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    source: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, chunk_size: int = 2048, overlap: int = 256) -> list[str]:
    """Simple character-based splitter that respects paragraph boundaries."""
    if len(text) <= chunk_size:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            parts.append(text[start:].strip())
            break
        # Try to split at paragraph boundary
        split_at = text.rfind("\n\n", start, end)
        if split_at == -1 or split_at <= start:
            # Fall back to sentence boundary
            split_at = text.rfind(". ", start, end)
        if split_at == -1 or split_at <= start:
            split_at = end
        else:
            split_at += 1  # include the period/newline
        parts.append(text[start:split_at].strip())
        start = max(split_at - overlap, start + 1)
    return [p for p in parts if p]


def chunk_pubmed_record(rec: dict) -> list[Chunk]:
    """PubMed abstract: title + abstract as a single chunk (usually fits)."""
    text = (rec["title"] + ". " + rec["abstract"]).strip()
    return [Chunk(
        chunk_id=f"pubmed:{rec['pmid']}:0",
        source="pubmed",
        doc_id=rec["pmid"],
        text=text,
        metadata={
            "title": rec["title"],
            "year": rec.get("year"),
            "authors": rec.get("authors", [])[:5],
            "journal": rec.get("journal"),
            "mesh_terms": rec.get("mesh_terms", []),
            "chunk_idx": 0,
            "total_chunks": 1,
        },
    )]


def chunk_pmc_record(rec: dict) -> list[Chunk]:
    """PMC full text: split each section, long sections get further split."""
    out: list[Chunk] = []
    idx = 0
    for sec in rec.get("sections", []):
        for piece in _split_text(sec["text"]):
            out.append(Chunk(
                chunk_id=f"pmc:{rec['pmcid']}:{idx}",
                source="pmc",
                doc_id=rec["pmcid"],
                text=piece,
                metadata={
                    "title": rec.get("title", ""),
                    "section": sec["name"],
                    "pmid": rec.get("pmid"),
                    "chunk_idx": idx,
                },
            ))
            idx += 1
    for c in out:
        c.metadata["total_chunks"] = len(out)
    return out


__all__ = ["Chunk", "chunk_pubmed_record", "chunk_pmc_record"]
