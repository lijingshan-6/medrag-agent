"""PubMed abstract ingestion via NCBI E-utilities (Biopython Entrez)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

from Bio import Entrez
from tenacity import retry, stop_after_attempt, wait_exponential


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _flatten_abstract_text(abst: Any) -> str:
    """Normalize BioPython's heterogeneous Abstract / AbstractText shapes."""
    if abst is None:
        return ""
    if isinstance(abst, str):
        return abst.strip()
    if isinstance(abst, dict):
        inner = abst.get("AbstractText")
        if inner is not None:
            return _flatten_abstract_text(inner)
        text = abst.get("#text")
        if text is not None:
            return str(text).strip()
        return ""
    if isinstance(abst, list):
        parts: list[str] = []
        for block in abst:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                chunk = block.get("#text")
                if chunk is None:
                    chunk = block.get("text")
                if isinstance(chunk, list):
                    parts.append(" ".join(str(t) for t in chunk))
                elif chunk is not None:
                    parts.append(str(chunk))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return " ".join(parts).strip()
    return str(abst).strip()


def _abstract_from_article(article: dict[str, Any]) -> str:
    raw_ab = article.get("Abstract")
    if raw_ab is None:
        return ""
    if isinstance(raw_ab, str):
        return raw_ab.strip()
    if isinstance(raw_ab, dict):
        return _flatten_abstract_text(raw_ab.get("AbstractText"))
    return _flatten_abstract_text(raw_ab)


def _publication_types(article: dict[str, Any]) -> list[str]:
    raw = article.get("PublicationTypeList")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        inner = raw.get("PublicationType")
        return [str(x) for x in _as_list(inner)]
    return [str(raw)]


def _mesh_terms(citation: dict[str, Any]) -> list[str]:
    mh = citation.get("MeshHeadingList")
    if not mh:
        return []
    out: list[str] = []
    for item in _as_list(mh):
        if not isinstance(item, dict):
            out.append(str(item))
            continue
        dn = item.get("DescriptorName")
        if isinstance(dn, dict):
            out.append(str(dn.get("#text", dn)))
        elif dn is not None:
            out.append(str(dn))
    return out


def _language(article: dict[str, Any]) -> str:
    lang = article.get("Language", ["eng"])
    if isinstance(lang, list) and lang:
        return str(lang[0])
    if isinstance(lang, str):
        return lang
    return "eng"


_BAD_PUB_TYPES = frozenset(
    {
        "Letter",
        "Editorial",
        "Comment",
        "Erratum",
        "Retracted Publication",
    }
)


@dataclass
class PubMedRecord:
    pmid: str
    title: str
    abstract: str
    authors: list[str]
    journal: str
    year: int
    pub_types: list[str]
    mesh_terms: list[str]
    language: str


def _build_query(keywords: list[str], year_from: int, year_to: int) -> str:
    kw = " OR ".join(f'"{k}"[Title/Abstract]' for k in keywords)
    return f"({kw}) AND ({year_from}:{year_to}[dp]) AND English[lang] AND hasabstract[text]"


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
def _esearch(query: str, retmax: int) -> list[str]:
    with Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort="pub_date") as handle:
        record = Entrez.read(handle)
    return list(record.get("IdList", []))


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
def _efetch_batch(pmids: list[str]) -> list[dict[str, Any]]:
    with Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="medline",
        retmode="xml",
    ) as handle:
        root = Entrez.read(handle)
    articles = root.get("PubmedArticle", [])
    if isinstance(articles, dict):
        return [articles]
    if isinstance(articles, list):
        return articles
    return []


def _parse_record(art: dict[str, Any]) -> PubMedRecord | None:
    """Extract fields from Entrez PubmedArticle dict; return None if filtered out."""
    try:
        cit = art["MedlineCitation"]
        article = cit["Article"]
        pmid = str(cit["PMID"])
        title = str(article.get("ArticleTitle", "") or "").strip()
        abstract = _abstract_from_article(article)
        if len(abstract.split()) < 80:
            return None

        date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year_raw = date.get("Year") or date.get("MedlineDate", "") or "0"
        year_str = str(year_raw).split()[0] if year_raw else "0"
        try:
            year = int(year_str)
        except ValueError:
            year = 0

        pub_types = _publication_types(article)
        if any(pt in _BAD_PUB_TYPES for pt in pub_types):
            return None

        authors: list[str] = []
        for a in _as_list(article.get("AuthorList")):
            if not isinstance(a, dict) or "LastName" not in a:
                continue
            authors.append(f"{a.get('LastName', '')} {a.get('Initials', '')}".strip())

        mesh = _mesh_terms(cit)
        journal = str(article.get("Journal", {}).get("Title", "") or "")
        language = _language(article)
        if language != "eng":
            return None

        return PubMedRecord(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            year=year,
            pub_types=pub_types,
            mesh_terms=mesh,
            language=language,
        )
    except Exception:
        return None


def fetch_pubmed_abstracts(
    keywords: list[str],
    year_from: int = 2020,
    year_to: int = 2026,
    max_results: int = 1500,
    email: str = "",
    api_key: str | None = None,
    batch_size: int = 200,
) -> Iterator[PubMedRecord]:
    """
    Search PubMed and yield parsed abstract records (filtered in `_parse_record`).

    NCBI requires a valid email on all requests.
    """
    email_clean = (email or "").strip()
    if not email_clean:
        raise ValueError(
            "NCBI E-utilities require a contact email. Pass email=... or set NCBI_EMAIL."
        )

    Entrez.email = email_clean
    Entrez.api_key = api_key

    query = _build_query(keywords, year_from, year_to)
    pmids = _esearch(query, retmax=max_results)
    print(f"[pubmed] esearch matched {len(pmids)} pmids")

    pause_s = 0.12 if api_key else 0.35

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        records = _efetch_batch(batch)
        for article in records:
            rec = _parse_record(article)
            if rec is not None:
                yield rec
        time.sleep(pause_s)


__all__ = ["PubMedRecord", "fetch_pubmed_abstracts"]
