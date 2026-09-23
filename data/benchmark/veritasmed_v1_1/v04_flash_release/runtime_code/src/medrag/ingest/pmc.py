"""PMC Open Access full-text ingestion via BioC XML API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class PMCRecord:
    pmcid: str
    pmid: str | None
    title: str
    full_text: str
    sections: list[dict]
    word_count: int


PMC_BIOC_URL = (
    "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{}/unicode"
)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
def _fetch_bioc_xml(pmcid: str, timeout: float = 30.0) -> str:
    url = PMC_BIOC_URL.format(pmcid)
    r = httpx.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _parse_bioc(xml_text: str) -> PMCRecord | None:
    try:
        root = ET.fromstring(xml_text)
        doc = root.find("document")
        if doc is None:
            return None
        pmcid = ""
        pmid = None
        for inf in doc.findall("infon"):
            k = inf.attrib.get("key", "")
            if k in ("article-id_pmc", "pmcid"):
                pmcid = inf.text or ""
            elif k in ("article-id_pmid", "pmid"):
                pmid = inf.text
        title = ""
        sections: list[dict] = []
        full_parts: list[str] = []
        cur_section = "body"
        for psg in doc.findall("passage"):
            sec_type = ""
            for inf in psg.findall("infon"):
                if inf.attrib.get("key") == "section_type":
                    sec_type = inf.text or ""
            text_el = psg.find("text")
            text = (text_el.text or "").strip() if text_el is not None else ""
            if not text:
                continue
            if sec_type == "TITLE" and not title:
                title = text
                continue
            if sec_type and sec_type != cur_section:
                cur_section = sec_type
            sections.append({"name": cur_section, "text": text})
            full_parts.append(text)
        full_text = "\n\n".join(full_parts).strip()
        wc = len(full_text.split())
        if wc < 1000:
            return None
        return PMCRecord(
            pmcid=pmcid, pmid=pmid, title=title,
            full_text=full_text, sections=sections, word_count=wc,
        )
    except Exception:
        return None


def fetch_pmc_records(pmcids: list[str], delay: float = 0.5):
    for pmcid in pmcids:
        try:
            xml = _fetch_bioc_xml(pmcid)
            rec = _parse_bioc(xml)
            if rec:
                yield rec
        except Exception as e:
            print(f"[pmc] {pmcid} failed: {e}")
        time.sleep(delay)


__all__ = ["PMCRecord", "fetch_pmc_records"]
