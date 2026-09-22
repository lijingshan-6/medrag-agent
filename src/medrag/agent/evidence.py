"""Source-bound answer components shared by generation, repair and the UI."""
from __future__ import annotations

import html
import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from medrag.retrieval.retriever import RetrievedChunk


class EvidenceSpan(BaseModel):
    chunk_id: str
    citation: str = ""
    quote: str


class AnswerComponent(BaseModel):
    id: str = ""
    requirement: str
    source_hint: str = ""
    status: Literal["supported", "partial", "missing"] = "missing"
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    required_details: list[str] = Field(default_factory=list)
    gap: str = ""
    answer: str = ""


def normalized(text: str) -> str:
    """Ignore presentation whitespace/HTML, preserving numbers and inequalities."""
    text = html.unescape(re.sub(r"</?[A-Za-z][^>]*>", "", text))
    return " ".join(text.split()).casefold()


def source_spans(chunks: list[RetrievedChunk]) -> dict[str, dict]:
    """Give source sentences stable local IDs so models need not transcribe quotes."""
    spans = {}
    for chunk in chunks:
        # A decimal or 'vs. 84%' is not a sentence break. All returned text is
        # still verbatim source material, including any original HTML markup.
        for quote in re.split(r"(?<=[.!?])\s+(?=[A-Z<])|\n{2,}", chunk.text):
            if quote.strip():
                spans[f"E{len(spans) + 1}"] = {
                    "chunk_id": chunk.chunk_id, "citation": chunk.citation, "quote": quote.strip(),
                }
    return spans


def _has_population_count(text: str) -> bool:
    return bool(re.search(
        r"\b\d[\d,]*\s+(?:[\w-]+\s+){0,3}(?:patients?|participants?|controls?|veterans?|subjects?|women|men|children|volunteers?|births|images|examinations|animals?|mice|rats)\b",
        normalized(text),
    ))


def bind_components(
    raw: object, requirements: list[str], chunks: list[RetrievedChunk], study_context: object = None,
) -> list[dict]:
    """Accept only quotations that exist in the declared retrieved chunk.

    This validates provenance, not whether the quoted finding answers the question.
    Citation keys come from the chunk rather than the model's proposed reference.
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    spans = source_spans(chunks)
    span_keys = list(spans)
    components = []
    for value in raw if isinstance(raw, list) else []:
        if isinstance(value, dict) and "evidence_ids" in value:
            ids = value["evidence_ids"] if isinstance(value["evidence_ids"], list) else []
            chosen = [key for key in ids if isinstance(key, str) and key in spans]
            reference_details = []
            # "Comparable" can refer to values in the preceding sentence. Only
            # bring that sentence in when no explicit comparator is given here;
            # an unrelated preceding numeric result is not a required detail.
            for key in list(chosen):
                index = span_keys.index(key)
                current = spans[key]
                comparison = normalized(current["quote"])
                if index and re.search(r"\bcomparab\w*\b", comparison) and not re.search(r"\b(vs|versus|compared)\b", comparison):
                    previous_key = span_keys[index - 1]
                    previous = spans[previous_key]
                    if previous["chunk_id"] == current["chunk_id"] and re.search(r"\d", previous["quote"]):
                        if previous_key not in chosen:
                            chosen.append(previous_key)
                        reference_details.append(previous["quote"])
            value = {**value, "evidence": [spans[key] for key in chosen],
                     "required_details": [*(value.get("required_details") or []), *reference_details]}
        try:
            component = AnswerComponent.model_validate(value)
        except (ValidationError, TypeError):
            continue
        if not component.requirement.strip():
            continue
        component.id = f"C{len(components) + 1}"
        valid = []
        for span in component.evidence:
            chunk = by_id.get(span.chunk_id)
            if chunk and normalized(span.quote) and normalized(span.quote) in normalized(chunk.text):
                if span.citation and span.citation != chunk.citation:
                    continue
                valid.append(span.model_copy(update={"citation": chunk.citation}))
        lost_evidence = len(valid) != len(component.evidence)
        component.evidence = valid
        # Required verbatim details must themselves be present in a bound quote.
        component.required_details = [
            detail for detail in component.required_details
            if normalized(detail) and any(normalized(detail) in normalized(s.quote) for s in valid)
        ]
        if not valid:
            if component.status != "missing":
                component.gap = f"A source passage could not be bound reliably for: {component.requirement.rstrip('.?')}."
            component.status = "missing"
        elif lost_evidence and component.status == "supported":
            component.status = "partial"
        if component.status != "supported" and not component.gap.strip():
            component.gap = f"The retrieved evidence does not establish: {component.requirement.rstrip('.?')}."
        if component.status == "supported":
            component.gap = ""
        component.answer = ""
        components.append(component.model_dump())
    if not components:
        components = [AnswerComponent(
            id=f"C{i + 1}", requirement=requirement,
            gap=f"No source-bound evidence was identified for: {requirement.rstrip('.?')}.",
        ).model_dump() for i, requirement in enumerate(requirements)]
    # Preserve plainly stated cohort counts even when the outline model forgets
    # its population field. This selects a verbatim methods sentence, not a
    # number inferred from a percentage or a benchmark-specific sample size.
    explicit_populations = [
        {**span, "required_details": [span["quote"]]}
        for span in spans.values()
        if _has_population_count(span["quote"])
        and re.search(r"\b(included|enrolled|recruited|analy[sz]ed|randomi[sz]ed|using|underwent|comprised)\b", normalized(span["quote"]))
    ]
    contexts = [*(study_context if isinstance(study_context, list) else []), *explicit_populations]
    # Study population belongs with the first supported result from that study,
    # not in an unbound global summary or repeated under every component.
    for value in contexts:
        if isinstance(value, dict) and "evidence_id" in value:
            value = {**value, **spans.get(value["evidence_id"], {})}
        try:
            span = EvidenceSpan.model_validate(value)
        except (ValidationError, TypeError):
            continue
        if not _has_population_count(span.quote):
            continue
        source = by_id.get(span.chunk_id)
        if not source or not normalized(span.quote) or normalized(span.quote) not in normalized(source.text):
            continue
        if span.citation and span.citation != source.citation:
            continue
        span.citation = source.citation
        for component in components:
            if component["status"] != "missing" and any(s["citation"] == source.citation for s in component["evidence"]):
                if span.model_dump() not in component["evidence"]:
                    component["evidence"].append(span.model_dump())
                for detail in value.get("required_details", []):
                    if isinstance(detail, str) and normalized(detail) and normalized(detail) in normalized(span.quote):
                        if detail not in component["required_details"]:
                            component["required_details"].append(detail)
                break
    return components


def outline_status(components: list[dict]) -> tuple[str, str]:
    if not components:
        return "insufficient", "No source-bound evidence was identified."
    gaps = list(dict.fromkeys(c["gap"] for c in components if c.get("gap")))
    if all(c["status"] == "supported" for c in components):
        return "complete", ""
    useful = any(c["status"] in {"supported", "partial"} and c["evidence"] for c in components)
    return ("partial" if useful else "insufficient"), " ".join(gaps)


def repair_gaps(components: list[dict], repairs: object, repair_ids: list[str]) -> list[dict]:
    """Allow targeted repair of the explanation, without inventing support."""
    by_id = {
        item.get("component_id"): str(item.get("gap", "")).strip()
        for item in repairs if isinstance(item, dict)
    } if isinstance(repairs, list) else {}
    result = []
    for component in components:
        component = dict(component)
        gap = by_id.get(component["id"], "")
        if (component["id"] in repair_ids and component["status"] != "supported" and gap
                and not re.search(r"\b(answer|component|audit|must)\b", gap, re.I)):
            component["gap"] = gap
        result.append(component)
    return result


def bind_claims(claims: list[dict], components: list[dict]) -> tuple[list[dict], list[str]]:
    """Reject claims whose citations do not belong to their requested component."""
    by_id = {c["id"]: c for c in components}
    accepted, issues = [], []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        component = by_id.get(claim.get("component_id"))
        if component and component["status"] == "missing":
            # Missing components are rendered as explicit gaps, never as adjacent
            # result claims. Discarding such a claim needs no regeneration loop.
            continue
        allowed = {s["citation"] for s in component["evidence"]} if component else set()
        citations = claim.get("cite", [])
        if (not component or component["status"] == "missing" or not citations
                or not isinstance(citations, list) or any(c not in allowed for c in citations)):
            issues.append(f"A claim has no matching component/source binding: {claim.get('text', '')}")
            continue
        accepted.append(claim)
    for component in components:
        if component["status"] != "missing" and component["evidence"]:
            if not any(c.get("component_id") == component["id"] for c in accepted):
                issues.append(f"{component['id']}: answer omitted {component['requirement']}")
    return accepted, issues


def missing_numeric_details(components: list[dict], claims: list[dict]) -> dict[str, list[str]]:
    """Catch omitted bound numbers; semantic support still needs source review."""
    def numbers(text: str) -> set[Decimal]:
        text = normalized(text)
        text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
        return {Decimal(n) for n in re.findall(r"(?<![\w.])(?:\d*\.\d+|\d+)(?!\w)", text)}

    missing = {}
    for component in components:
        if component["status"] == "missing":
            continue
        rendered = " ".join(c.get("text", "") for c in claims if c.get("component_id") == component["id"])
        absent = [d for d in component["required_details"] if numbers(d) - numbers(rendered)]
        if absent:
            missing[component["id"]] = absent
    return missing


def restore_numeric_quotes(components: list[dict], claims: list[dict]) -> list[dict]:
    """Restore omitted bound numbers without another generative call.

    Prefer a full source sentence over a disconnected number or an invented
    paraphrase. This does not infer an outcome or repair semantic contradictions.
    """
    missing = missing_numeric_details(components, claims)
    result = list(claims)
    for component in components:
        used = set()
        for detail in missing.get(component["id"], []):
            span = next((s for s in component["evidence"] if normalized(detail) in normalized(s["quote"])), None)
            if span and span["quote"] not in used:
                result.append({"component_id": component["id"],
                               "text": f'The study reports: "{span["quote"]}"',
                               "cite": [span["citation"]]})
                used.add(span["quote"])
    return result
