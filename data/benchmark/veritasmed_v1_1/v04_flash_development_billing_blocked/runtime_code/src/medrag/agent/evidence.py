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
    question_span: str = ""
    source_hint: str = ""
    status: Literal["supported", "partial", "missing"] = "missing"
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    required_details: list[str] = Field(default_factory=list)
    missing_outcome: str = ""
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


_COHORT_ROLES = {
    "development": r"\b(develop\w*|train\w*)\b",
    "calibration": r"\bcalibrat\w*\b",
    "validation": r"\bvalidat\w*\b",
    "test": r"\btest(?:ed|ing)?\b",
}
_METHOD_STEPS = re.compile(r"\b(combin(?:ed|es)|incorporat(?:e|ed|es)|us(?:e|ed|es)|calculated|optimi[sz]ed|applied)\b")


def _population_counts(text: str) -> set[str]:
    return {m.replace(",", "") for m in re.findall(
        r"\b(\d[\d,]*)\s+(?:[\w-]+\s+){0,3}(?:patients?|participants?|controls?|veterans?|subjects?|women|men|children|volunteers?|births|images|examinations|animals?|mice|rats)\b",
        normalized(text),
    )}


def population_scope_issue(text: str, component: dict) -> bool:
    """A cohort count cannot lose its explicit development/test/etc. role.

    This is a narrow guard for plainly labelled populations, not an entailment
    score. Ambiguous sentences with several roles still need semantic review.
    """
    if text.startswith('The study reports: "') and text.endswith('"'):
        quoted = normalized(text[len('The study reports: "'):-1])
        if any(quoted == normalized(s["quote"]) for s in component["evidence"]):
            return False  # An exact sentence has not paraphrased a cohort's role.
    counts = _population_counts(text)
    cohort_alias = r"\b(?:same|entire|whole|full|separate|independent|external|distinct|held[ -]out)\b[^.!?]{0,45}\b(?:data(?:set)?|set|cohort|population|patients|images)\b"
    # An anaphor can reassign a development population without repeating its
    # count. Neither shared nor independent validation data can be inferred
    # just from 'multicenter validation'.
    if (re.search(r"\b(?:validat\w*|test(?:ed|ing)?)\b", normalized(text))
            and re.search(cohort_alias, normalized(text))
            and not any(re.search(cohort_alias, normalized(s["quote"]))
                        and re.search(r"\b(?:validat\w*|test(?:ed|ing)?)\b", normalized(s["quote"]))
                        for s in component["evidence"])):
        return True
    for span in component["evidence"]:
        quote = normalized(span["quote"])
        if not counts.intersection(_population_counts(quote)):
            continue
        roles = [role for role, pattern in _COHORT_ROLES.items() if re.search(pattern, quote)]
        if len(roles) == 1 and not re.search(_COHORT_ROLES[roles[0]], normalized(text)):
            return True
    return False


def _bounded_gap(requirement: str, basis: str = "") -> str:
    # The missing component, not a free-form story about the underlying study,
    # names the evidence gap. In particular no invented design/data rationale.
    subject = re.sub(r"^(?:report\s+)?whether\s+|^report\s+", "", requirement, flags=re.I).rstrip('.?')
    statement = re.sub(r"^(?:does|do|did|can|could|has|have)\s+.+?\s+(?:establish|show|demonstrate|prove|confirm|support)\s+(?:that\s+)?", "", subject, flags=re.I)
    if statement != subject:
        if basis == "comparison":
            return f"The retrieved evidence does not provide the comparative data needed to establish that {statement}."
        if basis == "outcome":
            return f"The retrieved evidence does not provide outcome data establishing that {statement}."
        return f"The retrieved evidence does not establish that {statement}."
    if basis == "outcome":
        return f"The retrieved evidence does not provide outcome data establishing: {subject}."
    if basis == "comparison":
        return f"The retrieved evidence does not provide the required comparative data for: {subject}."
    return f"The retrieved evidence does not establish: {subject}."


def component_gap(component: dict, missing_outcome: str = "", basis: str = "") -> str:
    """Name only the absent part; a partial answer is not a blanket refusal."""
    phrase = missing_outcome.strip() if isinstance(missing_outcome, str) else ""
    safe_phrase = (5 <= len(phrase) <= 300 and not re.search(
        r"\b(because|due to|there were|there was)\b|^(?:no |identify |explain |what |which )", phrase, re.I))
    if basis:
        return _bounded_gap(component["requirement"], basis)
    if safe_phrase:
        return _bounded_gap(phrase)
    if component["status"] == "partial":
        return f"The retrieved passages only partly cover this requested aspect: {component['requirement'].rstrip('.?')}."
    return _bounded_gap(component["requirement"])


def _details(value: object) -> list[str]:
    """A single detail is one string, never an iterable of individual digits."""
    return [value] if isinstance(value, str) else [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def protocol_scope_issue(text: str, component: dict) -> bool:
    """Do not turn a design label into an unreported acquisition schedule.

    This narrow guard supplements semantic review, like the cohort-role guard.
    It does not decide whether a design or an inference is generally valid.
    """
    timing = r"\b(simultaneously|at the same time|at a single time point|at one time point)\b"
    if not re.search(timing, normalized(text)):
        return False
    evidence = " ".join(normalized(s["quote"]) for s in component["evidence"])
    return not re.search(timing + r"|\b(same visit|single visit|one visit)\b", evidence)


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
        missing_basis = value.get("missing_basis", "") if isinstance(value, dict) else ""
        missing_outcome = value.get("missing_outcome", "") if isinstance(value, dict) else ""
        if isinstance(value, dict):
            value = {**value, "required_details": _details(value.get("required_details"))}
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
        # A 'how does this method address the problem' component needs actual
        # method steps, not only the introductory purpose statement. Select
        # method sentences from the same already-bound source, never another study.
        method_requirement = re.search(r"\b(method\w*|technique\w*|reconstruction|filter\w*)\b", normalized(component.requirement))
        outcome_requirement = re.search(r"\b(result\w*|validation|performance)\b", normalized(component.requirement))
        if component.status != "missing" and method_requirement and not outcome_requirement:
            bound_chunks = {s.chunk_id for s in valid}
            for method_span in spans.values():
                if method_span["chunk_id"] in bound_chunks and _METHOD_STEPS.search(normalized(method_span["quote"])):
                    extra = EvidenceSpan.model_validate(method_span)
                    if extra not in valid:
                        valid.append(extra)
        component.evidence = valid
        # Required verbatim details must themselves be present in a bound quote.
        component.required_details = [
            detail for detail in component.required_details
            if normalized(detail) and any(normalized(detail) in normalized(s.quote) for s in valid)
        ]
        if value.get("required_details") and not component.required_details:
            # If proposed excerpts are paraphrases, keep the actual selected
            # source sentences as the contract instead of silently losing it.
            component.required_details = [span.quote for span in valid]
        for span in valid:
            # Preserve a whole selected contrast, including its null result.
            # A short positive half alone can reverse the meaning of a study.
            contrast = re.search(r"\b(whereas|while|but|versus)\b", normalized(span.quote))
            methods = (re.search(r"\b(how|method\w*|technique\w*|reconstruction|filter\w*)\b", normalized(component.requirement))
                       and _METHOD_STEPS.search(normalized(span.quote)))
            if (contrast or methods) and span.quote not in component.required_details:
                component.required_details.append(span.quote)
        if not valid:
            if component.status != "missing":
                component.gap = f"A source passage could not be bound reliably for: {component.requirement.rstrip('.?')}."
            component.status = "missing"
        elif lost_evidence and component.status == "supported":
            component.status = "partial"
        if component.status != "supported":
            component.gap = component_gap(component.model_dump(), missing_outcome, missing_basis)
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
                for detail in _details(value.get("required_details")):
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
    """Keep the question-bound gap; free prose cannot add an unrequested outcome.

    Kept as a compatibility helper for stored generation responses. Source
    review can revise a component's evidence status, but generator gap_repairs
    cannot silently change the question that the component is answering.
    """
    return [dict(component) for component in components]


def bind_claims(claims: list[dict], components: list[dict]) -> tuple[list[dict], list[str]]:
    """Reject claims whose citations do not belong to their requested component."""
    by_id = {c["id"]: c for c in components}
    accepted, issues = [], []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        text = str(claim.get("text", ""))
        if not text.startswith('The study reports: "'):
            text = re.sub(r"\bwe\b", "the authors", text, flags=re.I)
            text = re.sub(r"\bour\b", "their", text, flags=re.I)
            claim = {**claim, "text": text}
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
        if population_scope_issue(claim.get("text", ""), component):
            issues.append(f"{component['id']}: preserve the population's development/calibration/test/validation role. Report validation as stated without claiming shared, separate, independent or held-out data unless the source establishes that relationship.")
            continue
        if protocol_scope_issue(claim.get("text", ""), component):
            issues.append(f"{component['id']}: remove the unreported timing assertion, including 'at a single time point', 'at the same time' and 'simultaneously'. Keep the explanation that association does not establish the effect of an intervention, without describing when measurements occurred. Rejected draft: {claim.get('text', '')}")
            continue
        accepted.append(claim)
    for component in components:
        if component["status"] != "missing" and component["evidence"]:
            if not any(c.get("component_id") == component["id"] for c in accepted):
                issues.append(f"{component['id']}: answer omitted {component['requirement']}")
    return accepted, issues


def bind_additional_evidence(
    claims: list[dict], components: list[dict], chunks: list[RetrievedChunk],
) -> list[dict]:
    """Let generation recover a missed sentence, within the same bound study.

    Global sentence IDs resolve only to retrieved text. A generator cannot use
    them to upgrade a missing outcome or move another paper's result into this
    component. Semantic relevance remains the source reviewer's responsibility.
    """
    spans = source_spans(chunks)
    updated = [AnswerComponent.model_validate(c).model_dump() for c in components]
    by_id = {c["id"]: c for c in updated}
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        component = by_id.get(claim.get("component_id"))
        if not component or component["status"] == "missing":
            continue
        allowed = {s["citation"] for s in component["evidence"]}
        citations = claim.get("cite", [])
        ids = claim.get("evidence_ids", [])
        if not isinstance(ids, list) or not isinstance(citations, list):
            continue
        for key in ids:
            span = spans.get(key) if isinstance(key, str) else None
            if span and span["citation"] in allowed and span["citation"] in citations:
                if span not in component["evidence"]:
                    component["evidence"].append(dict(span))
                    if span["quote"] not in component["required_details"]:
                        component["required_details"].append(span["quote"])
    return updated


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
        citations = {span["citation"] for span in component["evidence"]}
        # A finding already explained under another component of the same study
        # need not be quoted twice. Other papers cannot satisfy this requirement.
        rendered = " ".join(c.get("text", "") for c in claims
                            if c.get("component_id") == component["id"]
                            or (c.get("cite") and set(c["cite"]) <= citations))
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
    for component in components:
        if component["status"] == "missing":
            continue
        rendered = normalized(" ".join(c.get("text", "") for c in claims if c.get("component_id") == component["id"]))
        for detail in component["required_details"]:
            # A vague 'used sampling' must not replace the actual method. Allow
            # concise paraphrases; quote only substantial lexical omissions.
            terms = set(re.findall(r"\b[a-z][a-z0-9-]{4,}\b", normalized(detail)))
            if (_METHOD_STEPS.search(normalized(detail)) and terms
                    and sum(term in rendered for term in terms) / len(terms) < 0.5):
                if detail not in missing.setdefault(component["id"], []):
                    missing[component["id"]].append(detail)
            # Preserve a selected contrast even if its second arm has no number.
            # This lexical fallback may quote a valid paraphrase unnecessarily;
            # it does not claim to establish semantic equivalence.
            contrast = re.search(r"\b(?:whereas|while|but)\b(.+)", normalized(detail))
            if contrast:
                tail_terms = set(re.findall(r"\b[a-z][a-z0-9-]{3,}\b", contrast[1])) - {
                    "with", "that", "this", "were", "remained", "substantially", "significantly",
                }
                if tail_terms and sum(t in rendered for t in tail_terms) / len(tail_terms) < 0.75:
                    if detail not in missing.setdefault(component["id"], []):
                        missing[component["id"]].append(detail)
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


def preserve_result_context(components: list[dict], claims: list[dict]) -> list[dict]:
    """Keep actors, denominators and qualifiers together in critical results.

    A paraphrase containing all the digits can still change whose performance
    was measured. For numerical/method components render the selected complete
    source sentences. Other components retain their generated explanation.
    This is extractive presentation, not an additional semantic judgment.
    """
    result = []
    for component in components:
        own = [c for c in claims if c.get("component_id") == component["id"]]
        if component["status"] == "missing":
            continue
        if re.search(r"\b(causal\w*|limitation\w*|why|prevent\w*|cannot)\b", normalized(component["requirement"])):
            # A design quotation supports an explanation but cannot replace it.
            # The semantic checker still reviews that generated explanation.
            result.extend(own)
            continue
        critical = any(re.search(r"\d", d) or _METHOD_STEPS.search(normalized(d))
                       for d in component["required_details"])
        if not critical:
            result.extend(own)
            continue
        selected = []
        for span in component["evidence"]:
            if any(normalized(d) in normalized(span["quote"]) for d in component["required_details"]):
                if span not in selected:
                    selected.append(span)
        if not selected:
            result.extend(own)
            continue
        # Preserve the source's text (including "we") inside explicit quotation
        # marks. The answer therefore does not impersonate the study authors.
        result.extend({"component_id": component["id"],
                       "text": f'The study reports: "{span["quote"]}"',
                       "cite": [span["citation"]]} for span in selected)
    return result
