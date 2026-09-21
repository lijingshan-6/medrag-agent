"""Create eighty source-first benchmark candidates with local Qwen wording."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from medrag.benchmark.inventory import build_normalized_chunks
from medrag.benchmark.review import ollama_json
from medrag.benchmark.schema import (
    Answerability,
    BenchmarkQuestion,
    BenchmarkSplit,
    ClaimImportance,
    EvidenceSpan,
    GoldClaim,
    QuestionReview,
    QuestionRubric,
    ReviewStatus,
    SourceRecord,
    SupportLevel,
    TaskType,
)

DOMAINS: dict[str, tuple[str, ...]] = {
    "cardiology": ("heart", "cardiac", "cardiovascular", "coronary", "atrial", "hypertension"),
    "oncology": ("cancer", "tumor", "carcinoma", "oncology", "neoplasm", "leukemia"),
    "neurology": ("brain", "stroke", "neurolog", "dementia", "parkinson", "epilep"),
    "endocrinology": ("diabetes", "thyroid", "adrenal", "hormone", "metabolic", "insulin"),
    "infectious_disease": ("infection", "infectious", "virus", "bacterial", "antimicrobial", "covid"),
    "nephrology": ("kidney", "renal", "dialysis", "nephro"),
    "gastroenterology": ("gastro", "intestinal", "liver", "hepatic", "pancrea", "bowel"),
    "pulmonology": ("lung", "pulmonary", "asthma", "respiratory", "copd"),
    "womens_health": ("pregnan", "women", "ovarian", "uter", "breast", "gynec"),
    "musculoskeletal": ("arthritis", "bone", "muscle", "orthop", "rheumat", "joint"),
}


class GeneratedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot: int = Field(ge=1, le=8)
    question: str = Field(min_length=15)
    claims: list[str]
    difficulty_reason: str = Field(min_length=10)
    missing_evidence: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)


class GeneratedBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[GeneratedCandidate] = Field(min_length=1, max_length=2)


@dataclass(frozen=True)
class Evidence:
    key: str
    chunk_id: str
    quote: str
    record: dict[str, Any]


@dataclass(frozen=True)
class Slot:
    task_type: TaskType
    answerability: Answerability
    evidence: tuple[Evidence, ...]
    negative_evidence: tuple[Evidence, ...]
    instruction: str

    @property
    def negative_chunks(self) -> tuple[str, ...]:
        return tuple(item.chunk_id for item in self.negative_evidence)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if 12 <= len(sentence.split()) <= 90
    ]


def _sentence_score(sentence: str) -> tuple[int, int]:
    lower = sentence.casefold()
    signals = (
        "result", "found", "associated", "compared", "significant", "conclusion",
        "suggest", "risk", "improved", "reduced", "increased", "sensitivity",
        "specificity", "patients", "participants", "limitations", "limited",
    )
    score = sum(term in lower for term in signals) + 2 * bool(re.search(r"\d", sentence))
    return score, -abs(len(sentence.split()) - 35)


def _best_sentences(record: dict[str, Any], count: int = 3) -> list[str]:
    sentences = _sentences(str(record.get("abstract") or ""))
    ranked = sorted(enumerate(sentences), key=lambda item: _sentence_score(item[1]), reverse=True)
    chosen = [sentence for _, sentence in ranked[:count]]
    if len(chosen) < count:
        raise ValueError(f"PMID {record.get('pmid')} lacks enough usable evidence sentences")
    return chosen


def _search_text(record: dict[str, Any]) -> str:
    return " ".join(
        [
            str(record.get("title") or ""),
            str(record.get("abstract") or ""),
            " ".join(str(term) for term in record.get("mesh_terms", [])),
        ]
    ).casefold()


def _topic_tokens(record: dict[str, Any]) -> set[str]:
    stop = {
        "humans", "male", "female", "adult", "aged", "study", "patients", "using",
        "effect", "analysis", "clinical", "results", "treatment", "disease",
    }
    mesh = {str(term).casefold() for term in record.get("mesh_terms", [])}
    title = set(re.findall(r"[a-z]{5,}", str(record.get("title") or "").casefold()))
    return {token for token in mesh | title if token not in stop}


def _select_documents(
    records: list[dict[str, Any]], keywords: tuple[str, ...]
) -> list[dict[str, Any]]:
    pool = [
        record
        for record in records
        if len(str(record.get("abstract") or "").split()) >= 120
        and not any(
            str(kind).casefold() == "retracted publication"
            for kind in record.get("pub_types", [])
        )
        and any(keyword in _search_text(record) for keyword in keywords)
        and len(_sentences(str(record.get("abstract") or ""))) >= 3
    ]
    if len(pool) < 6:
        raise ValueError(f"domain has only {len(pool)} usable documents")

    best_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_score = -1.0
    for left_index, left in enumerate(pool[:160]):
        left_tokens = _topic_tokens(left)
        for right in pool[left_index + 1 : 160]:
            right_tokens = _topic_tokens(right)
            shared = left_tokens & right_tokens
            score = len(shared) + len(shared) / max(len(left_tokens | right_tokens), 1)
            if score > best_score:
                best_score = score
                best_pair = (left, right)
    assert best_pair is not None
    selected = list(best_pair)
    used = {str(record["pmid"]) for record in selected}
    ranked_rest = sorted(
        (record for record in pool if str(record["pmid"]) not in used),
        key=lambda record: _sentence_score(_best_sentences(record, 1)[0]),
        reverse=True,
    )
    selected.extend(ranked_rest[:4])
    return selected


def _infer_study_design(record: dict[str, Any]) -> str:
    text = _search_text(record)
    patterns = [
        ("meta-analysis", "systematic review and meta-analysis"),
        ("systematic review", "systematic review"),
        ("randomized", "randomized controlled trial"),
        ("randomised", "randomized controlled trial"),
        ("case-control", "case-control study"),
        ("cross-sectional", "cross-sectional study"),
        ("cohort", "cohort study"),
        ("case report", "case report"),
        ("observational", "observational study"),
    ]
    for needle, label in patterns:
        if needle in text:
            return label
    types = [str(value) for value in record.get("pub_types", [])]
    return ", ".join(types) if types else "study design not stated in source metadata"


def _source(record: dict[str, Any], *, role: str) -> SourceRecord:
    limitations = [
        sentence
        for sentence in _sentences(str(record.get("abstract") or ""))
        if any(term in sentence.casefold() for term in ("limitation", "single-center", "small sample"))
    ][:2]
    publication_types = [str(value) for value in record.get("pub_types", [])]
    return SourceRecord(
        source_id=f"PMID:{record['pmid']}",
        pmid=str(record["pmid"]),
        title=str(record["title"]),
        year=record.get("year"),
        publication_type=publication_types[0] if publication_types else "not stated",
        study_design=_infer_study_design(record),
        role=role,
        limitations=limitations,
    )


def _make_evidence(record: dict[str, Any], sentence_index: int, key: str) -> Evidence:
    return Evidence(
        key=key,
        chunk_id=f"pubmed:{record['pmid']}:0",
        quote=_best_sentences(record)[sentence_index],
        record=record,
    )


def _slots(documents: list[dict[str, Any]]) -> list[Slot]:
    ev = {
        f"D{doc_index + 1}E{sentence_index + 1}": _make_evidence(
            record, sentence_index, f"D{doc_index + 1}E{sentence_index + 1}"
        )
        for doc_index, record in enumerate(documents)
        for sentence_index in range(3)
    }
    return [
        Slot(TaskType.SINGLE_EVIDENCE, Answerability.COMPLETE, (ev["D3E1"],), (ev["D4E3"],), "Ask for the precise finding in the passage."),
        Slot(TaskType.WITHIN_DOCUMENT_SYNTHESIS, Answerability.COMPLETE, (ev["D4E1"], ev["D4E2"]), (ev["D3E3"],), "Combine two complementary findings from the same study."),
        Slot(TaskType.WITHIN_DOCUMENT_SYNTHESIS, Answerability.COMPLETE, (ev["D5E1"], ev["D5E2"]), (ev["D6E3"],), "Combine the population or method with the reported outcome."),
        Slot(TaskType.CROSS_DOCUMENT_COMPARISON, Answerability.COMPLETE, (ev["D1E1"], ev["D2E1"]), (ev["D3E3"],), "Compare the related studies without claiming they are a head-to-head trial."),
        Slot(TaskType.CROSS_DOCUMENT_COMPARISON, Answerability.COMPLETE, (ev["D1E2"], ev["D2E2"]), (ev["D4E3"],), "Explain how the related studies differ in outcome, population, or interpretation."),
        Slot(TaskType.LIMITATIONS_AND_SAFETY, Answerability.COMPLETE, (ev["D6E1"], ev["D6E2"]), (ev["D5E3"],), "Ask what can responsibly be concluded and which qualifier limits interpretation."),
        Slot(TaskType.INSUFFICIENT_EVIDENCE, Answerability.PARTIAL, (ev["D3E2"],), (ev["D1E3"],), "Ask a two-part question: one part is supported, while long-term outcomes or a broader population are absent and must be named as missing."),
        Slot(TaskType.INSUFFICIENT_EVIDENCE, Answerability.UNANSWERABLE, (), (ev["D1E1"], ev["D2E1"]), "Ask for a direct head-to-head clinical recommendation that these related passages do not provide; the responsible answer must abstain."),
    ]


def _prompt(domain: str, slots: list[tuple[int, Slot]]) -> str:
    unique: dict[str, Evidence] = {}
    for _, slot in slots:
        for evidence in slot.evidence:
            unique[evidence.key] = evidence
    evidence_text = "\n\n".join(
        f"[{item.key}] PMID {item.record['pmid']} — {item.record['title']}\n{item.quote}"
        for item in unique.values()
    )
    assignments = "\n".join(
        f"Slot {index}: task={slot.task_type.value}; answerability={slot.answerability.value}; "
        f"evidence={','.join(item.key for item in slot.evidence) or 'none'}; {slot.instruction}"
        for index, slot in slots
    )
    return f"""You are drafting source-grounded evaluation questions for a medical literature RAG system.
Write in clear English for a medically literate user. Return exactly one item for each listed slot.
Use only the assigned evidence. Do not add clinical knowledge, causal language, or numerical values
absent from the passages. Each supported evidence passage requires exactly one atomic claim in the
same order as the evidence keys. Claims should paraphrase rather than copy the quote. For slot 8,
return zero claims. Partial and unanswerable items must name the requested evidence that is absent.
Questions must sound like plausible literature-search questions, not benchmark instructions.

Medical domain: {domain}

{assignments}

Evidence passages:
{evidence_text}
"""


def _build_question(
    question_id: str,
    domain: str,
    slot: Slot,
    generated: GeneratedCandidate,
) -> BenchmarkQuestion:
    if generated.slot < 1 or generated.slot > 8:
        raise ValueError(f"invalid slot {generated.slot}")
    if slot.evidence and not generated.claims:
        raise ValueError(
            f"slot {generated.slot}: supported candidates require at least one claim"
        )
    supported_claims = generated.claims[: len(slot.evidence)]
    unsupported_claims = generated.claims[len(slot.evidence) :]
    missing = generated.missing_evidence
    if slot.answerability is not Answerability.COMPLETE and not missing:
        missing = [
            (
                "Long-term outcomes or evidence in the broader population requested by the question are absent from the selected passages."
                if slot.answerability is Answerability.PARTIAL
                else "No direct head-to-head evidence or clinical recommendation matching the question is present in the selected passages."
            )
        ]

    evidence_groups = (
        [slot.evidence]
        if len(supported_claims) == 1
        else [(evidence,) for evidence in slot.evidence]
    )
    claims = []
    for index, (claim, evidence_group) in enumerate(
        zip(supported_claims, evidence_groups, strict=True), start=1
    ):
        claims.append(
            GoldClaim(
                claim_id=f"C{index}",
                text=claim,
                importance=ClaimImportance.REQUIRED,
                evidence=[
                    EvidenceSpan(
                        chunk_id=evidence.chunk_id,
                        quote=evidence.quote,
                        support=SupportLevel.DIRECT,
                    )
                    for evidence in evidence_group
                ],
            )
        )
    evidence_records: list[dict[str, Any]] = []
    seen_pmids: set[str] = set()
    for evidence in slot.evidence:
        pmid = str(evidence.record["pmid"])
        if pmid not in seen_pmids:
            seen_pmids.add(pmid)
            evidence_records.append(evidence.record)
    if not evidence_records:
        # For abstention items, near-topic sources are hard negatives rather than evidence.
        evidence_records = [item.record for item in slot.negative_evidence][:2]

    return BenchmarkQuestion(
        id=question_id,
        version="1.0.0",
        split=BenchmarkSplit.DEVELOPMENT,
        task_type=slot.task_type,
        answerability=slot.answerability,
        category=domain,
        difficulty_reason=generated.difficulty_reason,
        question=generated.question,
        gold_claims=claims,
        sources=[
            _source(record, role="hard_negative" if not slot.evidence else "primary_study")
            for record in evidence_records
        ],
        supporting_chunk_ids=list(dict.fromkeys(item.chunk_id for item in slot.evidence)),
        hard_negative_chunk_ids=list(dict.fromkeys(slot.negative_chunks)),
        forbidden_claims=list(dict.fromkeys(generated.forbidden_claims + unsupported_claims)),
        missing_evidence=missing,
        rubric=QuestionRubric(
            complete_if=(
                [f"All required claims C1-C{len(claims)} are present and correctly qualified."]
                if claims
                else ["The answer clearly states that the requested comparison is unsupported."]
            ),
            partial_if=["A supported point is given but a required qualifier or evidence boundary is omitted."],
            fail_if=["The answer introduces an unsupported clinical conclusion or recommendation."],
        ),
        review=QuestionReview(status=ReviewStatus.DRAFT),
    )
def build(root: Path, output_path: Path, model: str) -> list[BenchmarkQuestion]:
    pubmed_path = root / "data/raw/pubmed/abstracts.jsonl"
    pmc_path = root / "data/raw/pmc/full_texts.jsonl"
    records = _read_jsonl(pubmed_path)
    normalized_chunks = build_normalized_chunks(pubmed_path, pmc_path)
    available_chunks = {row["chunk_id"] for row in normalized_chunks}
    questions_by_id: dict[str, BenchmarkQuestion] = {}
    if output_path.exists():
        for row in _read_jsonl(output_path):
            question = BenchmarkQuestion.model_validate(row)
            questions_by_id[question.id] = question

    def save_checkpoint() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for question in sorted(questions_by_id.values(), key=lambda item: item.id):
                handle.write(question.model_dump_json() + "\n")

    for domain_index, (domain, keywords) in enumerate(DOMAINS.items()):
        documents = _select_documents(records, keywords)
        slots = _slots(documents)
        pending = [
            (slot_index, slot)
            for slot_index, slot in enumerate(slots, start=1)
            if f"VMG-{domain_index * 8 + slot_index:03d}" not in questions_by_id
        ]
        if not pending:
            print(f"[{domain}] 8/8 candidates already present", flush=True)
            continue
        for pair_start in range(len(pending)):
            pair = pending[pair_start : pair_start + 1]
            print(
                f"[{domain}] drafting slots {','.join(str(index) for index, _ in pair)}",
                flush=True,
            )
            result = ollama_json(
                model=model,
                prompt=_prompt(domain, pair),
                response_model=GeneratedBatch,
                timeout_seconds=300,
            )
            generated_by_slot = {
                item.slot: item for item in GeneratedBatch.model_validate(result.value).items
            }
            expected_slots = {index for index, _ in pair}
            if not expected_slots.issubset(generated_by_slot):
                raise ValueError(
                    f"{domain}: expected slots {sorted(expected_slots)}, "
                    f"got {sorted(generated_by_slot)}"
                )
            for slot_index, slot in pair:
                if any(chunk_id not in available_chunks for chunk_id in slot.negative_chunks):
                    raise ValueError(f"{domain} slot {slot_index}: missing hard-negative chunk")
                number = domain_index * 8 + slot_index
                question = _build_question(
                    f"VMG-{number:03d}", domain, slot, generated_by_slot[slot_index]
                )
                questions_by_id[question.id] = question
            save_checkpoint()
        print(f"[{domain}] complete", flush=True)

    questions = sorted(questions_by_id.values(), key=lambda item: item.id)
    save_checkpoint()
    return questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1/candidates.jsonl"),
    )
    args = parser.parse_args()
    questions = build(args.root.resolve(), args.output, args.model)
    print(f"wrote {len(questions)} source-first candidates to {args.output}")


if __name__ == "__main__":
    main()
