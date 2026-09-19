# VeritasMed Evidence-Grounded Benchmark v1

This is a 50-question engineering benchmark for evidence-grounded medical-literature RAG. It is designed to test whether an agent retrieves relevant passages, reconstructs supported claims, preserves study limits, cites evidence, and abstains when the supplied literature cannot answer a question.

It is **not clinician-reviewed**, is not a medical exam, and must not be used to make patient-care decisions.

## What is frozen

- Corpus snapshot: `6dd1f367df9b3653`
- Question file SHA-256: `dcdfb214f03dbf421e00a515278d7e3e476068728d009c10972e79ba353875c1`
- Split: 15 development questions and 35 holdout questions
- Tasks: {'single_evidence': 10, 'within_document_synthesis': 15, 'cross_document_comparison': 10, 'limitations_and_safety': 10, 'insufficient_evidence': 5}
- Answerability: {'complete': 40, 'partial': 5, 'unanswerable': 5}
- Domains: cardiac_surgery (1), cardio_infectious (2), cardiopulmonary (1), cardiovascular_imaging (2), clinical_ai (3), critical_care (1), electrophysiology (1), endocrinology (3), heart_failure (3), hepatology (3), infectious_disease (2), interventional_cardiology (1), musculoskeletal (4), nephrology (2), neurology (2), oncology (5), ophthalmology (1), pulmonology (3), radiology (3), womens_health (7)

Candidate IDs are preserved in `curation_decisions.jsonl`; final IDs are contiguous and stable. The append-only `review_log.jsonl` preserves the Qwen reconstruction, MedGemma challenge, and curator decision for every candidate.

## How the questions were made

Evidence was selected before question wording. Eighty candidates were drafted from exact PubMed passages. `qwen3:8b` independently reconstructed answers and `medgemma1.5:4b` challenged population, intervention/exposure, comparator, outcome, time point, and uncertainty. Model agreement never accepted an item automatically.

The curator then read all 80 question-claim-evidence bundles, rejected 30, and manually rewrote the selected 50. Required claims use verbatim evidence spans. A `derived` span is permitted only when it includes the explicit reasoning rule, such as why a cross-sectional association cannot establish an intervention effect.

## Composition

- 10 single-evidence questions
- 15 within-document synthesis questions
- 10 cross-document comparison questions
- 10 limitations and safety questions, including five partially answerable cases
- 5 insufficient-evidence questions that require abstention

The five abstention cases use same-topic hard negatives. They ask for an intervention effect or clinical outcome that the supplied observational, diagnostic, or single-arm evidence does not provide.

## Scoring intent

Score retrieval, claim completeness, claim support, citation coverage, qualification preservation, and answerability separately. A high-quality answer may be a refusal when the evidence does not contain the requested comparison. Do not award credit for medically plausible facts that are absent from the frozen passages.

## Known limits

- The corpus is a local snapshot and is not a systematic review for any clinical question.
- Most questions use PubMed abstracts rather than full articles, so the gold answer is bounded to the frozen passage.
- The dataset was curated by an engineering reviewer with local-model assistance, without clinician adjudication.
- Some 2026 records describe small, retrospective, preclinical, or technical studies; questions explicitly preserve those limits.
- Public repository users can inspect holdout labels. The holdout split prevents development leakage inside this project, not determined leaderboard gaming.

## Files

- `questions.jsonl`: final 50 questions
- `curation_decisions.jsonl`: final-to-candidate mapping and rejection reasons
- `review_log.jsonl`: append-only machine and curator review history
- `manifest.json`: frozen input hashes and corpus identity policy
- `model_manifest.json`: local-review model provenance
