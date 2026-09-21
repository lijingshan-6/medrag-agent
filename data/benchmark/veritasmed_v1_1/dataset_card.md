# VeritasMed Evidence-Grounded Benchmark v1.1

This is a source-disjoint engineering benchmark for evidence-grounded medical-literature RAG. It is not clinician-reviewed and must not be used as a clinical-safety claim.

## Why v1.1 exists

The v1.0 audit found source leakage across development and test, generic or contradictory study-design metadata, gold-contract defects, and unrelated hard negatives on answerable items. v1.1 preserves v1.0, repairs those defects, and assigns complete source-connected groups to one split.

## Frozen composition

- Corpus snapshot: `6dd1f367df9b3653`
- Question SHA-256: `d31c54cfe6847b53c72f96630f5047f90617dda87ab953c563a0e9246177056e`
- Questions: 50 (15 development, 35 test)
- Unique sources: 44
- Source overlap between splits: 0
- Evidence-chunk overlap between splits: 0
- Task types: {'cross_document_comparison': 10, 'insufficient_evidence': 5, 'limitations_and_safety': 10, 'single_evidence': 10, 'within_document_synthesis': 15}
- Answerability: {'complete': 40, 'partial': 5, 'unanswerable': 5}
- Domains: {'cardiac_surgery': 1, 'cardio_infectious': 2, 'cardiopulmonary': 1, 'cardiovascular_imaging': 2, 'clinical_ai': 3, 'critical_care': 1, 'electrophysiology': 1, 'endocrinology': 3, 'heart_failure': 3, 'hepatology': 3, 'infectious_disease': 2, 'interventional_cardiology': 1, 'musculoskeletal': 4, 'nephrology': 2, 'neurology': 2, 'oncology': 5, 'ophthalmology': 1, 'pulmonology': 3, 'radiology': 3, 'womens_health': 7}
- Source years: 2026 only

## Audit result

All 50 items received a source-first curator decision. 39 were retained and 11 gold contracts were revised. Exact evidence validation is run against all 44,768 normalized chunks. Answerable items no longer carry unrelated cross-domain hard negatives; the five abstention items retain same-topic, non-answering passages.

`qwen3.5:9b` generated a fresh 80-item candidate pool and independently re-audited the final 50. `llama3.1:8b` served only as an independent-family blind challenger. Their raw decisions and hashes are stored in the audit artifacts; source-first curator decisions in `quality_audit.jsonl` remain authoritative. `model_adjudications.jsonl` records how every model challenge was resolved.

## Intended use and limits

v1.1 is suitable for engineering iteration on this frozen corpus: retrieval coverage, evidence-bounded answering, citation support, and calibrated abstention. It is not representative of all medical literature because the local snapshot contains only 2026 records, most evidence is abstract-level, and no clinician independently adjudicated the labels. Public test labels deter casual development leakage but cannot provide a secret leaderboard.
