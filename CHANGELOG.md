# Changelog

## 0.4.0 repair — 2026-09-22 (unpublished development candidate)

- Preserve full user-derived subquestions before retrieval; match each study before final chunk selection and review its evidence separately.
- Keep development/test roles, actors, null contrasts and method steps in complete source sentences. Preserve cited design explanations separately.
- Separate missing outcome data from missing comparative data. Keep refusal wording tied to the requested outcome.
- Require a structured check for every component and avoid reasoning-only output exhaustion.
- Save two independent focus runs (6/6 and 6/6) and a separate full development run (15/15), with 0 unsupported additions and 0 execution errors in the full run. Preserve all earlier failures.
- Add `recompute_saved_agent.py --version v0.4-repaired`; older version selections retain their meaning.

See [the repair report](docs/agent-v0.4-repaired-report.md). No stable tag, push or held-out test evaluation is implied.

## 0.4.0 — 2026-09-22 (unpublished development candidate)

- Bind each answer component to its requested study, exact source quotation, required details and explicit evidence gap.
- Match study identity before drafting, preserve supported components during targeted repair, and restore omitted numeric details as attributed source quotations.
- Show complete, partial and insufficient evidence coverage with expandable quotations and source navigation. Guided examples remain labelled fixtures.
- Normalize Ollama wildcard listening addresses for Windows client connections; exercise a real partial answer after installing the locked demo in a fresh directory.
- Preserve two independent six-question repetitions (5/6 and 4/6 strict passes) and a separate full development run (9/15, 1 execution error). Publish every answer, source-first decision, log and timing, including failures and the user-requested pause/resume.
- Add `recompute_saved_agent.py --version v0.4`; the existing v0.3 default remains compatible.
- **Quality targets not met:** 3 unsupported or incorrectly scoped additions, 2 missing required qualifiers and only 12/13 answerable questions with all evidence retrieved. No overall answer-quality improvement is claimed. The 35-question test split remains unused.

Implementation and demonstration are available in the candidate branch; no stable v0.4 tag or GitHub publication is implied.

## 0.3.0 — 2026-09-22 (local showcase candidate)

- Plan component searches and retain evidence across requested sources; constrain single-study answers to the leading source.
- Preserve the original question, exact statistical qualifiers and explicitly requested toxicity outcomes throughout answer generation.
- Separate support, completeness and evidence-boundary checks; retry malformed JSON once in place and retain unresolved failures.
- Correct HTML cleanup that could remove statistical inequalities from evidence details.
- Publish a complete 15-question development run, source-first answer decisions and a model-free metric recalculation command. The 35-question test split remains unused.
- Document the current Agent workflow and add a walkthrough of measured answers beside the existing guided UI demo.

## 0.2.0 — 2026-09-21 (local showcase candidate)

- Added the source-disjoint VeritasMed v1.1 benchmark: 50 questions, 44 sources, 20 domains, exact evidence spans, explicit answerability boundaries and zero source or evidence-chunk overlap between development and test.
- Rebuilt the candidate pool with `qwen3.5:9b`, challenged it with `medgemma1.5:4b` and `llama3.1:8b`, and preserved every model finding, output hash and source-first adjudication.
- Ran the production LangGraph Agent on the 15-question development split with local `qwen3.5:9b`, BGE-M3 retrieval and CUDA reranking. The untouched 35-question test split remains reserved for a later declared milestone.
- Recorded 88.5% required-claim Recall@5, 90.0% claim completeness, 100.0% claim-support precision and 5/15 strict answer passes, with per-question failure analysis and curator overrides.
- Changed the active Ollama default from `qwen3:8b` to `qwen3.5:9b`; disabled hidden reasoning for bounded structured responses and restored the full JSON contract during answer regeneration.
- Added production-Agent benchmark runners, deterministic scoring, frozen manifests, an inspectable benchmark report and v0.2.0 release documentation.

No remote GitHub Release or Git tag is implied by this local candidate.

## 0.1.0 — 2026-09-18 (local showcase candidate)

- Added a labelled browser-only guided demo and an isolated live retrieval fixture, with source provenance and desktop/mobile screenshots.
- Added a Python 3.12 CPU dependency lock, clean-environment checks, offline CI and explicit optional Docker/Ollama configurations.
- Corrected WebSocket node starts, terminal error handling, deadlines, cancellation and per-request state isolation.
- Repaired malformed citation handling, backend health reporting and Ollama host selection.
- Removed the ineffective Ask pipeline selector; retained real P2/P3 selection in Explore. Repaired stale stream callbacks, repeated node updates, citation numbering and answer toolbar actions.
- Recomputed historical evaluation summaries with explicit error denominators, corrected Hit@K terminology and immutable input hashes.
- Replaced conflicting release claims with a reproducible quick-start, limitations and validation record.

No new full model benchmark was run. Live model completion, Docker execution and non-Windows execution are not signed off. No remote release or version tag is implied by this entry.
