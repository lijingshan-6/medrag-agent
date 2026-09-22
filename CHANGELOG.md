# Changelog

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
