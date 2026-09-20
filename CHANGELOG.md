# Changelog

## Unreleased — evidence-grounded benchmark v1

- Added a manually curated 50-question benchmark over a frozen 44,768-chunk corpus snapshot, with 44 unique sources, 20 domains, exact claim-level evidence and explicit answerability boundaries.
- Preserved the 80-item candidate pool, local Qwen/MedGemma review trail, 30 rejection decisions and final curator adjudications as inspectable artifacts.
- Added deterministic retrieval and answer scoring, full-set P1/P2 baselines, a development-set P3 baseline and a 15-question local answer baseline with visible curator corrections.
- Documented the dataset contract, reproduction commands, representative successes and failures, and current limitations in the benchmark report and README.

## 0.1.0 — 2026-09-18 (local showcase candidate)

- Added a labelled browser-only guided demo and an isolated live retrieval fixture, with source provenance and desktop/mobile screenshots.
- Added a Python 3.12 CPU dependency lock, clean-environment checks, offline CI and explicit optional Docker/Ollama configurations.
- Corrected WebSocket node starts, terminal error handling, deadlines, cancellation and per-request state isolation.
- Repaired malformed citation handling, backend health reporting and Ollama host selection.
- Removed the ineffective Ask pipeline selector; retained real P2/P3 selection in Explore. Repaired stale stream callbacks, repeated node updates, citation numbering and answer toolbar actions.
- Recomputed historical evaluation summaries with explicit error denominators, corrected Hit@K terminology and immutable input hashes.
- Replaced conflicting release claims with a reproducible quick-start, limitations and validation record.

No new full model benchmark was run. Live model completion, Docker execution and non-Windows execution are not signed off. No remote release or version tag is implied by this entry.
