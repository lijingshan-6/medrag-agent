# Independent release review — v0.1.0

Initial review: 2026-09-18. Final source and artifact recheck: 2026-09-19. This review covers the current working tree; it does not attribute every uncommitted change to the release work. Source files were not modified by the reviewer.

## Final verdict

**Approved for the documented local research showcase candidate scope, with two non-blocking Minor observations. No open Critical or Important code findings.** The two checkpoint defects discovered in the initial review were fixed and covered by real-LangGraph regressions. Source behavior, deterministic historical metrics, release documentation and screenshot artifacts are consistent with the stated single-turn showcase boundary.

This approval does not certify successful live model generation, reliable P3 execution under the host's current memory pressure, Docker/Ollama execution, non-Windows execution, hosted CI or a public deployment. The release documentation explicitly retains those unverified or unresolved boundaries. It makes no new-benchmark or clinical-validation claim.

## Open Minor observations

1. `frontend/src/App.tsx:162` and `frontend/src/components/QueryInput.tsx:30`: the interface still says "Recent threads" and "Thread" without an in-app single-turn explanation. These labels do not restore answers or supply conversational memory. README and release notes explain this accurately; a visible note or "Recent labels" wording would reduce confusion.
2. `frontend/src/components/AnswerPanel.tsx`: Guided Copy places only `result.answer` on the clipboard, unlike Guided Download, which includes fixture provenance. Add the same short guided-example prefix when copying from Guided mode. The visible banner and downloaded artifact already disclose the mode; this is a provenance polish issue, not a false live-run claim in the application.

## Initial review history

The initial verdict requested changes for the following two defects. Both are now resolved; the line references below identify the originally reviewed implementation.

## Resolved findings

### Resolved P1 — A request could return another request's answer

Location: `src/medrag/api/routes/ask.py:159`, `src/medrag/api/routes/ask.py:273-276`.

All invocations with the same user-supplied `thread_id` share the checkpoint key. After its own stream finishes, Ask reads `get_state(config)`, which means the latest checkpoint for that shared thread, not the final state of the invocation that just finished. Another Ask invocation can write that checkpoint before the read. The UI keeps the same thread ID for successive questions; overlapping API clients or a stop/retry while a synchronous node is still finishing expose the shared state problem. The frontend's stale-socket guard cannot correct an answer already mixed up at the backend.

Reproduction: a real one-node LangGraph with `InMemorySaver`, two real TestClient WebSockets using `thread_id="shared"`, and a controlled scheduling pause before the first `get_state` read. After request B completes, request A resumes. Observed result: request A answer = `request B`; request B answer = `request B`. No external services or model calls were used.

Recommended fix: isolate each single-turn Ask invocation under a unique internal checkpoint ID, or capture this invocation's final state directly from its own stream and serialize/guard shared checkpoint writes. Keep the public thread label separately if useful. Add a regression that overlaps two requests with the same public thread ID and asserts both answers belong to their own queries. Do not merely rely on frontend cancellation.

### Resolved P2 — Prior query rewrites leaked into the next question

Location: `src/medrag/api/routes/ask.py:46`, `src/medrag/agent/state.py:29`, `src/medrag/api/routes/ask.py:286`.

The initial state attempts to reset `rewritten_queries` using an empty list, but `AgentState` declares an additive reducer. For an existing checkpoint, adding `[]` preserves all rewrites from earlier questions. A new answer can therefore report old search rewrites even when its own `iterations` is zero. `history` is also additive, so supplying `[]` does not create genuinely isolated single-turn state.

Reproduction: invoke a real `StateGraph(AgentState)` twice with `_build_initial_state` and the same checkpoint ID. First invocation emits `['old query rewrite']`; second emits no rewrite. Observed second result: `answer='second'`, `rewritten_queries=['old query rewrite']`, `iterations=0`.

Recommended fix: the per-request checkpoint isolation above resolves this naturally. If intentionally retaining a shared checkpoint, explicitly overwrite fields that must be reset, and add a two-question regression. Do not remove the additive reducer blindly; it is needed to accumulate rewrites within one invocation.

## Verification performed independently

- `python -m pytest -q tests/test_api.py tests/test_citations.py tests/test_demo.py tests/test_eval_report.py tests/test_llm_config.py`: **20 passed**, one upstream Starlette/AnyIO deprecation warning.
- `python scripts/report_release.py --check`: **passed**, checked deterministic JSON and Markdown against saved inputs.
- Inspected stream cancellation/deadline behavior, terminal-event exclusivity, real task starts, citation normalization, embedded Qdrant singleton use, demo stable IDs/collection isolation, backend dependency lock, CI, Docker contexts and loopback ports, frontend stream ownership, and historical metric denominators.
- Hand-derived report tests distinguish Hit@5, macro evidence Recall@5, MRR, successful-only scores and failure-inclusive scores. Reports explicitly avoid claiming a fresh benchmark or Agent superiority.
- Readiness documentation explicitly says a successful backend probe does not prove model availability or end-to-end generation; this is an acknowledged limit, not an undisclosed readiness guarantee.

## Unverified / release boundaries

- No paid model or benchmark calls were made. The coordinator reported MiMo authentication failure (401), so live generation remains unverified in this environment.
- Docker daemon was unavailable to the coordinator. Docker configuration was reviewed statically; image builds, container startup, and Ollama generation were not independently run here.
- A guided fixed-example mode is acceptable only with its visible preset/no-live-call labels retained. It must not be counted as successful live RAG validation or benchmark evidence.
- The reviewer independently reran the full offline suite and frontend gates after the fix, as recorded below. Interactive browser operation and real-model retrieval measurements were performed by the coordinator; the reviewer inspected their documented boundaries and the resulting screenshot files.

## Resolution log

Rechecked 2026-09-19: both findings are resolved in the current source. `ask_ws` now creates a fresh internal UUID for every invocation while returning the public UI label unchanged. The new sequential and deliberately interleaved concurrent real-LangGraph tests exercise both original reproductions. `tests/test_api.py`: **10 passed**.

Independent full gates after the fix: **84 pytest passed** (one upstream Starlette/AnyIO deprecation warning); historical report `--check` passed; frontend **2 tests passed**; production frontend build passed (1,806 modules). The two original findings above are retained as review history, not open defects.

Final documentation/artifact pass completed 2026-09-19:

- Read README, demo guide, release notes, changelog, API reliability report and the completed validation record. Live generation blocked by MiMo HTTP 401, Docker/Ollama and non-Windows/hosted-CI gaps, single-turn semantics and historical-only metrics are stated consistently.
- README and validation record explicitly state that one P3 reranked retrieval succeeded earlier, while the final repetition failed with Windows OS error 1455 / insufficient page-file capacity. This remains an operational risk, not a passed final repeat.
- Compared README/release score values with `data/eval/release_summary.json`: strict 50 attempted / 49 scored / 1 error, scored composite 0.6711551020408163 and all-attempts composite 0.657732; hard 39 attempted and composite 0.8179487179487179. Displayed rounding is correct.
- Checked every local Markdown link in README, demo guide, release notes and validation record: no missing linked artifacts.
- Opened both checked-in screenshots and verified actual PNG dimensions: desktop 1280x720; mobile 390x844. Both show the Guided disclosure, illustrative checking and unmeasured confidence/time. The mobile image shows a readable stacked result without overlap in the captured viewport. Screenshots are UI examples, not evidence of successful model inference.
- Re-ran the locked Ruff executable after the final configuration change: **Ruff 0.16.8, all checks passed**. The explicit `E4,E7,E9,F` selection matches the documented release lint gate. No broader lint or formatting compliance is claimed.

No additional Critical or Important findings were found in this final pass.

The two Minor observations at the top remain non-blocking for this clearly labelled showcase candidate. Future live-deployment acceptance gates remain in `docs/validation-2026-09-18.md` and `docs/releases/v0.1.0.md`.
