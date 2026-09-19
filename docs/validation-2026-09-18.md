# v0.1.0 validation record — 2026-09-18 to 2026-09-19

This record separates release checks that passed from paths that were unavailable or incomplete on the validation host. Commands were run from `milestone/v0.1.0-showcase` in a fresh Python 3.12 environment named `.venv-release`; it is ignored by Git.

## Passed release gates

| Area | Check | Result |
|---|---|---|
| Python behavior | `.venv-release/Scripts/python.exe -m pytest -q` | **84 passed**, 1 upstream Starlette/AnyIO deprecation warning, 13.37 s |
| Python static checks | `.venv-release/Scripts/ruff.exe check src` | **Passed** with explicit stable `E4,E7,E9,F` release rules |
| Locked environment | `uv pip check --python .venv-release/Scripts/python.exe` | **164 packages compatible** |
| Historical report | `python scripts/report_release.py --check` | Summary JSON and Markdown match source artifacts |
| Frontend regression | `npm --prefix frontend test` | **2 passed**: stale/cancelled callbacks and terminal-event handling |
| Frontend production build | `npm --prefix frontend run build` | **Passed**, 1,806 modules transformed |
| Compose syntax | `docker compose config --quiet` | **Passed** for the default configuration |
| Python artifact | `uv build --wheel --out-dir output/release` | Built `medrag_agent-0.1.0-py3-none-any.whl` |
| Wheel smoke install | fresh Python 3.12 venv, install wheel with `--no-deps`, import package metadata | **Passed**, version `0.1.0` |
| Candidate secret scan | current tracked/untracked candidate, high-confidence token/private-key patterns | **0 matching paths across 170 files** |
| History secret scan | all Git patch history and sensitive filename patterns | **0 high-confidence value matches**; only `.env.example` and `frontend/.env.example` names found |

The secret scan is a focused release guard, not a substitute for GitHub secret scanning or credential rotation.

Local review artifacts are under `output/release/` and are ignored by Git: `veritasmed-v0.1.0-source.zip`, `medrag_agent-0.1.0-py3-none-any.whl`, and their external `SHA256SUMS` file.

The source archive was assembled from Git-tracked plus non-ignored candidate files. It excludes `.env`, Git metadata, local environments, model caches, `.demo-runtime`, `node_modules` and `output` itself.

## Local demo and API checks

- The real BGE-M3 CPU bootstrap inserted three stable points into the isolated `medrag_demo` collection. The idempotency regression also verifies that a second bootstrap remains at three points and does not alter `medrag_text`.
- `/api/live` returned `ok`. `/api/health` returned `degraded`: Qdrant was connected and the configured MiMo endpoint was disconnected because authentication returned HTTP 401.
- Real P2 hybrid search returned three fixture passages for `fastMRI`; a warm-up request on the final run took 5.04 s. The document endpoint returned the expected two passages for `PMC:PMC6996599`.
- P3 reranked search completed once earlier in this validation session with three passages (cold run about 11.06 s). A final repeat could not load the reranker because Windows reported OS error 1455, “page file too small,” while other model-training processes were using memory. Those unrelated processes were left untouched. Treat P3 memory headroom on this host as unresolved operational risk.
- Browser QA used a real Chromium session. Guided Ask, all three fixed examples, citation context, Explore text search, document navigation and query-parameter preservation were exercised. The browser console had no errors. The checked-in screenshots are actual renders at 1280×720 and 390×844; the mobile answer no longer overlaps its evidence/result sections.
- Guided mode made no backend/model call and continually displayed its fixture disclosure. Its confidence, elapsed time and reasoning steps were labelled unmeasured or illustrative.

## Independently reviewed fixes

The release reviewer reproduced two checkpoint defects before the fix: sequential questions could retain an old rewrite, and concurrent requests sharing a public UI label could return the other request's answer. Each Ask now receives a private checkpoint UUID while returning the original public label. Real LangGraph/InMemorySaver regression tests cover both cases. The independent final review reran the whole Python suite, report check, frontend tests and production build and found no remaining Critical or Important code issue. See [release-review.md](release-review.md).

This isolation deliberately makes Ask single-turn. The history endpoint cannot use a public UI label to restore these private checkpoints. The README and release notes state this limitation.

## Not passed or not executed

| Gate | Status and consequence |
|---|---|
| Successful live generated answer | **Blocked:** the available MiMo credential returned HTTP 401. No answer-success claim is made. Recheck normal, synthesis and insufficient-evidence questions with valid access. |
| Docker image build / startup | **Not executed:** Docker Engine was unavailable. Compose syntax passed only. |
| Ollama generation | **Not executed:** no configured local model was available. |
| macOS / Linux launcher | **Not executed:** implementation is cross-platform Python, but only Windows was exercised. |
| Hosted GitHub Actions | **Not executed:** workflow exists locally and needs its first run after push. |
| New full benchmark | **Intentionally not run:** published metrics are deterministic recalculations of historical artifacts. |
| Public deployment security | **Out of scope:** the web API has no public-user authentication, quotas or multi-tenant isolation. Keep it on loopback. |

## Release assessment

The tree is suitable as a transparent **local GitHub research showcase candidate**. It is not signed off as a hosted service, clinical system, fully verified Docker distribution or freshly benchmarked model release. A Git commit, tag, push and GitHub Release have not been created by this validation.
