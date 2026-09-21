# VeritasMed

**Evidence-grounded medical literature Q&A with an inspectable retrieval and checking workflow.**

VeritasMed connects a React interface to a LangGraph agent: retrieve literature passages, rerank them, assess the evidence, rewrite a weak query, generate a cited answer, and check it against the retrieved context. This repository is a **local research showcase**, not a clinically validated assistant.

**v0.2.0 showcase** · Python 3.12 · Node.js 22.12+ · Apache-2.0

![VeritasMed guided example: answer, workflow and cited evidence](docs/assets/guided-desktop.png)

*Actual application screenshot in Guided demo mode. Answers and animated steps are authored fixtures; they are not a live model run or a benchmark.*

## Try the interface without a key

Only Node.js is needed for this first path:

```sh
git clone https://github.com/lijingshan-6/medrag-agent.git
cd medrag-agent/frontend
npm ci
npm run dev
```

Open **http://127.0.0.1:5173/?demo=1**. Select one of the three example questions, inspect citations and passage context, or use Explore to browse the small fixture. Copy, download and rerun an example from the answer toolbar.

Guided mode runs in the browser. It uses fixed authored answers and text matching over three fixture passages; it makes no backend or model calls. Its disclosure banner stays visible. Arbitrary questions require Live mode.

## Run real retrieval and the agent

Requirements: Python **3.12**, Node.js **22.12+**, and [uv](https://docs.astral.sh/uv/). Initial installation and model downloads require internet access and several GB of disk space. The locked environment uses CPU PyTorch; GPU setup is outside this release's verified path.

From the repository root:

```sh
uv venv --python 3.12
uv pip sync requirements.lock --torch-backend cpu
uv pip install --no-deps -e .
```

Copy `.env.example` to `.env`, then set a valid `OPENAI_API_KEY` for the configured MiMo endpoint. The alternative Ollama configuration is in [the demo guide](docs/demo.md). Never commit `.env`.

Activate the environment:

| Shell | Command |
|---|---|
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| macOS / Linux | `source .venv/bin/activate` |

Start the demonstration:

```sh
python scripts/run_demo.py
```

The launcher indexes the bundled summaries with BGE-M3, starts the API and frontend, and prints **http://127.0.0.1:5173**. Later starts can use `python scripts/run_demo.py --skip-index`. Ports 8000 and 5173 must be free. If PowerShell blocks activation, invoke `.\.venv\Scripts\python.exe scripts/run_demo.py` directly.

The demo stores vectors and checkpoints under `.demo-runtime/`, uses collection `medrag_demo`, and does not reset the research collection `medrag_text`. **Explore works without an LLM key** after the models and fixture are installed. Live Ask also needs valid model access. The fixture contains three authored summaries about fastMRI and fastMRI+, not the original articles, medical records or MRI files; see its [provenance](data/demo/README.md).

**Verified here:** Windows, Python 3.12, CPU indexing, CUDA reranking, the production Agent graph with local Ollama `qwen3.5:9b`, local review with `medgemma1.5:4b`, API document access, frontend build and browser interactions. Docker, the MiMo cloud path and macOS/Linux execution remain unverified. Read the [v1.1 benchmark report](docs/benchmark-v1.1-report.md) and [validation record](docs/validation-2026-09-18.md) before treating this as a deployment recipe.

## What the project demonstrates

- **Retrieval:** BGE-M3 dense and sparse embeddings, Qdrant hybrid retrieval and a BGE cross-encoder reranker. Explore exposes P2 hybrid and P3 reranked search.
- **Agent control flow:** query routing, evidence grading, bounded rewriting and answer regeneration. Ask always uses the full agent workflow.
- **Inspectable answers:** source-linked citations, passage context and streamed node activity. A model evidence check is a self-assessment, not a guarantee of factual or clinical correctness.
- **Failure handling:** bounded LLM calls, response deadlines, one terminal stream result, cancellation between nodes and isolation between individual requests.
- **Reproducibility:** dependency lock, frozen corpus hashes, a claim-level 50-question benchmark, offline regression tests and inspectable evaluation artifacts.

```mermaid
flowchart LR
  UI[React interface] --> API[FastAPI / WebSocket]
  API --> Route[Route question]
  Route --> Retrieve[BGE-M3 + Qdrant]
  Retrieve --> Rerank[BGE reranker]
  Rerank --> Grade[Grade evidence]
  Grade -->|weak, within budget| Rewrite[Rewrite query]
  Rewrite --> Retrieve
  Grade --> Generate[Generate cited answer]
  Generate --> Check[Model evidence check]
  Check -->|revise, within budget| Generate
  Check --> Answer[Answer + sources + trace]
```

Each Ask is a standalone question. Session labels in the interface do **not** provide conversational memory or restore previous answers. Internal checkpoints are isolated per request to prevent cross-request result contamination.

## Evidence-grounded benchmark v1.1

The active benchmark contains 50 source-first questions over 44 PubMed sources and 20 domains, with exact claim-level quotes, explicit evidence gaps and five same-topic abstention cases. The 15-question development split and 35-question test split share no source or evidence chunk. All sources in this frozen snapshot are from 2026, and the set is engineering-reviewed rather than clinician-validated.

The v1.1 audit retained 39 contracts and revised 11. `qwen3.5:9b` generated a fresh 80-item candidate pool and re-audited the final set; `medgemma1.5:4b` challenged medical qualifiers; `llama3.1:8b` acted as an independent-family blind challenger. Every model finding was resolved against the frozen passages. Model agreement never created a final label.

The real production Agent graph was run on development only with `qwen3.5:9b`, BGE-M3 hybrid retrieval and CUDA reranking. The test split remains untouched.

| Development result | Score |
|---|---:|
| Required-claim Recall@5, 13 answerable questions | 88.5% |
| All required claims found@5 | 84.6% |
| nDCG@5 / MRR@5 | 0.8933 / 0.9231 |
| Claim completeness / citation coverage | 90.0% / 90.0% |
| Claim-support precision | 100.0% |
| Answerability score | 80.0% |
| Strict passes | **5/15 (33.3%)** |
| Mean end-to-end latency | 85.1 s |

The Agent's own faithfulness checker passed 15/15, but source-first adjudication found two missing cross-document answers, seven answers with omitted required qualifiers, one partial answer that did not state its evidence boundary, and failures on both unanswerable questions. The gap is a measured system weakness, not a benchmark score to hide.

[v1.1 report](docs/benchmark-v1.1-report.md) · [Dataset card](data/benchmark/veritasmed_v1_1/dataset_card.md) · [Frozen questions](data/benchmark/veritasmed_v1_1/questions.jsonl) · [Per-item adjudications](data/benchmark/veritasmed_v1_1/model_adjudications.jsonl) · [Development baseline manifest](data/benchmark/veritasmed_v1_1/baseline_agent_dev_manifest.json)

```sh
python scripts/benchmark/freeze_v1_1.py
python scripts/benchmark/audit_gold.py --questions data/benchmark/veritasmed_v1_1/questions.jsonl --output-dir data/benchmark/veritasmed_v1_1 --profile-only
```

The prior v1 metrics and direct-model component baseline remain preserved in the [v1 report](docs/benchmark-report.md); they are not treated as the production-Agent v1.1 result.

## Historical evaluation, with explicit denominators

These are recalculations of saved artifacts, **not new v0.1.0 benchmark runs**. Original run dates, source commit and full corpus snapshot were not preserved.

| Historical set | Attempted / scored / errors | Composite, scored only | Composite, errors as zero |
|---|---:|---:|---:|
| Strict standard set | 50 / 49 / 1 | 0.6712 | 0.6577 |
| Hard set | 39 / 39 / 0 | 0.8179 | 0.8179 |

| Historical retrieval | Hit@5 | Macro evidence Recall@5 | MRR@20 |
|---|---:|---:|---:|
| P1 dense baseline | 0.5000 | 0.2667 | 0.3547 |
| P2 hybrid | 0.6000 | 0.2933 | 0.4177 |
| P3 hybrid + reranking | 0.7000 | 0.3250 | 0.5134 |

The old retrieval artifact called the first metric Recall@5; it actually measures whether at least one reference is found, so this report calls it **Hit@5**. Strict and hard use different questions and cannot establish a version improvement. The same model family was involved in judging, and there is no matched agent-versus-plain-RAG ablation. These results establish neither clinical accuracy nor agent superiority.

[Full report and input hashes](docs/evaluation_report.md) · [Machine-readable summary](data/eval/release_summary.json)

```sh
python scripts/report_release.py --check
```

## Development checks

With the locked Python environment active:

```sh
python -m pytest -q
ruff check src/
python scripts/report_release.py --check
uv pip check
npm --prefix frontend test
npm --prefix frontend run build
```

Default tests isolate checkpoints, disable local dotenv configuration and do not call external models. Live integration tests require explicit `--run-live` plus separately configured infrastructure; they may incur API charges. [CI](.github/workflows/ci.yml) runs the offline gates and builds a wheel; hosted CI is not claimed as passed before a push.

## Data, privacy and operating limits

- BGE embedding/reranking and Qdrant storage run locally in the supported demo. With MiMo selected, questions and retrieved passages are sent to the configured cloud endpoint. Ollama is an alternative, not the default.
- Browser session labels live in local storage. Backend checkpoints can contain queries, evidence and generated answers. Keep personal health information out of this showcase.
- API and frontend bind to loopback. The HTTP/WebSocket API has no public-user authentication, quotas or multi-tenant isolation and is not ready for direct internet exposure. Optional MCP token handling is separate from the web API.
- Stop/disconnect prevents further graph steps once the current synchronous step yields; it cannot forcibly abort an already running model request. Model requests have separate timeouts.
- Health distinguishes process liveness (`/api/live`) from dependency reachability (`/api/health`). A healthy endpoint does not guarantee a populated collection or a successful generation.
- Original research data, model weights, checkpoints, credentials and generated local environments are not part of the source release. Third-party datasets and papers retain their own terms.

## Repository guide

| Path | Purpose |
|---|---|
| `src/medrag/agent/` | Graph, nodes, prompts and model configuration |
| `src/medrag/api/` | REST and WebSocket interface |
| `src/medrag/index/`, `src/medrag/retrieval/` | Embeddings, indexing and search |
| `frontend/` | React interface and labelled browser fixtures |
| `data/demo/` | Small authored demonstration corpus |
| `data/benchmark/veritasmed_v1_1/` | Active source-disjoint benchmark, reviews and production-Agent baseline |
| `data/benchmark/veritasmed_v1/` | Preserved historical v1 benchmark and component baselines |
| `data/eval/`, `data/golden/` | Historical evaluation artifacts and question sets |
| `tests/` | Offline behavior and contract regressions |
| `docs/` | Audit, plan, evaluation, validation and release notes |

[Demo walkthrough / optional Docker](docs/demo.md) · [v0.2.0 release notes](docs/releases/v0.2.0.md) · [Changelog](CHANGELOG.md) · [Benchmark plan](docs/superpowers/plans/2026-09-20-veritasmed-benchmark-v1.1-audit.md)

Code and repository-authored demo text are distributed under [Apache-2.0](LICENSE). The next research milestone is stronger claim-boundary enforcement on the new development set, followed by a paired agent-versus-plain-RAG comparison and separately validated live deployment path.
