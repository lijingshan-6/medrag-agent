# VeritasMed demonstration guide

## Two explicit modes

| Mode | Start | What actually runs |
|---|---|---|
| Guided | `cd frontend`, `npm ci`, `npm run dev`; open `/?demo=1` | Browser-only authored answers and illustrative steps; no measured scores |
| Live | Follow the root README, then `python scripts/run_demo.py` | Real BGE/Qdrant retrieval and, with valid model access, the LangGraph agent |

Do not present guided screenshots as live inference. The three fixed answers are checked-in fixtures, not recordings of successful model calls. Guided Explore performs simple text matching; switch to Live mode for P2/P3 retrieval.

## A short portfolio walkthrough

1. Open the guided page and point out its fixed-example banner.
2. Select **What kinds of data does the fastMRI knee dataset provide?** Show **Evidence covers the question**, expand its evidence component, and show the answer's source marker and its passage in the right column.
3. Select **How does fastMRI+ extend fastMRI, and does this evidence establish diagnostic accuracy?** Show the **Partially covered** state and its explicit diagnostic-accuracy gap. Expand **Additional pathology annotations**, read the source quotation, and click **View source**. Two passages from one document share a citation number.
4. Select **Does this evidence establish which treatment is best for an individual patient?** Show **Insufficient evidence**. A properly bounded refusal can pass the model check while the requested evidence is still insufficient.
5. Use **Explore**, search `fastMRI`, open a document and inspect its authored-summary label and original-paper link.
6. Open the [latest v0.4 repair report](agent-v0.4-repaired-report.md) and show an actual saved Agent answer next to its required evidence. Compare a repaired answer with its preserved original failed case. Guided response time and confidence are not measurements.

Copy and Download export the current answer; downloaded Markdown carries the guided-mode disclosure. Re-run repeats the selected example. Other questions in Guided mode show a clear instruction to use Live mode.

Actual Guided screenshots: [complete coverage](assets/v04-complete.png),
[partial coverage with an explicit gap](assets/v04-evidence-coverage.png), and
[insufficient evidence](assets/v04-insufficient.png). These show authored examples, not model runs.

A [real Live screenshot from the first candidate](assets/v04-live-partial.png) records the second question answered by
`qwen3.5:9b` after installation in a fresh Windows directory: 57.48 seconds, no query rewrites
or regenerations, with the annotation result and diagnostic-accuracy gap. Retrieval and generation
were live; the underlying three demo passages are still authored summaries, not benchmark papers.

## Live demo configuration

The launcher fixes `QDRANT_PATH`, `QDRANT_COLLECTION=medrag_demo` and `MEDRAG_DATA_DIR` to a separate `.demo-runtime` store. It uses one API process; do not attach several API processes to the same embedded store. Running it with `--skip-index` assumes that the first index run succeeded. Deleting or changing the research store is unnecessary.

For MiMo, edit `.env` with valid endpoint credentials. For an already installed and configured Ollama service:

```dotenv
LLM_BACKEND=ollama
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:9b
LLM_TIMEOUT_SECONDS=240
```

Download the chosen model with `ollama pull qwen3.5:9b` first. This model was exercised through the production Agent graph for the v1.1 baseline and v0.3/v0.4 development comparisons. The timeout above applies to each model call. The web API has a 300-second overall response deadline, with a 310-second browser fallback. In the first v0.4 candidate, a difficult boundary question failed after 381 seconds in the CLI; that attempt would time out earlier in the browser. Increasing the per-call timeout does not remove the web deadline. With the default MiMo backend, questions and retrieved passages leave the machine for the configured cloud service.

If Ollama was configured globally with a listening address such as `OLLAMA_HOST=0.0.0.0:11434`,
the application connects through `http://127.0.0.1:11434`. It also accepts scheme-free host:port
values and preserves explicitly configured remote or container hosts. There is no need to change
the service's listening address.

## Inspect a measured answer without running models

Open [the repaired development cases](agent-v0.4-repaired-cases.md) for all 15 real questions,
original answers, source passages and explicit adjudications. These saved answers are distinct from
the authored Guided demo. Re-entering a question performs a new model run and may produce a different answer.

| Real saved case | What to demonstrate |
|---|---|
| [VMG-018: two reconstruction methods](agent-v0.4-repaired-cases.md#vmg-018) | Follow each method's steps and validation findings back to its own study. |
| [VMG-009: samples and performance](agent-v0.4-repaired-cases.md#vmg-009) | Separate development data from the radiologists' validation performance; compare the [original failed answer](agent-v0.4-cases.md#vmg-009). |
| [VMG-006: supported findings plus a gap](agent-v0.4-repaired-cases.md#vmg-006) | Inspect the 21/20 population and the unanswered longitudinal question. |
| [VMG-014](agent-v0.4-repaired-cases.md#vmg-014) and [VMG-042](agent-v0.4-repaired-cases.md#vmg-042) | Distinguish absent clinical-outcome data from outcomes measured without the requested control comparison. |

The reports preserve failures and model self-check disagreements. Neither source quotations nor a
green model check are a clinical correctness guarantee. Review the full saved sequence rather than
selecting one successful rerun.

Recompute the saved development metrics:

```sh
python -m pip install "pydantic>=2.7,<3"
python scripts/benchmark/recompute_saved_agent.py --version v0.4-repaired
```

This reads committed answers and adjudications. It does not generate new answers or download
models. The full benchmark corpus is separate from the tiny fastMRI demo fixture; questions
from the benchmark should not be asked against that fixture expecting the same evidence.

Troubleshooting:

- **Needs setup / HTTP 401:** correct the model credential and restart the launcher. The release audit encountered this with the available credential; a successful generation must still be checked with valid access.
- **Ports busy:** stop the earlier demo/frontend before restarting. Defaults are API 8000 and frontend 5173.
- **First search is slow:** BGE weights are loaded on first use. Initial model download and cold startup are much slower than warm retrieval.
- **Answer timed out / evidence planning failed:** If a model call fails to return usable structured output, the request has failed; inspect the saved error and retry the request. This is an execution failure, not a medical conclusion or a valid refusal. Try the bundled demo questions to demonstrate the working path; inspect saved benchmark cases for the wider evidence assessment.
- **No collection:** run without `--skip-index` once. For research data use your explicitly selected external Qdrant collection and the ingestion scripts, not the tiny demo as a benchmark.
- **Stop button:** stops further client output and cooperatively stops future graph steps; an in-flight model call may finish in the background.

## Optional Docker path — configuration checked, execution unverified

Copy `.env.example` to `.env`, set real credentials and start Docker Engine. From the repository root:

```sh
docker compose config --quiet
docker compose up --build -d
docker compose exec backend python scripts/bootstrap_demo.py --device cpu
```

Open http://127.0.0.1:5173. The backend image includes the authored demo fixture; indexing is an explicit one-time step. The default Compose collection is `medrag_demo`. Ports are loopback-only; volumes retain vectors, checkpoints and downloaded model weights. `docker compose down` stops services without deleting volumes.

Optional Ollama override:

```sh
docker compose -f docker-compose.yml -f compose.ollama.yml up --build -d
docker compose -f docker-compose.yml -f compose.ollama.yml exec ollama ollama pull qwen3.5:9b
docker compose -f docker-compose.yml -f compose.ollama.yml exec backend python scripts/bootstrap_demo.py --device cpu
```

Docker startup and model downloads were not executed successfully on the release host because the Docker engine was unavailable. These commands are provided as configuration, not as a passed deployment test. The Windows `start_*.ps1` scripts are legacy research-environment helpers; the root README launcher is the supported showcase entry point.
