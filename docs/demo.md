# VeritasMed demonstration guide

## Two explicit modes

| Mode | Start | What actually runs |
|---|---|---|
| Guided | `cd frontend`, `npm ci`, `npm run dev`; open `/?demo=1` | Browser-only authored answers and illustrative steps; no measured scores |
| Live | Follow the root README, then `python scripts/run_demo.py` | Real BGE/Qdrant retrieval and, with valid model access, the LangGraph agent |

Do not present guided screenshots as live inference. The three fixed answers are checked-in fixtures, not recordings of successful model calls. Guided Explore performs simple text matching; switch to Live mode for P2/P3 retrieval.

## A short portfolio walkthrough

1. Open the guided page and point out its fixed-example banner.
2. Select **What kinds of data does the fastMRI knee dataset provide?** Show the answer's source marker and its passage in the right column.
3. Select **How does fastMRI+ extend fastMRI for imaging research?** Show two source IDs, the six illustrative workflow stages, and **View context**. Two passages from the same document share a citation number.
4. Select **Does this evidence establish which treatment is best for an individual patient?** The example explains the lack of supporting treatment evidence.
5. Use **Explore**, search `fastMRI`, open a document and inspect its authored-summary label and original-paper link.
6. Open the [v0.3 evaluation report](agent-v0.3-report.md) and show an actual saved Agent answer next to its required evidence. Guided response time and confidence are not measurements.

Copy and Download export the current answer; downloaded Markdown carries the guided-mode disclosure. Re-run repeats the selected example. Other questions in Guided mode show a clear instruction to use Live mode.

## Live demo configuration

The launcher fixes `QDRANT_PATH`, `QDRANT_COLLECTION=medrag_demo` and `MEDRAG_DATA_DIR` to a separate `.demo-runtime` store. It uses one API process; do not attach several API processes to the same embedded store. Running it with `--skip-index` assumes that the first index run succeeded. Deleting or changing the research store is unnecessary.

For MiMo, edit `.env` with valid endpoint credentials. For an already installed and configured Ollama service:

```dotenv
LLM_BACKEND=ollama
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:9b
LLM_TIMEOUT_SECONDS=240
```

Download the chosen model with `ollama pull qwen3.5:9b` first. This model was exercised through the production Agent graph for the v1.1 baseline and v0.3 development comparison. The timeout above applies to each model call; the browser has a 300-second overall response deadline. With the default MiMo backend, questions and retrieved passages leave the machine for the configured cloud service.

## Inspect a measured answer without running models

Open [the development cases](agent-v0.3-cases.md) for all 15 real questions, original answers,
gold claims, exact evidence quotes and final decisions. Suggested examples are VMG-002 for
numerical results, VMG-018 for a two-source methods comparison, and VMG-042 for an unsupported
comparative outcome. Also show one failed case from the report so the model's limits are visible.

To recalculate the saved metrics in a lightweight Python environment:

```sh
python -m pip install "pydantic>=2.7,<3"
python scripts/benchmark/recompute_saved_agent.py
```

This reads committed answers and adjudications. It does not generate new answers or download
models. The full benchmark corpus is separate from the tiny fastMRI demo fixture; questions
from the benchmark should not be asked against that fixture expecting the same evidence.

Troubleshooting:

- **Needs setup / HTTP 401:** correct the model credential and restart the launcher. The release audit encountered this with the available credential; a successful generation must still be checked with valid access.
- **Ports busy:** stop the earlier demo/frontend before restarting. Defaults are API 8000 and frontend 5173.
- **First search is slow:** BGE weights are loaded on first use. Initial model download and cold startup are much slower than warm retrieval.
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
