"""Deterministic release summary of saved, historical evaluation results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


INPUTS = {
    "strict": "data/eval/agent_eval_v2_strict.json",
    "hard": "data/eval/agent_eval_hard_v4.json",
    "retrieval": "data/eval/retrieval_eval_v2_gpu.json",
    "standard_questions": "data/golden/golden_dataset.jsonl",
    "hard_questions": "data/golden/golden_hard.jsonl",
}


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number")
    return float(value)


def _agent_summary(data: dict, label: str) -> dict:
    rows = data.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label}: results must be a nonempty list")
    if data.get("n") != len(rows):
        raise ValueError(f"{label}: n does not match attempted result rows")
    scored = []
    failures = []
    ids = set()
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in ids:
            raise ValueError(f"{label}: missing or duplicate result id {row_id!r}")
        ids.add(row_id)
        if "error" in row:
            if "composite" in row:
                raise ValueError(f"{label}/{row_id}: both score and error are present")
            error = row["error"]
            if not isinstance(error, str) or not error:
                raise ValueError(f"{label}/{row_id}: error must be nonempty text")
            failures.append({"id": row_id, "error": error})
            continue
        if "composite" not in row:
            raise ValueError(f"{label}/{row_id}: neither composite nor error is present")
        for key in ("faithfulness", "relevance", "correctness", "composite"):
            _number(row.get(key), f"{label}/{row_id}/{key}")
        scored.append(row)
    attempted = len(rows)
    if not scored:
        raise ValueError(f"{label}: no scored results")
    latencies = sorted(
        _number(row.get("agent_latency_s"), f"{label}/{row['id']}/agent_latency_s")
        for row in scored
    )
    return {
        "attempted": attempted,
        "scored": len(scored),
        "error_count": len(failures),
        "errors": failures,
        "judge": data.get("model_judge"),
        "pipeline": data.get("pipeline"),
        "faithfulness_scored": mean(row["faithfulness"] for row in scored),
        "relevance_scored": mean(row["relevance"] for row in scored),
        "correctness_scored": mean(row["correctness"] for row in scored),
        "composite_scored": mean(row["composite"] for row in scored),
        "composite_all_attempts": sum(row["composite"] for row in scored) / attempted,
        "latency_s": {
            "mean": mean(latencies),
            "median": median(latencies),
            "p95_nearest_rank": latencies[math.ceil(0.95 * len(latencies)) - 1],
            "n": len(latencies),
        },
    }


def _retrieval_summary(data: dict) -> dict:
    rows = data.get("results")
    if not isinstance(rows, list):
        raise ValueError("retrieval: results must be a list")
    by_pipeline: dict[str, dict[str, dict]] = {name: {} for name in ("p1", "p2", "p3")}
    for row in rows:
        pipeline = row.get("pipeline")
        if pipeline not in by_pipeline:
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in by_pipeline[pipeline]:
            raise ValueError(f"retrieval/{pipeline}: missing or duplicate result id {row_id!r}")
        gold = row.get("gold_chunk_ids")
        retrieved = row.get("retrieved_ids")
        if (not isinstance(gold, list) or not gold
                or any(not isinstance(chunk, str) or not chunk for chunk in gold)
                or not isinstance(retrieved, list)
                or any(not isinstance(chunk, str) or not chunk for chunk in retrieved)):
            raise ValueError(f"{row_id}/{pipeline}: invalid gold evidence or retrieved ranking")
        _number(row.get("latency_s"), f"{row_id}/{pipeline}/latency_s")
        by_pipeline[pipeline][row_id] = row
    expected_ids = set(by_pipeline["p1"])
    if not expected_ids or len(expected_ids) != data.get("n_questions"):
        raise ValueError("retrieval: P1 question count differs from n_questions")
    if any(set(group) != expected_ids for group in by_pipeline.values()):
        raise ValueError("retrieval: P1/P2/P3 question IDs differ")
    for row_id in expected_ids:
        evidence = {tuple(sorted(set(group[row_id]["gold_chunk_ids"])))
                    for group in by_pipeline.values()}
        if len(evidence) != 1:
            raise ValueError(f"retrieval/{row_id}: gold evidence differs across pipelines")
    output = {}
    for pipeline, group in by_pipeline.items():
        hits = []
        recalls = []
        reciprocals = []
        latencies = []
        for row_id in sorted(group):
            row = group[row_id]
            gold = set(row["gold_chunk_ids"])
            top5 = row["retrieved_ids"][:5]
            top20 = row["retrieved_ids"][:20]
            hits.append(float(bool(gold.intersection(top5))))
            recalls.append(len(gold.intersection(top5)) / len(gold))
            first_rank = next((index for index, chunk in enumerate(top20, 1) if chunk in gold), None)
            reciprocals.append(1 / first_rank if first_rank else 0.0)
            latencies.append(row["latency_s"])
        output[pipeline] = {
            "questions": len(group),
            "hit_at_5": mean(hits),
            "macro_evidence_recall_at_5": mean(recalls),
            "mrr_at_20": mean(reciprocals),
            "historical_mean_latency_s": mean(latencies),
        }
    return output


def build_summary(strict: dict, hard: dict, retrieval: dict, input_sha256: dict[str, str]) -> dict:
    """Recompute metrics from result rows, ignoring stored aggregate summaries."""
    return {
        "schema_version": 1,
        "provenance": "historical saved evaluation artifacts; no new model runs",
        "input_sha256": input_sha256,
        "strict": _agent_summary(strict, "strict"),
        "hard": _agent_summary(hard, "hard"),
        "retrieval": _retrieval_summary(retrieval),
    }


def load_release(root: Path) -> dict:
    """Read source files without changing them and hash their raw bytes."""
    raw = {name: (root / relative).read_bytes() for name, relative in INPUTS.items()}
    hashes = {INPUTS[name]: hashlib.sha256(content).hexdigest() for name, content in raw.items()}
    return build_summary(
        json.loads(raw["strict"]),
        json.loads(raw["hard"]),
        json.loads(raw["retrieval"]),
        hashes,
    )


def render_report(summary: dict) -> str:
    strict, hard, retrieval = summary["strict"], summary["hard"], summary["retrieval"]
    lines = [
        "# VeritasMed 历史评估报告",
        "",
        "本报告对应 2026-09-18 里程碑审计的历史工件；历史运行日期未记录。",
        "本报告从已保存的评估逐题结果重新计算；没有运行新的检索、模型或裁判评估。",
        "代码 commit、语料快照与硬件配置未随这些结果文件完整保存，",
        "因此不能把下面的延迟或成绩视为当前版本的实测表现。",
        "",
        "## Agent 问答评分",
        "",
        "| 历史题集 | 尝试 | 有评分 | 错误 | 成功题 Composite | 错误题按零 Composite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for title, result in (("strict 标准集", strict), ("hard 难题集", hard)):
        lines.append(
            f"| {title} | {result['attempted']} | {result['scored']} | "
            f"{result['error_count']} | {result['composite_scored']:.6f} | "
            f"{result['composite_all_attempts']:.6f} |"
        )
    lines += [
        "",
        "Composite 为逐题保存分数的均值。成功题口径只除以有评分题数；",
        "错误题按零口径以全部尝试数为分母。strict 与 hard 题集不同，不能直接比较为版本提升。",
        "",
        "| 历史题集 | 成功题 Faithfulness | Relevance | Correctness |",
        "|---|---:|---:|---:|",
    ]
    for title, result in (("strict", strict), ("hard", hard)):
        lines.append(
            f"| {title} | {result['faithfulness_scored']:.4f} | "
            f"{result['relevance_scored']:.4f} | {result['correctness_scored']:.4f} |"
        )
    lines += [
        "",
        f"历史文件记录的裁判：strict 为 `{strict['judge']}`，hard 为 `{hard['judge']}`。",
        "裁判与系统检查模型属于同一模型家族，评分独立性受限。",
        "Faithfulness 是对检索上下文的评分，不是医学正确率；Correctness 也不是临床验证结果。",
        "没有在同一冻结题集、语料、提示词和裁判条件下与普通 RAG 做配对消融，",
        "所以这些工件不能证明 Agent 优于普通 RAG。",
        "",
        "### 历史 Agent 端到端耗时（只含成功题，秒）",
        "",
        "| 题集 | n | 均值 | 中位数 | P95（nearest rank） |",
        "|---|---:|---:|---:|---:|",
    ]
    for title, result in (("strict", strict), ("hard", hard)):
        latency = result["latency_s"]
        lines.append(
            f"| {title} | {latency['n']} | {latency['mean']:.2f} | "
            f"{latency['median']:.2f} | {latency['p95_nearest_rank']:.2f} |"
        )
    lines += ["", "### 错误题", ""]
    for title, result in (("strict", strict), ("hard", hard)):
        if result["errors"]:
            for failure in result["errors"]:
                lines.append(f"- {title} {failure['id']}: `{failure['error']}`")
        else:
            lines.append(f"- {title}: 0 条。")
    lines += [
        "",
        "## 历史检索结果（标准集，P1/P2/P3）",
        "",
        "Hit@5 表示前五个结果至少命中一条标准证据；旧结果文件的 Recall@5 字段使用此定义。",
        "证据 Recall@5 先按每题计算命中证据数 / 标准证据数，再对题目取宏平均。",
        "MRR@20 对首条命中证据的名次取倒数，再对题目取平均；未命中记零。",
        "",
        "| 检索方案 | 题数 | Hit@5 | 宏平均证据 Recall@5 | MRR@20 | 历史平均检索耗时 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pipeline in ("p1", "p2", "p3"):
        item = retrieval[pipeline]
        lines.append(
            f"| {pipeline.upper()} | {item['questions']} | {item['hit_at_5']:.4f} | "
            f"{item['macro_evidence_recall_at_5']:.4f} | {item['mrr_at_20']:.4f} | "
            f"{item['historical_mean_latency_s']:.3f} 秒 |"
        )
    lines += [
        "",
        "检索耗时只包含历史检索步骤，不能与上面的 Agent 端到端耗时直接比较。",
        "这些逐题排名支持此历史评估中 P2/P3 命中与排序改善的观察，",
        "不构成 Agent 问答收益的证据。",
        "",
        "## 可追溯输入",
        "",
        "以下 SHA-256 校验文件原始字节。题集 hash 只标识当前保存的题集文件，",
        "历史运行缺少完整语料快照 hash，无法单凭这些文件复原运行环境。",
        "",
        "| 输入文件 | SHA-256 |",
        "|---|---|",
    ]
    for path, digest in sorted(summary["input_sha256"].items()):
        lines.append(f"| `{path}` | `{digest}` |")
    lines += [
        "",
        "重新生成：`python scripts/report_release.py`。检查工作树报告是否与输入一致：",
        "`python scripts/report_release.py --check`。两个命令都不会改写历史评估 JSON。",
        "",
    ]
    return "\n".join(lines)


def write_or_check(summary: dict, json_path: Path, report_path: Path, *, check: bool) -> None:
    """Write deterministic outputs or check their exact UTF-8 bytes without mutation."""
    expected = {
        json_path: (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        .encode("utf-8"),
        report_path: render_report(summary).encode("utf-8"),
    }
    for path, content in expected.items():
        if check:
            if not path.exists():
                raise ValueError(f"{path}: missing; run report_release.py to generate")
            if path.read_bytes() != content:
                raise ValueError(f"{path}: stale or modified; run report_release.py to regenerate")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
