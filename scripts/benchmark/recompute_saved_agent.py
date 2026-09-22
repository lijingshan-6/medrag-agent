"""Recompute the saved v0.3 development metrics; requires only Python and Pydantic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from medrag.benchmark.saved_run import recompute_saved_run, render_cases  # noqa: E402


def main() -> None:
    default_dir = ROOT / "data/benchmark/veritasmed_v1_1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=default_dir / "questions.jsonl")
    parser.add_argument("--answers", type=Path, default=default_dir / "agent_v03_dev_raw.json")
    parser.add_argument("--assessments", type=Path, default=default_dir / "answer_assessment_overrides_v03_dev.jsonl")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; otherwise print metrics only")
    parser.add_argument("--cases", type=Path, help="Optional Markdown file with every answer and gold evidence")
    args = parser.parse_args()
    report = recompute_saved_run(args.questions, args.answers, args.assessments)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.cases:
        args.cases.parent.mkdir(parents=True, exist_ok=True)
        args.cases.write_text(render_cases(report, args.questions), encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
