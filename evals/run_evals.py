#!/usr/bin/env python3
"""Run repeatable Anti-Slop Cannon evals against a fixture repo."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_ROOT = REPO_ROOT / "evals" / "fixtures" / "sloppy_repo"
DEFAULT_SLOP_DIR = REPO_ROOT / "evals" / "slop_examples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Anti-Slop Cannon fixture evals.")
    parser.add_argument("--provider", default="hash", choices=("hash", "google", "openai", "sentence-transformers"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dim", default=None, type=int)
    parser.add_argument("--root", default=str(DEFAULT_FIXTURE_ROOT))
    parser.add_argument("--slop-examples-dir", default=str(DEFAULT_SLOP_DIR))
    parser.add_argument("--threshold", default=0.80, type=float)
    parser.add_argument("--slop-match-threshold", default=0.70, type=float)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = os.environ.copy()
    load_env_file(Path(args.env_file), env)
    output_context = tempfile.TemporaryDirectory(prefix="anti-slop-eval.")
    output_dir = Path(output_context.name)
    if args.keep_output:
        output_context.cleanup()
        output_dir = Path(tempfile.mkdtemp(prefix="anti-slop-eval.keep."))

    command = [
        sys.executable,
        str(REPO_ROOT / "anti_slop_cannon.py"),
        str(Path(args.root).resolve()),
        "--provider",
        args.provider,
        "--granularity",
        "both",
        "--threshold",
        str(args.threshold),
        "--near-duplicate-threshold",
        "0.82",
        "--slop-examples-dir",
        str(Path(args.slop_examples_dir).resolve()),
        "--slop-match-threshold",
        str(args.slop_match_threshold),
        "--slop-top-matches",
        "0",
        "--output-dir",
        str(output_dir),
        "--no-cache",
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.output_dim:
        command.extend(["--output-dim", str(args.output_dim)])

    print(f"Running eval provider={args.provider} root={Path(args.root).resolve()}")
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr, end="", file=sys.stderr)
        return result.returncode

    report_path = output_dir / "anti_slop_report.json"
    html_path = output_dir / "anti_slop_map.html"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = evaluate_report(report, report_path, html_path)
    if args.keep_output:
        print(f"Kept eval output: {output_dir}")
    else:
        output_context.cleanup()

    if failures:
        print("Eval failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("Eval passed:")
    print(f"  files={report['file_count']} items={report['item_count']} clusters={report['cluster_count']}")
    print(f"  slop_examples={report['slop_example_count']} slop_matches={report['slop_match_count']}")
    return 0


def load_env_file(path: Path, env: dict[str, str]) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value


def evaluate_report(report: dict[str, Any], report_path: Path, html_path: Path) -> list[str]:
    failures: list[str] = []
    require(report_path.exists(), "missing JSON report", failures)
    require(html_path.exists(), "missing HTML map", failures)
    require(report.get("file_count") == 4, f"expected 4 fixture files, got {report.get('file_count')}", failures)
    require(report.get("item_count", 0) >= 8, f"expected at least 8 analysis items, got {report.get('item_count')}", failures)
    require(report.get("cluster_count", 0) >= 1, "expected at least one overlap cluster", failures)
    require(report.get("slop_example_count") == 1, "expected one slop example", failures)
    require(report.get("slop_match_count", 0) >= 2, "expected slop example to match duplicated total loops", failures)

    cluster_edges = report.get("cluster_edges", [])
    exact_total_edges = [
        edge
        for edge in cluster_edges
        if edge.get("relation") == "exact"
        and "parse_total_rows" in edge.get("a", "")
        and "parse_total_rows" in edge.get("b", "")
    ]
    require(exact_total_edges, "expected exact edge between duplicated parse_total_rows implementations", failures)

    matched_paths = {match.get("path") for match in report.get("slop_matches", [])}
    require("billing/invoice_totals.py" in matched_paths, "expected slop example match in billing totals", failures)
    require("payments/payment_totals.py" in matched_paths, "expected slop example match in payment totals", failures)
    return failures


def require(condition: object, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


if __name__ == "__main__":
    raise SystemExit(main())
