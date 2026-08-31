"""Run the Norma component benchmark over image pairs from a JSON dataset."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from .norma_benchmark import run_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to the benchmark dataset JSON.")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--tag", action="append", dest="tags")
    parser.add_argument("--write-remap", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _runtime_metadata() -> dict[str, Any]:
    revision = None
    script_path = Path(__file__).resolve()
    repo_root = next(
        (parent for parent in script_path.parents
         if (parent / ".git").exists()), None)
    if repo_root is not None:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root.as_posix()}",
             "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            revision = completed.stdout.strip() or None
    return {
        "git_revision": revision,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    matching = result.get("matching", {})
    remap = result.get("remap_validation", {})
    return {
        "id": result["id"],
        "success": result["success"],
        "matching_path": result.get("matching_path"),
        "random_seed": result.get("random_seed"),
        "final_pairs": matching.get("final_pairs"),
        "coverage_ratio": matching.get("coverage_ratio"),
        "outer_pairs": matching.get("outer_pairs"),
        "median_px": matching.get("median_px"),
        "p90_px": matching.get("p90_px"),
        "remap_status": remap.get("status"),
        "remap_evaluated_tiles": remap.get("evaluated_tiles"),
        "remap_evaluated_coverage": remap.get("evaluated_tile_coverage"),
        "remap_residual_shift_p90_px": remap.get(
            "residual_shift_p90_px"),
        "total_seconds": result.get("timing_seconds", {}).get("total"),
        "expectations_passed": result.get("expectations_passed"),
        "error": result.get("error", {}).get("message"),
    }


def _write_results(output_dir: Path, dataset_path: Path,
                   results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(dataset_path),
        "generated_at_epoch": time.time(),
        "runtime": _runtime_metadata(),
        "results": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    rows = [_summary_row(result) for result in results]
    if rows:
        with (output_dir / "summary.csv").open(
                "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _select_cases(payload: dict[str, Any], case_ids: list[str] | None,
                  tags: list[str] | None) -> list[dict[str, Any]]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Dataset root must contain a cases array")
    requested_ids = set(case_ids or [])
    requested_tags = set(tags or [])
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or "id" not in case:
            raise ValueError("Every case must be an object with an id")
        case_id = str(case["id"])
        if case_id in seen_ids:
            raise ValueError(f"Duplicate benchmark case id: {case_id}")
        seen_ids.add(case_id)
        if requested_ids and case_id not in requested_ids:
            continue
        labels = case.get("labels", [])
        label_set = set(labels if isinstance(labels, list) else labels.keys())
        if requested_tags and not requested_tags.issubset(label_set):
            continue
        selected.append(case)
    if not selected:
        raise ValueError("No benchmark cases matched the requested filters")
    return selected


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset root must be an object")
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be an object")
    selected = _select_cases(payload, args.case_ids, args.tags)
    output_dir = Path(args.output_dir).resolve()
    results: list[dict[str, Any]] = []
    for index, case in enumerate(selected, 1):
        logger.info("Benchmark case {}/{}: {}", index, len(selected), case["id"])
        result = run_case(
            case, defaults, dataset_path.parent, output_dir,
            args.write_remap, random_seed=args.seed)
        results.append(result)
        _write_results(output_dir, dataset_path, results)
        if result["success"]:
            logger.info(
                "Case {} succeeded: pairs={} p90={:.4f}px total={:.2f}s",
                result["id"], result["matching"]["final_pairs"],
                result["matching"]["p90_px"],
                result["timing_seconds"]["total"])
        else:
            logger.error("Case {} failed: {}", result["id"],
                         result["error"]["message"])
            if args.fail_fast:
                break

    failed = sum(
        not result["success"]
        and not (isinstance(result.get("expected"), dict)
                 and result["expected"].get("success") is False)
        for result in results)
    expectation_failures = sum(
        result.get("expectations_passed") is False for result in results)
    logger.info(
        "Benchmark complete: cases={} failed={} expectation_failures={} output={}",
        len(results), failed, expectation_failures, output_dir)
    return 1 if failed or expectation_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
