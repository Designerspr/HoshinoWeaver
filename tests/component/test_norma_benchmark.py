import json
import sys

import pytest

from benchmarks.norma_benchmark import _case_config, _expectation_checks


def test_remap_expectation_requires_sufficient_support():
    result = {
        "success": True,
        "matching": {
            "final_pairs": 100,
            "coverage_ratio": 0.5,
            "outer_pairs": 20,
            "p90_px": 0.2,
        },
        "remap_validation": {
            "status": "insufficient_support",
            "evaluated_tiles": 0,
            "evaluated_tile_coverage": 0.0,
            "residual_shift_p90_px": None,
        },
    }

    checks = _expectation_checks(
        {"max_residual_shift_p90_px": 1.0}, result)

    by_name = {check["name"]: check for check in checks}
    assert by_name["remap_status"]["passed"] is False
    assert by_name["max_residual_shift_p90_px"]["passed"] is False


def test_case_config_rejects_options_not_exposed_by_norma():
    with pytest.raises(ValueError, match="Unsupported alignment options"):
        _case_config({
            "alignment": {"guided_refine_mode": "triangle"},
        }, {})


def test_main_rejects_empty_case_selection(tmp_path, monkeypatch):
    import benchmarks.benchmark_norma_alignment as benchmark

    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({
        "cases": [{"id": "present", "reference": "a", "source": "b"}],
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "benchmarks.benchmark_norma_alignment", str(dataset),
        "--case", "missing",
    ])

    with pytest.raises(ValueError, match="No benchmark cases matched"):
        benchmark.main()


def test_main_rejects_duplicate_case_ids(tmp_path, monkeypatch):
    import benchmarks.benchmark_norma_alignment as benchmark

    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({
        "cases": [
            {"id": "duplicate", "reference": "a", "source": "b"},
            {"id": "duplicate", "reference": "c", "source": "d"},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "benchmarks.benchmark_norma_alignment", str(dataset),
    ])

    with pytest.raises(ValueError, match="Duplicate benchmark case id"):
        benchmark.main()


def test_expected_failure_does_not_fail_process(tmp_path, monkeypatch):
    import benchmarks.benchmark_norma_alignment as benchmark

    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({
        "cases": [{
            "id": "expected-failure",
            "reference": "a",
            "source": "b",
            "expected": {"success": False},
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "benchmarks.benchmark_norma_alignment", str(dataset),
        "--output-dir", str(tmp_path / "out"),
    ])
    monkeypatch.setattr(benchmark, "run_case", lambda *args, **kwargs: {
        "id": "expected-failure",
        "success": False,
        "expected": {"success": False},
        "expectations_passed": True,
        "error": {"message": "expected"},
    })
    monkeypatch.setattr(benchmark, "_write_results", lambda *args: None)

    assert benchmark.main() == 0


def test_cli_defaults_to_fixed_seed(monkeypatch):
    import benchmarks.benchmark_norma_alignment as benchmark

    monkeypatch.setattr(sys, "argv", [
        "benchmarks.benchmark_norma_alignment", "dataset.json",
    ])
    assert benchmark.parse_args().seed == 0


def test_summary_row_contains_only_core_metrics():
    import benchmarks.benchmark_norma_alignment as benchmark

    row = benchmark._summary_row({
        "id": "compact",
        "success": True,
        "matching_path": "asterism",
        "random_seed": 0,
        "matching": {
            "final_pairs": 30,
            "coverage_ratio": 0.4,
            "outer_pairs": 5,
            "median_px": 0.2,
            "p90_px": 0.5,
        },
        "remap_validation": {
            "status": "ok",
            "evaluated_tiles": 4,
            "evaluated_tile_coverage": 0.6,
            "residual_shift_p90_px": 0.75,
        },
        "timing_seconds": {"total": 1.5},
        "expectations_passed": True,
    })

    assert row["final_pairs"] == 30
    assert row["p90_px"] == pytest.approx(0.5)
    assert row["remap_residual_shift_p90_px"] == pytest.approx(0.75)
    assert "first_rotation_x_deg" not in row
