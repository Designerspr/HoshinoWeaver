from unittest.mock import Mock

from hoshicore.component import runtime_diagnostics


def test_runtime_component_probe_has_expected_sections():
    report = runtime_diagnostics.probe_runtime_components()

    assert set(report) == {
        "compiled_custom_ops", "cuda_runtime", "metal_runtime",
        "logical_ops", "modules",
    }
    assert report["compiled_custom_ops"]["status"] in {
        "available", "unavailable", "error",
    }
    assert "turbojpeg" in report["modules"]
    assert "pyexiv2" in report["modules"]
    assert report["logical_ops"]


def test_runtime_component_logging_runs_only_once(monkeypatch):
    report = {
        "compiled_custom_ops": {
            "status": "available", "compiler": "test", "openmp": True,
            "cuda": False,
        },
        "cuda_runtime": {
            "status": "not_built", "reason": "not built",
        },
        "metal_runtime": {
            "status": "unavailable", "reason": "not present",
        },
        "logical_ops": {
            "max_combine": {"backend": "openmp_cpu", "native": True,
                            "reason": None},
        },
        "modules": {
            "turbojpeg": {"status": "available"},
        },
    }
    monkeypatch.setattr(runtime_diagnostics, "_logged", False)
    monkeypatch.setattr(runtime_diagnostics, "_last_report", None)
    probe = Mock(return_value=report)
    monkeypatch.setattr(runtime_diagnostics, "probe_runtime_components", probe)
    logger = Mock()

    assert runtime_diagnostics.log_runtime_components(logger) is report
    assert runtime_diagnostics.log_runtime_components(logger) is None

    probe.assert_called_once_with()


def test_probe_reports_numpy_routes_when_compiled_module_is_absent(monkeypatch):
    monkeypatch.setattr(
        runtime_diagnostics, "load_compiled_module",
        lambda: (None, "module missing"))
    monkeypatch.setattr(
        runtime_diagnostics, "metal_device_info",
        lambda: {"available": False, "status": "unavailable"})
    monkeypatch.setattr(
        runtime_diagnostics, "_module_probe",
        lambda *args, **kwargs: {"status": "unavailable"})

    report = runtime_diagnostics.probe_runtime_components()

    assert report["compiled_custom_ops"]["status"] == "unavailable"
    assert report["cuda_runtime"]["status"] == "not_built"
    assert {item["backend"] for item in report["logical_ops"].values()} == {
        "numpy",
    }
