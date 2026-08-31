from ui.runtime_capabilities_dialog import format_runtime_capabilities_html


def test_capability_dialog_shows_native_import_failure_reason():
    report = {
        "compiled_custom_ops": {
            "status": "unavailable",
            "reason": "ImportError: DLL initialization routine failed",
        },
        "cuda_runtime": {"status": "not_built"},
        "metal_runtime": {"status": "unavailable"},
        "logical_ops": {},
        "modules": {},
    }

    text = format_runtime_capabilities_html(report)

    assert "原生算子不可用" in text
    assert "DLL initialization routine failed" in text
