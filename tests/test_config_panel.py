from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from ui.output_panel import OutputPanel
from ui.panel_builder import DynamicConfigPanel, PanelSchema
from ui.widgets import ConfigSpec


@pytest.fixture(scope="module")
def qt_app():
    import os

    previous = os.environ.get("QT_QPA_PLATFORM")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication([])
    yield app
    if previous is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = previous


@pytest.mark.asyncio
@pytest.mark.parametrize("align_mode", ["simple", "bundle"])
@pytest.mark.parametrize("enabled,override,expected_mode,expected_weight", [
    (True, {}, "native_relaxed", False),
    (False, {}, "auto", True),
    (True, {"star_detection_mode": "auto", "int_weight": True}, "auto", True),
])
async def test_hidden_configs_use_backend_defaults_like_cli(
        qt_app, tmp_path, monkeypatch, align_mode, enabled, override,
        expected_mode, expected_weight):
    import hoshicore.engine.wiring as wiring

    pipeline = Path(__file__).resolve().parents[1] / "hoshicore/dag/aligned_stack.meta.yaml"
    schema = PanelSchema.from_yaml(pipeline, pipeline.with_name("aligned_stack.ui.yaml"))
    schema.routes["align_mode"].default = align_mode
    panel = DynamicConfigPanel()
    panel.load_schema(schema)
    gui_configs = panel.collect_configs()
    hidden_keys = {spec.key for spec in schema.configs if spec.hidden}
    assert hidden_keys.isdisjoint(gui_configs)
    # Visible controls still submit values, even without a user edit.
    assert gui_configs["enable_ground"] is True

    raw_settings = {
        "star_detection_mode": {"enabled": enabled, "value": "native_relaxed"},
        "int_weight": {"enabled": enabled, "value": False},
        "star_detection_recovery": {"enabled": True, "value": False},
    }
    monkeypatch.setattr(wiring, "_read_raw_settings", lambda path=None: raw_settings)
    captured = []

    async def capture_run(dag, inputs, configs, *args, **kwargs):
        captured.append(configs.copy())
        return {}

    monkeypatch.setattr(wiring, "run_dag", capture_run)
    frame = tmp_path / "frame.tif"
    tifffile.imwrite(frame, np.zeros((10, 10, 3), dtype=np.uint16))
    output = {"output_filename": str(tmp_path / "output.tif")}
    routes = panel.collect_route_choices()
    # Exercise the shared GUI/CLI entry point; only image processing is stubbed.
    for configs in ({**gui_configs, **output, **override}, {**output, **override}):
        await wiring.run_from_yaml(
            str(pipeline), {"fnames": [str(frame)]}, configs,
            route_choices=routes, progress=False)

    for configs in captured:
        assert configs["star_detection_mode"] == expected_mode
        assert configs["int_weight"] is expected_weight
        assert configs["star_detection_recovery"] is False
    assert {key: captured[0][key] for key in hidden_keys} == {
        key: captured[1][key] for key in hidden_keys}
    panel.deleteLater()


def test_output_panel_submits_hidden_output_configs(qt_app):
    pipeline = Path(__file__).resolve().parents[1] / "hoshicore/dag/aligned_stack.meta.yaml"
    schema = PanelSchema.from_yaml(pipeline, pipeline.with_name("aligned_stack.ui.yaml"))
    panel = DynamicConfigPanel()
    panel.load_schema(schema)
    output = OutputPanel()
    output.load_specs(schema.outputs)
    output.apply_defaults({"output_format": "PNG", "output_dtype": "uint16",
                           "png_compressing": 3})
    output._regions[0]._path_edit.setText("output.png")
    configs = panel.collect_configs()
    assert "output_filename" not in configs
    assert "output_dtype" not in configs
    assert "png_compressing" not in configs
    configs.update(output.collect_outputs())
    assert configs["output_filename"] == "output.png"
    assert configs["output_dtype"] == "uint16"
    assert configs["png_compressing"] == 3
    panel.deleteLater()
    output.deleteLater()


def test_visible_when_does_not_omit_config(qt_app):
    schema = PanelSchema(meta_yaml_path="unused", configs=[
        ConfigSpec(key="show_option", type="bool", widget="switch", default=False),
        ConfigSpec(key="option", type="bool", widget="switch", default=True,
                   visible_when={"key": "show_option", "eq": True}),
    ])
    panel = DynamicConfigPanel()
    panel.load_schema(schema)
    assert panel._config_widgets["option"].isHidden()
    assert panel.collect_configs() == {"show_option": False, "option": True}
    panel.deleteLater()
