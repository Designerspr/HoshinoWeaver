from pathlib import Path

import pytest

from hoshicore.engine.build import _load_yaml, validate_and_build_order
from hoshicore.engine.flatten import flatten_sub_dags
from hoshicore.engine.meta import meta_resolve
from hoshicore.engine.wiring import instantiate_and_wire
from ui.yaml_loader import load_ui_yaml


_DAG_DIR = Path(__file__).parents[2] / "hoshicore" / "dag"


@pytest.mark.parametrize("route", [
    "mean", "sigma_clip", "median", "huber_mean", "max_mix", "max",
])
def test_bundle_reference_stack_dag_wires(route):
    spec = _load_yaml(str(_DAG_DIR / "norma_bundle_stack.meta.yaml"))
    spec = meta_resolve(spec, {"stacker": route, "ground_stacker": route})
    dag = validate_and_build_order(flatten_sub_dags(spec))
    _, feeders, _, _ = instantiate_and_wire(
        dag,
        {"fnames": ["a.tif", "b.tif"]},
        {"reference_frame_index": 0, "output_filename": "out.tif"},
    )
    for feeder in feeders:
        feeder.close()


def test_bundle_reference_stack_dag_wires_without_ground():
    spec = _load_yaml(str(_DAG_DIR / "norma_bundle_stack.meta.yaml"))
    spec = meta_resolve(
        spec, {"stacker": "mean", "ground_stacker": "mean"},
        {"enable_ground": False})
    dag = validate_and_build_order(flatten_sub_dags(spec))
    _, feeders, _, _ = instantiate_and_wire(
        dag,
        {"fnames": ["a.tif", "b.tif"]},
        {"reference_frame_index": 0, "output_filename": "out.tif",
         "enable_ground": False},
    )
    for feeder in feeders:
        feeder.close()


def test_bundle_reference_stack_keeps_ground_outside_remap_path():
    spec = _load_yaml(str(_DAG_DIR / "norma_bundle_stack.meta.yaml"))

    assert spec["nodes"]["bundle_adjust"]["inputs"]["data"] == (
        "ba_sky_mask.result")
    assert spec["nodes"]["sky_stacker"]["inputs"]["data"] == (
        "bundle_remap.result")
    assert spec["nodes"]["ground_stacker"]["inputs"]["data"] == (
        "remap_data_loader.result")
    assert spec["nodes"]["image_add"]["configs"] == {
        "image_a": "sky_apply_mask.result",
        "image_b": "ground_apply_mask.result",
    }


@pytest.mark.parametrize("route", [
    "mean", "max", "median", "sigma_clip",
])
def test_bundle_window_dag_wires(route):
    spec = _load_yaml(str(_DAG_DIR / "norma_bundle_window.meta.yaml"))
    spec = meta_resolve(spec, {"window_stacker": route})
    dag = validate_and_build_order(flatten_sub_dags(spec))
    _, feeders, _, _ = instantiate_and_wire(
        dag,
        {"fnames": ["a.tif", "b.tif"]},
        {"output_dir": "output"},
    )
    for feeder in feeders:
        feeder.close()


@pytest.mark.parametrize("name", [
    "norma_bundle_stack",
    "norma_bundle_window",
])
def test_bundle_workflow_ui_matches_meta(name):
    meta = _load_yaml(str(_DAG_DIR / f"{name}.meta.yaml"))
    ui = load_ui_yaml(_DAG_DIR / f"{name}.ui.yaml")

    assert set(ui["inputs"]) == set(meta["inputs"])
    assert set(ui["routes"]) == set(meta["routes"])
    for route_key, route in meta["routes"].items():
        assert set(ui["routes"][route_key]["options"]) == set(
            route["options"])
    assert set(ui["configs"]).issubset(meta["configs"])
