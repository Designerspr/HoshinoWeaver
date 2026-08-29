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
    spec = meta_resolve(spec, {"stacker": route})
    dag = validate_and_build_order(flatten_sub_dags(spec))
    _, feeders, _, _ = instantiate_and_wire(
        dag,
        {"fnames": ["a.tif", "b.tif"]},
        {"reference_frame_index": 0, "output_filename": "out.tif"},
    )
    for feeder in feeders:
        feeder.close()


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


@pytest.mark.parametrize(("name", "route_key"), [
    ("norma_bundle_stack", "stacker"),
    ("norma_bundle_window", "window_stacker"),
])
def test_bundle_workflow_ui_matches_meta(name, route_key):
    meta = _load_yaml(str(_DAG_DIR / f"{name}.meta.yaml"))
    ui = load_ui_yaml(_DAG_DIR / f"{name}.ui.yaml")

    assert set(ui["inputs"]) == set(meta["inputs"])
    assert set(ui["routes"]) == {route_key}
    assert set(ui["routes"][route_key]["options"]) == set(
        meta["routes"][route_key]["options"])
    assert set(ui["configs"]).issubset(meta["configs"])
