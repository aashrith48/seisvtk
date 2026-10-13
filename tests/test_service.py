"""Scene-manifest parsing for the deployable viewer entry point.

Only manifest resolution is tested (VTK-free); starting the trame server is not.
"""
import json

import numpy as np
import pytest

from seisvtk.io.ingest import write_array_to_zarr
from seisvtk.viz.service import load_scene

GEOM = {"iline_range": [100, 109], "xline_range": [300, 307],
        "z_range": [800.0, 1000.0], "sample_interval": 4.0}


def _cube(tmp_path):
    out = str(tmp_path / "cube.zarr")
    write_array_to_zarr(out, np.ones((10, 8, 51), dtype="float32"), GEOM)
    return out


def _manifest(tmp_path, **extra):
    m = {
        "zarr": _cube(tmp_path),
        "mode": "property", "colormap": "Viridis", "z_unit": "ms",
        "value_label": "GR_R", "well_radius": 15.0,
        "horizons": [{"name": "TopA", "il": [100, 101], "xl": [300, 301],
                      "z": [850.0, 860.0]}],
        "wells": [{"name": "W1", "il": [105] * 5, "xl": [303] * 5,
                   "z": [700.0, 850.0, 900.0, 950.0, 1100.0],
                   "logs": {"GR": [10.0, 15.0, 20.0, 25.0, 30.0]}}],
    }
    m.update(extra)
    p = tmp_path / "scene.json"
    p.write_text(json.dumps(m))
    return str(p)


def test_load_scene_parses_cube_overlays_and_kwargs(tmp_path):
    zarr, kwargs, horizons, wells = load_scene(_manifest(tmp_path))
    assert zarr.endswith("cube.zarr")
    assert kwargs["mode"] == "property" and kwargs["value_label"] == "GR_R"
    assert [h.name for h in horizons] == ["TopA"]
    assert [w.name for w in wells] == ["W1"]


def test_load_scene_clips_wells_to_cube_zrange_by_default(tmp_path):
    _, _, _, wells = load_scene(_manifest(tmp_path))
    # cube z-range is 800..1000; the 700 and 1100 samples are dropped
    assert np.all(wells[0].z >= 800.0) and np.all(wells[0].z <= 1000.0)
    assert wells[0].logs["GR"].tolist() == [15.0, 20.0, 25.0]


def test_load_scene_can_disable_clipping(tmp_path):
    _, _, _, wells = load_scene(_manifest(tmp_path, clip_wells_to_cube=False))
    assert wells[0].z.tolist() == [700.0, 850.0, 900.0, 950.0, 1100.0]


def test_load_scene_accepts_a_dict(tmp_path):
    with open(_manifest(tmp_path)) as f:
        manifest = json.load(f)
    zarr, _, horizons, _ = load_scene(manifest)
    assert horizons[0].name == "TopA"


def test_load_scene_well_from_npz_sidecar(tmp_path):
    npz = tmp_path / "w.npz"
    np.savez(npz, il=[105, 105], xl=[303, 303], z=[850.0, 900.0], GR=[5.0, 6.0])
    m = {"zarr": _cube(tmp_path), "clip_wells_to_cube": False,
         "wells": [{"name": "Wnpz", "npz": str(npz), "logs": ["GR"]}]}
    _, _, _, wells = load_scene(m)
    assert wells[0].name == "Wnpz"
    assert wells[0].logs["GR"].tolist() == [5.0, 6.0]
