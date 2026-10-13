"""SliceScene runtime volume swap: shape/ranges update, overlays clear, renders."""
import numpy as np
import pytest

from seisvtk.io.horizon import Horizon
from seisvtk.io.ingest import write_array_to_zarr
from seisvtk.viz.slice_scene import SliceScene


def _zarr(tmp_path, name, shape, geom):
    arr = np.random.default_rng(0).standard_normal(shape).astype("float32")
    out = str(tmp_path / name)
    write_array_to_zarr(out, arr, geom)
    return out


G1 = {"iline_range": [100, 109], "xline_range": [300, 307],
      "z_range": [0.0, 124.0], "sample_interval": 4.0}
G2 = {"iline_range": [500, 515], "xline_range": [20, 33],
      "z_range": [0.0, 60.0], "sample_interval": 2.0}


def test_load_volume_swaps_shape_and_ranges(tmp_path):
    z1 = _zarr(tmp_path, "a.zarr", (10, 8, 32), G1)
    z2 = _zarr(tmp_path, "b.zarr", (16, 14, 31), G2)
    sc = SliceScene(z1, iline=105, xline=303, z=60.0, mode="amplitude")
    assert (sc.n_il, sc.n_xl, sc.n_samp) == (10, 8, 32)

    sc.load_volume(z2)
    assert (sc.n_il, sc.n_xl, sc.n_samp) == (16, 14, 31)
    assert sc.md["iline_range"] == [500, 515]
    assert sc.iline == (500 + 515) // 2          # planes recentre on the new cube
    sc.rw.Render()                               # renders the swapped cube cleanly


def test_load_volume_clears_overlays(tmp_path):
    z1 = _zarr(tmp_path, "a.zarr", (10, 8, 32), G1)
    z2 = _zarr(tmp_path, "b.zarr", (16, 14, 31), G2)
    sc = SliceScene(z1, iline=105, xline=303, z=60.0, mode="amplitude")
    sc.add_horizon(Horizon.from_arrays("H", il=[100, 101], xl=[300, 301],
                                       z=[40.0, 44.0]))
    assert sc.horizons == ["H"]
    sc.load_volume(z2)
    assert sc.horizons == []                     # old-grid overlays dropped
