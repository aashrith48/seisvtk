"""Generic array -> chunked Zarr ingest: geometry attrs, live mask, NaN, stride."""
import numpy as np
import pytest
import zarr

from seisvtk.io.ingest import write_array_to_zarr
from seisvtk.io.survey import GridTransform
from seisvtk.io.volume import Volume

GEOM = {
    "iline_range": [100, 109],
    "xline_range": [300, 307],
    "z_range": [0.0, 124.0],
    "sample_interval": 4.0,
}


def test_round_trip_attrs_and_values(tmp_path):
    rng = np.random.default_rng(0)
    cube = rng.standard_normal((10, 8, 32)).astype("float32")
    cube[0, 0, :] = np.nan                          # a dead trace
    out = str(tmp_path / "c.zarr")
    attrs = write_array_to_zarr(out, cube, GEOM)

    assert attrs["iline_range"] == [100, 109]
    assert attrs["sample_interval"] == 4.0
    assert attrs["grid_cells"] == 80
    assert attrs["live_traces"] == 79               # one dead trace
    assert attrs["missing_traces"] == 1

    z = zarr.open_array(out, mode="r")
    back = z[:]
    live = zarr.open_array(f"{out}/live", mode="r")[:]
    assert back.shape == (10, 8, 32)
    assert not live[0, 0]                            # dead trace masked
    assert live[1, 1]
    finite = np.isfinite(cube)
    assert np.allclose(back[finite], cube[finite])
    assert np.isnan(back[0, 0, 0])                  # NaN preserved


def test_stride_decimates(tmp_path):
    cube = np.ones((8, 8, 16), dtype="float32")
    out = str(tmp_path / "s.zarr")
    geom = {**GEOM, "iline_range": [100, 106], "xline_range": [300, 306]}
    write_array_to_zarr(out, cube, geom, stride=(2, 2, 2))
    z = zarr.open_array(out, mode="r")
    assert z.shape == (4, 4, 8)                      # halved on each axis


def test_survey_transform_round_trips_to_volume(tmp_path):
    # a host that passes its georeference must get it back through Volume.transform
    xf = GridTransform.from_coeffs(600000, 6070000, 24.0, -7.0, 7.0, 24.0)
    cube = np.ones((10, 8, 32), dtype="float32")
    out = str(tmp_path / "t.zarr")
    write_array_to_zarr(out, cube, {**GEOM, "transform": xf})
    vol = Volume(zarr_path=out)
    assert vol.transform is not None
    il, xl = vol.transform.world_to_grid(*xf.grid_to_world(103, 305))
    assert il == pytest.approx(103) and xl == pytest.approx(305)


def test_no_transform_when_geometry_omits_it(tmp_path):
    cube = np.ones((10, 8, 32), dtype="float32")
    out = str(tmp_path / "n.zarr")
    write_array_to_zarr(out, cube, GEOM)
    assert Volume(zarr_path=out).transform is None
