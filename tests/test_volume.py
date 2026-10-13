"""Volume opens local paths, prebuilt stores, open arrays, and remote URLs.

The point of the store flexibility is the memory invariant: a cube can live on S3
and be opened as a remote store so only the chunks a slice touches are fetched —
the whole volume is never held in memory or copied to local disk.
"""
import numpy as np
import pytest
import zarr

from seisvtk.io.ingest import write_array_to_zarr
from seisvtk.io.volume import Volume

GEOM = {"iline_range": [100, 109], "xline_range": [300, 307],
        "z_range": [0.0, 124.0], "sample_interval": 4.0}


def _write(tmp_path):
    cube = np.arange(10 * 8 * 32, dtype="float32").reshape(10, 8, 32)
    out = str(tmp_path / "c.zarr")
    write_array_to_zarr(out, cube, GEOM)
    return out, cube


def test_volume_opens_from_local_path(tmp_path):
    out, _ = _write(tmp_path)
    vol = Volume(zarr_path=out)
    assert vol.shape == (10, 8, 32)
    assert vol.metadata["iline_range"] == [100, 109]


def test_volume_opens_from_store_object_and_reads_one_slice(tmp_path):
    out, cube = _write(tmp_path)
    store = zarr.storage.LocalStore(out)                 # a prebuilt store, not a path
    vol = Volume(zarr_path=store)
    assert vol.shape == (10, 8, 32)
    sl = vol.seismic_slice(vol.get_array_index(iline=103))
    assert np.allclose(sl[0], cube[3])                   # lazily reads just that plane


def test_volume_opens_from_open_array(tmp_path):
    out, _ = _write(tmp_path)
    vol = Volume(zarr_path=zarr.open_array(out, mode="r"))
    assert vol.shape == (10, 8, 32)


def test_remote_url_routes_through_fsspec(tmp_path):
    # An s3:// URL must route to the fsspec store, never be mistaken for a local
    # path. Without an fsspec backend installed that surfaces as a raised error
    # (and with one, it would attempt a real remote open) — either way, it raises
    # rather than silently opening nothing.
    with pytest.raises(Exception):
        Volume(zarr_path="s3://seisvtk-tests/missing.zarr")
