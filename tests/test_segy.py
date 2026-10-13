"""SEG-Y -> Zarr conversion: auto geometry, explicit bytes, Volume round-trip.

Skips entirely where ``segyio`` is not installed (the slice viewer does not need
it), building a tiny synthetic 3D SEG-Y on the fly for the cases that do run.
"""
import numpy as np
import pytest

segyio = pytest.importorskip("segyio")

from seisvtk.io.loaders import convert_segy, suggest_zarr_path
from seisvtk.io.segy2zarr import _fit_survey_transform
from seisvtk.io.survey import GridTransform
from seisvtk.io.volume import Volume

_GEO = GridTransform.from_coeffs(1000.0, 2000.0, 24.0, -7.0, 7.0, 24.0)


def _grid_headers(nil, nxl, il0=100, xl0=300):
    """Inline/crossline arrays in XL-fastest trace order (the usual SEG-Y layout)."""
    il = np.repeat(np.arange(il0, il0 + nil), nxl).astype(float)
    xl = np.tile(np.arange(xl0, xl0 + nxl), nil).astype(float)
    return il, xl

IL0, XL0, NIL, NXL, NS, DT_US = 100, 300, 6, 5, 20, 4000
COORD_SCALAR = -100                                   # SEG-Y: divide stored ints by 100


def _make_segy(path, *, nil=NIL, nxl=NXL, ns=NS, xf=None):
    """Write a small dense 3D SEG-Y with standard IL/XL headers (189/193).

    With ``xf`` (a :class:`GridTransform`), also writes CDP-X/Y (181/185) for each
    trace from that affine, scaled by ``COORD_SCALAR`` — so the converter can
    recover the survey georeference from the headers.
    """
    spec = segyio.spec()
    spec.format = 5                                   # 4-byte IEEE float
    spec.samples = np.arange(ns) * (DT_US / 1000.0)   # ms
    spec.ilines = np.arange(IL0, IL0 + nil)
    spec.xlines = np.arange(XL0, XL0 + nxl)
    spec.sorting = 2
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = DT_US
        tr = 0
        for i, il in enumerate(spec.ilines):
            for x, xl in enumerate(spec.xlines):
                hdr = {segyio.su.iline: int(il), segyio.su.xline: int(xl),
                       segyio.su.ns: ns, segyio.su.dt: DT_US}
                if xf is not None:
                    wx, wy = xf.grid_to_world(int(il), int(xl))
                    hdr[segyio.su.scalco] = COORD_SCALAR
                    hdr[segyio.su.cdpx] = int(round(float(wx) * 100))
                    hdr[segyio.su.cdpy] = int(round(float(wy) * 100))
                f.header[tr] = hdr
                f.trace[tr] = np.full(ns, i * 10 + x, dtype=np.float32)
                tr += 1
    return str(path)


def test_convert_auto_geometry(tmp_path):
    segy = _make_segy(tmp_path / "cube.segy")
    out = str(tmp_path / "cube.zarr")
    attrs = convert_segy(segy, out)                   # iline/xline auto-detected
    assert attrs["iline_range"] == [IL0, IL0 + NIL - 1]
    assert attrs["xline_range"] == [XL0, XL0 + NXL - 1]
    assert attrs["sample_interval"] == pytest.approx(4.0)
    assert attrs["grid_cells"] == NIL * NXL
    assert attrs["live_traces"] == NIL * NXL          # dense survey


def test_convert_progress_is_called(tmp_path):
    segy = _make_segy(tmp_path / "c.segy")
    seen = []
    convert_segy(segy, str(tmp_path / "c.zarr"),
                 progress=lambda done, total: seen.append((done, total)))
    assert seen, "progress callback never fired"
    assert seen[-1][0] == seen[-1][1]                 # ends at 100%


def test_convert_volume_round_trip(tmp_path):
    segy = _make_segy(tmp_path / "v.segy")
    out = str(tmp_path / "v.zarr")
    convert_segy(segy, out)
    vol = Volume(zarr_path=out)
    assert vol.shape == (NIL, NXL, NS)
    # trace (i=2, x=3) was filled with the constant i*10+x = 23; the inline slab
    # keeps a length-1 inline axis, so it is sl[0, xl].
    sl = vol.seismic_slice(vol.get_array_index(iline=IL0 + 2))
    assert np.allclose(sl[0, 3], 23.0)


def test_convert_explicit_bytes(tmp_path):
    segy = _make_segy(tmp_path / "e.segy")
    attrs = convert_segy(segy, str(tmp_path / "e.zarr"),
                         iline=segyio.TraceField.INLINE_3D,
                         xline=segyio.TraceField.CROSSLINE_3D)
    assert attrs["iline_range"] == [IL0, IL0 + NIL - 1]


def test_convert_missing_file(tmp_path):
    with pytest.raises(ValueError):
        convert_segy(str(tmp_path / "nope.segy"), str(tmp_path / "x.zarr"))


def test_convert_recovers_survey_transform(tmp_path):
    # CDP X/Y in the headers -> a world affine the cube carries, so a well's
    # world (X, Y) ties straight to inline/crossline.
    xf = GridTransform.from_coeffs(600000.0, 6070000.0, 24.0, -7.0, 7.0, 24.0)
    out = str(tmp_path / "geo.zarr")
    convert_segy(_make_segy(tmp_path / "geo.segy", xf=xf), out)
    vol = Volume(zarr_path=out)
    assert vol.transform is not None
    x, y = xf.grid_to_world(IL0 + 2, XL0 + 3)
    il, xl = vol.transform.world_to_grid(x, y)
    assert il == pytest.approx(IL0 + 2, abs=1e-2)
    assert xl == pytest.approx(XL0 + 3, abs=1e-2)


def test_convert_without_coords_has_no_transform(tmp_path):
    out = str(tmp_path / "flat.zarr")
    convert_segy(_make_segy(tmp_path / "flat.segy"), out)   # no CDP X/Y written
    assert Volume(zarr_path=out).transform is None


def test_fit_survives_stride_aligned_with_crossline_count():
    # XL-fastest order with max_points forcing step == n_xl: a naive stride would
    # sample a single crossline and fit a rank-deficient (wrong) affine.
    nil, nxl = 20, 5
    il, xl = _grid_headers(nil, nxl)
    x, y = _GEO.grid_to_world(il, xl)
    d = _fit_survey_transform(il, xl, x, y, max_points=nil)      # would-be step = nxl
    assert d is not None
    got = GridTransform.from_dict(d)
    ril, rxl = got.world_to_grid(*_GEO.grid_to_world(110.0, 303.0))
    assert ril == pytest.approx(110, abs=1e-6) and rxl == pytest.approx(303, abs=1e-6)


def test_fit_masks_zero_coordinate_traces():
    il, xl = _grid_headers(10, 4)
    x, y = _GEO.grid_to_world(il, xl)
    x[0] = y[0] = 0.0                                            # dead / padding traces
    x[13] = y[13] = 0.0
    got = GridTransform.from_dict(_fit_survey_transform(il, xl, x, y))
    ril, rxl = got.world_to_grid(*_GEO.grid_to_world(105.0, 302.0))
    assert ril == pytest.approx(105, abs=1e-6) and rxl == pytest.approx(302, abs=1e-6)


def test_fit_none_when_single_line_or_all_zero():
    il, xl = _grid_headers(1, 8)                                 # one inline only
    x, y = _GEO.grid_to_world(il, xl)
    assert _fit_survey_transform(il, xl, x, y) is None
    il2, xl2 = _grid_headers(5, 4)
    zeros = np.zeros(il2.size)
    assert _fit_survey_transform(il2, xl2, zeros, zeros) is None


def test_suggest_zarr_path():
    assert suggest_zarr_path(r"D:\data\survey.segy") == r"D:\data\survey.zarr"
    assert suggest_zarr_path("/x/y/CUBE.SGY") == "/x/y/CUBE.zarr"
