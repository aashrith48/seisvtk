"""Horizon readers: delimiters, IL/XL, rows (multi-horizon), columns, XYZ."""
import numpy as np
import pytest

from seisvtk.io.horizon import (Horizon, read_horizon_columns, read_horizon_ilxl,
                                 read_horizon_rows, read_horizon_xyz)
from seisvtk.io.survey import GridTransform


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_from_arrays_drops_nulls_and_grids_holes():
    # (11,21) carries the null, so it drops out and becomes a hole in the grid
    h = Horizon.from_arrays("H", il=[10, 10, 11, 11], xl=[20, 21, 20, 21],
                            z=[100.0, 101.0, 102.0, -999.25])
    assert h.il.size == 3                                  # null pick removed
    iu, xu, Z = h.to_grid()
    assert iu.tolist() == [10, 11]
    assert xu.tolist() == [20, 21]
    assert np.isnan(Z[1, 1])                               # (11,21) null -> hole
    assert Z[0, 0] == 100.0


def test_read_horizon_ilxl_comma(tmp_path):
    path = _write(tmp_path, "h.csv", "IL,XL,Z\n100,200,850\n100,201,852\n")
    h = read_horizon_ilxl(path)
    assert h.il.tolist() == [100, 100]
    assert h.xl.tolist() == [200, 201]
    assert np.allclose(h.z, [850, 852])


def test_read_horizon_ilxl_zscale(tmp_path):
    path = _write(tmp_path, "h.txt", "100 200 0.85\n100 201 0.852\n")   # seconds
    h = read_horizon_ilxl(path, z_scale=1000.0)
    assert np.allclose(h.z, [850, 852])                   # -> ms


def test_read_horizon_rows_multiple(tmp_path):
    txt = ("Name,IL,XL,Z\n"
           "Top,100,200,850\nTop,100,201,852\n"
           "Base,100,200,980\nBase,100,201,982\n")
    hs = read_horizon_rows(_write(tmp_path, "rows.csv", txt))
    assert [h.name for h in hs] == ["Top", "Base"]        # first-seen order
    assert np.allclose(hs[0].z, [850, 852])
    assert np.allclose(hs[1].z, [980, 982])


def test_read_horizon_columns(tmp_path):
    txt = "IL XL Top Base\n100 200 850 980\n100 201 852 982\n"
    hs = read_horizon_columns(_write(tmp_path, "cols.txt", txt))
    assert [h.name for h in hs] == ["Top", "Base"]
    assert np.allclose(hs[0].z, [850, 852])
    assert hs[1].il.tolist() == [100, 100]


def test_read_horizon_xyz_inverts_transform(tmp_path):
    # identity survey: world (x, y) == grid (il, xl)
    xf = GridTransform.fit(il=[0, 1, 0, 1], xl=[0, 0, 1, 1],
                           x=[0, 1, 0, 1], y=[0, 0, 1, 1])
    path = _write(tmp_path, "xyz.txt", "10 20 850\n11 20 852\n")
    h = read_horizon_xyz(path, xf)
    assert h.il.tolist() == [10, 11]
    assert h.xl.tolist() == [20, 20]
    assert np.allclose(h.z, [850, 852])
