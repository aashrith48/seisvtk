"""Panel orchestration: path + options -> Horizon / Well model objects."""
import numpy as np
import pytest

from seisvtk.io.loaders import (HORIZON_LAYOUTS, default_log, load_horizons,
                                 load_well)
from seisvtk.io.survey import GridTransform


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ---- horizons ----------------------------------------------------------------
def test_load_horizons_ilxl_returns_single_list(tmp_path):
    path = _write(tmp_path, "h.csv", "IL,XL,Z\n100,200,850\n100,201,852\n")
    hs = load_horizons(path, "ilxl")
    assert len(hs) == 1
    assert hs[0].il.tolist() == [100, 100]


def test_load_horizons_rows_splits_by_name(tmp_path):
    txt = ("Name,IL,XL,Z\nTop,100,200,850\nTop,100,201,852\n"
           "Base,100,200,980\nBase,100,201,982\n")
    hs = load_horizons(_write(tmp_path, "rows.csv", txt), "rows")
    assert [h.name for h in hs] == ["Top", "Base"]


def test_load_horizons_columns_one_per_z(tmp_path):
    txt = "IL XL Top Base\n100 200 850 980\n100 201 852 982\n"
    hs = load_horizons(_write(tmp_path, "cols.txt", txt), "columns")
    assert [h.name for h in hs] == ["Top", "Base"]


def test_load_horizons_zscale_applies(tmp_path):
    path = _write(tmp_path, "h.txt", "100 200 0.85\n100 201 0.852\n")   # seconds
    (h,) = load_horizons(path, "ilxl", z_scale=1000.0)
    assert np.allclose(h.z, [850, 852])                                 # -> ms


def test_load_horizons_xyz_needs_transform(tmp_path):
    path = _write(tmp_path, "xyz.txt", "10 20 850\n11 20 852\n")
    with pytest.raises(ValueError):
        load_horizons(path, "xyz")                                      # no transform


def test_load_horizons_xyz_with_transform(tmp_path):
    xf = GridTransform.fit(il=[0, 1, 0, 1], xl=[0, 0, 1, 1],
                           x=[0, 1, 0, 1], y=[0, 0, 1, 1])              # identity
    path = _write(tmp_path, "xyz.txt", "10 20 850\n11 20 852\n")
    (h,) = load_horizons(path, "xyz", transform=xf)
    assert h.il.tolist() == [10, 11]


def test_load_horizons_bad_layout(tmp_path):
    path = _write(tmp_path, "h.csv", "IL,XL,Z\n100,200,850\n")
    with pytest.raises(ValueError):
        load_horizons(path, "nope")


def test_horizon_layouts_constant():
    assert set(HORIZON_LAYOUTS) == {"ilxl", "rows", "columns", "xyz"}


# ---- wells -------------------------------------------------------------------
def test_load_well_vertical_depth_cube(tmp_path):
    path = _write(tmp_path, "l.csv", "MD,GR,RHOB\n100,55,2.3\n110,60,2.4\n120,65,2.5\n")
    w = load_well(path, il=250, xl=500)
    assert np.allclose(w.z, [100, 110, 120])            # depth cube: z == md
    assert w.il.tolist() == [250, 250, 250]             # placed at (il, xl)
    assert w.xl.tolist() == [500, 500, 500]
    assert set(w.logs) == {"GR", "RHOB"}


def test_load_well_name_defaults_to_stem(tmp_path):
    path = _write(tmp_path, "A-12.csv", "MD,GR\n10,5\n20,6\n")
    assert load_well(path, il=1, xl=1).name == "A-12"


def test_load_well_checkshot_maps_time(tmp_path):
    logs = _write(tmp_path, "l.csv", "MD,GR\n100,5\n200,6\n")
    cs = _write(tmp_path, "cs.txt", "100 1.0\n200 2.0\n")   # md -> twt seconds
    w = load_well(logs, il=1, xl=1, checkshot=cs, checkshot_unit="s")
    assert np.allclose(w.z, [1000.0, 2000.0])              # seconds -> ms twt
    assert not np.allclose(w.z, [100, 200])                # not raw depth


def test_load_well_no_curves_raises(tmp_path):
    path = _write(tmp_path, "depthonly.csv", "MD\n100\n110\n")
    with pytest.raises(ValueError):
        load_well(path, il=1, xl=1)


def test_load_well_from_las_uses_header_name(f3_las):
    w = load_well(str(f3_las), il=250, xl=500)
    assert w.name == "F02-1"
    assert default_log(w) == "GR"


def test_load_well_deviated_bends_in_plan(tmp_path):
    logs = _write(tmp_path, "l.csv", "MD,GR\n0,5\n100,6\n200,7\n")
    dev = _write(tmp_path, "d.csv", "MD,INC,AZI\n0,0,0\n100,0,0\n200,60,90\n")  # east turn
    xf = GridTransform.from_binning(bin_il=25.0, bin_xl=25.0, azimuth_deg=0.0)
    w = load_well(logs, il=250, xl=500, deviation=dev, transform=xf)
    assert w.il.tolist() == [250, 250, 250]        # due-east turn leaves inline fixed
    assert w.xl[0] == pytest.approx(500)           # wellhead on the crossline
    assert w.xl[-1] > 500                          # deepest sample stepped east
    assert w.z[-1] < 200                           # TVD shorter than MD after the turn


def test_load_well_deviation_without_transform_raises(tmp_path):
    logs = _write(tmp_path, "l.csv", "MD,GR\n0,5\n100,6\n")
    dev = _write(tmp_path, "d.csv", "MD,INC,AZI\n0,0,0\n100,0,0\n")
    with pytest.raises(ValueError):
        load_well(logs, il=1, xl=1, deviation=dev)          # geometry is mandatory
