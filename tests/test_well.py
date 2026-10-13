"""Well readers: delimited log tables, LAS, dispatch, coordinates."""
import numpy as np
import pytest

from seisvtk.io.deviation import minimum_curvature
from seisvtk.io.survey import GridTransform
from seisvtk.io.well import (Well, find_curve, read_las, read_logs,
                             read_well_coordinates, read_well_table)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_read_well_table_csv_with_header(tmp_path):
    path = _write(tmp_path, "logs.csv", "MD,GR,RHOB\n100,55,2.3\n100.5,60,2.4\n101,-999.25,2.5\n")
    md, curves, header = read_well_table(path)
    assert np.allclose(md, [100, 100.5, 101])
    assert set(curves) == {"GR", "RHOB"}
    assert np.isnan(curves["GR"][2])                      # null -> NaN
    assert header == {}


def test_read_well_table_semicolon_and_depth_alias(tmp_path):
    # depth column is not first; it must be found by its DEPT alias
    path = _write(tmp_path, "logs.txt", "GR;DEPT;NPHI\n55;100;0.3\n60;101;0.28\n")
    md, curves, _ = read_well_table(path)
    assert np.allclose(md, [100, 101])
    assert set(curves) == {"GR", "NPHI"}


def test_read_well_table_headerless_1based_depth(tmp_path):
    path = _write(tmp_path, "logs.txt", "100 55 2.3\n101 60 2.4\n")
    md, curves, _ = read_well_table(path, depth=1)        # 1-based column number
    assert np.allclose(md, [100, 101])
    assert set(curves) == {"2", "3"}


def test_read_las_f3(f3_las):
    md, curves, header = read_las(str(f3_las))
    assert md[0] == pytest.approx(48.0)
    assert find_curve(curves, "GR") == "GR"
    assert header.get("WELL") == "F02-1"
    assert np.isnan(curves["GR"]).any()                   # LAS NULL applied


def test_read_logs_dispatch(tmp_path, f3_las):
    assert read_logs(str(f3_las))[2].get("WELL") == "F02-1"   # -> read_las
    csv = _write(tmp_path, "l.csv", "MD,GR\n10,5\n20,6\n")
    assert read_logs(csv)[2] == {}                            # -> read_well_table


def test_read_well_coordinates(tmp_path):
    path = _write(tmp_path, "coords.csv",
                  "Well,X,Y,EKB\nA-1,500,1000,30\nB-2,600,1100,25\n")
    coords = read_well_coordinates(path)
    assert coords["A-1"] == {"x": 500.0, "y": 1000.0, "ekb": 30.0}
    assert coords["B-2"]["y"] == 1100.0


def test_well_from_arrays_scalar_broadcast():
    w = Well.from_arrays("W", il=10, xl=20, z=[100, 200, 300], logs={"GR": [1, 2, 3]})
    assert w.il.tolist() == [10, 10, 10]
    assert w.xl.tolist() == [20, 20, 20]
    assert w.logs["GR"].tolist() == [1, 2, 3]


# ---- deviated wells ----------------------------------------------------------
_BINNING = GridTransform.from_binning(bin_il=25.0, bin_xl=25.0, azimuth_deg=0.0)


def test_from_world_track_ties_world_xy_to_lines():
    # the primitive: a trajectory already in world X/Y/Z -> (il, xl, z) via the affine
    xf = GridTransform.from_binning(25.0, 25.0, azimuth_deg=0.0)
    x, y = xf.grid_to_world(np.array([100, 100.0]), np.array([200, 210.0]))
    w = Well.from_world_track("W", x, y, z=[850, 870.0], logs={"GR": [1, 2.0]},
                              transform=xf)
    assert np.allclose(w.il, [100, 100])
    assert np.allclose(w.xl, [200, 210])
    assert np.allclose(w.z, [850, 870])


def test_deviated_vertical_survey_reduces_to_vertical():
    # an all-vertical survey (ekb 0, md from 0) must place like Well.vertical
    md = np.array([0, 100, 200, 300.0])
    dev = minimum_curvature(md, inc_deg=np.zeros(4), azi_deg=np.zeros(4))
    logs = {"GR": [1, 2, 3, 4.0]}
    w = Well.deviated("W", md=md, logs=logs, deviation=dev, transform=_BINNING,
                      head_il=100, head_xl=200)
    straight = Well.vertical("W", 100, 200, md, logs)
    assert np.allclose(w.il, straight.il)
    assert np.allclose(w.xl, straight.xl)
    assert np.allclose(w.z, straight.z)            # z == md for a depth cube, ekb 0


def test_deviated_slant_offsets_laterally_and_shortens_tvd():
    # 60-degree slant due east: +X offset -> crosslines grow, inline fixed, TVD < MD
    md = np.array([0, 100.0])
    dev = minimum_curvature(md, inc_deg=[60, 60.0], azi_deg=[90, 90.0])
    w = Well.deviated("W", md=md, logs={"GR": [1, 2.0]}, deviation=dev,
                      transform=_BINNING, head_il=100, head_xl=200)
    assert np.allclose(w.il, [100, 100])                       # due east, inline unchanged
    assert w.xl[-1] == pytest.approx(200 + 100 * np.sin(np.radians(60)) / 25.0)
    assert w.z[-1] == pytest.approx(50.0)                      # TVD = MD*cos60, not 100


def test_deviated_ekb_shifts_to_tvdss():
    # ekb (Kelly bushing, +up) lifts the datum: z = tvd - ekb, like QEarth
    md = np.array([0, 100.0])
    dev = minimum_curvature(md, inc_deg=np.zeros(2), azi_deg=np.zeros(2))
    w = Well.deviated("W", md=md, logs={"GR": [1, 2.0]}, deviation=dev,
                      transform=_BINNING, head_il=100, head_xl=200, ekb=30.0)
    assert np.allclose(w.z, md - 30.0)                         # TVDSS


def test_deviated_time_cube_bends_laterally_but_z_is_twt():
    from seisvtk.io.well import TimeDepth
    md = np.array([0, 100.0])
    dev = minimum_curvature(md, inc_deg=[60, 60.0], azi_deg=[90, 90.0])
    td = TimeDepth.from_checkshot([0, 100.0], [0, 80.0])       # MD->TWT ms
    w = Well.deviated("W", md=md, logs={"GR": [1, 2.0]}, deviation=dev,
                      transform=_BINNING, head_il=100, head_xl=200, time_depth=td)
    assert w.xl[-1] > 200                                      # still bends in plan view
    assert np.allclose(w.z, td(md))                            # vertical axis is TWT, not TVD


def test_deviated_wellhead_in_world_coords():
    # wellhead may be given directly in world X/Y (QEarth's well.x / well.y)
    md = np.array([0, 100.0])
    dev = minimum_curvature(md, inc_deg=np.zeros(2), azi_deg=np.zeros(2))
    xf = GridTransform.from_binning(25.0, 25.0, azimuth_deg=0.0)
    head_x, head_y = xf.grid_to_world(100.0, 200.0)
    w = Well.deviated("W", md=md, logs={"GR": [1, 2.0]}, deviation=dev,
                      transform=xf, head_xy=(float(head_x), float(head_y)))
    assert np.allclose(w.il, [100, 100]) and np.allclose(w.xl, [200, 200])


def test_deviated_tie_in_survey_anchors_tvd():
    # survey starts at MD 500 (a tie-in); logs reach above it. The hole is taken
    # vertical down to the first station, so z stays == md for a vertical survey.
    md = np.array([400, 500, 600.0])
    dev = minimum_curvature([500, 600.0], [0, 0.0], [0, 0.0])
    w = Well.deviated("W", md=md, logs={"GR": [1, 2, 3.0]}, deviation=dev,
                      transform=_BINNING, head_il=100, head_xl=200)
    assert np.allclose(w.z, md)              # anchored, not 500 m too shallow


def test_deviated_needs_a_wellhead():
    md = np.array([0, 100.0])
    dev = minimum_curvature(md, inc_deg=np.zeros(2), azi_deg=np.zeros(2))
    with pytest.raises(ValueError):
        Well.deviated("W", md=md, logs={"GR": [1, 2.0]}, deviation=dev,
                      transform=_BINNING)                       # no head_xy / head_il+xl
