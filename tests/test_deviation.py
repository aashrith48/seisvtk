"""Minimum-curvature math and deviation-file readers."""
import numpy as np
import pytest

from seisvtk.io.deviation import Deviation, minimum_curvature, read_deviation


def test_vertical_well_is_straight_down():
    md = np.array([0, 100, 200, 300.0])
    dev = minimum_curvature(md, inc_deg=np.zeros(4), azi_deg=np.zeros(4))
    assert np.allclose(dev.east, 0)
    assert np.allclose(dev.north, 0)
    assert np.allclose(dev.tvd, md)                       # TVD == MD when vertical


def test_horizontal_segment_adds_offset_not_tvd():
    # drop vertically 100 m, then turn due east and run 100 m horizontal
    md = np.array([0, 100, 200.0])
    inc = np.array([0, 90, 90.0])      # second leg fully horizontal
    azi = np.array([90, 90, 90.0])     # due east
    dev = minimum_curvature(md, inc, azi)
    assert dev.tvd[-1] < 200                               # horizontal leg adds little TVD
    assert dev.east[-1] > dev.north[-1]                    # heads east, not north
    assert dev.east[-1] > 50


def test_known_slant_tvd():
    # single 60-degree slant leg: TVD gain = MD * cos(60) = 50
    dev = minimum_curvature([0, 100.0], [60, 60.0], [0, 0.0])
    assert dev.tvd[-1] == pytest.approx(50.0, abs=1e-6)
    assert dev.north[-1] == pytest.approx(100 * np.sin(np.radians(60)), abs=1e-6)


def test_deviation_at_interpolates():
    dev = minimum_curvature([0, 100.0], [0, 0.0], [0, 0.0])
    e, n, t = dev.at(50.0)
    assert t == pytest.approx(50.0)


def test_read_deviation_directional(tmp_path):
    p = tmp_path / "dev.csv"
    p.write_text("MD,INC,AZI\n0,0,0\n100,0,0\n200,90,90\n")
    dev = read_deviation(str(p), kind="directional")
    assert isinstance(dev, Deviation)
    assert dev.tvd[1] == pytest.approx(100.0)
    assert dev.east[-1] > 0


def test_read_deviation_xyz_is_wellhead_relative(tmp_path):
    p = tmp_path / "dev.txt"
    p.write_text("0 500 1000 0\n100 500 1000 100\n200 560 1000 190\n")
    dev = read_deviation(str(p), kind="xyz")
    assert dev.east[0] == 0 and dev.north[0] == 0          # offsets from first row
    assert dev.east[-1] == pytest.approx(60.0)
    assert dev.tvd[-1] == pytest.approx(190.0)


def test_read_deviation_bad_kind(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("MD,INC,AZI\n0,0,0\n")
    with pytest.raises(ValueError):
        read_deviation(str(p), kind="nope")


def test_at_extrapolates_above_and_below_survey():
    # vertical survey 100..200: outside the range it continues vertically, not flat
    dev = minimum_curvature([100, 200.0], [0, 0.0], [0, 0.0])
    e, n, t = dev.at(50.0)                 # above the first station
    assert e == pytest.approx(0.0) and n == pytest.approx(0.0)
    assert t == pytest.approx(-50.0)       # tvd offset extrapolated along md-slope 1
    _, _, t2 = dev.at(300.0)               # below the last station
    assert t2 == pytest.approx(200.0)      # not clamped at 100


def test_read_deviation_drops_nan_and_sorts(tmp_path):
    p = tmp_path / "dev.csv"
    p.write_text("MD,INC,AZI\n200,0,0\n100,0,0\nnan,0,0\n0,0,0\n")   # unsorted + NaN row
    dev = read_deviation(str(p), kind="directional")
    assert dev.md.tolist() == [0, 100, 200]          # NaN dropped, ascending
    assert np.all(np.diff(dev.md) > 0)
