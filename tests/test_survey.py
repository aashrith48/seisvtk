"""Survey affine: inversion round-trip, least-squares fit, OpendTect parse."""
import numpy as np
import pytest

from seisvtk.io.survey import GridTransform, read_opendtect_survey


def test_world_grid_round_trip():
    xf = GridTransform.from_coeffs(x0=600000, y0=6070000,
                                   a_il=25.0, a_xl=-0.7, b_il=0.7, b_xl=25.0)
    il = np.array([100, 200, 300.0])
    xl = np.array([300, 400, 500.0])
    x, y = xf.grid_to_world(il, xl)
    il2, xl2 = xf.world_to_grid(x, y)
    assert np.allclose(il, il2)
    assert np.allclose(xl, xl2)


def test_fit_recovers_affine():
    truth = GridTransform.from_coeffs(1000, 2000, 25.0, -0.5, 0.5, 25.0)
    il = np.array([0, 10, 0, 10, 5.0])
    xl = np.array([0, 0, 10, 10, 5.0])
    x, y = truth.grid_to_world(il, xl)
    fit = GridTransform.fit(il, xl, x, y)
    assert fit.a_il == pytest.approx(25.0, abs=1e-6)
    assert fit.b_xl == pytest.approx(25.0, abs=1e-6)
    assert fit.x0 == pytest.approx(1000.0, abs=1e-6)


def test_degenerate_transform_raises():
    bad = GridTransform.from_coeffs(0, 0, 1.0, 1.0, 1.0, 1.0)   # parallel axes
    with pytest.raises(ValueError):
        bad.world_to_grid(1.0, 2.0)


def test_from_binning_axis_aligned():
    # azimuth 0: inline runs north (+Y), crossline runs east (+X)
    xf = GridTransform.from_binning(bin_il=25.0, bin_xl=12.5, azimuth_deg=0.0)
    x0, y0 = xf.grid_to_world(0, 0)
    x1, _ = xf.grid_to_world(0, 1)                 # +1 crossline -> +bin_xl east
    _, y1 = xf.grid_to_world(1, 0)                 # +1 inline    -> +bin_il north
    assert x1 - x0 == pytest.approx(12.5)
    assert y1 - y0 == pytest.approx(25.0)


def test_from_binning_rotated_round_trip():
    xf = GridTransform.from_binning(bin_il=25.0, bin_xl=25.0, azimuth_deg=37.0)
    il = np.array([10, 20, 30.0])
    xl = np.array([5, 15, 25.0])
    x, y = xf.grid_to_world(il, xl)
    il2, xl2 = xf.world_to_grid(x, y)
    assert np.allclose(il, il2) and np.allclose(xl, xl2)


def test_delta_world_to_grid_ignores_origin():
    # a pure metre offset maps to a line-number offset, independent of x0/y0
    xf = GridTransform.from_binning(bin_il=25.0, bin_xl=25.0, azimuth_deg=0.0,
                                    x0=123456.0, y0=7000000.0)
    dil, dxl = xf.delta_world_to_grid(east=50.0, north=25.0)
    assert dil == pytest.approx(1.0)               # 25 m north  = 1 inline
    assert dxl == pytest.approx(2.0)               # 50 m east   = 2 crosslines


def test_from_binning_rejects_nonpositive_bins():
    with pytest.raises(ValueError):
        GridTransform.from_binning(0.0, 25.0)
    with pytest.raises(ValueError):
        GridTransform.from_binning(25.0, -5.0)


def test_is_valid_relative_check():
    assert not GridTransform.from_coeffs(0, 0, 1, 1, 1, 1).is_valid     # parallel axes
    assert GridTransform.from_binning(25.0, 25.0, 30.0).is_valid
    # tiny but non-degenerate coefficients (feet-ish scaled down) still valid
    assert GridTransform.from_coeffs(0, 0, 1e-3, 0, 0, 1e-3).is_valid


def test_from_dict_missing_coefficient_raises():
    with pytest.raises(ValueError):
        GridTransform.from_dict({"x0": 0, "y0": 0})          # incomplete


def test_as_dict_from_dict_round_trip():
    xf = GridTransform.from_binning(25.0, 12.5, 41.0, x0=1000.0, y0=2000.0)
    assert GridTransform.from_dict(xf.as_dict()) == xf


def test_read_opendtect_survey_f3(f3_survey):
    s = read_opendtect_survey(str(f3_survey))
    assert s["iline_range"] == [100, 750]
    assert s["xline_range"] == [300, 1250]
    assert s["z_range"][1] == pytest.approx(1.848)              # seconds in the file
    # a known F3 set point: IL/XL 99/104 -> (600938.13, 6073394.5)
    x, y = s["transform"].grid_to_world(99, 104)
    assert x == pytest.approx(600938.13, abs=2.0)
    assert y == pytest.approx(6073394.5, abs=2.0)
