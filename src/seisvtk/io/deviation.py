"""Well deviation surveys: turn a trajectory file into XY/TVD along the hole.

A well is rarely a clean vertical line; to drape it on a cube we need the
horizontal position at every logged depth. Surveys come in two shapes:

* **Directional** — ``(MD, inclination, azimuth)``. The classic minimum-curvature
  method (:func:`minimum_curvature`) integrates these angle stations into
  east/north/TVD offsets from the wellhead. This is the industry-standard
  survey-to-position calculation.
* **XYZ** — ``(MD, X, Y, Z)`` positions already computed by the acquisition
  software; read straight through.

Both land in a :class:`Deviation`: measured depth plus east/north offsets from
the first station and true vertical depth, which :mod:`seisvtk.io.well` combines
with a wellhead ``(X, Y)`` and a time-depth relation to place the log.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisvtk.io.tables import pick_column, read_table, to_float


@dataclass
class Deviation:
    """A trajectory sampled on measured depth.

    ``east``/``north`` are horizontal offsets from the first station (metres in
    the survey's own units); ``tvd`` is true vertical depth below it. Add the
    wellhead ``(X, Y)`` to ``east``/``north`` for absolute position.
    """

    md: np.ndarray
    east: np.ndarray
    north: np.ndarray
    tvd: np.ndarray

    def at(self, md) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(east, north, tvd)`` offsets at measured depths ``md``.

        Linear within the surveyed interval and *linearly extrapolated* along the
        end tangents outside it (not clamped): a log sample above the first
        station continues up the opening trajectory — vertical for a survey that
        starts vertical — and one below the last station continues down the final
        tangent, rather than freezing horizontally at the last station.
        """
        md = np.asarray(md, float)
        return (self._interp(md, self.east),
                self._interp(md, self.north),
                self._interp(md, self.tvd))

    def _interp(self, md, vals) -> np.ndarray:
        t = np.interp(md, self.md, vals)
        if self.md.size >= 2:                                  # extend the end tangents
            s0 = (vals[1] - vals[0]) / (self.md[1] - self.md[0])
            s1 = (vals[-1] - vals[-2]) / (self.md[-1] - self.md[-2])
            t = np.where(md < self.md[0], vals[0] + s0 * (md - self.md[0]), t)
            t = np.where(md > self.md[-1], vals[-1] + s1 * (md - self.md[-1]), t)
        return t


def minimum_curvature(md, inc_deg, azi_deg) -> Deviation:
    """Minimum-curvature trajectory from angle stations.

    ``inc_deg`` is inclination from vertical (0 = straight down) and ``azi_deg``
    is azimuth clockwise from north, both in degrees. Returns cumulative
    east/north/TVD offsets from the first station (which sits at the origin).
    """
    md = np.asarray(md, float)
    inc = np.radians(np.asarray(inc_deg, float))
    azi = np.radians(np.asarray(azi_deg, float))
    n = md.size
    east = np.zeros(n)
    north = np.zeros(n)
    tvd = np.zeros(n)

    for i in range(1, n):
        dmd = md[i] - md[i - 1]
        i1, i2 = inc[i - 1], inc[i]
        a1, a2 = azi[i - 1], azi[i]
        # dogleg angle between the two station directions
        cos_dl = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1))
        dl = np.arccos(np.clip(cos_dl, -1.0, 1.0))
        rf = 1.0 if dl < 1e-9 else (2.0 / dl) * np.tan(dl / 2.0)  # ratio factor
        half = dmd / 2.0 * rf
        north[i] = north[i - 1] + half * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2))
        east[i] = east[i - 1] + half * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2))
        tvd[i] = tvd[i - 1] + half * (np.cos(i1) + np.cos(i2))

    return Deviation(md, east, north, tvd)


def read_deviation(path: str, *, kind: str = "directional",
                   cols: tuple[int, ...] | None = None,
                   delimiter: str | None = None) -> Deviation:
    """Read a deviation survey file into a :class:`Deviation`.

    ``kind="directional"`` expects ``(MD, inc, azi)`` columns (default columns
    1,2,3); ``kind="xyz"`` expects ``(MD, X, Y, Z)`` (default 1,2,3,4) and the
    XY are stored as offsets from the first row so the model stays wellhead-
    relative. ``cols`` overrides the 1-based column numbers; ``delimiter`` is
    sniffed when ``None``. Rows with a non-finite value are dropped and the
    survey is sorted by measured depth, since minimum-curvature accumulates down
    the hole and :meth:`Deviation.at` needs ``md`` ascending.
    """
    df = read_table(path, delimiter=delimiter)

    if kind == "directional":
        c = cols or (1, 2, 3)
        md, (inc, azi) = _clean_survey(df, c[0], (c[1], c[2]))
        return minimum_curvature(md, inc, azi)

    if kind == "xyz":
        c = cols or (1, 2, 3, 4)
        md, (x, y, z) = _clean_survey(df, c[0], (c[1], c[2], c[3]))
        return Deviation(md, x - x[0], y - y[0], z - z[0])

    raise ValueError(f"kind must be 'directional' or 'xyz', got {kind!r}")


def _clean_survey(df, md_col, value_cols):
    """Pull MD + value columns, drop non-finite rows, sort ascending by MD."""
    md = to_float(df[pick_column(df, md_col)])
    values = [to_float(df[pick_column(df, c)]) for c in value_cols]
    good = np.isfinite(md)
    for v in values:
        good &= np.isfinite(v)
    order = np.argsort(md[good], kind="stable")
    return md[good][order], tuple(v[good][order] for v in values)
