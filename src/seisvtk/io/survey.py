"""Survey geometry: the affine map between line numbers and world coordinates.

A 3D survey places each ``(inline, crossline)`` bin at a world ``(X, Y)`` via a
2-D affine (a rigid rotation + scale, written into every SEG-Y trace header and
every interpretation export)::

    X = x0 + a_il * IL + a_xl * XL
    Y = y0 + b_il * IL + b_xl * XL

Data exported in world coordinates (horizons as ``X Y Z``, well heads, fault
sticks) needs the *inverse* of this map to land on the cube's grid.
``GridTransform`` carries the forward and inverse maps and is numpy-vectorised.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridTransform:
    """Affine between survey line numbers ``(inline, crossline)`` and world ``(X, Y)``.

    Construct from the six coefficients (``from_coeffs``) or from an OpendTect
    ``.survey`` ``Coord-*-BinID`` pair (``from_opendtect``).
    """

    x0: float
    y0: float
    a_il: float
    a_xl: float
    b_il: float
    b_xl: float

    _COEFFS = ("x0", "y0", "a_il", "a_xl", "b_il", "b_xl")

    @classmethod
    def from_coeffs(cls, x0, y0, a_il, a_xl, b_il, b_xl) -> "GridTransform":
        return cls(float(x0), float(y0), float(a_il), float(a_xl),
                   float(b_il), float(b_xl))

    def as_dict(self) -> dict:
        """The six coefficients as a JSON-friendly dict (for a cube's attributes)."""
        return {k: float(getattr(self, k)) for k in self._COEFFS}

    @classmethod
    def from_dict(cls, d) -> "GridTransform":
        """Inverse of :meth:`as_dict`; also accepts an existing transform."""
        if isinstance(d, GridTransform):
            return d
        missing = [k for k in cls._COEFFS if k not in d]
        if missing:
            raise ValueError(f"survey transform is missing coefficients: {missing}")
        return cls.from_coeffs(*(d[k] for k in cls._COEFFS))

    @classmethod
    def fit(cls, il, xl, x, y) -> "GridTransform":
        """Least-squares forward affine ``(IL, XL) -> (X, Y)`` from control points.

        For surveys whose affine isn't recorded but that have an export carrying
        both line numbers and world coordinates (a horizon grid, a navigation
        file): pass matching ``il, xl, x, y`` arrays and get a transform back.
        """
        il = np.asarray(il, float)
        xl = np.asarray(xl, float)
        A = np.column_stack([il, xl, np.ones_like(il)])
        (a_il, a_xl, x0), *_ = np.linalg.lstsq(A, np.asarray(x, float), rcond=None)
        (b_il, b_xl, y0), *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
        return cls(float(x0), float(y0), float(a_il), float(a_xl), float(b_il), float(b_xl))

    @classmethod
    def from_binning(cls, bin_il: float, bin_xl: float, azimuth_deg: float = 0.0,
                     *, x0: float = 0.0, y0: float = 0.0) -> "GridTransform":
        """Build the affine from bin spacing and grid rotation alone.

        ``azimuth_deg`` is the heading of the **increasing-inline** axis, degrees
        clockwise from North (world ``X`` = easting, ``Y`` = northing); crosslines
        run 90 degrees clockwise of that. ``bin_il``/``bin_xl`` are the world
        distances between adjacent inlines/crosslines.

        ``bin_il``/``bin_xl`` are metres per unit **inline/crossline number** (so a
        survey that numbers lines in steps of 2 wants half the physical bin). The
        azimuth is grid north; any grid-vs-true-north convergence is the caller's
        to fold in. This is the fallback when a cube never recorded its world
        georeference: with a wellhead line position it is enough to drape a
        deviated well's metre-offsets, even though absolute ``(X, Y)`` is unknown
        (``x0``/``y0`` default to the origin). A georeferenced cube should instead
        carry its real transform (:meth:`from_coeffs` / :meth:`fit` / :meth:`from_opendtect`).
        """
        if bin_il <= 0 or bin_xl <= 0:
            raise ValueError(f"bin spacing must be positive, got {bin_il}, {bin_xl}")
        az = math.radians(azimuth_deg)
        s, c = math.sin(az), math.cos(az)
        # inline axis -> (sin az, cos az); crossline = inline rotated 90 deg CW.
        return cls(float(x0), float(y0),
                   bin_il * s, bin_xl * c,      # a_il, a_xl  (dX/dIL, dX/dXL)
                   bin_il * c, -bin_xl * s)     # b_il, b_xl  (dY/dIL, dY/dXL)

    @classmethod
    def from_opendtect(cls, coord_x_binid, coord_y_binid) -> "GridTransform":
        """Build from OpendTect ``Coord-X-BinID`` / ``Coord-Y-BinID`` triples.

        Each is ``origin, per-inline, per-crossline`` — e.g. the backtick-split
        values of ``Coord-X-BinID: 598408.2476`-0.6980`24.990`` in a ``.survey``.
        """
        x0, a_il, a_xl = (float(v) for v in coord_x_binid)
        y0, b_il, b_xl = (float(v) for v in coord_y_binid)
        return cls(x0, y0, a_il, a_xl, b_il, b_xl)

    @property
    def is_valid(self) -> bool:
        """True when the axes are not parallel, by a *relative* determinant test.

        A unit-independent check (metres, feet, or scaled integers all pass the
        same way), used to vet a fitted transform before trusting it.
        """
        det = self.a_il * self.b_xl - self.a_xl * self.b_il
        scale = abs(self.a_il * self.b_xl) + abs(self.a_xl * self.b_il)
        return abs(det) > 1e-9 * max(scale, 1.0)

    @property
    def _det(self) -> float:
        det = self.a_il * self.b_xl - self.a_xl * self.b_il
        if abs(det) < 1e-12:
            raise ValueError("degenerate survey transform (inline/crossline axes are parallel)")
        return det

    def grid_to_world(self, il, xl):
        """``(inline, crossline)`` line numbers -> ``(X, Y)`` world coordinates."""
        il = np.asarray(il, dtype=float)
        xl = np.asarray(xl, dtype=float)
        x = self.x0 + self.a_il * il + self.a_xl * xl
        y = self.y0 + self.b_il * il + self.b_xl * xl
        return x, y

    def world_to_grid(self, x, y):
        """``(X, Y)`` world coordinates -> ``(inline, crossline)`` (float, un-rounded)."""
        x = np.asarray(x, dtype=float) - self.x0
        y = np.asarray(y, dtype=float) - self.y0
        inv = 1.0 / self._det
        il = inv * (self.b_xl * x - self.a_xl * y)
        xl = inv * (-self.b_il * x + self.a_il * y)
        return il, xl

    def delta_world_to_grid(self, east, north):
        """A world ``(east, north)`` *offset* -> an ``(inline, crossline)`` offset.

        The rotation/scale part of :meth:`world_to_grid` without the origin shift,
        so a wellhead-relative survey offset (metres) becomes a line-number offset
        to add to the wellhead's ``(inline, crossline)``.
        """
        de = np.asarray(east, dtype=float)
        dn = np.asarray(north, dtype=float)
        inv = 1.0 / self._det
        dil = inv * (self.b_xl * de - self.a_xl * dn)
        dxl = inv * (-self.b_il * de + self.a_il * dn)
        return dil, dxl


def read_opendtect_survey(path: str) -> dict:
    """Parse an OpendTect ``.survey`` / ``survey.txt`` for geometry.

    Returns ``{transform, iline_range, xline_range, z_range, sample_interval}``.
    ``z`` is reported in the file's own unit (OpendTect stores seconds for time
    surveys); callers that work in milliseconds should scale.
    """
    fields: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, _, v = line.partition(":")
                fields[k.strip()] = v.strip()

    def _triple(key):
        return [float(v) for v in fields[key].split("`")]

    def _range(key):
        lo, hi, step = (float(v) for v in fields[key].split("`")[:3])
        return lo, hi, step

    il_lo, il_hi, _ = _range("In-line range")
    xl_lo, xl_hi, _ = _range("Cross-line range")
    z_lo, z_hi, z_step = _range("Z range")
    return {
        "transform": GridTransform.from_opendtect(
            _triple("Coord-X-BinID"), _triple("Coord-Y-BinID")),
        "iline_range": [int(il_lo), int(il_hi)],
        "xline_range": [int(xl_lo), int(xl_hi)],
        "z_range": [z_lo, z_hi],
        "sample_interval": z_step,
    }
