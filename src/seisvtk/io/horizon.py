"""Interpreted horizon surfaces: read common export formats into one model.

A :class:`Horizon` is a set of interpreted picks in *survey coordinates* —
``(inline, crossline, z)`` where ``z`` is the vertical value in the cube's unit
(two-way time in ms, or depth). It is deliberately scene-independent: the same
horizon drapes onto any cube that shares the survey, and the viewer converts
line numbers to array indices when it builds the surface actor.

Readers cover the usual interpretation exports:

* :func:`read_horizon_xyz` — ``X Y Z`` world coordinates (no line numbers); needs
  a :class:`~seisvtk.io.survey.GridTransform` to invert world -> grid. This is
  the OpendTect / Petrel "XYZ" point export (e.g. the F3 horizons).
* :func:`read_horizon_ilxl` — columns that already include inline/crossline.
* :meth:`Horizon.from_arrays` — build directly from arrays, for a host app that
  reads its own internal horizon type (map it to ``il/xl/z`` and hand it over).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisvtk.io.survey import GridTransform
from seisvtk.io.tables import read_table, to_float

# Common "no value" sentinels in interpretation exports.
NULLS = (-999.25, 999.25, 1.0e30, -1.0e30)


@dataclass
class Horizon:
    """Interpreted picks on a survey grid: parallel ``il``, ``xl``, ``z`` arrays."""

    name: str
    il: np.ndarray  # inline line numbers (int)
    xl: np.ndarray  # crossline line numbers (int)
    z: np.ndarray   # vertical value (ms or m), same units as the cube's z axis

    @classmethod
    def from_arrays(cls, name, il, xl, z) -> "Horizon":
        il = np.asarray(il).astype(np.int64)
        xl = np.asarray(xl).astype(np.int64)
        z = np.asarray(z).astype(float)
        keep = np.isfinite(z) & ~np.isin(z, NULLS)
        return cls(name, il[keep], xl[keep], z[keep])

    def to_grid(self):
        """Scatter the picks onto their ``(inline, crossline)`` lattice.

        Returns ``(il_axis, xl_axis, Z)`` where ``il_axis`` / ``xl_axis`` are the
        sorted unique line numbers and ``Z`` is a ``(n_il, n_xl)`` array of ``z``
        with ``NaN`` in cells that were never picked (the surface's holes).
        """
        iu = np.unique(self.il)
        xu = np.unique(self.xl)
        Z = np.full((iu.size, xu.size), np.nan, dtype=float)
        Z[np.searchsorted(iu, self.il), np.searchsorted(xu, self.xl)] = self.z
        return iu, xu, Z

    @property
    def z_range(self) -> tuple[float, float]:
        return float(np.nanmin(self.z)), float(np.nanmax(self.z))


def _load_columns(path: str, cols: tuple[int, ...], *,
                  delimiter: str | None = None) -> np.ndarray:
    """Read numeric columns ``cols`` (0-based) as a float array, in that order.

    Any delimiter (comma/tab/semicolon/whitespace) is sniffed, and a header row,
    comment lines and null sentinels are handled by :mod:`seisvtk.io.tables`.
    """
    df = read_table(path, delimiter=delimiter)
    return np.column_stack([to_float(df.iloc[:, c]) for c in cols])


def read_horizon_xyz(path: str, transform: GridTransform, *,
                     name: str | None = None,
                     cols: tuple[int, int, int] = (0, 1, 2),
                     z_scale: float = 1.0,
                     delimiter: str | None = None) -> Horizon:
    """Read an ``X Y Z`` world-coordinate horizon, inverting ``transform`` to grid.

    ``z_scale`` converts the file's vertical unit to the cube's (e.g. ``1000`` for
    a seconds export against a millisecond cube). Line numbers are rounded to the
    nearest bin. ``delimiter`` is sniffed when ``None``.
    """
    data = _load_columns(path, cols, delimiter=delimiter)
    x, y, z = data[:, 0], data[:, 1], data[:, 2] * z_scale
    il_f, xl_f = transform.world_to_grid(x, y)
    il = np.rint(il_f).astype(np.int64)
    xl = np.rint(xl_f).astype(np.int64)
    return Horizon.from_arrays(name or _stem(path), il, xl, z)


def read_horizon_ilxl(path: str, *,
                      name: str | None = None,
                      cols: tuple[int, int, int] = (0, 1, 2),
                      z_scale: float = 1.0,
                      delimiter: str | None = None) -> Horizon:
    """Read a horizon whose columns already carry inline/crossline.

    ``cols`` selects ``(inline, crossline, z)`` column indices (0-based); the
    default suits an ``IL XL Z`` file, and e.g. ``cols=(2, 3, 4)`` an
    ``X Y IL XL Z`` Petrel/Kingdom grid. ``delimiter`` is sniffed when ``None``.
    """
    data = _load_columns(path, cols, delimiter=delimiter)
    return Horizon.from_arrays(name or _stem(path),
                               data[:, 0], data[:, 1], data[:, 2] * z_scale)


def read_horizon_rows(path: str, *, cols: tuple[int, int, int, int] = (0, 1, 2, 3),
                      z_scale: float = 1.0,
                      delimiter: str | None = None) -> list[Horizon]:
    """Read a ``Name  IL  XL  Z`` file holding one or more horizons.

    ``cols`` is the 0-based ``(name, inline, crossline, z)`` columns. Rows are
    grouped by the name column, so a single file can carry several horizons;
    returns one :class:`Horizon` per distinct name, in first-seen order.
    """
    df = read_table(path, delimiter=delimiter)
    name_c, il_c, xl_c, z_c = (df.columns[c] for c in cols)
    il = to_float(df[il_c])
    xl = to_float(df[xl_c])
    z = to_float(df[z_c]) * z_scale
    names = df[name_c].astype(str).to_numpy()
    out = []
    for nm in dict.fromkeys(names):              # distinct, first-seen order
        m = names == nm
        out.append(Horizon.from_arrays(nm.strip(), il[m], xl[m], z[m]))
    return out


def read_horizon_columns(path: str, *, il_col: int = 0, xl_col: int = 1,
                         z_scale: float = 1.0,
                         delimiter: str | None = None) -> list[Horizon]:
    """Read an ``IL  XL  H1  H2  ...`` file: one horizon per Z column.

    Needs a header row naming the horizon columns; ``il_col``/``xl_col`` are the
    0-based inline/crossline columns and every *other* column is a horizon whose
    name is its header. Cells holding a null become holes in that surface.
    """
    df = read_table(path, header=True, delimiter=delimiter)
    il = to_float(df.iloc[:, il_col])
    xl = to_float(df.iloc[:, xl_col])
    out = []
    for j, col in enumerate(df.columns):
        if j in (il_col, xl_col):
            continue
        out.append(Horizon.from_arrays(str(col).strip(), il, xl,
                                       to_float(df[col]) * z_scale))
    return out


def _stem(path: str) -> str:
    import os
    return os.path.splitext(os.path.basename(path))[0]
