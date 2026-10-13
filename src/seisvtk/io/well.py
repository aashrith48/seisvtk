"""Wells: read logs, trajectory and time-depth into one drapeable model.

A :class:`Well` is a trajectory in *survey coordinates* — parallel ``il``, ``xl``,
``z`` arrays down the hole — plus a dict of logs sampled on the same points. As
with :class:`~seisvtk.io.horizon.Horizon`, it is scene-independent: the viewer
maps line numbers and ``z`` to array indices when it builds the tube.

Placing a well in a *time* cube needs three ingredients, each from its own file
in a typical project, and each with a classic trap:

* **trajectory** — where the hole is. Vertical wells (the common case here) are a
  surface ``(X, Y)`` held constant down the hole (:func:`read_welltrack`).
* **time-depth** — measured depth (the log index) to two-way time. Checkshots and
  sonic "dt models" disagree in *unit* (seconds vs milliseconds) and checkshots
  can be non-monotonic with duplicate depths; :class:`TimeDepth` sorts, dedupes
  and enforces monotonic time.
* **logs** — :func:`read_las` (via lasio), NaN already applied for the LAS NULL.

A host app that keeps wells in its own database maps them to arrays and calls
:meth:`Well.vertical` / :meth:`Well.vertical_from_world` directly.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from seisvtk.io.survey import GridTransform
from seisvtk.io.tables import NULLS, pick_column, read_table, to_float

_NULL = -900.0  # below this a log sample is treated as missing (covers -999.25)

# Depth/index curves — the sample axis, never a modelable log.
_DEPTH_ALIASES = ("DEPTH", "DEPT", "MD", "TVD", "TVDSS", "TVDKB", "TVDML")


@dataclass
class TimeDepth:
    """Monotonic measured-depth -> two-way-time (ms) map with linear extrapolation."""

    md: np.ndarray   # strictly increasing
    twt: np.ndarray  # ms, non-decreasing, aligned to md

    @classmethod
    def from_checkshot(cls, md, twt) -> "TimeDepth":
        """Clean raw checkshot pairs: sort by depth, average duplicate depths,
        force time non-decreasing."""
        md = np.asarray(md, float)
        twt = np.asarray(twt, float)
        good = np.isfinite(md) & np.isfinite(twt)
        md, twt = md[good], twt[good]
        umd, inv = np.unique(md, return_inverse=True)         # sorts + dedupes depth
        utwt = np.bincount(inv, weights=twt) / np.bincount(inv)
        utwt = np.maximum.accumulate(utwt)                    # time can't go back up-hole
        return cls(umd, utwt)

    def __call__(self, md) -> np.ndarray:
        md = np.asarray(md, float)
        t = np.interp(md, self.md, self.twt)
        if self.md.size >= 2:                                  # linear extrapolation outside
            s0 = (self.twt[1] - self.twt[0]) / (self.md[1] - self.md[0])
            s1 = (self.twt[-1] - self.twt[-2]) / (self.md[-1] - self.md[-2])
            t = np.where(md < self.md[0], self.twt[0] + s0 * (md - self.md[0]), t)
            t = np.where(md > self.md[-1], self.twt[-1] + s1 * (md - self.md[-1]), t)
        return t


@dataclass
class Well:
    """A well trajectory in survey coords with logs sampled along it."""

    name: str
    il: np.ndarray
    xl: np.ndarray
    z: np.ndarray                 # ms (time cube) or depth, same unit as the cube
    logs: dict[str, np.ndarray]

    @classmethod
    def from_arrays(cls, name, il, xl, z, logs) -> "Well":
        """Build directly from trajectory arrays — the host-adapter entry point.

        ``il``/``xl`` may be scalars (a vertical well) or arrays along the hole;
        scalars broadcast to ``z``'s length. ``logs`` maps curve name -> values
        aligned to ``z``.
        """
        z = np.asarray(z, float)
        n = z.size
        il = np.full(n, float(il)) if np.ndim(il) == 0 else np.asarray(il, float)
        xl = np.full(n, float(xl)) if np.ndim(xl) == 0 else np.asarray(xl, float)
        return cls(name, il, xl, z, {k: np.asarray(v, float) for k, v in logs.items()})

    @classmethod
    def vertical(cls, name, il, xl, md, logs, *, time_depth: "TimeDepth | None" = None) -> "Well":
        """A vertical well at fixed ``(il, xl)`` sampled on measured depth ``md``.

        ``z`` is ``time_depth(md)`` for a time cube, or ``md`` itself for a depth
        cube. ``logs`` maps curve name -> values aligned to ``md``.
        """
        md = np.asarray(md, float)
        z = time_depth(md) if time_depth is not None else md
        n = md.size
        return cls(name, np.full(n, float(il)), np.full(n, float(xl)), z,
                   {k: np.asarray(v, float) for k, v in logs.items()})

    @classmethod
    def vertical_from_world(cls, name, x, y, transform: GridTransform, md, logs,
                            *, time_depth: "TimeDepth | None" = None) -> "Well":
        """Like :meth:`vertical`, but locate the well from world ``(X, Y)``."""
        il, xl = transform.world_to_grid(x, y)
        return cls.vertical(name, float(il), float(xl), md, logs, time_depth=time_depth)

    @classmethod
    def from_world_track(cls, name, x, y, z, logs, transform: GridTransform) -> "Well":
        """Place a trajectory already resolved to world coordinates.

        ``x``/``y`` are the along-hole world coordinates and ``z`` the cube's
        vertical value at each sample (TVDSS for a depth cube, two-way time for a
        time cube — the caller decides). Each ``(X, Y)`` is mapped to a fractional
        ``(inline, crossline)`` through ``transform`` — the one operation that ties
        a well to the seismic grid. ``logs`` are aligned to the same samples.
        """
        il, xl = transform.world_to_grid(x, y)
        return cls.from_arrays(name, il, xl, z, logs)

    @classmethod
    def deviated(cls, name, md, logs, deviation: "Deviation", transform: GridTransform,
                 *, head_xy: "tuple[float, float] | None" = None,
                 head_il: float | None = None, head_xl: float | None = None,
                 ekb: float = 0.0, time_depth: "TimeDepth | None" = None) -> "Well":
        """A deviated well from a survey + a wellhead + the cube's survey geometry.

        ``deviation`` supplies wellhead-relative ``(east, north, tvd)`` offsets
        (see :mod:`seisvtk.io.deviation`), sampled at the log depths ``md``. The
        wellhead is given either in world coordinates (``head_xy=(X, Y)`` — e.g.
        a LAS location) or as a line position (``head_il``/``head_xl``), which is
        converted to world through ``transform``. The along-hole world track is
        then tied to the grid via :meth:`from_world_track`.

        Vertically, a depth cube gets true vertical depth sub-sea. A survey's
        first station sits at measured depth ``deviation.md[0]`` with ``tvd`` 0,
        so the absolute TVD is anchored there (``md0 + tvd``) — the hole is taken
        vertical from the surface down to the first station, which keeps a tie-in
        survey (first station well below 0) at the right depth. Then
        ``z = (md0 + tvd) - ekb`` (``ekb`` the Kelly-bushing elevation, positive
        up). A time cube instead gets ``z = time_depth(md)``, where ``time_depth``
        is keyed on **measured depth** (as checkshots are read here and in QEarth);
        ``ekb`` does not enter the time branch. With a vertical survey, ``ekb=0``
        and ``md`` from zero, this reduces exactly to :meth:`vertical`.
        """
        md = np.asarray(md, float)
        east, north, tvd = deviation.at(md)
        if head_xy is not None:
            x0, y0 = float(head_xy[0]), float(head_xy[1])
        elif head_il is not None and head_xl is not None:
            x0, y0 = transform.grid_to_world(float(head_il), float(head_xl))
        else:
            raise ValueError("a deviated well needs a wellhead: head_xy=(X, Y) "
                             "or head_il and head_xl")
        x = x0 + east
        y = y0 + north
        if time_depth is not None:
            z = time_depth(md)
        else:
            z = (float(deviation.md[0]) + tvd) - float(ekb)   # anchored TVDSS
        return cls.from_world_track(name, x, y, z, logs, transform)


# ---- readers -----------------------------------------------------------------
def read_las(path: str):
    """Read a LAS file -> ``(md, curves, header)``.

    ``md`` is the depth index; ``curves`` maps mnemonic -> array (NaN already
    applied for the LAS NULL); ``header`` maps the Well-section mnemonics to
    values (``WELL``, ``LOC``, ``EKB``, ...).
    """
    import lasio
    las = lasio.read(path)
    md = np.asarray(las.index, float)
    curves = {c.mnemonic: np.asarray(las[c.mnemonic], float) for c in las.curves}
    header = {item.mnemonic: item.value for item in las.well}
    return md, curves, header


def _find_depth_column(columns) -> str | None:
    """The first column whose name is a depth/index alias (case-insensitive)."""
    norm = {str(c).strip().upper().replace("_", ""): c for c in columns}
    return next((norm[a] for a in _DEPTH_ALIASES if a in norm), None)


def read_well_table(path: str, *, depth: str | int | None = None,
                    delimiter: str | None = None,
                    nulls=NULLS):
    """Read a delimited CSV/TXT well-log table -> ``(md, curves, header)``.

    The depth column is ``depth`` (a header name, or a 1-based column number for
    headerless files); when ``None`` it is found by alias (``DEPTH``/``MD``/...),
    falling back to the first column. Every other column becomes a curve with
    ``nulls`` mapped to ``NaN``. Mirrors the shape of :func:`read_las` so callers
    treat LAS and delimited logs alike.
    """
    df = read_table(path, delimiter=delimiter)
    depth_col = (pick_column(df, depth) if depth is not None
                 else (_find_depth_column(df.columns) or df.columns[0]))
    md = to_float(df[depth_col], nulls)
    curves = {str(c): to_float(df[c], nulls) for c in df.columns if c != depth_col}
    return md, curves, {}


def read_logs(path: str, **kwargs):
    """Read well logs from either a LAS or a delimited CSV/TXT file.

    Dispatches on extension: ``.las`` -> :func:`read_las`, anything else ->
    :func:`read_well_table`. Returns the same ``(md, curves, header)`` triple.
    """
    if os.path.splitext(path)[1].lower() == ".las":
        return read_las(path)
    return read_well_table(path, **kwargs)


def read_well_coordinates(path: str, *, cols: tuple[int, ...] = (1, 2, 3, 4),
                          delimiter: str | None = None) -> dict[str, dict]:
    """Read a ``Name  X  Y  [EKB]`` coordinates table into ``{name: {x,y,ekb}}``.

    ``cols`` is the 1-based ``(name, x, y, ekb)`` column numbers; pass a 3-tuple
    to omit EKB. Useful when logs carry no location header and X/Y live in a
    separate sheet.
    """
    df = read_table(path, delimiter=delimiter)
    name_c = pick_column(df, cols[0])
    x = to_float(df[pick_column(df, cols[1])])
    y = to_float(df[pick_column(df, cols[2])])
    ekb = to_float(df[pick_column(df, cols[3])]) if len(cols) > 3 else np.full(len(df), np.nan)
    out: dict[str, dict] = {}
    for i, nm in enumerate(df[name_c].astype(str)):
        out[nm.strip()] = {"x": float(x[i]), "y": float(y[i]), "ekb": float(ekb[i])}
    return out


def find_curve(curves: dict, prefix: str = "GR") -> str | None:
    """First curve whose name starts with ``prefix`` (case/underscore-insensitive)."""
    p = prefix.upper().replace("_", "")
    return next((k for k in curves if k.upper().replace("_", "").startswith(p)), None)


def read_welltrack(path: str) -> tuple[float, float]:
    """Surface ``(X, Y)`` from a well track. Assumes a vertical well (constant X/Y)."""
    arr = np.loadtxt(path, dtype=float, comments=("#", "!"))
    arr = np.atleast_2d(arr)
    return float(arr[0, 0]), float(arr[0, 1])


def las_location(header: dict) -> tuple[float, float] | None:
    """Parse ``X = .. Y = ..`` out of a LAS ``LOC`` header value, if present."""
    loc = str(header.get("LOC", ""))
    m = re.search(r"X\s*=\s*([-\d.]+).*?Y\s*=\s*([-\d.]+)", loc)
    return (float(m.group(1)), float(m.group(2))) if m else None


def read_checkshot(path: str, *, unit: str = "ms",
                   cols: tuple[int, int] = (0, 1)) -> TimeDepth:
    """Read a ``MD  TWT`` time-depth table into a :class:`TimeDepth` (ms internally).

    ``cols`` selects the (measured-depth, time) columns; ``unit`` is the time
    column's unit (``"s"`` scales by 1000, ``"ms"`` leaves it).
    """
    data = np.loadtxt(path, usecols=cols, dtype=float, comments=("#", "!"))
    data = np.atleast_2d(data)
    md, twt = data[:, 0], data[:, 1]
    if unit == "s":
        twt = twt * 1000.0
    elif unit != "ms":
        raise ValueError(f"unit must be 's' or 'ms', got {unit!r}")
    return TimeDepth.from_checkshot(md, twt)
