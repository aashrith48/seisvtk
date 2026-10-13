"""Turn a file path + a few options into drapeable model objects.

This is the orchestration layer the Data panel calls: it maps a format/layout
choice to the right low-level reader in :mod:`seisvtk.io` and hands back ready
:class:`~seisvtk.io.horizon.Horizon` / :class:`~seisvtk.io.well.Well` objects.
It is scene-independent and VTK-free, so it unit-tests on its own; the viewer
takes what it returns and drapes it.
"""
from __future__ import annotations

import os

from seisvtk.io.deviation import read_deviation
from seisvtk.io.horizon import (Horizon, read_horizon_columns, read_horizon_ilxl,
                                 read_horizon_rows, read_horizon_xyz)
from seisvtk.io.survey import GridTransform
from seisvtk.io.well import Well, find_curve, read_checkshot, read_logs

# Horizon file layouts the panel offers, in menu order.
HORIZON_LAYOUTS = ("ilxl", "rows", "columns", "xyz")


def load_horizons(path: str, layout: str = "ilxl", *,
                  z_scale: float = 1.0,
                  delimiter: str | None = None,
                  transform: GridTransform | None = None) -> list[Horizon]:
    """Read one or more horizons from *path* using the named *layout*.

    * ``ilxl``    — ``IL XL Z`` columns              -> one horizon.
    * ``rows``    — ``Name IL XL Z``                 -> one horizon per name.
    * ``columns`` — ``IL XL H1 H2 ...`` (needs header) -> one horizon per Z column.
    * ``xyz``     — ``X Y Z`` world points           -> one horizon; needs *transform*.

    Always returns a list (even for single-surface layouts) so the caller drapes
    the result uniformly. ``z_scale`` converts the file's vertical unit to the
    cube's (e.g. ``1000`` for a seconds file against a millisecond cube).
    """
    if layout not in HORIZON_LAYOUTS:
        raise ValueError(f"layout must be one of {HORIZON_LAYOUTS}, got {layout!r}")
    if layout == "ilxl":
        return [read_horizon_ilxl(path, z_scale=z_scale, delimiter=delimiter)]
    if layout == "rows":
        return read_horizon_rows(path, z_scale=z_scale, delimiter=delimiter)
    if layout == "columns":
        return read_horizon_columns(path, z_scale=z_scale, delimiter=delimiter)
    if transform is None:
        raise ValueError("layout 'xyz' needs a survey transform to invert X/Y to grid")
    return [read_horizon_xyz(path, transform, z_scale=z_scale, delimiter=delimiter)]


def _well_name(path: str, header: dict) -> str:
    """A well's display name: its LAS ``WELL`` header, else the file stem."""
    well = str(header.get("WELL", "")).strip()
    return well or os.path.splitext(os.path.basename(path))[0]


def load_well(path: str, *, il: float, xl: float,
              name: str | None = None,
              depth: str | int | None = None,
              delimiter: str | None = None,
              checkshot: str | None = None,
              checkshot_unit: str = "ms",
              deviation: str | None = None,
              deviation_kind: str = "directional",
              dev_cols: tuple[int, ...] | None = None,
              ekb: float = 0.0,
              transform: GridTransform | None = None) -> Well:
    """Read a well's logs into a :class:`Well` placed at wellhead ``(il, xl)``.

    Logs come from LAS or a delimited CSV/TXT via :func:`read_logs`; ``depth``
    picks the depth column for delimited files (header name or 1-based number).
    A *checkshot* path maps measured depth to two-way time for a **time** cube
    (``checkshot_unit`` ``'s'`` or ``'ms'``); without one the cube is taken to be
    in **depth** (``z`` = true vertical depth). Raises ``ValueError`` if the file
    holds no curves.

    Pass a *deviation* survey file to drape a **deviated** well: ``deviation_kind``
    is ``"directional"`` (``MD inc azi``) or ``"xyz"`` (``MD X Y Z``), ``dev_cols``
    overrides the column numbers, and ``ekb`` is the Kelly-bushing elevation for
    the depth datum. A deviated well needs the cube's survey geometry, so a
    ``transform`` is **required** when *deviation* is given (``il``/``xl`` are then
    the wellhead line position). Without *deviation* the well is vertical and
    ``transform`` is ignored.
    """
    md, curves, header = read_logs(path, depth=depth, delimiter=delimiter)
    if not curves:
        raise ValueError(f"no log curves found in {os.path.basename(path)}")
    time_depth = read_checkshot(checkshot, unit=checkshot_unit) if checkshot else None
    well_name = name or _well_name(path, header)
    if deviation:
        if transform is None:
            raise ValueError("a deviated well needs the cube's survey geometry "
                             "(transform=...); it maps the trajectory to inline/crossline")
        dev = read_deviation(deviation, kind=deviation_kind, cols=dev_cols,
                             delimiter=delimiter)
        return Well.deviated(well_name, md, curves, dev, transform,
                             head_il=float(il), head_xl=float(xl), ekb=ekb,
                             time_depth=time_depth)
    return Well.vertical(well_name, float(il), float(xl), md, curves,
                         time_depth=time_depth)


def default_log(well: Well) -> str | None:
    """The log to colour a well by: GR if present, else the first curve."""
    return find_curve(well.logs, "GR") or (next(iter(well.logs), None))


def suggest_zarr_path(segy_path: str) -> str:
    """A sensible output ``.zarr`` path beside a SEG-Y file (same stem)."""
    root, _ = os.path.splitext(segy_path)
    return root + ".zarr"


def convert_segy(segy_path: str, out_path: str, *,
                 iline: int | None = None, xline: int | None = None,
                 progress=None) -> dict:
    """Convert a 3D SEG-Y to a chunked Zarr; return the store's geometry attrs.

    ``iline``/``xline`` are the inline/crossline header **byte** offsets; when
    either is ``None`` they are auto-detected from the trace headers. ``progress``
    is forwarded to the converter as ``progress(done_bands, total_bands)``.

    ``segyio`` is imported lazily so the viewer runs where it is not installed;
    a clear :class:`RuntimeError` is raised only if conversion is actually asked
    for without it.
    """
    if not os.path.isfile(segy_path):
        raise ValueError(f"no such SEG-Y file: {segy_path}")
    try:                                   # lazy: the slice viewer needs no segyio
        from seisvtk.io.segy2zarr import convert_segy_to_zarr
        from seisvtk.io.segy_scan import detect_geometry
    except ImportError as e:               # pragma: no cover - env-dependent
        raise RuntimeError("SEG-Y conversion needs 'segyio' "
                           "(pip install segyio)") from e

    if iline is None or xline is None:
        geom = detect_geometry(segy_path)  # scans headers, raises if implausible
        iline = iline or geom["iline_byte"]
        xline = xline or geom["xline_byte"]
    return convert_segy_to_zarr(segy_path, int(iline), int(xline), out_path,
                                progress=progress)
