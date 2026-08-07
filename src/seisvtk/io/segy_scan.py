"""Automatic inline/crossline header-byte detection for 3D SEG-Y."""

from __future__ import annotations

import numpy as np
import segyio
from segyio import TraceField as TF

# header field -> prior
#
# TraceField members are the SEG-Y rev1 byte offsets themselves
# (TraceField.INLINE_3D == 189), so each key doubles as the byte offset that
# ends up in the scan results.
#
# The prior encodes how likely a byte is to hold a real line number rather than
# a counter that forms a valid grid by accident. Numeric tests alone cannot
# separate these: many writers duplicate line numbers across several slots, and
# per-line trace counters produce a correctly shaped cube with wrong labels.
CANDIDATES: dict[int, float] = {
    TF.INLINE_3D:           1.00,  # SEG-Y rev1 standard for 3D
    TF.CROSSLINE_3D:        1.00,
    TF.CDP_X:               0.90,  # commonly repurposed for IL/XL
    TF.CDP_Y:               0.90,
    TF.FieldRecord:         0.80,  # classic pre-rev1 convention
    TF.CDP:                 0.80,
    TF.EnergySourcePoint:   0.60,
    TF.ShotPoint:           0.60,
    TF.TraceNumber:         0.30,  # sequence counters
    TF.CDP_TRACE:           0.30,
    TF.TRACE_SEQUENCE_LINE: 0.20,
    TF.TRACE_SEQUENCE_FILE: 0.20,
}

MIN_FILL = 0.2  # below this the field is a coordinate, not a line number


def _regularity(values: np.ndarray) -> float:
    """1.0 when the unique values form a perfect arithmetic sequence."""
    if len(values) < 2:
        return 0.0
    steps = np.diff(values)
    return float((steps == steps[0]).mean())


def scan_geometry(path: str, top: int = 4) -> list[dict]:
    """Rank candidate (iline_byte, xline_byte) pairs for a 3D SEG-Y file.

    Returns dicts sorted best-first, each with the byte offsets, grid size,
    line-number ranges, ``fill`` (traces / grid cells, so 1.0 is a dense cube
    and anything less is missing traces) and ``missing``.
    """
    with segyio.open(path, "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        fields = {}
        for byte, prior in CANDIDATES.items():
            values = f.attributes(byte)[:]
            uniq = np.unique(values)
            # Drop constants and one-value-per-trace counters.
            if 2 <= len(uniq) < n_traces:
                fields[byte] = (values, uniq, prior)

    results: list[dict] = []
    for il_byte, (il_vals, il_uniq, il_prior) in fields.items():
        for xl_byte, (xl_vals, xl_uniq, xl_prior) in fields.items():
            if il_byte == xl_byte:
                continue

            cells = len(il_uniq) * len(xl_uniq)
            if cells < n_traces:
                continue  # too few cells to hold every trace

            # Every trace must occupy its own (inline, crossline) cell.
            key = il_vals.astype(np.int64) * (int(xl_uniq.max()) + 1) + xl_vals
            if len(np.unique(key)) != n_traces:
                continue

            fill = n_traces / cells
            if fill < MIN_FILL:
                continue  # coordinates, not line numbers

            # Inline varies slowly, crossline fast. Also drops the mirror pair.
            if (np.diff(il_vals) != 0).sum() >= (np.diff(xl_vals) != 0).sum():
                continue

            regularity = 0.5 * (_regularity(il_uniq) + _regularity(xl_uniq))
            results.append({
                "iline_byte": il_byte,
                "xline_byte": xl_byte,
                "n_iline": len(il_uniq),
                "n_xline": len(xl_uniq),
                "iline_range": (int(il_uniq.min()), int(il_uniq.max())),
                "xline_range": (int(xl_uniq.min()), int(xl_uniq.max())),
                "fill": fill,
                "missing": cells - n_traces,
                "score": fill * (0.5 + 0.5 * regularity) * (il_prior * xl_prior),
            })

    results.sort(key=lambda r: -r["score"])
    return results[:top]


def detect_geometry(path: str) -> dict:
    """Best-guess geometry for a 3D SEG-Y. Raises if none is plausible."""
    candidates = scan_geometry(path, top=1)
    if not candidates:
        raise ValueError(f"could not infer 3D geometry for {path}")
    return candidates[0]


def open_auto(path: str, **kwargs):
    """``segyio.open`` with inline/crossline bytes detected automatically.

    Use as a context manager, exactly like ``segyio.open``.
    """
    geom = detect_geometry(path)
    return segyio.open(
        path, "r",
        iline=geom["iline_byte"],
        xline=geom["xline_byte"],
        **kwargs,
    )