"""Robust reader for the delimited ASCII tables wells and horizons arrive in.

Interpretation exports are maddeningly inconsistent: comma, tab, semicolon or
whitespace separated; with or without a header row; peppered with ``#`` / ``!``
comment lines and blank lines. This module turns any of them into a tidy
``pandas.DataFrame`` so the typed readers (:mod:`seisvtk.io.well`,
:mod:`seisvtk.io.horizon`) can just pick columns.

Two conventions make the rest of the code simple:

* **Headerless files get 1-based string column names** ``"1"``, ``"2"`` … so a
  caller can select a column by the 1-based number interpreters quote in their
  docs, or by name when a header is present — see :func:`pick_column`.
* **Null sentinels become ``NaN``** only on request (the typed reader knows which
  columns are numeric), via :func:`to_float`.
"""
from __future__ import annotations

import io
import re
from typing import Sequence

import numpy as np
import pandas as pd

# Single-character delimiters we sniff for, in priority order. Whitespace is the
# fallback when none of these separates the columns.
_DELIMITERS = (",", "\t", ";")

# "No value" markers common to LAS/ODT/Petrel/Kingdom exports.
NULLS: tuple[float, ...] = (-999.25, -999.0, 999.25, 1.0e30, -1.0e30)


def _clean_lines(path: str, comment: Sequence[str]) -> list[str]:
    """File lines with blanks and ``comment``-led lines removed."""
    out: list[str] = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s and s[0] not in comment:
                out.append(line.rstrip("\n").rstrip("\r"))
    return out


def detect_delimiter(lines: Sequence[str]) -> str | None:
    """Sniff the column delimiter from sample lines; ``None`` means whitespace.

    A candidate wins only if it appears the *same* number of times (>=1) in
    every probed line — the signature of a real column separator, as opposed to
    a comma that happens to sit inside one field.
    """
    probe = [ln for ln in lines[:25] if ln.strip()]
    if not probe:
        return None
    for d in _DELIMITERS:
        counts = [ln.count(d) for ln in probe]
        if min(counts) >= 1 and len(set(counts)) == 1:
            return d
    return None


def _looks_numeric(token: str) -> bool:
    token = token.strip()
    if token == "":
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def _split(line: str, delimiter: str | None) -> list[str]:
    pattern = re.escape(delimiter) if delimiter is not None else r"\s+"
    return re.split(pattern, line.strip())


def read_table(path: str, *, delimiter: str | None = None,
               header: str | bool = "auto",
               comment: Sequence[str] = ("#", "!")) -> pd.DataFrame:
    """Read a delimited ASCII table into a DataFrame.

    ``delimiter`` is sniffed when ``None`` (``","``/``"\\t"``/``";"``/whitespace).
    ``header`` ``"auto"`` treats the first data row as names only when it holds a
    non-numeric token; a headerless table gets 1-based string column names.
    """
    lines = _clean_lines(path, comment)
    if not lines:
        raise ValueError(f"no data rows in {path!r}")

    d = delimiter if delimiter is not None else detect_delimiter(lines)
    if header == "auto":
        has_header = any(not _looks_numeric(t) for t in _split(lines[0], d))
    else:
        has_header = bool(header)

    sep = re.escape(d) if d is not None else r"\s+"
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep,
                     header=0 if has_header else None, engine="python")
    if not has_header:
        df.columns = [str(i + 1) for i in range(df.shape[1])]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def pick_column(df: pd.DataFrame, selector: str | int) -> str:
    """Resolve a column *selector* to a real column name.

    ``int`` is a **1-based** position (interpreters quote columns this way);
    ``str`` matches a header name case- and underscore-insensitively.
    """
    if isinstance(selector, (int, np.integer)):
        idx = int(selector) - 1
        if not 0 <= idx < df.shape[1]:
            raise IndexError(f"column {selector} out of range (1..{df.shape[1]})")
        return df.columns[idx]
    key = str(selector).strip().upper().replace("_", "")
    for c in df.columns:
        if str(c).strip().upper().replace("_", "") == key:
            return c
    raise KeyError(f"column {selector!r} not found in {list(df.columns)}")


def to_float(series, nulls: Sequence[float] = NULLS) -> np.ndarray:
    """A column as float with ``nulls`` (and non-finite) mapped to ``NaN``."""
    v = np.asarray(series, dtype=float)
    out = v.copy()
    for nv in nulls:
        out[np.isclose(v, nv, rtol=0, atol=1e-6)] = np.nan
    return out
