"""Generic volume -> chunked Zarr ingest, shared by every input format.

``convert_segy_to_zarr`` (ragged SEG-Y traces) and any other source — a dense
numpy cube, an HDF5 dataset, a property volume produced by a modelling tool —
all funnel through here, so every store ends up with identical chunking,
compression, ``live`` mask and geometry-attribute conventions. Downstream
readers (:class:`seisvtk.io.volume.Volume`) therefore never care where a cube
came from.

Two entry points:

* :func:`write_bands_to_zarr` — the low-level writer. The caller supplies an
  iterator of inline bands ``(b0, b1, buf, mask)``; this creates the arrays,
  writes each band once, and stamps the geometry attributes. Ragged sources
  (SEG-Y) that must scatter traces into a grid use this directly.
* :func:`write_array_to_zarr` — convenience for a source that is already a
  dense ``(n_inline, n_crossline, n_sample)`` array-like (a numpy array or a
  lazily-sliceable ``h5py``/``zarr`` dataset). It streams one inline band at a
  time, so peak memory is a single band, never the whole cube.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Mapping

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

# (inline, crossline, sample). The sample axis is short and read whole far more
# often than the lateral axes, so it gets the largest chunk.
DEFAULT_CHUNKS: tuple[int, int, int] = (64, 64, 128)


def default_compressor() -> BloscCodec:
    """The shared on-disk codec: Blosc+zstd, byte-shuffled, level 3."""
    return BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)


def _logger(logger: logging.Logger | None) -> logging.Logger:
    return logger or logging.getLogger(__name__)


def _json_fill(fill: float) -> float | str:
    """Zarr attributes are JSON, which has no NaN literal; store it as a string."""
    return "nan" if (isinstance(fill, float) and math.isnan(fill)) else float(fill)


def _geometry_attrs(
    geometry: Mapping[str, Any],
    shape: tuple[int, int, int],
    fill: float,
    n_live: int,
    extra: Mapping[str, Any] | None,
) -> dict:
    """Assemble the canonical attribute dict every store carries.

    ``geometry`` must provide survey-coordinate ranges and the vertical sampling
    so a reader can translate inline/crossline/z *labels* to array indices:
    ``iline_range``, ``xline_range``, ``z_range``, ``sample_interval`` (and
    optionally ``z_start``, defaulting to ``z_range[0]``).
    """
    n_il, n_xl, n_samp = shape
    il_lo, il_hi = geometry["iline_range"]
    xl_lo, xl_hi = geometry["xline_range"]
    z_lo, z_hi = geometry["z_range"]
    attrs = {
        "iline_range": [int(il_lo), int(il_hi)],
        "xline_range": [int(xl_lo), int(xl_hi)],
        "z_range": [float(z_lo), float(z_hi)],
        "sample_interval": float(geometry["sample_interval"]),
        "z_start": float(geometry.get("z_start", z_lo)),
        "n_samples": int(n_samp),
        "grid_cells": int(n_il * n_xl),
        "live_traces": int(n_live),
        "missing_traces": int(n_il * n_xl - n_live),
        "fill_value": _json_fill(fill),
    }
    transform = geometry.get("transform")
    if transform is not None:                   # a host that knows its georeference
        from seisvtk.io.survey import GridTransform
        attrs["survey_transform"] = GridTransform.from_dict(transform).as_dict()
    if extra:
        attrs.update(dict(extra))
    return attrs


def write_bands_to_zarr(
    out: str,
    *,
    shape: tuple[int, int, int],
    geometry: Mapping[str, Any],
    bands: Iterable[tuple[int, int, np.ndarray, np.ndarray]],
    chunks: tuple[int, int, int] = DEFAULT_CHUNKS,
    fill: float = 0.0,
    dtype: str = "float32",
    compressor: BloscCodec | None = None,
    extra_attrs: Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict:
    """Write a cube to a chunked, compressed Zarr from an iterator of inline bands.

    Each item in ``bands`` is ``(b0, b1, buf, mask)``: the half-open inline range
    ``[b0, b1)``, the dense samples ``buf`` of shape ``(b1 - b0, n_xl, n_samp)``,
    and a boolean ``mask`` of shape ``(b1 - b0, n_xl)`` marking which (inline,
    crossline) cells hold a real trace. Bands are written once, in order; nothing
    is accumulated, so memory is bounded by the largest band the caller yields.

    Returns the store's final attribute dict.
    """
    n_il, n_xl, n_samp = shape
    comp = compressor or default_compressor()
    log = _logger(logger)

    cube = zarr.create_array(
        store=out, shape=(n_il, n_xl, n_samp), chunks=chunks,
        dtype=dtype, fill_value=fill, compressors=[comp],
    )
    live = zarr.create_array(
        store=f"{out}/live", shape=(n_il, n_xl),
        chunks=(chunks[0], n_xl), dtype="bool", fill_value=False,
    )

    log.info(f"Writing {n_il} x {n_xl} x {n_samp} -> {out}")
    written = 0
    for b0, b1, buf, mask in bands:
        cube[b0:b1] = buf
        live[b0:b1] = mask
        written += b1 - b0
        log.info(f"  inlines {b0}..{b1} ({written}/{n_il})")

    n_live = int(live[:].sum())
    attrs = _geometry_attrs(geometry, shape, fill, n_live, extra_attrs)
    cube.attrs.update(attrs)
    log.info(f"Done. live {n_live}/{n_il * n_xl}, missing {n_il * n_xl - n_live}")
    return dict(cube.attrs)


def write_array_to_zarr(
    out: str,
    array: Any,
    geometry: Mapping[str, Any],
    *,
    chunk_il: int = 64,
    chunk_xl: int = 64,
    chunk_samp: int = 128,
    fill: float = float("nan"),
    stride: tuple[int, int, int] = (1, 1, 1),
    logger: logging.Logger | None = None,
    extra_attrs: Mapping[str, Any] | None = None,
) -> dict:
    """Stream a dense ``(n_inline, n_crossline, n_sample)`` array into a Zarr.

    ``array`` is anything that supports lazy band slicing ``array[b0:b1]`` and has
    ``.shape`` — a numpy array, an ``h5py`` dataset, another ``zarr`` array. Only
    one inline band is in memory at a time.

    Undefined cells are expected to be ``NaN`` (the natural "no data" marker for a
    property volume living between horizons). ``fill`` defaults to ``NaN`` so the
    stored cube preserves them exactly, and a cell is ``live`` if any sample in its
    column is finite. ``stride`` decimates ``(il, xl, z)`` for a quick preview.

    Note: geometry ``iline_range``/``xline_range``/``z_range`` describe the data
    *as written*, so a caller that strides must pass already-strided ranges/step.
    """
    log = _logger(logger)
    si, sj, sk = stride
    n_il = len(range(0, array.shape[0], si))
    n_xl = len(range(0, array.shape[1], sj))
    n_samp = len(range(0, array.shape[2], sk))
    chunks = (chunk_il, chunk_xl, chunk_samp)
    src_band = chunk_il * si  # source inlines consumed per written band

    def bands():
        for b0 in range(0, n_il, chunk_il):
            s0 = b0 * si
            s1 = min(s0 + src_band, array.shape[0])
            raw = np.asarray(array[s0:s1], dtype="float32")
            buf = raw[::si, ::sj, ::sk]
            b1 = b0 + buf.shape[0]
            mask = np.isfinite(buf).any(axis=2)
            yield b0, b1, buf, mask

    return write_bands_to_zarr(
        out, shape=(n_il, n_xl, n_samp), geometry=geometry, bands=bands(),
        chunks=chunks, fill=fill, logger=logger, extra_attrs=extra_attrs,
    )
