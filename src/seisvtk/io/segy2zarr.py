import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from seisvtk.io.logging_config import setup_logging
import logging
from typing import Callable
import segyio
import numpy as np

# SEG-Y Rev1 trace-header byte locations for the bin-centre world coordinates
# and the coordinate scalar (same defaults QEarth's geometry inference uses).
CDP_X_BYTE, CDP_Y_BYTE, SCALAR_BYTE = 181, 185, 71


def _apply_coord_scalar(coords, scalar):
    """Apply the SEG-Y SourceGroupScalar: >0 multiplies, <0 divides, 0 is 1."""
    coords = np.asarray(coords, dtype=float)
    scalar = np.asarray(scalar, dtype=float)
    factor = np.where(scalar > 0, scalar,
                      np.where(scalar < 0, 1.0 / np.abs(np.where(scalar == 0, 1, scalar)), 1.0))
    return coords * factor


def _fit_survey_transform(il_hdr, xl_hdr, x, y, *, max_points=20000):
    """Least-squares world affine from trace coordinates; ``None`` if unusable.

    Returns the six-coefficient dict for a store attribute, or ``None`` when the
    SEG-Y carried no dependable CDP X/Y — so a coordinate-free survey simply
    stores no transform rather than a plausible-looking wrong one. Guards:

    * drop non-finite and ``(0, 0)`` traces (dead / un-populated coordinate words);
    * require both inline and crossline to actually vary (else the fit is rank
      deficient and ``lstsq`` would hand back a quietly wrong answer);
    * sample *evenly across the file* (``linspace``), never a fixed stride, which
      can align with the crossline period and collect a single line;
    * reject the fit if its residual exceeds half a bin, i.e. it does not
      reproduce the traces it was built from.
    """
    from seisvtk.io.survey import GridTransform

    il_hdr = np.asarray(il_hdr, float)
    xl_hdr = np.asarray(xl_hdr, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    good = np.isfinite(x) & np.isfinite(y) & ~((x == 0) & (y == 0))
    if good.sum() < 3:                                    # too few real coordinates
        return None
    il_g, xl_g, x_g, y_g = il_hdr[good], xl_hdr[good], x[good], y[good]
    if np.unique(il_g).size < 2 or np.unique(xl_g).size < 2:
        return None                                       # one line only -> rank deficient

    n = il_g.size
    idx = np.unique(np.linspace(0, n - 1, min(n, max_points)).astype(int))
    tf = GridTransform.fit(il_g[idx], xl_g[idx], x_g[idx], y_g[idx])
    if not tf.is_valid:
        return None

    # Does the fit actually reproduce the coordinates? (catches a bad solve.)
    px, py = tf.grid_to_world(il_g[idx], xl_g[idx])
    residual = float(np.max(np.hypot(px - x_g[idx], py - y_g[idx])))
    bin_il = np.hypot(tf.a_il, tf.b_il)
    bin_xl = np.hypot(tf.a_xl, tf.b_xl)
    if residual > 0.5 * min(bin_il, bin_xl):
        return None
    return tf.as_dict()


def convert_segy_to_zarr(
      filepath: str,
      iline: int,
      xline: int,
      out: str,
      *,
      chunk_il: int = 64,
      chunk_xl: int = 64,
      chunk_samp: int = 128,
      fill: float = 0.0,
      logger: logging.Logger | None = None,
      progress: Callable[[int, int], None] | None = None,
  ) -> dict:
      """Convert a 3D SEG-Y to a chunked, compressed Zarr array.

      One unified path handles dense and ragged surveys: every trace is scattered
      into its (inline, crossline) grid cell, and any cell that receives no trace
      keeps ``fill``. A dense survey simply fills every cell. A boolean ``live``
      mask records which cells are real, so downstream code never has to guess.

      Streams in inline bands aligned to the chunk boundary, so peak memory is one
      band (chunk_il x n_xline x n_samples), not the whole cube.

      ``progress``, if given, is called ``progress(done_bands, total_bands)`` after
      each inline band is written, so a caller can drive a progress bar.
      """
      log = logger or logging.getLogger(__spec__.name if __spec__ else __name__)
      comp = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)

      with segyio.open(filepath, "r", ignore_geometry=True) as f:
          n_samp = len(f.samples)
          il_hdr = f.attributes(iline)[:]
          xl_hdr = f.attributes(xline)[:]

          # World coordinates (CDP X/Y) -> the survey affine, so a well's X/Y can
          # later be tied to inline/crossline straight from the cube.
          cdp_x = _apply_coord_scalar(f.attributes(CDP_X_BYTE)[:],
                                      f.attributes(SCALAR_BYTE)[:])
          cdp_y = _apply_coord_scalar(f.attributes(CDP_Y_BYTE)[:],
                                      f.attributes(SCALAR_BYTE)[:])
          survey_transform = _fit_survey_transform(il_hdr, xl_hdr, cdp_x, cdp_y)

          il_labels = np.unique(il_hdr)
          xl_labels = np.unique(xl_hdr)
          n_il, n_xl = len(il_labels), len(xl_labels)

          # Map each trace's header value to a 0-based grid index.
          il_idx = np.searchsorted(il_labels, il_hdr)
          xl_idx = np.searchsorted(xl_labels, xl_hdr)

          cube = zarr.create_array(
              store=out, shape=(n_il, n_xl, n_samp),
              chunks=(chunk_il, chunk_xl, chunk_samp),
              dtype="float32", fill_value=fill, compressors=[comp],
          )
          live = zarr.create_array(
              store=f"{out}/live", shape=(n_il, n_xl),
              chunks=(chunk_il, n_xl), dtype="bool", fill_value=False,
          )

          log.info(f"Grid {n_il} x {n_xl} x {n_samp}, {f.tracecount} traces -> {out}")

          # Scatter one inline band at a time; write each band once.
          band_starts = range(0, n_il, chunk_il)
          n_bands = len(band_starts)
          for bi, b0 in enumerate(band_starts):
              b1 = min(b0 + chunk_il, n_il)
              in_band = np.flatnonzero((il_idx >= b0) & (il_idx < b1))

              buf = np.full((b1 - b0, n_xl, n_samp), fill, dtype=np.float32)
              mask = np.zeros((b1 - b0, n_xl), dtype=bool)
              for t in in_band:
                  buf[il_idx[t] - b0, xl_idx[t], :] = f.trace[t]
                  mask[il_idx[t] - b0, xl_idx[t]] = True

              cube[b0:b1] = buf
              live[b0:b1] = mask
              if progress is not None:
                  progress(bi + 1, n_bands)

          n_live = n_il * n_xl - int((~live[:]).sum())
          attrs = {
              "iline_byte": int(iline),
              "xline_byte": int(xline),
              "iline_range": [int(il_labels[0]), int(il_labels[-1])],
              "xline_range": [int(xl_labels[0]), int(xl_labels[-1])],
              "n_samples": int(n_samp),
              "sample_interval": float(segyio.tools.dt(f) / 1000.0),  # ms
              "z_range": [float(f.samples[0]), float(f.samples[-1])],
              "z_start": float(f.samples[0]),
              "n_traces": int(f.tracecount),
              "grid_cells": int(n_il * n_xl),
              "live_traces": int(n_live),
              "missing_traces": int(n_il * n_xl - n_live),
              "fill_value": float(fill),
          }
          if survey_transform is not None:
              attrs["survey_transform"] = survey_transform
          cube.attrs.update(attrs)

      log.info(f"Done. live {n_live}/{n_il * n_xl}, "
               f"missing {n_il * n_xl - n_live}")
      return dict(cube.attrs)