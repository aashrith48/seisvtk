import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from seisvtk.logging_config import setup_logging
import logging
import segyio
import numpy as np


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
  ) -> dict:  
      """Convert a 3D SEG-Y to a chunked, compressed Zarr array.

      One unified path handles dense and ragged surveys: every trace is scattered
      into its (inline, crossline) grid cell, and any cell that receives no trace
      keeps ``fill``. A dense survey simply fills every cell. A boolean ``live``
      mask records which cells are real, so downstream code never has to guess.

      Streams in inline bands aligned to the chunk boundary, so peak memory is one
      band (chunk_il x n_xline x n_samples), not the whole cube.
      """
      log = logger or logging.getLogger(__spec__.name if __spec__ else __name__)
      comp = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)

      with segyio.open(filepath, "r", ignore_geometry=True) as f:
          n_samp = len(f.samples)
          il_hdr = f.attributes(iline)[:]
          xl_hdr = f.attributes(xline)[:]

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
          for b0 in range(0, n_il, chunk_il):
              b1 = min(b0 + chunk_il, n_il)
              in_band = np.flatnonzero((il_idx >= b0) & (il_idx < b1))

              buf = np.full((b1 - b0, n_xl, n_samp), fill, dtype=np.float32)
              mask = np.zeros((b1 - b0, n_xl), dtype=bool)
              for t in in_band:
                  buf[il_idx[t] - b0, xl_idx[t], :] = f.trace[t]
                  mask[il_idx[t] - b0, xl_idx[t]] = True

              cube[b0:b1] = buf
              live[b0:b1] = mask

          n_live = n_il * n_xl - int((~live[:]).sum())
          cube.attrs.update({
              "iline_byte": int(iline),
              "xline_byte": int(xline),
              "iline_range": [int(il_labels[0]), int(il_labels[-1])],
              "xline_range": [int(xl_labels[0]), int(xl_labels[-1])],
              "n_samples": int(n_samp),
              "sample_interval": float(segyio.tools.dt(f) / 1000.0),  # ms
              "z_start": float(f.samples[0]),
              "n_traces": int(f.tracecount),
              "grid_cells": int(n_il * n_xl),
              "live_traces": int(n_live),
              "missing_traces": int(n_il * n_xl - n_live),
              "fill_value": float(fill),
          })

      log.info(f"Done. live {n_live}/{n_il * n_xl}, "
               f"missing {n_il * n_xl - n_live}")
      return dict(cube.attrs)