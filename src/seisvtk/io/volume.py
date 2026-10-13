import re

import zarr
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

# A URL scheme like ``s3://`` / ``gs://`` / ``https://`` (but not a bare path or a
# Windows drive ``D:\``) — opened through an fsspec-backed Zarr store so chunks
# are fetched over the network on demand, never bulk-downloaded.
_URL_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)


def _open_readonly_array(src) -> "zarr.Array":
    """Open *src* as a read-only Zarr array, local or remote, without materialising it.

    Accepts a local path, a remote URL (``s3://bucket/cube.zarr`` and the like, via
    fsspec — the backend, e.g. ``s3fs``, must be installed), a pre-built Zarr store,
    or an already-open array. In every case only chunks that are later indexed are
    read, so a 500 GB cube on S3 costs a few range requests per slice, not a copy.
    """
    if isinstance(src, zarr.Array):
        return src
    from zarr.abc.store import Store
    if isinstance(src, Store):
        return zarr.open_array(store=src, mode="r")
    if isinstance(src, str) and _URL_SCHEME.match(src):
        from zarr.storage import FsspecStore
        return zarr.open_array(store=FsspecStore.from_url(src, read_only=True), mode="r")
    return zarr.open_array(src, mode="r")


@dataclass
class Volume:
    """Lazy, out-of-core reader for a seismic cube stored as a Zarr array.

    Wraps a Zarr store written by ``convert_segy_to_zarr`` and serves slices by
    *survey coordinate* (inline / crossline number, or vertical value) rather than
    raw array index. The underlying array is never materialised; only the chunks a
    requested slice touches are read — from local disk or, for an ``s3://`` (or
    other fsspec) URL, from object storage over the network. The cube itself never
    lives in memory, so even a multi-GB volume on S3 costs only the few chunks a
    visible slice touches.

    Attributes:
        zarr_path: The store to open — a local path, a remote URL (``s3://…`` via
            fsspec), a pre-built Zarr store, or an already-open ``zarr.Array``.
        array: Lazy ``zarr.Array`` handle. Indexing it (``array[i, :, :]``) is what
            triggers the actual read; the object itself holds no samples.
        metadata: The store's ``.attrs`` as a plain dict — survey geometry written
            at convert time (``iline_range``, ``xline_range``, ``z_range``,
            ``sample_interval``, live-trace counts, etc.).
        shape: ``(n_inlines, n_crosslines, n_samples)`` — the array's dimensions in
            index space. Distinct from the survey ranges in ``metadata``, which are
            the axis *labels*.
    """

    zarr_path: "Path | str | object"

    def __post_init__(self):
        """Open the store lazily and cache its geometry.

        Reads only the Zarr metadata (``zarr.json``), not the sample data, so this
        is instant even for a 500 GB cube, and the same whether the store is local
        or on S3.
        """
        self.array = _open_readonly_array(self.zarr_path)
        self.metadata = dict(self.array.attrs)
        self.shape = self.array.shape

    @property
    def transform(self):
        """The cube's survey affine as a :class:`~seisvtk.io.survey.GridTransform`.

        Rebuilt from the ``survey_transform`` attribute (the six affine
        coefficients) written at convert time when the SEG-Y carried world
        coordinates, or by a host that knows its own georeference. ``None`` when
        the store never recorded one — deviated wells then need an explicit
        transform (e.g. from bin spacing). This is the map a well's world ``(X, Y)``
        goes through to land on inline/crossline.
        """
        if not hasattr(self, "_transform"):
            st = self.metadata.get("survey_transform")
            if st:
                from seisvtk.io.survey import GridTransform
                self._transform = GridTransform.from_dict(st)
            else:
                self._transform = None
        return self._transform

    def seismic_slice(self, idx):
        """Index the underlying Zarr array directly, in *array-index* space.

        A raw passthrough with no survey-coordinate conversion. Pass it the tuple
        from :meth:`get_array_index`, e.g. ``vol[vol.get_array_index(iline=300)]``.
        """
        return self.array[idx]

    def get_array_index(self,
                        xline: int | None = None,
                        iline: int | None = None,
                        z: float | None = None):
        """Convert survey coordinates to a Zarr index tuple for one slice.

        Each axis left as ``None`` becomes a full slice (``:``), so passing a single
        coordinate yields the plane orthogonal to that axis:

            * ``iline=300``  -> ``(idx, :, :)``  one inline section
            * ``xline=500``  -> ``(:, idx, :)``  one crossline section
            * ``z=1200.0``   -> ``(:, :, idx)``  one time/depth slice

        Inline and crossline convert by subtracting the range start
        (``label - range[0]``); the vertical coordinate additionally divides by the
        sample interval and rounds to the nearest sample.

        Args:
            xline: Crossline *number* (survey label), not an array index.
            iline: Inline *number* (survey label), not an array index.
            z: Vertical coordinate in the cube's units (ms or m), from ``z_range``.

        Returns:
            A ``(iline_idx, xline_idx, z_idx)`` tuple for use with ``__getitem__``;
            unspecified axes are ``slice(None)``.
        """

        # Default every axis to a full slice (":")
        iline_arr_idx: int | slice = slice(None)
        xline_arr_idx: int | slice = slice(None)
        z_arr_idx: int | slice = slice(None)

        #Get slice index if requested
        if iline is not None:
            iline_arr_n = iline - self.metadata['iline_range'][0]
            iline_arr_idx = slice(iline_arr_n,iline_arr_n+1)
        if xline is not None:
            xline_arr_n = xline - self.metadata['xline_range'][0]
            xline_arr_idx = slice(xline_arr_n,xline_arr_n+1)
        if z is not None:
            z_arr_n = int(round(
                (z - self.metadata['z_range'][0]) / self.metadata['sample_interval']
            ))
            z_arr_idx = slice(z_arr_n,z_arr_n+1)

        return (iline_arr_idx, xline_arr_idx, z_arr_idx)