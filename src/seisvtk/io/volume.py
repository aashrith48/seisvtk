import zarr
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Volume:
    """Lazy, out-of-core reader for a seismic cube stored as a Zarr array.

    Wraps a Zarr store written by ``convert_segy_to_zarr`` and serves slices by
    *survey coordinate* (inline / crossline number, or vertical value) rather than
    raw array index. The underlying array is never materialised; only the chunks a
    requested slice touches are read from disk (or, later, from object storage).

    Attributes:
        zarr_path: Path to the ``.zarr`` store.
        array: Lazy ``zarr.Array`` handle. Indexing it (``array[i, :, :]``) is what
            triggers the actual read; the object itself holds no samples.
        metadata: The store's ``.attrs`` as a plain dict — survey geometry written
            at convert time (``iline_range``, ``xline_range``, ``z_range``,
            ``sample_interval``, live-trace counts, etc.).
        shape: ``(n_inlines, n_crosslines, n_samples)`` — the array's dimensions in
            index space. Distinct from the survey ranges in ``metadata``, which are
            the axis *labels*.
    """

    zarr_path: Path | str

    def __post_init__(self):
        """Open the store lazily and cache its geometry.

        Reads only the Zarr metadata (``zarr.json``), not the sample data, so this
        is instant even for a 500 GB cube.
        """
        self.array = zarr.open_array(self.zarr_path, mode="r")
        self.metadata = dict(self.array.attrs)
        self.shape = self.array.shape

    def __getitem__(self, idx):
        """Index the underlying Zarr array directly, in *array-index* space.

        A raw passthrough with no survey-coordinate conversion. Pass it the tuple
        from :meth:`get_array_index`, e.g. ``vol[vol.get_array_index(inline=300)]``.
        """
        return self.array[idx]

    def get_array_index(self,
                        xline: int | None = None,
                        inline: int | None = None,
                        z: float | None = None):
        """Convert survey coordinates to a Zarr index tuple for one slice.

        Each axis left as ``None`` becomes a full slice (``:``), so passing a single
        coordinate yields the plane orthogonal to that axis:

            * ``inline=300`` -> ``(idx, :, :)``  one inline section
            * ``xline=500``  -> ``(:, idx, :)``  one crossline section
            * ``z=1200.0``   -> ``(:, :, idx)``  one time/depth slice

        Inline and crossline convert by subtracting the range start
        (``label - range[0]``); the vertical coordinate additionally divides by the
        sample interval and rounds to the nearest sample.

        Args:
            xline: Crossline *number* (survey label), not an array index.
            inline: Inline *number* (survey label), not an array index.
            z: Vertical coordinate in the cube's units (ms or m), from ``z_range``.

        Returns:
            A ``(inline_idx, xline_idx, z_idx)`` tuple for use with ``__getitem__``;
            unspecified axes are ``slice(None)``.
        """

        # Default every axis to a full slice (":")
        iline_arr_idx: int | slice = slice(None)
        xline_arr_idx: int | slice = slice(None)
        z_arr_idx: int | slice = slice(None)

        #Get slice index if requested
        if inline is not None:
            iline_arr_idx = inline - self.metadata['iline_range'][0]
        if xline is not None:
            xline_arr_idx = xline - self.metadata['xline_range'][0]
        if z is not None:
            z_arr_idx = int(round(
                (z - self.metadata['z_range'][0]) / self.metadata['sample_interval']
            ))

        return (iline_arr_idx, xline_arr_idx, z_arr_idx)