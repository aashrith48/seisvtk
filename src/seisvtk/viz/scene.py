from pathlib import Path

from seisvtk.viz.slice_scene import SliceScene


def render_three_slices(zarr_path: Path | str, *,
                        iline: int,
                        xline: int,
                        z: float,
                        vertical_exaggeration: float = 1.0,
                        out: str = "three_slice.png") -> str:
    """Render the inline/crossline/z slices together in one 3D box to a PNG.

    Thin convenience wrapper around :class:`SliceScene` for a one-shot render.
    """
    scene = SliceScene(zarr_path, iline=iline, xline=xline, z=z,
                       vertical_exaggeration=vertical_exaggeration)
    return scene.render_to_png(out)
