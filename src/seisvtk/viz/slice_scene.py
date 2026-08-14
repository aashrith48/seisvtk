import vtk
import numpy as np
from pathlib import Path

from seisvtk.io.volume import Volume
from seisvtk.viz.bridge import slice_to_image_data
from seisvtk.viz.actors import grey_image_slice
from seisvtk.viz.colormaps import build_lut


class SliceScene:
    """Three orthogonal seismic sections in one 3D box, with out-of-core refill.

    The scene (renderer, three image-slice actors, cube outline, camera) is built
    once. Moving a plane via :meth:`set_inline` / :meth:`set_crossline` /
    :meth:`set_z` re-reads *only that slab* from the ``Volume`` — touching just the
    chunks it crosses — and swaps that actor's input and position. The greyscale
    clip is fixed at construction so brightness stays stable while scrubbing.

    Hand :attr:`render_window` to trame (Phase 4) for interactive streaming.
    """

    def __init__(self, zarr_path: Path | str, *,
                 iline: int, xline: int, z: float,
                 vertical_exaggeration: float = 1.0,
                 offscreen: bool = True):
        self.vol = Volume(zarr_path=Path(zarr_path))
        self.md = self.vol.metadata
        self.n_il, self.n_xl, self.n_samp = self.vol.shape
        self.ve = vertical_exaggeration
        self.spacing = (1.0, 1.0, self.ve)
        self.iline, self.xline, self.z = iline, xline, z

        # Initial slabs + one shared greyscale clip (fixed for stable scrubbing).
        il = self._read(iline=iline)
        xl = self._read(xline=xline)
        zs = self._read(z=z)
        pooled = np.concatenate([il.ravel(), xl.ravel(), zs.ravel()])
        self.clip = float(np.percentile(np.abs(pooled), 98))

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.12, 0.12, 0.14)

        self._a_il = grey_image_slice(il, self.clip, self.spacing)
        self._a_xl = grey_image_slice(xl, self.clip, self.spacing)
        self._a_z = grey_image_slice(zs, self.clip, self.spacing)
        self._actors = (self._a_il, self._a_xl, self._a_z)
        for a in self._actors:
            self.ren.AddActor(a)
        self._place_all()
        self.colormap = "Greyscale"
        self.set_colormap(self.colormap)

        self._add_outline()

        self.rw = vtk.vtkRenderWindow()
        self.rw.SetOffScreenRendering(offscreen)
        self.rw.AddRenderer(self.ren)
        self.rw.SetSize(1000, 1000)
        self._setup_camera()
        

    # ---- survey-coordinate -> array-index helpers ----
    def _il_index(self, n: int) -> int:
        return n - self.md["iline_range"][0]

    def _xl_index(self, n: int) -> int:
        return n - self.md["xline_range"][0]

    def _z_index(self, v: float) -> int:
        return int(round((v - self.md["z_range"][0]) / self.md["sample_interval"]))

    def _read(self, **kw) -> np.ndarray:
        """Read a single slab (touches only the chunks it crosses)."""
        return self.vol.seismic_slice(self.vol.get_array_index(**kw))

    def _refill(self, actor: vtk.vtkImageSlice, slab: np.ndarray) -> None:
        """Swap an actor's image in place; orientation/window-level are unchanged."""
        img = slice_to_image_data(slab)
        img.SetSpacing(*self.spacing)
        mapper = actor.GetMapper()
        mapper.SetInputData(img)
        mapper.SetSliceNumber(0)

    def _place_all(self) -> None:
        self._a_il.SetPosition(self._il_index(self.iline), 0, 0)
        self._a_xl.SetPosition(0, self._xl_index(self.xline), 0)
        self._a_z.SetPosition(0, 0, self._z_index(self.z) * self.ve)

    def _add_outline(self) -> None:
        outline = vtk.vtkOutlineSource()
        outline.SetBounds(0, self.n_il - 1, 0, self.n_xl - 1,
                          0, (self.n_samp - 1) * self.ve)
        om = vtk.vtkPolyDataMapper()
        om.SetInputConnection(outline.GetOutputPort())
        oa = vtk.vtkActor()
        oa.SetMapper(om)
        oa.GetProperty().SetColor(1, 1, 1)
        self.ren.AddActor(oa)
        self._outline = outline

    def _setup_camera(self) -> None:
        cam = self.ren.GetActiveCamera()
        cam.SetViewUp(0, 0, -1)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetPosition(1, -1, -1)
        self.ren.ResetCamera()
        cam.Zoom(1.4)
        self.ren.ResetCameraClippingRange()

    # ---- public refill API: re-reads only the moved slab ----
    def set_inline(self, n: int) -> None:
        self.iline = n
        self._refill(self._a_il, self._read(iline=n))
        self._a_il.SetPosition(self._il_index(n), 0, 0)

    def set_crossline(self, n: int) -> None:
        self.xline = n
        self._refill(self._a_xl, self._read(xline=n))
        self._a_xl.SetPosition(0, self._xl_index(n), 0)

    def set_z(self, v: float) -> None:
        self.z = v
        self._refill(self._a_z, self._read(z=v))
        self._a_z.SetPosition(0, 0, self._z_index(v) * self.ve)

    def reset_camera(self) -> None:
        """Refit the camera to the default 3/4 view with depth pointing down."""
        self._setup_camera()

    def set_vertical_exaggeration(self, ve: float) -> None:
        """Rescale the depth (z) axis live: image spacing, z-slice position, box."""
        self.ve = ve
        self.spacing = (1.0, 1.0, ve)
        for a in self._actors:
            a.GetMapper().GetInput().SetSpacing(*self.spacing)
        self._a_z.SetPosition(0, 0, self._z_index(self.z) * ve)
        self._outline.SetBounds(0, self.n_il - 1, 0, self.n_xl - 1,
                                0, (self.n_samp - 1) * ve)
        self.ren.ResetCameraClippingRange()

    def set_colormap(self, name: str) -> None:
        """Apply a named color table to all three slices (see colormaps.NAMES)."""
        lut = build_lut(name, self.clip)
        for a in self._actors:
            prop = a.GetProperty()
            prop.SetLookupTable(lut)
            prop.UseLookupTableScalarRangeOn()
        self.colormap = name

    @property
    def render_window(self) -> vtk.vtkRenderWindow:
        return self.rw

    def render_to_png(self, out: str = "three_slice.png") -> str:
        self.rw.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.rw)
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(out)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        return out
