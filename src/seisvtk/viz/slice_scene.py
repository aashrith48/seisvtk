import vtk
import numpy as np
from pathlib import Path

from seisvtk.io.volume import Volume
from seisvtk.io.horizon import Horizon
from seisvtk.io.well import Well
from seisvtk.viz.bridge import slice_to_image_data
from seisvtk.viz.actors import grey_image_slice
from seisvtk.viz.colormaps import build_lut, build_property_lut, NAMES, PROPERTY_NAMES
from seisvtk.viz.horizon_actor import horizon_surface_actor, PALETTE
from seisvtk.viz.well_actor import well_tube_actor, gr_log_lut


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
                 offscreen: bool = True,
                 mode: str = "amplitude",
                 colormap: str | None = None):
        self.vol = Volume(zarr_path=zarr_path)        # local path, s3:// URL, or store
        self.md = self.vol.metadata
        self.n_il, self.n_xl, self.n_samp = self.vol.shape
        self.ve = vertical_exaggeration
        self.spacing = (1.0, 1.0, self.ve)
        self.iline, self.xline, self.z = iline, xline, z
        if mode not in ("amplitude", "property"):
            raise ValueError(f"mode must be 'amplitude' or 'property', got {mode!r}")
        self.mode = mode
        self.colormap_names = PROPERTY_NAMES if mode == "property" else NAMES

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.12, 0.12, 0.14)

        # Draped overlays (horizons, wells): (name, actor) pairs, kept so vertical
        # exaggeration and visibility toggles reach them too.
        self._horizon_actors: list[tuple[str, vtk.vtkActor]] = []
        self._well_actors: list[tuple[str, vtk.vtkActor]] = []
        self._scalar_bars: list[tuple[str, vtk.vtkScalarBarActor]] = []
        self._section_bar: vtk.vtkScalarBarActor | None = None

        # Read the three middle slabs, fix the color scale, build the slice actors.
        self._install_slabs(colormap)

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

    def _install_slabs(self, preferred_cmap: str | None) -> None:
        """Read the three middle slabs, fix the color scale, (re)build the slices.

        Shared by ``__init__`` and :meth:`load_volume`. The greyscale/property clip
        is computed once here (NaN-aware: property cubes are mostly no-data between
        horizons, and even amplitude cubes can carry dead/NaN traces) so brightness
        stays put while scrubbing. Any previous slice actors are removed first, so
        this is safe to call again on a volume swap.
        """
        il = self._read(iline=self.iline)
        xl = self._read(xline=self.xline)
        zs = self._read(z=self.z)
        pooled = np.concatenate([il.ravel(), xl.ravel(), zs.ravel()])
        finite = pooled[np.isfinite(pooled)]
        if self.mode == "property":
            if finite.size:
                self.vmin, self.vmax = (float(v) for v in np.percentile(finite, [2, 98]))
            else:
                self.vmin, self.vmax = 0.0, 1.0
            if self.vmax <= self.vmin:
                self.vmax = self.vmin + 1.0
            self.clip = self.vmax
            default_cmap = preferred_cmap or "Viridis"
        else:
            self.clip = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
            if self.clip <= 0:
                self.clip = 1.0
            self.vmin, self.vmax = -self.clip, self.clip
            default_cmap = preferred_cmap or "Greyscale"

        for a in getattr(self, "_actors", ()):      # drop old slices on a swap
            self.ren.RemoveActor(a)
        self._a_il = grey_image_slice(il, self.clip, self.spacing)
        self._a_xl = grey_image_slice(xl, self.clip, self.spacing)
        self._a_z = grey_image_slice(zs, self.clip, self.spacing)
        self._actors = (self._a_il, self._a_xl, self._a_z)
        for a in self._actors:
            self.ren.AddActor(a)
        self._place_all()
        self.colormap = default_cmap
        self.set_colormap(self.colormap)

    def load_volume(self, zarr_path: Path | str) -> None:
        """Swap the displayed cube in place, keeping the same render window.

        Rebuilds the three slices (and the bounding box) from a different Zarr
        store without touching the renderer or render window, so a trame
        ``VtkRemoteView`` already bound to :attr:`render_window` keeps streaming.
        Draped overlays belong to the old survey grid, so they are cleared; the
        planes recentre on the new cube and the camera refits. The caller is
        responsible for pushing the new survey ranges to any UI sliders.
        """
        for _, a in self._horizon_actors + self._well_actors:
            self.ren.RemoveActor(a)
        self._horizon_actors.clear()
        self._well_actors.clear()

        self.vol = Volume(zarr_path=zarr_path)        # local path, s3:// URL, or store
        self.md = self.vol.metadata
        self.n_il, self.n_xl, self.n_samp = self.vol.shape
        il_lo, il_hi = self.md["iline_range"]
        xl_lo, xl_hi = self.md["xline_range"]
        z_lo, z_hi = self.md["z_range"]
        self.iline = (il_lo + il_hi) // 2
        self.xline = (xl_lo + xl_hi) // 2
        self.z = (z_lo + z_hi) / 2

        self._install_slabs(self.colormap)
        self._outline.SetBounds(0, self.n_il - 1, 0, self.n_xl - 1,
                                0, (self.n_samp - 1) * self.ve)
        self._setup_camera()

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
        self._outline_actor = oa

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
        for _, a in self._horizon_actors + self._well_actors:
            a.SetScale(1.0, 1.0, ve)
        self.ren.ResetCameraClippingRange()

    def set_colormap(self, name: str) -> None:
        """Apply a named color table to all three slices.

        Amplitude mode uses the diverging maps in ``colormaps.NAMES`` over
        ``[-clip, +clip]``; property mode uses the sequential maps in
        ``colormaps.PROPERTY_NAMES`` over ``[vmin, vmax]`` with transparent
        no-data. ``self.colormap_names`` lists the valid names for the mode.
        """
        if self.mode == "property":
            lut = build_property_lut(name, self.vmin, self.vmax)
        else:
            lut = build_lut(name, self.clip)
        for a in self._actors:
            prop = a.GetProperty()
            prop.SetLookupTable(lut)
            prop.UseLookupTableScalarRangeOn()
        self.colormap = name
        # Keep the section legend (if shown) pointing at the live color table.
        self._section_lut = lut
        if getattr(self, "_section_bar", None) is not None:
            self._section_bar.SetLookupTable(lut)

    # ---- draped overlays ----
    def add_horizon(self, horizon: Horizon, *,
                    color: tuple[float, float, float] | None = None,
                    opacity: float = 1.0,
                    name: str | None = None) -> vtk.vtkActor:
        """Drape an interpreted horizon over the sections.

        The horizon's survey ``(il, xl, z)`` picks are gridded and mapped into the
        scene's index frame: ``il - iline_range[0]``, ``xl - xline_range[0]`` and
        ``(z - z_range[0]) / sample_interval``. Picks must be in the same vertical
        unit as the cube. Colors cycle through ``PALETTE`` unless one is given.
        """
        iu, xu, Z = horizon.to_grid()
        il_index = iu - self.md["iline_range"][0]
        xl_index = xu - self.md["xline_range"][0]
        z_index = (Z - self.md["z_range"][0]) / self.md["sample_interval"]
        color = color or PALETTE[len(self._horizon_actors) % len(PALETTE)]
        actor = horizon_surface_actor(il_index, xl_index, z_index,
                                      color=color, opacity=opacity)
        actor.SetScale(1.0, 1.0, self.ve)
        self.ren.AddActor(actor)
        self._horizon_actors.append((name or horizon.name, actor))
        return actor

    @property
    def horizons(self) -> list[str]:
        return [n for n, _ in self._horizon_actors]

    def set_horizon_visibility(self, name: str, visible: bool) -> None:
        for n, a in self._horizon_actors:
            if n == name:
                a.SetVisibility(1 if visible else 0)

    # ---- section / box visibility ----
    _SECTION = {"inline": "_a_il", "crossline": "_a_xl", "z": "_a_z"}

    def set_section_visibility(self, axis: str, visible: bool) -> None:
        """Show/hide one section plane (``axis`` in 'inline'/'crossline'/'z')."""
        getattr(self, self._SECTION[axis]).SetVisibility(1 if visible else 0)

    def set_outline_visibility(self, visible: bool) -> None:
        self._outline_actor.SetVisibility(1 if visible else 0)

    def add_well(self, well: Well, *,
                 color_by: str | None = None,
                 radius: float = 6.0,
                 lut: vtk.vtkScalarsToColors | None = None,
                 vmin: float | None = None, vmax: float | None = None,
                 name: str | None = None) -> vtk.vtkActor:
        """Drape a well as a log-colored tube over the sections.

        ``color_by`` picks the log (defaults to the first); ``vmin``/``vmax`` set
        the color range (robust 5/95 percentiles of the log if unset). The
        trajectory is mapped into the scene's index frame like a horizon, and
        samples with no valid two-way time are dropped.
        """
        log_name = color_by or (next(iter(well.logs)) if well.logs else None)
        values = (np.asarray(well.logs[log_name], float) if log_name
                  else np.zeros(well.z.size))

        il_index = np.asarray(well.il, float) - self.md["iline_range"][0]
        xl_index = np.asarray(well.xl, float) - self.md["xline_range"][0]
        z_index = (np.asarray(well.z, float) - self.md["z_range"][0]) / self.md["sample_interval"]

        keep = np.isfinite(z_index)
        il_index, xl_index, z_index, values = (
            il_index[keep], xl_index[keep], z_index[keep], values[keep])

        finite = values[np.isfinite(values)]
        if vmin is None:
            vmin = float(np.percentile(finite, 5)) if finite.size else 0.0
        if vmax is None:
            vmax = float(np.percentile(finite, 95)) if finite.size else 150.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        values = np.nan_to_num(values, nan=vmin)
        lut = lut or gr_log_lut(vmin, vmax)

        actor = well_tube_actor(il_index, xl_index, z_index, values,
                                lut=lut, vrange=(vmin, vmax), radius=radius)
        actor.SetScale(1.0, 1.0, self.ve)
        self.ren.AddActor(actor)
        self._well_actors.append((name or well.name, actor))
        return actor

    @property
    def wells(self) -> list[str]:
        return [n for n, _ in self._well_actors]

    def set_well_visibility(self, name: str, visible: bool) -> None:
        for n, a in self._well_actors:
            if n == name:
                a.SetVisibility(1 if visible else 0)

    def add_scalar_bar(self, lut: vtk.vtkScalarsToColors, title: str, *,
                       name: str | None = None, slot: int = 0,
                       n_labels: int = 5) -> vtk.vtkScalarBarActor:
        """Add a named color-scale legend to the view.

        ``name`` keys it for :meth:`set_scalar_bar_visibility`; ``slot`` stacks
        bars left of the right edge (0 = rightmost) so several don't overlap.
        """
        sb = vtk.vtkScalarBarActor()
        sb.SetLookupTable(lut)
        sb.SetTitle(title)
        sb.SetNumberOfLabels(n_labels)
        sb.UnconstrainedFontSizeOn()
        sb.SetWidth(0.07)
        sb.SetHeight(0.38)
        sb.SetBarRatio(0.22)
        sb.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        sb.GetPositionCoordinate().SetValue(0.92 - 0.12 * slot, 0.09)
        for tp in (sb.GetTitleTextProperty(), sb.GetLabelTextProperty()):
            tp.SetColor(1, 1, 1)
            tp.SetFontSize(13)
            tp.ItalicOff()
            tp.ShadowOn()
        sb.SetLabelFormat("%.0f")
        self.ren.AddActor2D(sb)
        self._scalar_bars.append((name or title, sb))
        return sb

    def add_section_scalar_bar(self, title: str, *, slot: int = 0,
                               n_labels: int = 5) -> vtk.vtkScalarBarActor:
        """Legend for the section color scale; it follows colormap/range changes."""
        bar = self.add_scalar_bar(self._section_lut, title, name="__section__",
                                  slot=slot, n_labels=n_labels)
        self._section_bar = bar
        return bar

    def set_scalar_bar_visibility(self, name: str, visible: bool) -> None:
        for n, sb in self._scalar_bars:
            if n == name:
                sb.SetVisibility(1 if visible else 0)

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
