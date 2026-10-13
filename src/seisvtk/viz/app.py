import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import vtk
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import html, vuetify3, vtk as vtk_widgets

from seisvtk.io.volume import Volume
from seisvtk.io.browse import browse
from seisvtk.io.loaders import (convert_segy, default_log, load_horizons,
                                load_well, suggest_zarr_path)
from seisvtk.io.survey import GridTransform
from seisvtk.viz.slice_scene import SliceScene
from seisvtk.viz.colormaps import NAMES, PROPERTY_NAMES
from seisvtk.viz.horizon_actor import PALETTE
from seisvtk.viz.well_actor import gr_log_lut
from seisvtk.viz.interactor import OrbitPanStyle

# Horizon layouts offered in the Data panel. ``xyz`` is omitted until the viewer
# carries a survey transform to invert world X/Y (loaders.load_horizons supports
# it, so it drops in here once that lands); the three grid layouts need none.
_HZ_LAYOUTS = ("ilxl", "rows", "columns")
_HZ_LAYOUT_LABELS = {
    "ilxl": "IL / XL / Z  (one surface)",
    "rows": "Name / IL / XL / Z  (many)",
    "columns": "IL / XL / Z-per-column",
}
_WELL_SWATCH = "amber"


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(255 * c) for c in rgb)


def build(zarr_path: str | Path, *,
          mode: str = "amplitude",
          colormap: str | None = None,
          ve: float | None = None,
          horizons=None,
          wells=None,
          well_radius: float = 6.0,
          z_unit: str = "",
          value_label: str = ""):
    # Read geometry first so the initial planes start at the middle of whatever
    # cube was loaded (opening a Volume only reads metadata, not samples). Pass the
    # store through unchanged so an s3:// URL is not mangled by Path().
    md = Volume(zarr_path=zarr_path).metadata
    il_lo, il_hi = md["iline_range"]
    xl_lo, xl_hi = md["xline_range"]
    z_lo, z_hi = md["z_range"]
    z_step = md["sample_interval"]

    # Property cubes usually have far fewer samples than traces, so they want a
    # taller default exaggeration than zero-centred amplitude seismic.
    if ve is None:
        ve = 3.0 if mode == "property" else 0.25
    colormaps = PROPERTY_NAMES if mode == "property" else NAMES
    value_label = value_label or ("Property" if mode == "property" else "Amplitude")

    scene = SliceScene(
        zarr_path,
        iline=(il_lo + il_hi) // 2,
        xline=(xl_lo + xl_hi) // 2,
        z=(z_lo + z_hi) / 2,
        vertical_exaggeration=ve,
        offscreen=True,   # render headless; trame streams the framebuffer, no GL window
        mode=mode,
        colormap=colormap,
    )

    server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller

    # Overlays (horizons, wells) live in two reactive lists so the drawer can
    # render a toggle per item with `v_for` — that is what lets a file loaded at
    # runtime through the Data panel grow a new switch, not just the ones passed
    # in at build time. Each item: id (v_for key + toggle address), label
    # (display), key (the exact name handed to the scene), color, visible.
    state.hz_items = []
    state.wl_items = []

    def _add_horizon(h, *, opacity: float = 0.6):
        """Drape a horizon and give it a drawer toggle with a matching swatch."""
        color = PALETTE[len(state.hz_items) % len(PALETTE)]
        key = f"{h.name}##hz{len(state.hz_items)}"      # unique even if names repeat
        scene.add_horizon(h, color=color, opacity=opacity, name=key)
        state.hz_items = state.hz_items + [
            {"id": key, "label": h.name, "key": key,
             "color": _hex(color), "visible": True}]

    def _add_well(w, *, color_by=None, lut=None, vmin=None, vmax=None,
                  radius: float = well_radius):
        key = f"{w.name}##wl{len(state.wl_items)}"
        scene.add_well(w, color_by=color_by, radius=radius,
                       lut=lut, vmin=vmin, vmax=vmax, name=key)
        state.wl_items = state.wl_items + [
            {"id": key, "label": w.name, "key": key,
             "color": _WELL_SWATCH, "visible": True}]

    # Build-time overlays go through the same path as panel-loaded ones.
    for h in (horizons or []):
        _add_horizon(h)

    # Build-time wells share one log color scale (so colors compare across wells)
    # and the view carries a single matching legend.
    log_name = None
    if wells:
        log_name = next(iter(wells[0].logs))
        pooled = np.concatenate([np.asarray(w.logs[log_name], float) for w in wells
                                 if log_name in w.logs])
        pooled = pooled[np.isfinite(pooled)]
        wl_vmin, wl_vmax = ((float(v) for v in np.percentile(pooled, [5, 95]))
                            if pooled.size else (0.0, 150.0))
        wl_vmin, wl_vmax = float(wl_vmin), float(wl_vmax)
        well_lut = gr_log_lut(wl_vmin, wl_vmax)
        for w in wells:
            _add_well(w, color_by=log_name, lut=well_lut, vmin=wl_vmin, vmax=wl_vmax)
        scene.add_scalar_bar(well_lut, f"{log_name} (well)", name="well", slot=1)

    # Legends for every property (seismic section + the pooled well log) are
    # optional: added once but hidden, and the drawer toggles each on/off.
    scene.add_section_scalar_bar(value_label, slot=0)
    scene.set_scalar_bar_visibility("__section__", False)
    if wells:
        scene.set_scalar_bar_visibility("well", False)

    # trame's remote view drives the render loop through an interactor; the
    # render window must have one attached (never Start()ed here). The custom
    # style lets the right button (or both buttons) pan, so a zoomed-in plane
    # can be recentred without losing the orbit on the left button.
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(scene.render_window)
    interactor.SetInteractorStyle(OrbitPanStyle())

    # Lock depth (z) to screen-vertical: after every camera move, re-pin the
    # view-up to -z so the section can orbit but never roll off vertical.
    def _keep_z_vertical(caller, event):
        scene.ren.GetActiveCamera().SetViewUp(0, 0, -1)

    interactor.AddObserver("InteractionEvent", _keep_z_vertical)

    # Slider -> refill only that slab -> push a fresh frame.
    @state.change("iline")
    def _on_iline(iline, **_):
        scene.set_inline(int(iline))
        ctrl.view_update()

    @state.change("xline")
    def _on_xline(xline, **_):
        scene.set_crossline(int(xline))
        ctrl.view_update()

    @state.change("z")
    def _on_z(z, **_):
        scene.set_z(float(z))
        ctrl.view_update()

    @state.change("colormap")
    def _on_colormap(colormap, **_):
        scene.set_colormap(colormap)
        ctrl.view_update()

    @state.change("ve")
    def _on_ve(ve, **_):
        scene.set_vertical_exaggeration(float(ve))
        ctrl.view_update()

    def reset_view():
        scene.reset_camera()
        ctrl.view_update()

    ctrl.reset_view = reset_view

    # ---- overlay visibility: one controller for every horizon/well toggle ----
    # The switches are controlled (one-way `model_value`); a click addresses the
    # item by id, flips it server-side, drives the matching actor, and reassigns
    # the list (a new object, so the view re-renders) — no per-item state keys.
    def toggle_overlay(oid):
        for items_key, setter in (("hz_items", scene.set_horizon_visibility),
                                   ("wl_items", scene.set_well_visibility)):
            items = state[items_key]
            if any(o["id"] == oid for o in items):
                new = []
                for o in items:
                    if o["id"] == oid:
                        o = {**o, "visible": not o["visible"]}
                        setter(o["key"], o["visible"])
                    new.append(o)
                state[items_key] = new
                ctrl.view_update()
                return

    ctrl.toggle_overlay = toggle_overlay

    # ---- section / box visibility ----
    state.sec_inline = state.sec_crossline = state.sec_z = True
    state.show_box = True

    @state.change("sec_inline", "sec_crossline", "sec_z")
    def _on_sections(**_):
        scene.set_section_visibility("inline", state.sec_inline)
        scene.set_section_visibility("crossline", state.sec_crossline)
        scene.set_section_visibility("z", state.sec_z)
        ctrl.view_update()

    @state.change("show_box")
    def _on_box(**_):
        scene.set_outline_visibility(state.show_box)
        ctrl.view_update()

    # Color bars start hidden; toggling drives the matching legend's visibility.
    state.cbar_seismic = False
    state.cbar_well = False

    @state.change("cbar_seismic")
    def _on_cbar_seismic(**_):
        scene.set_scalar_bar_visibility("__section__", state.cbar_seismic)
        ctrl.view_update()

    if wells:
        @state.change("cbar_well")
        def _on_cbar_well(**_):
            scene.set_scalar_bar_visibility("well", state.cbar_well)
            ctrl.view_update()

    # ---- Data panel: load a horizon/well file straight into the live scene ----
    state.data_kind = "horizon"
    state.data_path = ""
    state.hz_layout = "ilxl"
    state.hz_zscale = 1.0
    state.wl_il = (il_lo + il_hi) // 2        # drop a new well where the planes start
    state.wl_xl = (xl_lo + xl_hi) // 2
    state.wl_checkshot = ""
    state.wl_cs_unit = "ms"
    # Deviated-well inputs: a survey file bends the well off vertical.
    state.wl_deviation = ""
    state.wl_dev_kind = "directional"         # directional (MD/inc/azi) or xyz
    state.wl_ekb = 0.0                         # Kelly-bushing elevation for the datum
    state.wl_bin_il = 25.0                     # fallback survey geometry when the
    state.wl_bin_xl = 25.0                     # cube carries no world transform
    state.wl_azimuth = 0.0
    state.cube_georef = scene.vol.transform is not None
    state.data_msg = ""
    state.data_error = False

    def _report(msg, *, error=False):
        state.data_msg = msg
        state.data_error = error

    def load_data():
        path = (state.data_path or "").strip().strip('"')
        if not path:
            return _report("Enter a file path to load.", error=True)
        if not os.path.isfile(path):
            return _report(f"No such file: {path}", error=True)
        try:
            if state.data_kind == "horizon":
                hs = load_horizons(path, state.hz_layout,
                                   z_scale=float(state.hz_zscale or 1.0))
                for h in hs:
                    _add_horizon(h)
                names = ", ".join(h.name for h in hs)
                _report(f"Loaded {len(hs)} horizon(s): {names}")
            else:
                checkshot = (state.wl_checkshot or "").strip().strip('"') or None
                if checkshot and not os.path.isfile(checkshot):
                    return _report(f"No such checkshot file: {checkshot}", error=True)
                deviation = (state.wl_deviation or "").strip().strip('"') or None
                if deviation and not os.path.isfile(deviation):
                    return _report(f"No such deviation file: {deviation}", error=True)
                try:
                    ekb = float(state.wl_ekb or 0.0)
                except (TypeError, ValueError):
                    return _report("Kelly-bushing elevation must be a number.", error=True)
                # A deviated well needs survey geometry to tie its trajectory to
                # inline/crossline: use the cube's own transform, else the bins.
                transform = None
                if deviation:
                    transform = scene.vol.transform
                    if transform is None:                  # cube has no georeference
                        try:
                            bin_il, bin_xl = float(state.wl_bin_il), float(state.wl_bin_xl)
                            azimuth = float(state.wl_azimuth or 0.0)
                        except (TypeError, ValueError):
                            return _report("Bin spacing and azimuth must be numbers.",
                                           error=True)
                        if bin_il <= 0 or bin_xl <= 0:
                            return _report("Bin spacing must be positive (metres per "
                                           "inline/crossline).", error=True)
                        transform = GridTransform.from_binning(bin_il, bin_xl, azimuth)
                w = load_well(path, il=float(state.wl_il), xl=float(state.wl_xl),
                              checkshot=checkshot, checkshot_unit=state.wl_cs_unit,
                              deviation=deviation, deviation_kind=state.wl_dev_kind,
                              ekb=ekb, transform=transform)
                _add_well(w, color_by=default_log(w))
                shape = "deviated" if deviation else "vertical"
                _report(f"Loaded {shape} well '{w.name}' at IL "
                        f"{int(float(state.wl_il))}, XL {int(float(state.wl_xl))}.")
            ctrl.view_update()
        except Exception as e:                 # surface the reason; keep the server up
            _report(f"Could not load: {e}", error=True)

    ctrl.load_data = load_data

    # ---- survey ranges drive the Section sliders reactively, so swapping the
    # displayed cube (below) just re-points them; they are not baked into the
    # template as literals. ----
    state.il_min, state.il_max = il_lo, il_hi
    state.xl_min, state.xl_max = xl_lo, xl_hi
    state.z_min, state.z_max, state.z_slider_step = z_lo, z_hi, z_step

    # ---- Volume: convert a SEG-Y to Zarr and/or open a store into the scene ----
    state.segy_path = ""
    state.zarr_out = ""
    state.open_path = ""
    state.vol_busy = False
    state.vol_pct = 0
    state.vol_msg = ""
    state.vol_error = False

    @state.change("segy_path")
    def _suggest_out(segy_path, **_):
        # Auto-fill a sensible output path the first time a SEG-Y is named.
        clean = (segy_path or "").strip().strip('"')
        if clean and not state.zarr_out:
            state.zarr_out = suggest_zarr_path(clean)

    def _open_into_scene(path):
        """Swap the live cube and re-point the sliders/overlays at the new survey."""
        scene.load_volume(path)
        m = scene.md
        state.il_min, state.il_max = m["iline_range"]
        state.xl_min, state.xl_max = m["xline_range"]
        state.z_min, state.z_max = m["z_range"]
        state.z_slider_step = m["sample_interval"]
        state.iline, state.xline, state.z = scene.iline, scene.xline, scene.z
        state.cube_georef = scene.vol.transform is not None
        state.hz_items = []            # overlays were dropped with the old grid
        state.wl_items = []
        state.sec_inline = state.sec_crossline = state.sec_z = True
        ctrl.view_update()

    async def convert_and_view():
        segy = (state.segy_path or "").strip().strip('"')
        out = (state.zarr_out or "").strip().strip('"') or (
            suggest_zarr_path(segy) if segy else "")
        if not segy or not os.path.isfile(segy):
            state.vol_error, state.vol_msg = True, f"No such SEG-Y file: {segy}"
            return
        if os.path.exists(out):
            state.vol_error, state.vol_msg = True, f"Output already exists: {out}"
            return
        state.zarr_out = out
        state.vol_busy, state.vol_error, state.vol_pct = True, False, 0
        state.vol_msg = "Scanning geometry…"
        state.flush()

        prog = {"pct": 0}
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(
            None, lambda: convert_segy(
                segy, out, progress=lambda d, t: prog.update(pct=int(100 * d / t) if t else 0)))
        while not fut.done():                   # stream progress while it runs
            state.vol_pct = prog["pct"]
            state.vol_msg = f"Converting…  {prog['pct']}%"
            state.flush()
            await asyncio.sleep(0.4)
        try:
            attrs = await fut
            state.vol_pct = 100
            state.vol_msg = (f"Converted {attrs['iline_range']} IL × "
                             f"{attrs['xline_range']} XL — opening…")
            state.flush()
            _open_into_scene(out)
            state.vol_msg = f"Viewing {os.path.basename(out)}."
        except Exception as e:                  # keep the server alive on failure
            state.vol_error, state.vol_msg = True, f"Convert failed: {e}"
        finally:
            state.vol_busy = False
            state.flush()

    ctrl.convert_and_view = convert_and_view

    def open_existing():
        p = (state.open_path or "").strip().strip('"')
        if not p or not os.path.exists(p):
            state.vol_error, state.vol_msg = True, f"No such .zarr store: {p}"
            return
        try:
            _open_into_scene(p)
            state.vol_error, state.vol_msg = False, f"Viewing {os.path.basename(p)}."
        except Exception as e:
            state.vol_error, state.vol_msg = True, f"Could not open: {e}"

    ctrl.open_volume = open_existing

    # ---- server-side file picker: browse the render host's disk in a popup
    # (a web page can't hand back a native path, and the cubes live server-side),
    # then drop the chosen path into whichever field opened it. ----
    start_dir = os.path.dirname(os.path.abspath(str(zarr_path)))
    state.fb_open = False
    state.fb_cwd = start_dir
    state.fb_parent = None
    state.fb_crumbs = []
    state.fb_query = ""            # live name filter within the current folder
    state.fb_title = "Choose a file"
    state.fb_entries = []
    state.fb_target = ""           # the state field the pick fills
    state.fb_pick = "file"         # "file" or "store" (a .zarr directory)
    state.fb_exts = []             # file extensions to show (dirs always shown)

    def fb_navigate(path):
        data = browse(path, exts=set(state.fb_exts) if state.fb_exts else None)
        state.fb_cwd = data["cwd"]
        state.fb_parent = data["parent"]
        state.fb_entries = data["entries"]
        state.fb_crumbs = data["crumbs"]
        state.fb_query = ""

    def open_browser(target, title, exts, pick):
        state.fb_target, state.fb_title = target, title
        state.fb_exts, state.fb_pick = exts, pick
        fb_navigate(state.fb_cwd or start_dir)
        state.fb_open = True

    ctrl.open_browser = open_browser

    def fb_up():
        if state.fb_parent is not None:
            fb_navigate(state.fb_parent)

    ctrl.fb_up = fb_up
    ctrl.fb_go = fb_navigate            # breadcrumb jump

    def fb_click(entry):
        kind = entry.get("type")
        if kind in ("up", "dir", "drive"):
            fb_navigate(entry["path"])
        elif kind == "store" and state.fb_pick != "store":
            fb_navigate(entry["path"])            # descend into a .zarr when not picking one
        else:                                     # file, or a store when picking a store
            state[state.fb_target] = entry["path"]
            state.fb_open = False

    ctrl.fb_click = fb_click

    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("seisvtk")

        with layout.toolbar:
            vuetify3.VSpacer()
            vuetify3.VSelect(
                v_model=("colormap", scene.colormap),
                items=("colormaps", colormaps),
                label="Colormap", hide_details=True,
                density="compact", style="max-width: 160px",
            )
            vuetify3.VSlider(
                v_model=("ve", scene.ve),
                min=0.05, max=8.0, step=0.05,
                label="V.E.", hide_details=True, thumb_label=True,
                density="compact", style="max-width: 180px", class_="ml-3",
            )
            vuetify3.VBtn(
                "Center", click=ctrl.reset_view,
                variant="tonal", density="compact", class_="mx-2",
            )

        with layout.content:
            view = vtk_widgets.VtkRemoteView(scene.render_window, interactive_ratio=1)
            ctrl.view_update = view.update
            ctrl.on_server_ready.add(view.update)

        with layout.drawer as drawer:
            drawer.width = 320
            with vuetify3.VContainer(fluid=True, classes="pa-3"):
                _sections_section(z_unit, scene)
                _overlays_section(ctrl)
                _data_section(ctrl)
                _colorbars_section(value_label, log_name, bool(wells))

                vuetify3.VDivider(classes="my-3")
                vuetify3.VSwitch(v_model=("show_box",), label="Bounding box",
                                 color="grey", density="compact", hide_details=True)

            _file_browser_dialog(ctrl)

    return server


# File extensions each picker offers (dirs are always navigable; a .zarr store
# is pickable only when a volume is wanted). ``[".zarr"]`` as a *file* filter
# shows no files, so the "open store" picker lists only folders and stores.
_EXTS_DATA = [".txt", ".csv", ".dat", ".xyz", ".hor", ".las"]
_EXTS_SEGY = [".segy", ".sgy"]
_EXTS_CHECKSHOT = [".txt", ".csv", ".dat", ".chk"]
_EXTS_STORE = [".zarr"]
_EXTS_DEV = [".txt", ".csv", ".dat", ".dev", ".wl", ".xyz", ".asc"]


def _js_args(*items) -> str:
    """A JS array literal for a trame event's args, single-quoted so it nests
    safely inside the double-quoted ``@click`` attribute trame generates."""
    return json.dumps(list(items)).replace('"', "'")


def _path_field(ctrl, model, label, placeholder, browse_args, *, show=None):
    """A path text field with a folder button that opens the server file picker.

    ``browse_args`` is the JS arg array for ``ctrl.open_browser`` —
    ``[target, title, exts, pick]``. ``show`` sets ``v-show`` on the row.
    """
    row_kw = {"v_show": show} if show else {}
    with vuetify3.VRow(no_gutters=True, align="center", classes="mb-2", **row_kw):
        with vuetify3.VCol():
            vuetify3.VTextField(v_model=(model,), label=label, placeholder=placeholder,
                                density="compact", hide_details=True, clearable=True)
        with vuetify3.VCol(cols="auto", classes="pl-1"):
            vuetify3.VBtn(icon="mdi-folder-open", size="small", variant="tonal",
                          click=(ctrl.open_browser, browse_args))


def _file_browser_dialog(ctrl):
    """A modern server-side file picker: breadcrumb + Up, search, click to pick.

    Lists the render host's directories/files. Folders/drives navigate, a ``.zarr``
    store is selectable, files are filtered by type. The visible list is filtered
    live by the search box (client-side, so it stays snappy).
    """
    with vuetify3.VDialog(v_model=("fb_open",), max_width="760", scrollable=True):
        with vuetify3.VCard(rounded="lg"):
            with vuetify3.VToolbar(density="comfortable", color="surface", flat=True):
                vuetify3.VIcon("mdi-folder-search-outline", classes="mx-3",
                               color="primary")
                vuetify3.VToolbarTitle("{{ fb_title }}", classes="text-subtitle-1")
                vuetify3.VBtn(icon="mdi-close", variant="text", size="small",
                              classes="mr-1", click="fb_open = false")

            # Breadcrumb trail + an Up button (disabled at the drive list).
            with vuetify3.VSheet(classes="d-flex align-center px-3 py-2 flex-nowrap",
                                 color="surface", style="overflow-x: auto"):
                vuetify3.VBtn(icon="mdi-arrow-up", size="small", variant="tonal",
                              classes="mr-2", disabled=("fb_parent === null",),
                              click=ctrl.fb_up)
                vuetify3.VBtn(
                    "{{ c.label }}", v_for="(c, i) in fb_crumbs", key="c.path",
                    variant=("i === fb_crumbs.length - 1 ? 'tonal' : 'text'",),
                    append_icon=("i < fb_crumbs.length - 1 ? 'mdi-chevron-right' : null",),
                    size="small", classes="text-none px-2", click=(ctrl.fb_go, "[c.path]"),
                )
            vuetify3.VDivider()

            vuetify3.VTextField(
                v_model=("fb_query",), placeholder="Search this folder",
                prepend_inner_icon="mdi-magnify", density="compact", variant="solo-filled",
                flat=True, hide_details=True, clearable=True, classes="px-3 py-2",
            )
            vuetify3.VDivider()

            with vuetify3.VList(density="comfortable", nav=True, lines="one",
                                style="max-height: 46vh; overflow-y: auto"):
                with vuetify3.VListItem(
                    v_for=("item in fb_entries.filter(e => e.type !== 'up' && "
                           "(!fb_query || e.name.toLowerCase().includes(fb_query.toLowerCase())))"),
                    key="item.path", rounded="lg", classes="mx-2",
                    click=(ctrl.fb_click, "[item]"),
                ):
                    with vuetify3.VListItemTitle(classes="d-flex align-center"):
                        vuetify3.VIcon(icon=("item.icon",), color=("item.color",),
                                       size="small", classes="mr-3")
                        html.Span("{{ item.name }}", classes="text-truncate")
                # empty-folder hint
                html.Div("This folder is empty",
                         v_show="!fb_entries.some(e => e.type !== 'up')",
                         classes="text-center text-disabled text-body-2 py-6")

            vuetify3.VDivider()
            with vuetify3.VCardActions():
                vuetify3.VSpacer()
                vuetify3.VBtn("Cancel", variant="text", click="fb_open = false")


# ---- drawer sections (kept small; each builds one block of the panel) --------
def _sections_section(z_unit, scene):
    # Slider ranges bind to state (il_min/il_max/…) so a volume swap re-points them.
    vuetify3.VLabel("Sections", classes="text-overline")
    z_sfx = f" {z_unit}" if z_unit else ""
    vuetify3.VSwitch(v_model=("sec_inline",), label=("`Inline  —  IL ${iline}`",),
                     color="primary", density="compact", hide_details=True)
    vuetify3.VSlider(v_model=("iline", scene.iline), min=("il_min",), max=("il_max",),
                     step=1, thumb_label=True, density="compact", hide_details=True)
    vuetify3.VSwitch(v_model=("sec_crossline",), label=("`Crossline  —  XL ${xline}`",),
                     color="primary", density="compact", hide_details=True)
    vuetify3.VSlider(v_model=("xline", scene.xline), min=("xl_min",), max=("xl_max",),
                     step=1, thumb_label=True, density="compact", hide_details=True)
    vuetify3.VSwitch(v_model=("sec_z",), label=(f"`Depth  —  ${{z}}{z_sfx}`",),
                     color="primary", density="compact", hide_details=True)
    vuetify3.VSlider(v_model=("z", scene.z), min=("z_min",), max=("z_max",),
                     step=("z_slider_step",), thumb_label=True, density="compact",
                     hide_details=True)


def _overlays_section(ctrl):
    # One reactive toggle per loaded horizon / well. The divider + label only
    # show when the list is non-empty, so a cube with no overlays stays clean.
    vuetify3.VDivider(classes="my-3", v_if="hz_items.length")
    vuetify3.VLabel("Horizons", classes="text-overline", v_if="hz_items.length")
    vuetify3.VSwitch(
        v_for="item in hz_items", key="item.id",
        model_value=("item.visible",), label=("item.label",), color=("item.color",),
        click=(ctrl.toggle_overlay, "[item.id]"),
        density="compact", hide_details=True,
    )
    vuetify3.VDivider(classes="my-3", v_if="wl_items.length")
    vuetify3.VLabel("Wells", classes="text-overline", v_if="wl_items.length")
    vuetify3.VSwitch(
        v_for="item in wl_items", key="item.id",
        model_value=("item.visible",), label=("item.label",), color=("item.color",),
        click=(ctrl.toggle_overlay, "[item.id]"),
        density="compact", hide_details=True,
    )


def _data_section(ctrl):
    # The arg array is embedded inside a double-quoted @click attribute, so its
    # string literals must use single quotes (our labels/exts carry no quotes).
    data_args = _js_args("data_path", "Choose a horizon or well file", _EXTS_DATA, "file")
    segy_args = _js_args("segy_path", "Choose a SEG-Y file", _EXTS_SEGY, "file")
    open_args = _js_args("open_path", "Choose a .zarr store", _EXTS_STORE, "store")
    cs_args = _js_args("wl_checkshot", "Choose a checkshot file", _EXTS_CHECKSHOT, "file")
    dev_args = _js_args("wl_deviation", "Choose a deviation survey", _EXTS_DEV, "file")

    vuetify3.VDivider(classes="my-3")
    vuetify3.VLabel("Load data", classes="text-overline")
    with vuetify3.VBtnToggle(
        v_model=("data_kind",), mandatory=True, density="compact",
        variant="outlined", divided=True, class_="mb-2",
    ):
        vuetify3.VBtn("Horizon", value="horizon", size="small")
        vuetify3.VBtn("Well", value="well", size="small")
        vuetify3.VBtn("Volume", value="volume", size="small")

    # Shared file-path field — horizons & wells only (volume has its own fields).
    _path_field(ctrl, "data_path", "File path", "pick a horizon or well file",
                data_args, show="data_kind !== 'volume'")

    # Horizon-only options.
    with vuetify3.VSheet(v_show="data_kind === 'horizon'", color="transparent"):
        vuetify3.VSelect(
            v_model=("hz_layout",),
            items=("hz_layouts", [{"title": t, "value": v}
                                  for v, t in _HZ_LAYOUT_LABELS.items()]),
            label="Layout", density="compact", hide_details=True, class_="mb-2",
        )
        vuetify3.VTextField(
            v_model=("hz_zscale",), label="Z scale (×)", type="number",
            density="compact", hide_details=True, class_="mb-2",
            hint="1000 if file is in seconds and the cube is in ms", persistent_hint=True,
        )

    # Well-only options.
    with vuetify3.VSheet(v_show="data_kind === 'well'", color="transparent"):
        with vuetify3.VRow(no_gutters=True, class_="mb-2"):
            with vuetify3.VCol(cols=6, class_="pr-1"):
                vuetify3.VTextField(v_model=("wl_il",), label="IL", type="number",
                                    density="compact", hide_details=True)
            with vuetify3.VCol(cols=6, class_="pl-1"):
                vuetify3.VTextField(v_model=("wl_xl",), label="XL", type="number",
                                    density="compact", hide_details=True)
        with vuetify3.VRow(no_gutters=True, class_="mb-1"):
            vuetify3.VLabel("Wellhead inline / crossline", classes="text-caption text-medium-emphasis")
        _path_field(ctrl, "wl_checkshot", "Checkshot file (optional)",
                    "MD / TWT table for a time cube", cs_args)
        vuetify3.VSelect(
            v_model=("wl_cs_unit",), items=("cs_units", ["ms", "s"]),
            label="Checkshot time unit", density="compact", hide_details=True,
            class_="mb-2", v_show="wl_checkshot",
        )

        # Deviation survey -> a deviated well. Left blank, the well is vertical.
        vuetify3.VDivider(classes="my-2")
        _path_field(ctrl, "wl_deviation", "Deviation survey (optional)",
                    "MD/inc/azi or MD/X/Y/Z trajectory", dev_args)
        with vuetify3.VSheet(v_show="wl_deviation", color="transparent"):
            with vuetify3.VBtnToggle(
                v_model=("wl_dev_kind",), mandatory=True, density="compact",
                variant="outlined", divided=True, class_="mb-2",
            ):
                vuetify3.VBtn("MD / inc / azi", value="directional", size="small")
                vuetify3.VBtn("MD / X / Y / Z", value="xyz", size="small")
            vuetify3.VTextField(
                v_model=("wl_ekb",), label="Kelly-bushing elevation (EKB)",
                type="number", density="compact", hide_details=True, class_="mb-2",
                hint="depth datum: z = TVD − EKB", persistent_hint=True,
            )
            # Georeferenced cube -> use its survey geometry; otherwise the user
            # gives bin spacing + azimuth so metre offsets become inline/crossline.
            vuetify3.VAlert(
                v_show="cube_georef", type="info", density="compact", variant="tonal",
                class_="mb-2", text="Tied to the cube's own survey geometry.",
            )
            with vuetify3.VSheet(v_show="!cube_georef", color="transparent"):
                vuetify3.VLabel("Survey geometry (cube has none)",
                                classes="text-caption text-medium-emphasis")
                with vuetify3.VRow(no_gutters=True, class_="mb-2 mt-1"):
                    with vuetify3.VCol(cols=4, class_="pr-1"):
                        vuetify3.VTextField(v_model=("wl_bin_il",), label="Bin IL (m)",
                                            type="number", density="compact",
                                            hide_details=True)
                    with vuetify3.VCol(cols=4, class_="px-1"):
                        vuetify3.VTextField(v_model=("wl_bin_xl",), label="Bin XL (m)",
                                            type="number", density="compact",
                                            hide_details=True)
                    with vuetify3.VCol(cols=4, class_="pl-1"):
                        vuetify3.VTextField(v_model=("wl_azimuth",), label="Azimuth °",
                                            type="number", density="compact",
                                            hide_details=True)

    # Horizon/well load button + its own status alert.
    with vuetify3.VSheet(v_show="data_kind !== 'volume'", color="transparent"):
        vuetify3.VBtn("Load into scene", click=ctrl.load_data, block=True,
                      color="primary", variant="tonal", class_="mb-2")
        vuetify3.VAlert(
            v_show="data_msg", text=("data_msg",), density="compact", variant="tonal",
            type=("data_error ? 'error' : 'success'",), class_="mb-1",
        )

    # Volume-only options: convert a SEG-Y to Zarr, or open an existing store.
    with vuetify3.VSheet(v_show="data_kind === 'volume'", color="transparent"):
        _path_field(ctrl, "segy_path", "SEG-Y file", "pick a .segy / .sgy file",
                    segy_args)
        vuetify3.VTextField(
            v_model=("zarr_out",), label="Output .zarr", density="compact",
            hide_details="auto", class_="mb-2", clearable=True,
            hint="auto-filled from the SEG-Y name; geometry is auto-detected",
            persistent_hint=True,
        )
        vuetify3.VBtn("Convert → view", click=ctrl.convert_and_view, block=True,
                      color="primary", variant="tonal", class_="mb-2",
                      loading=("vol_busy",), disabled=("vol_busy",))
        vuetify3.VProgressLinear(
            v_show="vol_busy", model_value=("vol_pct",), height=6,
            color="primary", rounded=True, class_="mb-2",
        )
        vuetify3.VDivider(classes="my-2")
        _path_field(ctrl, "open_path", "Open existing .zarr",
                    "pick a .zarr store", open_args)
        vuetify3.VBtn("Open in viewer", click=ctrl.open_volume, block=True,
                      variant="outlined", density="compact", class_="mb-2",
                      disabled=("vol_busy",))
        vuetify3.VAlert(
            v_show="vol_msg", text=("vol_msg",), density="compact", variant="tonal",
            type=("vol_error ? 'error' : 'success'",), class_="mb-1",
        )


def _colorbars_section(value_label, log_name, has_wells):
    vuetify3.VDivider(classes="my-3")
    vuetify3.VLabel("Color bars", classes="text-overline")
    vuetify3.VSwitch(v_model=("cbar_seismic",), label=f"Section — {value_label}",
                     color="primary", density="compact", hide_details=True)
    if has_wells:
        vuetify3.VSwitch(v_model=("cbar_well",), label=f"Wells — {log_name}",
                         color="amber", density="compact", hide_details=True)


def main():
    # Grab our required zarr path, then hand the rest (e.g. --server, --port)
    # to trame's own CLI parser via server.start().
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("zarr_path", help="Path to a .zarr seismic store")
    parser.add_argument("--mode", choices=["amplitude", "property"], default="amplitude",
                        help="amplitude = zero-centred seismic; property = sequential (GR, impedance, ...)")
    parser.add_argument("--colormap", default=None, help="initial colormap name")
    parser.add_argument("--ve", type=float, default=None, help="vertical exaggeration")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    server = build(args.zarr_path, mode=args.mode, colormap=args.colormap, ve=args.ve)
    server.start()


if __name__ == "__main__":
    main()
