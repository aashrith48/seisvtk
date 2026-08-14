import argparse
import sys
from pathlib import Path

import vtk
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3, vtk as vtk_widgets

from seisvtk.io.volume import Volume
from seisvtk.viz.slice_scene import SliceScene
from seisvtk.viz.colormaps import NAMES as COLORMAPS


def build(zarr_path: str | Path):
    # Read geometry first so the initial planes start at the middle of whatever
    # cube was loaded (opening a Volume only reads metadata, not samples).
    md = Volume(zarr_path=Path(zarr_path)).metadata
    il_lo, il_hi = md["iline_range"]
    xl_lo, xl_hi = md["xline_range"]
    z_lo, z_hi = md["z_range"]
    z_step = md["sample_interval"]

    scene = SliceScene(
        zarr_path,
        iline=(il_lo + il_hi) // 2,
        xline=(xl_lo + xl_hi) // 2,
        z=(z_lo + z_hi) / 2,
        vertical_exaggeration=0.25,
        offscreen=True,   # render headless; trame streams the framebuffer, no GL window
    )

    # trame's remote view drives the render loop through an interactor; the
    # render window must have one attached (never Start()ed here).
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(scene.render_window)
    interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

    # Lock depth (z) to screen-vertical: after every camera move, re-pin the
    # view-up to -z so the section can orbit but never roll off vertical.
    def _keep_z_vertical(caller, event):
        scene.ren.GetActiveCamera().SetViewUp(0, 0, -1)

    interactor.AddObserver("InteractionEvent", _keep_z_vertical)

    server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller

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

    with SinglePageLayout(server) as layout:
        layout.title.set_text("seisvtk")

        with layout.content:
            view = vtk_widgets.VtkRemoteView(scene.render_window, interactive_ratio=1)
            ctrl.view_update = view.update
            ctrl.on_server_ready.add(view.update)

        with layout.toolbar:
            vuetify3.VSelect(
                v_model=("colormap", scene.colormap),
                items=("colormaps", COLORMAPS),
                label="Colormap", hide_details=True,
                density="compact", style="max-width: 160px",
            )
            vuetify3.VSlider(
                v_model=("ve", scene.ve),
                min=0.05, max=2.0, step=0.05,
                label="V.E.", hide_details=True, thumb_label=True,
                density="compact", style="max-width: 200px",
            )
            vuetify3.VBtn(
                "Center", click=ctrl.reset_view,
                variant="tonal", density="compact", class_="mx-2",
            )
            vuetify3.VSpacer()
            vuetify3.VSlider(
                v_model=("iline", scene.iline),
                min=il_lo, max=il_hi, step=1,
                label="Inline", hide_details=True, thumb_label=True,
                density="compact", style="max-width: 260px",
            )
            vuetify3.VSlider(
                v_model=("xline", scene.xline),
                min=xl_lo, max=xl_hi, step=1,
                label="Crossline", hide_details=True, thumb_label=True,
                density="compact", style="max-width: 260px",
            )
            vuetify3.VSlider(
                v_model=("z", scene.z),
                min=z_lo, max=z_hi, step=z_step,
                label="Z", hide_details=True, thumb_label=True,
                density="compact", style="max-width: 260px",
            )

    return server


def main():
    # Grab our required zarr path, then hand the rest (e.g. --server, --port)
    # to trame's own CLI parser via server.start().
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("zarr_path", help="Path to a .zarr seismic store")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    server = build(args.zarr_path)
    server.start()


if __name__ == "__main__":
    main()
