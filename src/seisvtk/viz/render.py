import vtk
import numpy as np
from seisvtk.io.volume import Volume
from seisvtk.viz.actors import grey_image_slice
from pathlib import Path


def render_vtkimage(zarr_path: Path | str, *,
                    iline: int | None = None,
                    xline: int | None = None,
                    z: float | None = None,
                    vertical_exaggeration: float = 1.0):
    path = Path(zarr_path)
    vol = Volume(zarr_path=path)

    #Get seismic slice 3D array from user requested positions
    slice = vol.seismic_slice(vol.get_array_index(xline=xline,iline=iline,z=z))

    #Get data percentiles clip
    clip = np.percentile(np.abs(slice),98)

    #Build oriented greyscale slice actor.
    # World frame: VTK axes are (x=inline, y=crossline, z=samples/depth).
    # Vertical exaggeration scales the sample (z) axis relative to the
    # 1-unit-per-trace horizontal axes. A section has far more samples than
    # traces, so pass VE < 1 to de-squish.
    actor = grey_image_slice(slice, clip, spacing=(1.0, 1.0, vertical_exaggeration))

    #Render
    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(800, 800)

    # Look down the axis the slice was cut on. For iline/xline sections, depth
    # (z) points down on screen and the in-plane trace axis runs horizontally;
    # a z-slice is a map view looking straight down. Parallel projection = no
    # perspective distortion on a flat section.
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()
    if iline is not None:
        cam.SetPosition(-1, 0, 0)   # look along +x: y=crossline right, z=depth down
        cam.SetViewUp(0, 0, -1)
    elif xline is not None:
        cam.SetPosition(0, 1, 0)    # look along -y: x=inline right, z=depth down
        cam.SetViewUp(0, 0, -1)
    elif z is not None:
        cam.SetPosition(0, 0, 1)    # map view looking down z: x=inline, y=crossline
        cam.SetViewUp(0, 1, 0)
    cam.SetFocalPoint(0, 0, 0)
    ren.ResetCamera()
    rw.Render()

    #Write to PNG
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(f"{iline}_{xline}_{z}_slice.png")
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

