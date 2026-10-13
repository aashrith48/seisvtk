"""Build a VTK tube actor for a well, colored by a log, in the scene index frame.

Mirrors :mod:`seisvtk.viz.horizon_actor`: the caller supplies the trajectory
already in the scene's world frame (inline index, crossline index, sample index)
plus per-sample log values; this returns one tube actor colored by those values.
Vertical exaggeration is applied by the caller as an actor scale.
"""
from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support as nps


def gr_log_lut(vmin: float = 0.0, vmax: float = 150.0) -> vtk.vtkColorTransferFunction:
    """Gamma-ray color table: low (sand) cream -> high (shale) green."""
    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(vmin, 0.95, 0.93, 0.70)
    ctf.AddRGBPoint(0.5 * (vmin + vmax), 0.85, 0.75, 0.30)
    ctf.AddRGBPoint(vmax, 0.20, 0.55, 0.25)
    return ctf


def well_tube_actor(il_index: np.ndarray, xl_index: np.ndarray, z_index: np.ndarray,
                    values: np.ndarray, *,
                    lut: vtk.vtkScalarsToColors,
                    vrange: tuple[float, float],
                    radius: float = 6.0, sides: int = 12) -> vtk.vtkActor:
    """A log-colored tube along ``(il_index, xl_index, z_index)``.

    ``values`` colors each point through ``lut`` over ``vrange``; ``radius`` is in
    inline/crossline index units (≈ traces). The caller applies V.E. via
    ``actor.SetScale(1, 1, ve)``.
    """
    n = len(z_index)
    coords = np.column_stack([
        np.asarray(il_index, float), np.asarray(xl_index, float),
        np.asarray(z_index, float)]).astype(np.float32)

    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(coords, deep=True))
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(n)
    for i in range(n):
        line.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(cells)

    scal = nps.numpy_to_vtk(np.asarray(values, np.float32), deep=True)
    scal.SetName("log")
    poly.GetPointData().SetScalars(scal)

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(poly)
    tube.SetRadius(radius)
    tube.SetNumberOfSides(sides)
    tube.CappingOn()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(*vrange)
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor
