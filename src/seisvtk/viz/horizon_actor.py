"""Build a VTK surface actor for a horizon, in the slice scene's index frame.

The scene's world frame is ``x = inline index, y = crossline index,
z = sample index`` (see :mod:`seisvtk.viz.bridge`), with vertical exaggeration
applied as an actor scale. :func:`horizon_surface_actor` takes a horizon already
gridded into *that* frame — inline/crossline indices for the lattice and a
``(n_il, n_xl)`` grid of sample indices — and returns a single surface actor.

Holes (cells the interpreter never picked, ``NaN``) are blanked with a VTK
HIDDENPOINT ghost array rather than drawn flat at zero, so a partial horizon
reads as a real surface with gaps.
"""
from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support as nps

# Distinct, muted colors for stacking several horizons.
PALETTE = [
    (0.30, 0.80, 0.45), (0.96, 0.62, 0.22), (0.42, 0.70, 0.95),
    (0.85, 0.45, 0.85), (0.95, 0.85, 0.30), (0.40, 0.85, 0.80),
    (0.90, 0.50, 0.45),
]


def horizon_surface_actor(il_index: np.ndarray, xl_index: np.ndarray,
                          z_index_grid: np.ndarray, *,
                          color: tuple[float, float, float] = PALETTE[0],
                          opacity: float = 1.0) -> vtk.vtkActor:
    """Surface actor from a gridded horizon in the scene's index frame.

    Args:
        il_index: 1-D inline indices for the grid rows (``il_label - il_min``).
        xl_index: 1-D crossline indices for the grid columns.
        z_index_grid: ``(len(il_index), len(xl_index))`` sample indices, ``NaN``
            where the surface has no pick.
        color: solid RGB in 0..1.
        opacity: surface opacity.

    Vertical exaggeration is *not* baked in — the caller applies it with
    ``actor.SetScale(1, 1, ve)`` so one scale drives slices and horizons alike.
    """
    n_il, n_xl = len(il_index), len(xl_index)
    if z_index_grid.shape != (n_il, n_xl):
        raise ValueError(f"z grid {z_index_grid.shape} != ({n_il}, {n_xl})")

    # VTK structured points run x (inline) fastest, so lay the grid out (xl, il).
    zt = np.asarray(z_index_grid, dtype=float).T                # (n_xl, n_il)
    X = np.repeat(np.asarray(il_index, float)[None, :], n_xl, axis=0)
    Y = np.repeat(np.asarray(xl_index, float)[:, None], n_il, axis=1)
    coords = np.column_stack(
        [X.ravel(), Y.ravel(), np.nan_to_num(zt).ravel()]).astype(np.float32)

    sg = vtk.vtkStructuredGrid()
    sg.SetDimensions(n_il, n_xl, 1)
    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(coords, deep=True))
    sg.SetPoints(pts)

    # Blank the holes: HIDDENPOINT ghosts drop cells touching an unpicked corner.
    valid = np.isfinite(zt).ravel()
    ghosts = np.where(valid, 0, vtk.vtkDataSetAttributes.HIDDENPOINT).astype(np.uint8)
    ga = nps.numpy_to_vtk(ghosts, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    ga.SetName(vtk.vtkDataSetAttributes.GhostArrayName())
    sg.GetPointData().AddArray(ga)

    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(sg)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(geom.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(opacity)
    prop.SetAmbient(0.35)          # lift it out of the dark background
    prop.SetDiffuse(0.7)
    return actor
