import vtk

# Colormap names exposed in the UI, in menu order.
NAMES = ["Greyscale", "Seismic", "Red-Blue", "Rainbow"]


def _ctf(points) -> vtk.vtkColorTransferFunction:
    """Build a color transfer function from (scalar, r, g, b) control points."""
    ctf = vtk.vtkColorTransferFunction()
    for x, r, g, b in points:
        ctf.AddRGBPoint(x, r, g, b)
    return ctf


def build_lut(name: str, clip: float) -> vtk.vtkColorTransferFunction:
    """A color table spanning ``[-clip, +clip]`` for a seismic amplitude slice.

    Amplitudes are zero-centered, so all maps are diverging about 0. Returned as
    a ``vtkColorTransferFunction`` (a ``vtkScalarsToColors``) usable directly as an
    image property's lookup table with ``UseLookupTableScalarRangeOn``.
    """
    c = clip
    if name == "Greyscale":
        return _ctf([(-c, 0, 0, 0), (c, 1, 1, 1)])
    if name == "Seismic":  # blue (-) -> white (0) -> red (+)
        return _ctf([(-c, 0, 0, 1), (0, 1, 1, 1), (c, 1, 0, 0)])
    if name == "Red-Blue":  # red (-) -> white (0) -> blue (+)
        return _ctf([(-c, 1, 0, 0), (0, 1, 1, 1), (c, 0, 0, 1)])
    if name == "Rainbow":
        return _ctf([(-c, 0, 0, 1), (-c / 2, 0, 1, 1), (0, 0, 1, 0),
                     (c / 2, 1, 1, 0), (c, 1, 0, 0)])
    raise ValueError(f"unknown colormap: {name!r}")
