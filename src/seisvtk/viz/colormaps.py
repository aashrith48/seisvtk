import vtk

# Diverging amplitude maps (zero-centred), exposed in the UI in menu order.
NAMES = ["Greyscale", "Seismic", "Red-Blue", "Rainbow"]

# Sequential maps for *property* volumes (gamma ray, impedance, porosity, ...),
# which are not zero-centred. Control points are (t, r, g, b) with t in [0, 1].
PROPERTY_NAMES = ["Viridis", "Plasma", "Turbo", "Greyscale"]

_SEQUENTIAL: dict[str, list[tuple[float, float, float, float]]] = {
    "Viridis": [(0.0, 0.267, 0.005, 0.329), (0.25, 0.254, 0.265, 0.530),
                (0.5, 0.128, 0.567, 0.551), (0.75, 0.369, 0.789, 0.383),
                (1.0, 0.993, 0.906, 0.144)],
    "Plasma": [(0.0, 0.050, 0.030, 0.528), (0.25, 0.417, 0.000, 0.658),
               (0.5, 0.692, 0.165, 0.564), (0.75, 0.881, 0.393, 0.383),
               (1.0, 0.940, 0.975, 0.131)],
    "Turbo": [(0.0, 0.190, 0.072, 0.232), (0.25, 0.098, 0.671, 0.859),
              (0.5, 0.407, 0.980, 0.337), (0.75, 0.976, 0.647, 0.133),
              (1.0, 0.479, 0.012, 0.012)],
    "Greyscale": [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)],
}


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


def build_property_lut(name: str, vmin: float, vmax: float, *,
                       n: int = 256,
                       nan_rgba: tuple[float, float, float, float] = (0, 0, 0, 0),
                       ) -> vtk.vtkLookupTable:
    """A sequential color table over ``[vmin, vmax]`` for a property volume.

    Unlike :func:`build_lut`, this is *not* diverging about zero — the range is
    taken straight from the data (typically robust percentiles), so a gamma-ray
    or impedance cube uses its full color span. Returned as a
    ``vtkLookupTable`` (a ``vtkScalarsToColors``) so it drops into an image
    property's lookup table exactly like ``build_lut``.

    Undefined samples (``NaN``) map to ``nan_rgba`` — transparent by default, so
    the large no-data regions of a horizon-bounded property volume read as empty
    space rather than opaque blocks.
    """
    points = _SEQUENTIAL.get(name)
    if points is None:
        raise ValueError(f"unknown property colormap: {name!r}")

    ramp = _ctf([(t, r, g, b) for t, r, g, b in points])  # interpolate in [0, 1]
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    for i in range(n):
        r, g, b = ramp.GetColor(i / (n - 1))
        lut.SetTableValue(i, r, g, b, 1.0)
    lut.SetTableRange(vmin, vmax)
    lut.SetNanColor(*nan_rgba)          # RGBA: alpha 0 -> transparent no-data
    lut.SetUseAboveRangeColor(False)
    lut.SetUseBelowRangeColor(False)
    lut.Build()
    return lut
