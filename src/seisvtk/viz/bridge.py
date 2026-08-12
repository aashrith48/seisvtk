import vtk
import numpy as np
from vtkmodules.util import numpy_support


def slice_to_image_data(arr_slice: np.ndarray) -> vtk.vtkImageData:
    """Wrap a seismic slab as a ``vtkImageData`` in the shared world frame.

    The slab is a 3D array ``(n_inline, n_crossline, n_samples)`` with exactly
    one axis of length 1 (the slice normal). We map the array axes straight to
    VTK's world axes so every slice orientation shares one coordinate system::

        VTK x = inline    VTK y = crossline    VTK z = samples (depth/time)

    VTK stores scalars x-fastest, so we ravel in Fortran order (inline varies
    fastest, then crossline, then samples) to match ``SetDimensions(shape)``
    without reversing. The caller positions the slab along its normal with
    ``actor.SetPosition(...)`` and sets vertical exaggeration via z-spacing.
    """
    img = vtk.vtkImageData()
    img.SetDimensions(arr_slice.shape)
    vtk_array = numpy_support.numpy_to_vtk(
        arr_slice.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT
    )
    img.GetPointData().SetScalars(vtk_array)
    return img
