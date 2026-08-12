import vtk
import numpy as np

from seisvtk.viz.bridge import slice_to_image_data


def grey_image_slice(slab: np.ndarray, clip: float,
                     spacing=(1.0, 1.0, 1.0)) -> vtk.vtkImageSlice:
    """Build a greyscale, correctly-oriented image slice actor from a seismic slab.

    The slab is a 3D array with one length-1 axis (the slice normal). We set the
    mapper's orientation to that axis so VTK displays the true section plane
    rather than defaulting to the Z plane. Window/level maps ``[-clip, +clip]``
    to black->white. The caller positions the actor along its normal with
    ``actor.SetPosition(...)``.
    """
    img = slice_to_image_data(slab)
    img.SetSpacing(*spacing)

    axis = int(np.argmin(slab.shape))  # length-1 axis = slice normal (0=x,1=y,2=z)
    mapper = vtk.vtkImageSliceMapper()
    mapper.SetInputData(img)
    mapper.SetOrientation(axis)
    mapper.SetSliceNumber(0)

    actor = vtk.vtkImageSlice()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColorWindow(2 * clip)
    actor.GetProperty().SetColorLevel(0.0)
    return actor
