import vtk

_VTKIS_NONE = 0  # vtkInteractorStyle "no active gesture" state


class OrbitPanStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball camera tuned for inspecting zoomed-in sections.

    * Left-drag   : orbit (trackball default)
    * Wheel       : zoom (trackball default)
    * Right-drag  : pan -- the move you need to recentre after zooming in
    * Left+Right  : pan -- the two-button grab that feels natural for dragging

    Trackball's default right-button *dolly* is dropped (the wheel already
    zooms), which frees the right button for panning. Out of the box, panning
    lives only on the middle button or Shift+Left-drag, both awkward to reach
    through a browser remote view, so a plane you zoom into is hard to keep
    framed. Here either the right button or both buttons grab and slide it.

    VTK's single-gesture state machine means only one of orbit/pan runs at a
    time; the handlers below hand off between them as buttons go down and up so
    the active gesture always matches which buttons are currently held.
    """

    def __init__(self):
        self._left = False
        self._right = False
        self.AddObserver("LeftButtonPressEvent", self._left_down)
        self.AddObserver("LeftButtonReleaseEvent", self._left_up)
        self.AddObserver("RightButtonPressEvent", self._right_down)
        self.AddObserver("RightButtonReleaseEvent", self._right_up)

    def _left_down(self, _obj, _evt):
        self._left = True
        if self._right:
            self._start_pan()            # second button down -> pan
        else:
            self.OnLeftButtonDown()      # orbit

    def _left_up(self, _obj, _evt):
        self._left = False
        if self.GetState() != _VTKIS_NONE:
            self.EndPan() if self._right else self.OnLeftButtonUp()
        if self._right:                  # right still held -> resume pan
            self._start_pan()

    def _right_down(self, _obj, _evt):
        self._right = True
        if self._left and self.GetState() != _VTKIS_NONE:
            self.OnLeftButtonUp()        # stop orbit before panning
        self._start_pan()

    def _right_up(self, _obj, _evt):
        self._right = False
        if self.GetState() != _VTKIS_NONE:
            self.EndPan()
        if self._left:                   # left still held -> resume orbit
            self.OnLeftButtonDown()

    def _start_pan(self):
        if self.GetState() == _VTKIS_NONE:
            self.StartPan()
