# seisvtk

Server-side 3D seismic cross-section viewer (trame + VTK) with a lightweight,
VTK-free I/O core. The viewer renders inline / crossline / Z slices of a cube and
streams the rendered frames to the browser, so the client stays thin and the cube
is read **out-of-core** — only the chunks a visible slice touches are loaded, from
local disk or straight from object storage (`s3://`). Wells (including deviated)
and horizons drape on the same grid.

## Install

```bash
pip install seisvtk            # I/O core: SEG-Y -> Zarr, wells, horizons, geometry
pip install "seisvtk[viz]"     # + the trame/VTK viewer
pip install "seisvtk[s3]"      # + read cubes directly from s3://
```

The core has no VTK dependency, so a data producer (e.g. a pipeline step that
only converts SEG-Y and writes overlays) installs just `seisvtk`. For a **headless
/ CPU** render host, install the `vtk-osmesa` wheel in place of `vtk` (see
`docker/Dockerfile`).

## Use

Serve a single cube:

```bash
python -m seisvtk.viz.app path/to/cube.zarr --mode property --port 8910
```

Serve a scene manifest (a cube plus well/horizon overlays — `s3://` cubes work):

```bash
seisvtk-viewer scene.json --port 8910
```

Convert a SEG-Y to a chunked Zarr from Python:

```python
from seisvtk.io.loaders import convert_segy
convert_segy("survey.segy", "survey.zarr")   # geometry auto-detected from headers
```

## What's in the box

- **I/O (`seisvtk.io`, no VTK):** streaming SEG-Y→Zarr conversion with survey-geometry
  recovery; an out-of-core `Volume` reader (local or `s3://`); readers for wells
  (LAS/CSV, deviation surveys, checkshots), horizons, and the survey affine
  (`GridTransform`).
- **Viewer (`seisvtk.viz`):** a trame app streaming inline/crossline/Z slice planes,
  well tubes coloured by a log, and horizon surfaces; amplitude and property
  colormaps; vertical exaggeration; runtime cube swap.

## License

[MIT](LICENSE) © 2026 Aashrit
