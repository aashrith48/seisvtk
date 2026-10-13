# Headless (CPU/OSMesa) seisvtk viewer image.
#
# The slice viewer renders a handful of planes + well tubes + horizon surfaces,
# which software GL (OSMesa / llvmpipe) handles comfortably — no GPU, so this runs
# on an ordinary CPU node. We install the I/O core + the viz extra, but swap the
# default `vtk` (GLX/X11) wheel for `vtk-osmesa` (offscreen software GL), and add
# the S3 backend so a cube can be opened straight from `s3://` (chunks stream in on
# demand; the volume is never downloaded whole).
#
# NOTE: this needs a real build in the target environment to validate the OSMesa
# render path; it has not been built in the authoring session.
#
# Build:  docker build -f docker/Dockerfile -t seisvtk-viewer:osmesa .
# Run  :  docker run --rm -p 8910:8910 -e SCENE=/data/scene.json \
#            -v /path/to/data:/data seisvtk-viewer:osmesa
FROM python:3.11-slim-bookworm

# Minimal runtime libs the vtk-osmesa wheel links against (it bundles OSMesa).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \
    PORT=8910

WORKDIR /app
COPY . /app

# Core (numpy/pandas/segyio/lasio/zarr) from the package, then the viz stack with
# the OSMesa VTK wheel (not plain `vtk`) + the S3 store backend.
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir \
        vtk-osmesa trame trame-vtk trame-vuetify \
        fsspec s3fs

EXPOSE 8910

# Phase 0 default: serve a single scene manifest (local path or s3:// cube inside).
# Phase 2 replaces this entrypoint with the per-session manager that mints a scene
# per viewer session from a chart id + token.
CMD ["sh", "-c", "seisvtk-viewer \"$SCENE\" --port \"$PORT\" --host 0.0.0.0"]
