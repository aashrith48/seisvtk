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
# Run  :  docker run --rm -p 8914:8914 \
#            -e SEISVTK_VIEWER_SECRET=<shared-secret> \
#            -e SEISVTK_VIEWER_PUBLIC_BASE=https://viewer.example \
#            seisvtk-viewer:osmesa
FROM python:3.11-slim-bookworm

# Minimal runtime libs the vtk-osmesa wheel links against (it bundles OSMesa).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \
    PORT=8914

WORKDIR /app
COPY . /app

# Core (numpy/pandas/segyio/lasio/zarr) from the package, then the viz stack with
# the OSMesa VTK wheel (not plain `vtk`) + aiohttp (session service) + the S3
# store backend (so a cube opens straight from s3:// — chunks stream on demand).
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir \
        vtk-osmesa trame trame-vtk trame-vuetify \
        aiohttp fsspec s3fs

EXPOSE 8914

# The long-running multi-session service: it mints a trame worker per viewer
# session (each worker runs `seisvtk-viewer` on an internal port) and reverse-
# proxies the browser to it. A scene comes from the signed token, not the image,
# so one container serves every chart. Token auth is enabled via
# SEISVTK_VIEWER_SECRET; SEISVTK_VIEWER_PUBLIC_BASE sets the external URL.
CMD ["seisvtk-viewer-service"]
