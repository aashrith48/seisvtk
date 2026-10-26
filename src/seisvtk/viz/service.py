"""Serve a viewer from a *scene manifest* — the deployable entry point.

A scene manifest (``scene.json``) names a cube and its overlays so a host (e.g. a
pipeline that has already exported the data) can launch the viewer without any
bespoke Python. It is the generalisation of the ad-hoc launch scripts: one JSON
in, a streaming trame+VTK viewer out.

Manifest schema (all overlay arrays are plain JSON lists; a cube may be a local
path or an ``s3://`` URL — opened as a remote store so only touched chunks load)::

    {
      "zarr": "s3://bucket/run756/GR_R.zarr",   # or a local path
      "mode": "property",                        # or "amplitude"
      "colormap": "Viridis",
      "z_unit": "ms",
      "value_label": "GR_R",
      "well_radius": 15.0,
      "ve": null,                                # None -> sensible default
      "clip_wells_to_cube": true,                # drop samples outside z-range
      "wells": [
        {"name": "HEWITT", "il": [...], "xl": [...], "z": [...],
         "logs": {"GR": [...]}}
      ],
      "horizons": [
        {"name": "TopA", "il": [...], "xl": [...], "z": [...]}
      ]
    }

An overlay entry may instead carry ``"npz": "<path-or-url>"`` with array keys
``il``/``xl``/``z`` (+ ``logs`` as ``<name>`` keys for a well), to keep a big
surface out of the JSON. Nothing here holds the cube in memory — the manifest is
tiny and the volume streams chunk-by-chunk at render time.
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import re

import numpy as np

from seisvtk.io.horizon import Horizon
from seisvtk.io.volume import Volume
from seisvtk.io.well import Well

# ``build`` (and VTK/trame) is imported lazily inside ``serve`` so manifest parsing
# stays importable wherever only the I/O core is installed.

_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _read_manifest(src: str):
    """Load a scene.json from a local path or a URL (s3://…); return (dict, base_dir).

    *base_dir* is the directory the manifest lives in, used to resolve overlay
    paths given relative to it, so a scene bundle stays portable (the node writes
    the cube name relative to the manifest; the viewer joins it with wherever the
    manifest actually is — a shared volume, or an ``s3://`` prefix)."""
    if _URL_SCHEME.match(src):
        import fsspec  # part of the [s3] extra; present in the viewer image
        with fsspec.open(src, "r") as f:
            manifest = json.load(f)
        return manifest, src.rsplit("/", 1)[0]
    with open(src) as f:
        manifest = json.load(f)
    return manifest, os.path.dirname(os.path.abspath(src))


def _resolve(ref: str, base: str) -> str:
    """Resolve an overlay/cube *ref* against the manifest's *base* dir.

    Absolute refs (a URL, or an absolute local path) are returned unchanged;
    a relative ref is joined onto *base* — in URL space when *base* is a URL."""
    if not isinstance(ref, str) or _URL_SCHEME.match(ref) or os.path.isabs(ref):
        return ref
    if _URL_SCHEME.match(base):
        return f"{base.rstrip('/')}/{ref}"
    return os.path.normpath(os.path.join(base, ref))


def _arrays(entry: dict, keys):
    """Pull named arrays from an entry, inline or from its ``npz`` sidecar."""
    if "npz" in entry:
        with np.load(entry["npz"], allow_pickle=True) as d:
            return {k: np.asarray(d[k], float) for k in keys}
    return {k: np.asarray(entry[k], float) for k in keys}


def _horizon(entry: dict) -> Horizon:
    a = _arrays(entry, ("il", "xl", "z"))
    return Horizon.from_arrays(entry.get("name", "horizon"), a["il"], a["xl"], a["z"])


def _well(entry: dict, z_range=None) -> Well | None:
    a = _arrays(entry, ("il", "xl", "z"))
    if "npz" in entry:
        with np.load(entry["npz"], allow_pickle=True) as d:
            logs = {k: np.asarray(d[k], float) for k in entry.get("logs", [])}
    else:
        logs = {k: np.asarray(v, float) for k, v in entry.get("logs", {}).items()}
    il, xl, z = a["il"], a["xl"], a["z"]
    if z_range is not None:                              # keep only samples in the cube
        lo, hi = z_range
        m = (z >= lo) & (z <= hi)
        if m.sum() < 2:
            return None
        il, xl, z = il[m], xl[m], z[m]
        logs = {k: v[m] for k, v in logs.items()}
    return Well.from_arrays(entry.get("name", "well"), il, xl, z, logs)


def load_scene(manifest: dict | str):
    """Resolve a manifest (dict or path/URL to JSON) into ``build`` arguments.

    Returns ``(zarr, build_kwargs, horizons, wells)``. Only the cube's metadata is
    read here (for optional well clipping) — never its samples.
    """
    base = ""
    if isinstance(manifest, str):
        manifest, base = _read_manifest(manifest)

    zarr = _resolve(manifest["zarr"], base)
    z_range = None
    if manifest.get("clip_wells_to_cube", True):
        z_range = Volume(zarr_path=zarr).metadata["z_range"]

    def _rebase(entries):
        for e in entries:
            if "npz" in e:
                e = {**e, "npz": _resolve(e["npz"], base)}
            yield e

    horizons = [_horizon(h) for h in _rebase(manifest.get("horizons", []))]
    wells = [w for w in (_well(e, z_range) for e in _rebase(manifest.get("wells", []))) if w]

    kwargs = {k: manifest[k] for k in
              ("mode", "colormap", "ve", "well_radius", "z_unit", "value_label")
              if k in manifest}
    return zarr, kwargs, horizons, wells


def serve(manifest: dict | str, *, port: int = 8910, host: str = "0.0.0.0",
          open_browser: bool = False):
    """Build and start the viewer for *manifest* (blocks serving)."""
    from seisvtk.viz.app import build                    # pulls VTK/trame — serve only
    zarr, kwargs, horizons, wells = load_scene(manifest)
    print(f"[seisvtk] serving {zarr} "
          f"({len(horizons)} horizons, {len(wells)} wells) on :{port}", flush=True)
    server = build(zarr, horizons=horizons, wells=wells, **kwargs)
    server.start(port=port, host=host, open_browser=open_browser,
                 exit_on_close=False, timeout=0)
    return server


def main(argv=None):
    p = argparse.ArgumentParser(description="Serve the seisvtk viewer from a scene manifest.")
    p.add_argument("manifest", help="Path (or URL) to a scene.json")
    p.add_argument("--port", type=int, default=8910)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--open-browser", action="store_true")
    args = p.parse_args(argv)
    serve(args.manifest, port=args.port, host=args.host, open_browser=args.open_browser)


if __name__ == "__main__":
    main()
