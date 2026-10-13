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

import numpy as np

from seisvtk.io.horizon import Horizon
from seisvtk.io.volume import Volume
from seisvtk.io.well import Well

# ``build`` (and VTK/trame) is imported lazily inside ``serve`` so manifest parsing
# stays importable wherever only the I/O core is installed.


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
    if isinstance(manifest, str):
        with open(manifest) as f:
            manifest = json.load(f)

    zarr = manifest["zarr"]
    z_range = None
    if manifest.get("clip_wells_to_cube", True):
        z_range = Volume(zarr_path=zarr).metadata["z_range"]

    horizons = [_horizon(h) for h in manifest.get("horizons", [])]
    wells = [w for w in (_well(e, z_range) for e in manifest.get("wells", [])) if w]

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
