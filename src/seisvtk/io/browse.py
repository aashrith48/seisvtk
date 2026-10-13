"""Server-side directory listing for the viewer's file-picker popup.

The trame viewer renders where the data lives, so files are chosen by browsing
the *server's* filesystem (its own disk when run locally), not by uploading from
the browser — which only yields bytes, never a path, and cannot handle multi-GB
cubes. :func:`browse` turns a directory into a sorted, typed entry list the popup
renders; it is pure and unit-tested, with the trame dialog kept in the app layer.
"""
from __future__ import annotations

import os
import string

# A Zarr "file" is really a directory; treat ``*.zarr`` as a selectable store
# rather than a folder to descend into.
STORE_SUFFIXES = (".zarr",)

_ICON = {"up": "mdi-arrow-up-left", "drive": "mdi-harddisk",
         "dir": "mdi-folder", "store": "mdi-database", "file": "mdi-file-outline"}
# Vuetify theme colours, so the list reads at a glance (folders warm, data cool).
_COLOR = {"up": "grey", "drive": "blue-grey", "dir": "amber-darken-1",
          "store": "blue", "file": "blue-grey-lighten-1"}


def _entry(name, path, kind):
    return {"name": name, "path": path, "type": kind,
            "icon": _ICON[kind], "color": _COLOR[kind]}


def _crumbs(path: str) -> list[dict]:
    """Breadcrumb trail for *path*: ``[{label, path}]`` from "This PC" down."""
    crumbs = [{"label": "This PC", "path": ""}]
    if not path:
        return crumbs
    path = os.path.abspath(path)
    drive, tail = os.path.splitdrive(path)
    if drive:
        acc = drive + os.sep
        crumbs.append({"label": drive, "path": acc})
    else:
        acc = os.sep
        crumbs.append({"label": "/", "path": acc})
    for part in [p for p in tail.split(os.sep) if p]:
        acc = os.path.join(acc, part)
        crumbs.append({"label": part, "path": acc})
    return crumbs


def list_drives() -> list[str]:
    """Mounted Windows drive roots (``['C:\\\\', 'D:\\\\', ...]``); ``[]`` off Windows.

    Uses the ``GetLogicalDrives`` bitmask, which reports assigned drive letters
    instantly without spinning up removable media — probing ``A:\\`` or an empty
    optical drive with ``os.path.exists`` can block for tens of seconds.
    """
    if os.name != "nt":
        return []
    import ctypes

    bitmask = ctypes.windll.kernel32.GetLogicalDrives()
    return [f"{string.ascii_uppercase[i]}:\\" for i in range(26) if bitmask & (1 << i)]


def _is_drive_root(path: str) -> bool:
    drive, tail = os.path.splitdrive(os.path.abspath(path))
    return bool(drive) and tail in ("", os.sep, "/")


def _parent(path: str) -> str | None:
    """The directory above *path*; ``""`` at a drive root (→ drive list), None at top."""
    if _is_drive_root(path):
        return ""                                   # one level up from C:\ is "pick a drive"
    up = os.path.dirname(os.path.abspath(path).rstrip("\\/"))
    return up or None


def _drives_view() -> dict:
    """The top level: a row per drive (Windows) or root (POSIX)."""
    roots = list_drives() or ["/"]
    entries = [_entry(r, r, "drive") for r in roots]
    return {"cwd": "", "parent": None, "entries": entries, "crumbs": _crumbs("")}


def browse(path: str | None, *, exts=None) -> dict:
    """List *path* for the picker: ``{cwd, parent, entries}``.

    Each entry is ``{name, path, type, icon}`` with ``type`` one of ``up`` (the
    parent row), ``dir`` (navigable folder), ``store`` (a selectable ``*.zarr``),
    or ``file`` (selectable). Files are kept only if their extension is in ``exts``
    (lowercase, dotted, e.g. ``{'.segy', '.sgy'}``); ``None`` keeps all. Folders
    are always listed so the user can navigate. Hidden/system entries and
    unreadable children are skipped rather than raising. An empty/None *path*
    yields the drive list.
    """
    if not path:
        return _drives_view()
    path = os.path.abspath(path)
    entries: list[dict] = []
    parent = _parent(path)
    if parent is not None:
        entries.append(_entry("..", parent, "up"))

    try:
        children = list(os.scandir(path))
    except OSError:                                  # permission denied, gone, etc.
        children = []

    dirs, stores, files = [], [], []
    exts = {e.lower() for e in exts} if exts else None
    for e in children:
        name = e.name
        if name.startswith(".") or name.startswith("$"):
            continue
        try:
            is_dir = e.is_dir()
        except OSError:
            continue
        if is_dir:
            if name.lower().endswith(STORE_SUFFIXES):
                stores.append(_entry(name, e.path, "store"))
            else:
                dirs.append(_entry(name, e.path, "dir"))
        elif exts is None or os.path.splitext(name)[1].lower() in exts:
            files.append(_entry(name, e.path, "file"))

    key = lambda d: d["name"].lower()
    entries += sorted(dirs, key=key) + sorted(stores, key=key) + sorted(files, key=key)
    return {"cwd": path, "parent": parent, "entries": entries, "crumbs": _crumbs(path)}
