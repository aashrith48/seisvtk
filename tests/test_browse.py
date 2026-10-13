"""Directory listing for the file-picker: typing, filtering, ordering, parents."""
import os

import pytest

from seisvtk.io.browse import browse


def _tree(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "cube.zarr").mkdir()          # a store (selectable), not a folder
    (tmp_path / "wells.csv").write_text("x")
    (tmp_path / "survey.segy").write_text("x")
    (tmp_path / ".hidden").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    return tmp_path


def test_lists_dirs_stores_files_in_order(tmp_path):
    out = browse(str(_tree(tmp_path)))
    types = [e["type"] for e in out["entries"]]
    # parent row first, then dir, then store, then files (hidden dropped)
    assert types[0] == "up"
    assert "dir" in types and "store" in types and "file" in types
    assert types.index("dir") < types.index("store") < types.index("file")
    names = [e["name"] for e in out["entries"]]
    assert ".hidden" not in names
    assert out["cwd"] == os.path.abspath(str(tmp_path))


def test_zarr_is_a_store_not_a_dir(tmp_path):
    out = browse(str(_tree(tmp_path)))
    store = next(e for e in out["entries"] if e["name"] == "cube.zarr")
    assert store["type"] == "store"


def test_ext_filter_keeps_only_matching_files(tmp_path):
    out = browse(str(_tree(tmp_path)), exts={".segy", ".sgy"})
    files = [e["name"] for e in out["entries"] if e["type"] == "file"]
    assert files == ["survey.segy"]              # wells.csv / notes.txt filtered out
    # folders and stores are still listed regardless of the file filter
    assert any(e["type"] == "store" for e in out["entries"])


def test_parent_row_navigates_up(tmp_path):
    sub = _tree(tmp_path) / "sub"
    out = browse(str(sub))
    up = out["entries"][0]
    assert up["type"] == "up"
    assert up["path"] == os.path.abspath(str(tmp_path))


def test_empty_path_lists_drives():
    out = browse("")
    assert out["parent"] is None                 # nothing above the drive list
    assert out["entries"]                         # at least one drive/root
    assert all(e["type"] == "drive" for e in out["entries"])


def test_missing_dir_is_empty_not_error(tmp_path):
    out = browse(str(tmp_path / "does-not-exist"))
    # no children, but it still returns a parent row rather than raising
    assert [e for e in out["entries"] if e["type"] != "up"] == []


def test_entries_carry_icon_and_color(tmp_path):
    out = browse(str(_tree(tmp_path)))
    store = next(e for e in out["entries"] if e["type"] == "store")
    assert store["icon"] and store["color"]          # both set for the UI


def test_crumbs_trace_the_path(tmp_path):
    sub = _tree(tmp_path) / "sub"
    crumbs = browse(str(sub))["crumbs"]
    assert crumbs[0] == {"label": "This PC", "path": ""}   # jump-to-drives root
    assert crumbs[-1]["label"] == "sub"
    assert crumbs[-1]["path"] == os.path.abspath(str(sub))
    # each crumb's path is an ancestor of the next
    for a, b in zip(crumbs[1:], crumbs[2:]):
        assert b["path"].startswith(a["path"])
