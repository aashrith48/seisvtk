"""Delimited-table sniffing: delimiter, header, column pick, nulls."""
import numpy as np
import pytest

from seisvtk.io.tables import (NULLS, detect_delimiter, pick_column, read_table,
                               to_float)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_detect_delimiter_variants():
    assert detect_delimiter(["a,b,c", "1,2,3"]) == ","
    assert detect_delimiter(["a;b;c", "1;2;3"]) == ";"
    assert detect_delimiter(["a\tb\tc", "1\t2\t3"]) == "\t"
    assert detect_delimiter(["a b c", "1 2 3"]) is None          # whitespace
    # a comma inside one field, not a separator, must not win
    assert detect_delimiter(["1 2 3", "4 5 6", "7,0 8 9"]) is None


def test_header_autodetect_and_1based_names(tmp_path):
    hdr = read_table(_write(tmp_path, "h.csv", "MD,GR\n10,55\n20,60\n"))
    assert list(hdr.columns) == ["MD", "GR"]
    headerless = read_table(_write(tmp_path, "n.csv", "10,55\n20,60\n"))
    assert list(headerless.columns) == ["1", "2"]                 # 1-based


def test_comment_and_blank_lines_skipped(tmp_path):
    txt = "# comment\n! another\n\nMD GR\n10 55\n20 60\n"
    df = read_table(_write(tmp_path, "c.txt", txt))
    assert df.shape == (2, 2)
    assert list(df.columns) == ["MD", "GR"]


def test_pick_column_by_name_and_1based():
    import pandas as pd
    frame = pd.DataFrame({"MD": [1.0], "GR_R": [2.0]})
    assert pick_column(frame, "md") == "MD"                       # case-insensitive
    assert pick_column(frame, "GRR") == "GR_R"                    # underscore-insensitive
    assert pick_column(frame, 2) == "GR_R"                        # 1-based
    with pytest.raises(IndexError):
        pick_column(frame, 9)
    with pytest.raises(KeyError):
        pick_column(frame, "NOPE")


def test_to_float_maps_nulls_to_nan():
    v = to_float([10.0, -999.25, 20.0, 1.0e30], NULLS)
    assert np.isnan(v[1]) and np.isnan(v[3])
    assert v[0] == 10.0 and v[2] == 20.0
