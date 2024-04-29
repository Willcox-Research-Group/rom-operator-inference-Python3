# utils/test_repr.py
"""Tests for utils._repr."""

import opinf


def test_str2repr(s="test string representation"):
    """Test utils._repr.str2repr()."""

    class Dummy:
        def __init__(self, s):
            self.__s = str(s)

        def __str__(self):
            return self.__s

    d = Dummy(s)
    rep = opinf.utils.str2repr(d)

    assert rep.startswith("<Dummy object at ")
    lines = rep.split("\n")
    assert len(lines) == 2
    assert str(hex(id(d))) in lines[0]
    assert lines[1] == s
