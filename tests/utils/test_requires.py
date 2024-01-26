# utils/test_requires.py
"""Tests for utils._requires."""

import pytest

import opinf


def test_requires():
    """Test utils._requires.requires()."""

    class Dummy:
        def __init__(self, attr=None):
            self.attr = attr

        @opinf.utils.requires("attr")
        def do_something(self):
            pass

    d = Dummy()
    with pytest.raises(AttributeError) as ex:
        d.do_something()
    assert ex.value.args[0] == "attr not set"

    d.attr = 10
    d.do_something()
