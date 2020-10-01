# lstsq/__init__.py
"""Tests for rom_operator_inference.lstsq.__init__.py."""

import pytest

import rom_operator_inference as roi


def test_lstsq_size():
    """Test lstsq.lstsq_size()."""
    m, r = 3, 7

    # Try with bad input combinations.
    with pytest.raises(ValueError) as ex:
        roi.lstsq.lstsq_size("cAHB", r)
    assert ex.value.args[0] == "argument m > 0 required since 'B' in modelform"

    with pytest.raises(ValueError) as ex:
        roi.lstsq.lstsq_size("cAH", r, m=10)
    assert ex.value.args[0] == "argument m=10 invalid since 'B' in modelform"

    # Test without inputs.
    assert roi.lstsq.lstsq_size("c", r) == 1
    assert roi.lstsq.lstsq_size("A", r) == r
    assert roi.lstsq.lstsq_size("cA", r) == 1 + r
    assert roi.lstsq.lstsq_size("cAH", r) == 1 + r + r*(r+1)//2
    assert roi.lstsq.lstsq_size("cG", r) == 1 + r*(r+1)*(r+2)//6

    # Test with inputs.
    assert roi.lstsq.lstsq_size("cB", r, m) == 1 + m
    assert roi.lstsq.lstsq_size("AB", r, m) == r + m
    assert roi.lstsq.lstsq_size("cAB", r, m) == 1 + r + m
    assert roi.lstsq.lstsq_size("AHB", r, m) == r + r*(r+1)//2 + m
    assert roi.lstsq.lstsq_size("GB", r, m) == r*(r+1)*(r+2)//6 + m

    # Test with affines.
    assert roi.lstsq.lstsq_size("c", r, affines={"c":[0,0]}) == 2
    assert roi.lstsq.lstsq_size("A", r, affines={"A":[0,0]}) == 2*r
