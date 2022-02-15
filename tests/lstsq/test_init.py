# lstsq/__init__.py
"""Tests for rom_operator_inference.lstsq.__init__.py."""

import pytest

import rom_operator_inference as opinf


def test_lstsq_size():
    """Test lstsq.lstsq_size()."""
    m, r = 3, 7

    # Try with bad input combinations.
    with pytest.raises(ValueError) as ex:
        opinf.lstsq.lstsq_size("cAHB", r)
    assert ex.value.args[0] == "argument m > 0 required since 'B' in modelform"

    with pytest.raises(ValueError) as ex:
        opinf.lstsq.lstsq_size("cAH", r, m=10)
    assert ex.value.args[0] == "argument m=10 invalid since 'B' in modelform"

    # Test without inputs.
    assert opinf.lstsq.lstsq_size("c", r) == 1
    assert opinf.lstsq.lstsq_size("A", r) == r
    assert opinf.lstsq.lstsq_size("cA", r) == 1 + r
    assert opinf.lstsq.lstsq_size("cAH", r) == 1 + r + r*(r+1)//2
    assert opinf.lstsq.lstsq_size("cG", r) == 1 + r*(r+1)*(r+2)//6

    # Test with inputs.
    assert opinf.lstsq.lstsq_size("cB", r, m) == 1 + m
    assert opinf.lstsq.lstsq_size("AB", r, m) == r + m
    assert opinf.lstsq.lstsq_size("cAB", r, m) == 1 + r + m
    assert opinf.lstsq.lstsq_size("AHB", r, m) == r + r*(r+1)//2 + m
    assert opinf.lstsq.lstsq_size("GB", r, m) == r*(r+1)*(r+2)//6 + m

    # Test with affines.
    assert opinf.lstsq.lstsq_size("c", r, affines={"c":[0,0]}) == 2
    assert opinf.lstsq.lstsq_size("A", r, affines={"A":[0,0]}) == 2*r
