# core/nonparametric/test_frozen.py
"""Tests for core.nonparametric._frozen."""

import pytest

import opinf


class TestFrozenMixin:
    """Test core.nonparametric._frozen._FrozenMixin."""

    class Dummy(opinf.core.nonparametric._frozen._FrozenMixin,
                opinf.core.nonparametric._base._NonparametricOpInfROM):
        """Instantiable version of _FrozenMixin."""
        def predict(*args, **kwargs):
            pass

    def test_disabled(self, ModelClass=None):
        """Test core.nonparametric._frozen._FrozenMixin.fit()."""
        if ModelClass is None:
            ModelClass = self.Dummy
        rom = ModelClass("A")

        # Test disabled data_matrix_ property.
        assert rom.data_matrix_ is None
        rom.solver_ = "A"
        assert rom.data_matrix_ is None

        # Test disabled fit().
        with pytest.raises(NotImplementedError) as ex:
            rom.fit(None, None, known_operators=None)
        assert ex.value.args[0] == \
            ("fit() is disabled for this class, call fit() "
             "on the parametric ROM object")


# TODO: test each child of _FrozenMixin.
