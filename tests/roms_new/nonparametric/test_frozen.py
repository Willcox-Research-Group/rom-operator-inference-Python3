# roms/nonparametric/test_frozen.py
"""Tests for roms.nonparametric._frozen."""

import pytest

import opinf


class TestFrozenMixin:
    """Test roms.nonparametric._frozen._FrozenMixin."""

    class Dummy(
        opinf.roms_new.nonparametric._frozen._FrozenMixin,
        opinf.roms_new.nonparametric._base._NonparametricROM,
    ):
        """Instantiable version of _FrozenMixin."""

        def predict(*args, **kwargs):
            pass

    def test_disabled(self, ModelClass=None):
        """Test roms.nonparametric._frozen._FrozenMixin.fit()."""
        if ModelClass is None:
            ModelClass = self.Dummy
        rom = ModelClass("A")

        # Test disabled data_matrix_ property.
        assert rom.data_matrix_ is None
        rom.solver_ = "A"
        assert rom.data_matrix_ is None
        assert rom.d is None

        # Test disabled fit().
        with pytest.raises(NotImplementedError) as ex:
            rom.fit(None, None, known_operators=None)
        assert ex.value.args[0] == (
            "fit() is disabled for this class, call fit() "
            "on the parametric ROM object"
        )
