# models/nonparametric/test_frozen.py
"""Tests for models.nonparametric._frozen."""

import pytest

import opinf


class TestFrozenMixin:
    """Test models.nonparametric._frozen._FrozenMixin."""

    class Dummy(
        opinf.models.nonparametric._frozen._FrozenMixin,
        opinf.models.nonparametric._base._NonparametricModel,
    ):
        """Instantiable version of _FrozenMixin."""

        def predict(*args, **kwargs):
            pass

    def test_disabled(self, ModelClass=None):
        """Test models.nonparametric._frozen._FrozenMixin.fit()."""
        if ModelClass is None:
            ModelClass = self.Dummy
        model = ModelClass("A")

        # Test disabled data_matrix_ property.
        assert model.data_matrix_ is None
        model.solver_ = "A"
        assert model.data_matrix_ is None
        assert model.operator_matrix_dimension is None

        # Test disabled fit().
        with pytest.raises(NotImplementedError) as ex:
            model.fit(None, None, known_operators=None)
        assert ex.value.args[0] == (
            "fit() is disabled for this class, call fit() "
            "on the parametric model object"
        )

        # Test disabled _clear().
        with pytest.raises(NotImplementedError) as ex:
            model._clear()
        assert ex.value.args[0] == (
            "_clear() is disabled for this class, call fit() "
            "on the parametric model object"
        )
