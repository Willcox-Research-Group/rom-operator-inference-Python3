# roms/test_parametric.py
"""Tests for roms._parametric.py."""

import pytest

import opinf

from .test_base import _TestBaseROM


_module = opinf.roms


class TestROM(_TestBaseROM):
    """Test roms.ROM."""

    ROM = _module.ParametricROM
    ModelClasses = (
        opinf.models.ParametricContinuousModel,
        opinf.models.ParametricDiscreteModel,
        opinf.models.InterpContinuousModel,
        opinf.models.InterpDiscreteModel,
    )

    def _get_models(self):
        """Return a list of valid model instantiations."""
        return [
            opinf.models.InterpContinuousModel("A"),
            opinf.models.InterpDiscreteModel("AB"),
        ]

    def test_init(self):
        """Test __init__() and properties."""

        # Model error.
        with pytest.raises(TypeError) as ex:
            self.ROM(10)
        assert ex.value.args[0] == (
            "'model' must be a parametric model instance"
        )

        # Other arguments.
        super().test_init()

    # DONE TO HERE.
    def test_fit(self):
        """Test fit()."""
        raise NotImplementedError

    def test_predict(self):
        """Test predict()."""
        raise NotImplementedError
