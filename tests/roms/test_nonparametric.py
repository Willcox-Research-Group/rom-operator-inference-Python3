# roms/test_nonparametric.py
"""Tests for roms._nonparametric.py."""

import pytest

import opinf

from .test_base import _TestBaseROM


_module = opinf.roms


class TestROM(_TestBaseROM):
    """Test roms.ROM."""

    ROM = _module.ROM
    ModelClasses = (
        opinf.models.ContinuousModel,
        opinf.models.DiscreteModel,
    )

    def _get_models(self):
        """Return a list of valid model instantiations."""
        return [
            opinf.models.ContinuousModel("A"),
            opinf.models.DiscreteModel("AB"),
        ]

    def test_init(self):
        """Test __init__() and properties."""

        # Model error.
        with pytest.raises(TypeError) as ex:
            self.ROM(10)
        assert ex.value.args[0] == (
            "'model' must be a models.ContinuousModel "
            "or models.DiscreteModel instance"
        )

        # Other arguments.
        super().test_init()

    def test_fit(self):
        """Test fit()."""
        raise NotImplementedError

    def test_predict(self):
        """Test predict()."""
        raise NotImplementedError
