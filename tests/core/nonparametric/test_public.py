# core/nonparametric/test_public.py
"""Tests for rom_operator_inference.core.nonparametric._public."""

# TODO: test fit(), predict().

# import pytest
# import numpy as np
# from scipy import linalg as la

import rom_operator_inference as opinf

from .. import _get_data
from .test_base import TestNonparametricOpInfROM


class TestSteadyOpInfROM:
    """Test core.nonparametric._public.SteadyOpInfROM."""
    ModelClass = opinf.core.nonparametric._public.SteadyOpInfROM

    def test_evaluate(self):
        """Test core.nonparametric._public.SteadyOpInfROM.evaluate()."""
        raise NotImplementedError

    def test_fit(self):
        """Test core.nonparametric._public.SteadyOpInfROM.fit()."""
        TestNonparametricOpInfROM.test_fit(self, self.ModelClass)

    def test_predict(self):
        """Test core.nonparametric._public.SteadyOpInfROM.predict()."""
        raise NotImplementedError


class TestDiscreteOpInfROM:
    """Test core.nonparametric._public.DiscreteOpInfROM."""
    ModelClass = opinf.core.nonparametric._public.DiscreteOpInfROM

    def test_evaluate(self):
        """Test core.nonparametric._public.DiscreteOpInfROM.evaluate()."""
        raise NotImplementedError

    def test_fit(self):
        """Test core.nonparametric._public.DiscreteOpInfROM.fit()."""
        TestNonparametricOpInfROM.test_fit(self, self.ModelClass)

        Q, Qnext, U = _get_data(20, 500, 3)
        U1d = U[0,:]

        rom = self.ModelClass("AB")
        rom.fit(None, Q, Qnext, U)
        rom.fit(None, Q, inputs=U)
        rom.fit(None, Q, inputs=U1d)

    def test_predict(self):
        """Test core.nonparametric._public.DiscreteOpInfROM.predict()."""
        raise NotImplementedError


class TestContinuousOpInfROM:
    """Test core.nonparametric._public.ContinuousOpInfROM."""
    ModelClass = opinf.core.nonparametric._public.ContinuousOpInfROM

    def test_evaluate(self):
        """Test core.nonparametric._public.ContinuousOpInfROM.evaluate()."""
        raise NotImplementedError

    def test_fit(self):
        """Test core.nonparametric._public.ContinuousOpInfROM.fit()."""
        TestNonparametricOpInfROM.test_fit(self, self.ModelClass)

    def test_predict(self):
        """Test core.nonparametric._public.ContinuousOpInfROM.predict()."""
        raise NotImplementedError
