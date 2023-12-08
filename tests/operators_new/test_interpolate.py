# operators/test_interpoloate.py
"""Tests for operators._interpolate."""

# import pytest
import numpy as np

import opinf


_module = opinf.operators_new._interpolate

__d = 8
__Dblock = np.random.random((4, __d))


class _DummyOperator(opinf.operators_new._base._NonparametricOperator):
    """Instantiable version of _NonparametricOperator."""

    def set_entries(*args, **kwargs):
        _module._NonparametricOperator.set_entries(*args, **kwargs)

    def input_dimension(*args, **kwargs):
        pass

    def _str(*args, **kwargs):
        pass

    def apply(*args, **kwargs):
        return -1

    def galerkin(self, *args, **kwargs):
        return self

    def datablock(self, states, inputs=None):
        return __Dblock

    def operator_dimension(self, r, m):
        return __d


class TestInterpolatedOperator:
    """Test operators._interpolate._InterpolatedOperator."""

    class Dummy(_module._InterpolatedOperator):
        """Instantiable version of _InterpolatedOperator."""

        _OperatorClass = _DummyOperator

    def test_init(self):
        """Test _InterpolatedOperator.__init__()."""
        raise NotImplementedError

    def test_properties(self):
        """Test _InterpolatedOperator.set_entries(),
        entries, shape, state_dimension, training_parameters,
        interpolator(), and __len__().
        """
        raise NotImplementedError

    def test_eq(self):
        """Test _InterpolatedOperator.__eq__()."""
        raise NotImplementedError

    def test_evaluate(self):
        """Test _InterpolatedOperator.evaluate()."""
        raise NotImplementedError

    def test_galerkin(self):
        """Test _InterpolatedOperator.galerkin()."""
        raise NotImplementedError

    def test_datablock(self):
        """Test _InterpolatedOperator.datablock()."""
        raise NotImplementedError

    def test_operator_dimension(self):
        """Test _InterpolatedOperator.operator_dimension()."""
        raise NotImplementedError

    def test_copy(self):
        """Test _InterpolatedOperator.copy()."""
        raise NotImplementedError

    def test_save(self):
        """Test _InterpolatedOperator.save()."""
        raise NotImplementedError

    def test_load(self):
        """Test _InterpolatedOperator.load()."""
        raise NotImplementedError
