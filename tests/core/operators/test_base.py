# core/operators/test_base.py
"""Tests for rom_operator_inference.core.operators._base."""

import pytest
import numpy as np

import rom_operator_inference as opinf


_module = opinf.core.operators._base


class TestBaseNonparametricOperator:
    """Test core.operators._nonparametric._BaseNonparametricOperator."""
    class Dummy(_module._BaseNonparametricOperator):
        """Instantiable version of _BaseNonparametricOperator."""
        def __init__(self, entries):
            super().__init__(entries)

        def __call__(*args, **kwargs):
            pass

    class Dummy2(Dummy):
        """A distinct instantiable version of _BaseNonparametricOperator."""
        pass

    def test_init(self):
        """Test _BaseNonparametricOperator.__init__()."""
        A = np.random.random((10,11))
        op = self.Dummy(A)
        assert op.entries is A

    def test_validate_entries(self):
        """Test _BaseNonparametricOperator._validate_entries()."""
        func = _module._BaseNonparametricOperator._validate_entries
        with pytest.raises(TypeError) as ex:
            func([1, 2, 3, 4])
        assert ex.value.args[0] == "operator entries must be NumPy array"

        A = np.arange(12, dtype=float).reshape((4,3)).T
        A[0,0] = np.nan
        with pytest.raises(ValueError) as ex:
            func(A)
        assert ex.value.args[0] == "operator entries must not be NaN"

        A[0,0] = np.inf
        with pytest.raises(ValueError) as ex:
            func(A)
        assert ex.value.args[0] == "operator entries must not be Inf"

        # Valid argument, no exceptions raised.
        A[0,0] = 0
        func(A)

    def test_getitem(self):
        """Test _BaseNonparametricOperator.__getitem__()."""
        A = np.random.random((8, 6))
        op = self.Dummy(A)
        for s in [slice(2), (slice(1), slice(1,3)), slice(1, 4, 2)]:
            assert np.all(op[s] == A[s])

    def test_eq(self):
        """Test _BaseNonparametricOperator.__eq__()."""
        A = np.arange(12).reshape((4,3))
        opA = self.Dummy(A)
        opA2 = self.Dummy2(A)
        assert opA != opA2

        opAT = self.Dummy(A.T)
        assert opA != opAT

        opB = self.Dummy(A + 1)
        assert opA != opB

        opC = self.Dummy(A + 0)
        assert opA == opC


class TestBaseParametricOperator:
    """Test core.operators._nonparametric._BaseNonparametricOperator."""
    class Dummy(_module._BaseParametricOperator):
        """Instantiable version of _BaseParametricOperator."""
        _OperatorClass = _module._BaseNonparametricOperator

        def __init__(self):
            super().__init__()

        def __call__(*args, **kwargs):
            pass

    class BadDummy(_module._BaseParametricOperator):
        """Instantiable but incorrect version of _BaseNonparametricOperator."""
        _OperatorClass = int

        def __init__(self):
            super().__init__()

        def __call__(*args, **kwargs):
            pass

    def test_init(self):
        """Test _BaseParametricOperator.__init__() and .OperatorClass."""
        # Make sure Dummy is instantiable and has the right OperatorClass.
        dummy = self.Dummy()
        assert dummy.OperatorClass is _module._BaseNonparametricOperator

        # Try changing the OperatorClass.
        with pytest.raises(AttributeError) as ex:
            dummy.OperatorClass = int
        assert ex.value.args[0] == "can't set attribute"

        # Try instantiating a class with an invalid OperatorClass.
        with pytest.raises(RuntimeError) as ex:
            self.BadDummy()
        assert ex.value.args[0] == "invalid OperatorClass 'int'"

    def test_check_shape_consistency(self):
        """Test _BaseNonparametricOperator._check_shape_consistency()."""
        X = np.arange(12).reshape((3, 4))

        matrices = [X, X, X.T, X]
        with pytest.raises(ValueError) as ex:
            self.Dummy._check_shape_consistency(matrices, "input")
        assert ex.value.args[0] == "input shapes inconsistent"

        matrices = [X, X+1, X/2, X*0]
        self.Dummy._check_shape_consistency(matrices, "arguments")
