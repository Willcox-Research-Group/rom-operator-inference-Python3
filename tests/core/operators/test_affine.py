# core/operators/test_affine.py
"""Tests for rom_operator_inference.core.operators._affine."""

import pytest
import numpy as np

import rom_operator_inference as opinf


_module = opinf.core.operators._affine
_base = opinf.core.operators._base


class _OperatorDummy(_base._BaseNonparametricOperator):
    """Instantiable version of _BaseNonparametricOperator."""
    def __init__(self, entries):
        _base._BaseNonparametricOperator.__init__(self, entries)

    def __call__(*args, **kwargs):
        return 0


class TestAffineOperator:
    """Test opinf.core.operators._affine._AffineOperator."""

    class Dummy(_module._AffineOperator):
        """Instantiable version of core.operators._affine._AffineOperator."""
        _OperatorClass = _OperatorDummy

    @staticmethod
    def _set_up_affine_expansion(n=5):
        """Set up a valid affine expansion."""
        coefficient_functions = [np.sin, np.cos, np.exp]
        matrices = list(np.random.random((len(coefficient_functions), n, n)))
        return coefficient_functions, matrices

    def test_init(self):
        """Test core.operators._affine._AffineOperator.__init__()."""
        funcs, matrices = self._set_up_affine_expansion()

        # Try with non-callables.
        with pytest.raises(TypeError) as ex:
            self.Dummy(matrices, matrices)
        assert ex.value.args[0] == \
            "coefficient functions of affine operator must be callable"

        # Try with different number of functions and matrices.
        with pytest.raises(ValueError) as ex:
            self.Dummy(funcs, matrices[:-1])
        assert ex.value.args[0] == \
            f"{len(funcs)} = len(coefficient_functions) " \
            f"!= len(matrices) = {len(matrices[:-1])}"

        # Try with matrices of different shapes.
        with pytest.raises(ValueError) as ex:
            self.Dummy(funcs, matrices[:-1] + [np.random.random((10,2))])
        assert ex.value.args[0] == "operator matrix shapes inconsistent"

        # Correct usage.
        self.Dummy(funcs, matrices)

    def test_properties(self):
        """Test _AffineOperator.[coefficient_functions|matrices|shape]."""
        funcs, matrices = self._set_up_affine_expansion()
        op = self.Dummy(funcs, matrices)

        # Check coefficient_functions / matrices attributes.
        assert op.coefficient_functions is funcs
        assert op.matrices is matrices

        # Check shape attribute.
        for A in matrices:
            assert op.shape == A.shape

        # Ensure these attributes are all properties.
        for attr in ["coefficient_functions", "matrices", "shape"]:
            with pytest.raises(AttributeError) as ex:
                setattr(op, attr, 10)
            assert ex.value.args[0] == "can't set attribute"

    def test_validate_coefficient_functions(self):
        """Test _AffineOperator._validate_coefficient_functions()."""
        funcs, matrices = self._set_up_affine_expansion()

        # Try with non-callables.
        with pytest.raises(TypeError) as ex:
            self.Dummy._validate_coefficient_functions(matrices, matrices)
        assert ex.value.args[0] == \
            "coefficient functions of affine operator must be callable"

        # Try with vector-valued functions.
        def f1(t):
            return np.array([t, t**2])

        with pytest.raises(ValueError) as ex:
            self.Dummy._validate_coefficient_functions([f1, f1], 10)
        assert ex.value.args[0] == \
            "coefficient functions of affine operator must return a scalar"

        # Correct usage.
        self.Dummy._validate_coefficient_functions(funcs, 10)

    def test_call(self):
        """Test _AffineOperator.__call__()."""
        funcs, matrices = self._set_up_affine_expansion()

        op = self.Dummy(funcs, matrices)
        A = op(10)
        assert isinstance(A, _OperatorDummy)
        assert A.shape == op.shape

        A_true = sum([funcs[i](10)*matrices[i] for i in range(len(op))])
        assert np.allclose(A.entries, A_true)

    def test_eq(self):
        """Test _AffineOperator.__eq__()."""
        funcs, matrices = self._set_up_affine_expansion()
        op1 = self.Dummy(funcs, matrices)
        op2 = self.Dummy(funcs[:-1], matrices[:-1])

        assert op1 != 1
        assert op1 != op2

        op2 = self.Dummy(funcs, [A[:,:-1] for A in matrices])
        assert op1 != op2

        op2 = self.Dummy(funcs, matrices)
        assert op1 == op2
