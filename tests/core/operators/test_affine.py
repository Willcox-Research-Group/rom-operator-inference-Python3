# core/operators/test_affine.py
"""Tests for rom_operator_inference.core.operators._affine."""

import pytest
import numpy as np

import rom_operator_inference as opinf

from .. import _get_operators


_module = opinf.core.operators._affine
_base = opinf.core.operators._base


class _OperatorDummy(_base._BaseNonparametricOperator):
    """Instantiable version of _BaseNonparametricOperator."""
    def __init__(self, entries):
        _base._BaseNonparametricOperator.__init__(self, entries)

    def evaluate(*args, **kwargs):
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
            self.Dummy(funcs, matrices[:-1] + [np.random.random((10, 2))])
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

        op2 = self.Dummy(funcs, [A[:, :-1] for A in matrices])
        assert op1 != op2

        op2 = self.Dummy(funcs, matrices)
        assert op1 == op2


def test_affineoperators(r=10, m=3, s=5):
    """Test all affine operator classes by instantiating and calling."""
    # Get nominal operators to play with.
    c, A, H, G, B = _get_operators(r, m)

    # Get interpolation data for each type of operator.
    funcs = [np.sin, np.cos, np.exp]
    cs = list(np.random.standard_normal((3,) + c.shape))
    As = list(np.random.standard_normal((3,) + A.shape))
    Hs = list(np.random.standard_normal((3,) + H.shape))
    Gs = list(np.random.standard_normal((3,) + G.shape))
    Bs = list(np.random.standard_normal((3,) + B.shape))

    # Instantiate each affine operator.
    caffineop = _module.AffineConstantOperator(funcs, cs)
    Aaffineop = _module.AffineLinearOperator(funcs, As)
    Haffineop = _module.AffineQuadraticOperator(funcs, Hs)
    Gaffineop = _module.AffineCubicOperator(funcs, Gs)
    Baffineop = _module.AffineLinearOperator(funcs, Bs)

    # Call each affine operator on a new parameter.
    p = .314159
    c_new = caffineop(p)
    assert isinstance(c_new, opinf.core.operators.ConstantOperator)
    assert c_new.shape == c.shape

    A_new = Aaffineop(p)
    assert isinstance(A_new, opinf.core.operators.LinearOperator)
    assert A_new.shape == A.shape

    H_new = Haffineop(p)
    assert isinstance(H_new, opinf.core.operators.QuadraticOperator)
    assert H_new.shape == H.shape

    G_new = Gaffineop(p)
    assert isinstance(G_new, opinf.core.operators.CubicOperator)
    assert G_new.shape == G.shape

    B_new = Baffineop(p)
    assert isinstance(B_new, opinf.core.operators.LinearOperator)
    assert B_new.shape == B.shape
