# core/operators/test_interpolate.py
"""Tests for rom_operator_inference.core.operators._interpolate."""

import pytest
import numpy as np

import rom_operator_inference as opinf

from .. import _get_operators

_module = opinf.core.operators._interpolate
_base = opinf.core.operators._base


class _OperatorDummy(_base._BaseNonparametricOperator):
    """Instantiable version of _BaseNonparametricOperator."""
    def __init__(self, entries):
        _base._BaseNonparametricOperator.__init__(self, entries)

    def __call__(*args, **kwargs):
        return 0


class _InterpolatorDummy:
    """Dummy class for interpolator that obeys the following syntax.
    >>> interpolator_object = _InterpolatorDummy(data_points, data_values)
    >>> interpolator_evaluation = interpolator_object(new_data_point)
    Since this is a dummy, no actual interpolation or evaluation takes place.
    """
    def __init__(self, points, values, **kwargs):
        self.__values = values[0]

    def __call__(self, newpoint):
        return self.__values


class TestIterpolatedOperator:
    """Test opinf.core.operators._interpolate._InterpolatedOperator."""

    class Dummy(_module._InterpolatedOperator):
        """Instantiable version of _InterpolatedOperator."""
        _OperatorClass = _OperatorDummy
        _InterpolatorClass = _InterpolatorDummy

    @staticmethod
    def _set_up_interpolation_data(numpoints=4, shape=(10, 10)):
        """Set up points / values for dummy interpolation data."""
        parameters = np.linspace(0, 1, numpoints)
        matrices = list(np.random.random(((numpoints,) + shape)))
        return parameters, matrices

    def test_init(self):
        """Test _InterpolatedOperator.__init__()."""
        params, matrices = self._set_up_interpolation_data()

        # Try with different number of interpolation points and matrices.
        with pytest.raises(ValueError) as ex:
            self.Dummy(params, matrices[:-1])
        assert ex.value.args[0] == \
            f"{len(params)} = len(parameter_values) " \
            f"!= len(matrices) = {len(matrices[:-1])}"

        # Try with parameters of different shapes.
        params_bad = list(np.random.random((len(matrices), 2)))
        params_bad[0] = np.random.random(3)
        with pytest.raises(ValueError) as ex:
            self.Dummy(params_bad, matrices)
        assert ex.value.args[0] == "parameter sample shapes inconsistent"

        # Try with matrices of different shapes.
        with pytest.raises(ValueError) as ex:
            self.Dummy(params, matrices[:-1] + [np.random.random((10,2))])
        assert ex.value.args[0] == "operator matrix shapes inconsistent"

        # Correct usage.
        self.Dummy(params, matrices)

    def test_properties(self):
        """Test _InterpolatedOperator properties,
        parameter_values, matrices, shape, and interpolator.
        """
        params, matrices = self._set_up_interpolation_data()
        op = self.Dummy(params, matrices)

        # Check parameter_values / matrices attributes.
        assert op.parameter_values is params
        assert op.matrices is matrices
        assert isinstance(op.interpolator, _InterpolatorDummy)

        # Check shape attribute.
        for A in matrices:
            assert op.shape == A.shape

        # Ensure these attributes are all properties.
        for attr in ["parameter_values", "matrices", "shape", "interpolator"]:
            with pytest.raises(AttributeError) as ex:
                setattr(op, attr, 10)
            assert ex.value.args[0] == "can't set attribute"

    def test_call(self):
        """Test _InterpolatedOperator.__call__()."""
        params, matrices = self._set_up_interpolation_data()

        op = self.Dummy(params, matrices)
        A = op(.314159)
        assert isinstance(A, _OperatorDummy)
        assert A.shape == op.shape

    def test_eq(self):
        """Test _InterpolatedOperator.__eq__()."""
        params, matrices = self._set_up_interpolation_data()
        op1 = self.Dummy(params, matrices)
        op2 = self.Dummy(params[:-1], matrices[:-1])

        assert op1 != 1
        assert op1 != op2

        op2 = self.Dummy(params, [A[:,:-1] for A in matrices])
        assert op1 != op2

        op2 = self.Dummy(np.random.random((len(params), 11)), matrices)
        assert op1 != op2

        op2 = self.Dummy(params - 1, matrices)
        assert op1 != op2

        op2 = self.Dummy(params, matrices)
        assert op1 == op2


def test_spline1Doperators(r=10, m=3, s=5):
    """Test all Spline1d operator classes by instantiating and calling."""
    # Get nominal operators to play with.
    c, A, H, G, B = _get_operators(r, m)

    # Get interpolation data for each type of operator.
    params = np.linspace(0, 1, s)
    cs = [c + p**2 + np.random.standard_normal(c.shape)/20 for p in params]
    As = [A + p**2 + np.random.standard_normal(A.shape)/20 for p in params]
    Hs = [H + p**2 + np.random.standard_normal(H.shape)/20 for p in params]
    Gs = [G + p**2 + np.random.standard_normal(G.shape)/20 for p in params]
    Bs = [B + p**2 + np.random.standard_normal(B.shape)/20 for p in params]

    # Instantiate each Spline1d operator.
    csplineop = _module.Spline1dConstantOperator(params, cs)
    Asplineop = _module.Spline1dLinearOperator(params, As)
    Hsplineop = _module.Spline1dQuadraticOperator(params, Hs)
    Gsplineop = _module.Spline1dCubicOperator(params, Gs)
    Bsplineop = _module.Spline1dLinearOperator(params, Bs)

    # Call each Spline1d operator on a new parameter.
    p = .314159
    c_new = csplineop(p)
    assert isinstance(c_new, opinf.core.operators.ConstantOperator)
    assert c_new.shape == c.shape

    A_new = Asplineop(p)
    assert isinstance(A_new, opinf.core.operators.LinearOperator)
    assert A_new.shape == A.shape

    H_new = Hsplineop(p)
    assert isinstance(H_new, opinf.core.operators.QuadraticOperator)
    assert H_new.shape == H.shape

    G_new = Gsplineop(p)
    assert isinstance(G_new, opinf.core.operators.CubicOperator)
    assert G_new.shape == G.shape

    B_new = Bsplineop(p)
    assert isinstance(B_new, opinf.core.operators.LinearOperator)
    assert B_new.shape == B.shape


# TODO: interpolation options other than 1D cubic splines.
