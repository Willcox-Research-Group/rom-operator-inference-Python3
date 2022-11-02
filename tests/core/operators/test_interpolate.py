# core/operators/test_interpolate.py
"""Tests for core.operators._interpolate."""

import pytest
import numpy as np
import scipy.interpolate as interp

import opinf

from .. import _get_operators

_module = opinf.core.operators._interpolate
_base = opinf.core.operators._base


class _OperatorDummy(_base._BaseNonparametricOperator):
    """Instantiable version of _BaseNonparametricOperator."""
    def __init__(self, entries):
        _base._BaseNonparametricOperator.__init__(self, entries)

    def evaluate(*args, **kwargs):
        return 0

    def jacobian(*args, **kwargs):
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


class TestInterpolatedOperator:
    """Test opinf.core.operators._interpolate._InterpolatedOperator."""

    class Dummy(_module._InterpolatedOperator):
        """Instantiable version of _InterpolatedOperator."""
        _OperatorClass = _OperatorDummy

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
            self.Dummy(params, matrices[:-1], _InterpolatorDummy)
        assert ex.value.args[0] == \
            f"{len(params)} = len(parameters) " \
            f"!= len(matrices) = {len(matrices[:-1])}"

        # Try with parameters of different shapes.
        params_bad = list(np.random.random((len(matrices), 2)))
        params_bad[0] = np.random.random(3)
        with pytest.raises(ValueError) as ex:
            self.Dummy(params_bad, matrices, _InterpolatorDummy)
        assert ex.value.args[0] == "parameter sample shapes inconsistent"

        # Try with matrices of different shapes.
        with pytest.raises(ValueError) as ex:
            self.Dummy(params,
                       matrices[:-1] + [np.random.random((10, 2))],
                       _InterpolatorDummy)
        assert ex.value.args[0] == "operator matrix shapes inconsistent"

        # Correct usage.
        self.Dummy(params, matrices, _InterpolatorDummy)

    def test_properties(self):
        """Test _InterpolatedOperator properties,
        parameters, matrices, shape, and interpolator.
        """
        params, matrices = self._set_up_interpolation_data()
        op = self.Dummy(params, matrices, _InterpolatorDummy)

        # Check parameters / matrices attributes.
        assert op.parameters is params
        assert np.allclose(op.matrices, matrices)
        assert isinstance(op.interpolator, _InterpolatorDummy)

        # Check shape attribute.
        for A in matrices:
            assert op.shape == A.shape

        # Ensure these attributes are all properties.
        for attr in ["parameters", "matrices", "shape", "interpolator"]:
            with pytest.raises(AttributeError) as ex:
                setattr(op, attr, 10)
            assert ex.value.args[0] == "can't set attribute"

    def test_call(self):
        """Test _InterpolatedOperator.__call__()."""
        params, matrices = self._set_up_interpolation_data()

        op = self.Dummy(params, matrices, _InterpolatorDummy)

        A = op(.314159)
        assert isinstance(A, _OperatorDummy)
        assert A.shape == op.shape

    def test_eq(self):
        """Test _InterpolatedOperator.__eq__()."""
        params, matrices = self._set_up_interpolation_data()
        op1 = self.Dummy(params, matrices, _InterpolatorDummy)
        op2 = self.Dummy(params[:-1], matrices[:-1], _InterpolatorDummy)

        assert op1 != 1
        assert op1 != op2

        op2 = self.Dummy(params, [A[:, :-1] for A in matrices],
                         _InterpolatorDummy)
        assert op1 != op2

        op2 = self.Dummy(np.random.random((len(params), 11)), matrices,
                         _InterpolatorDummy)
        assert op1 != op2

        op2 = self.Dummy(params - 1, matrices, _InterpolatorDummy)
        assert op1 != op2

        op2 = self.Dummy(params, matrices, _InterpolatorDummy)
        assert op1 == op2


def test_1Doperators(r=10, m=3, s=5):
    """Test InterpolatedOperator classes with using all 1D interpolators
    from scipy.interpolate.
    """
    InterpolatorClass = interp.CubicSpline

    # Get nominal operators to play with.
    c, A, H, G, B = _get_operators(r, m)

    # Get interpolation data for each type of operator.
    params = np.sort(np.linspace(0, 1, s) + np.random.standard_normal(s)/40)
    cs = [c + p**2 + np.random.standard_normal(c.shape)/20 for p in params]
    As = [A + p**2 + np.random.standard_normal(A.shape)/20 for p in params]
    Hs = [H + p**2 + np.random.standard_normal(H.shape)/20 for p in params]
    Gs = [G + p**2 + np.random.standard_normal(G.shape)/20 for p in params]
    Bs = [B + p**2 + np.random.standard_normal(B.shape)/20 for p in params]

    # Instantiate each 1d-parametric operator.
    cinterp = _module.InterpolatedConstantOperator(params, cs,
                                                   InterpolatorClass)
    Ainterp = _module.InterpolatedLinearOperator(params, As, InterpolatorClass)
    Hinterp = _module.InterpolatedQuadraticOperator(params, Hs,
                                                    InterpolatorClass)
    Ginterp = _module.InterpolatedCubicOperator(params, Gs, InterpolatorClass)
    Binterp = _module.InterpolatedLinearOperator(params, Bs, InterpolatorClass)

    # Call each parametric operator on a new parameter.
    parameter = .314159
    for IC in [
        interp.Akima1DInterpolator,
        interp.BarycentricInterpolator,
        interp.CubicSpline,
        interp.KroghInterpolator,
        interp.PchipInterpolator,
    ]:
        for operator in [cinterp, Ainterp, Hinterp, Ginterp, Binterp]:
            operator.set_interpolator(IC)

        c_new = cinterp(parameter)
        assert isinstance(c_new, opinf.core.operators.ConstantOperator)
        assert c_new.shape == c.shape

        A_new = Ainterp(parameter)
        assert isinstance(A_new, opinf.core.operators.LinearOperator)
        assert A_new.shape == A.shape

        H_new = Hinterp(parameter)
        assert isinstance(H_new, opinf.core.operators.QuadraticOperator)
        assert H_new.shape == H.shape

        G_new = Ginterp(parameter)
        assert isinstance(G_new, opinf.core.operators.CubicOperator)
        assert G_new.shape == G.shape

        B_new = Binterp(parameter)
        assert isinstance(B_new, opinf.core.operators.LinearOperator)
        assert B_new.shape == B.shape

        with pytest.raises(ValueError) as ex:
            Ainterp([parameter, parameter, parameter])
        assert ex.value.args[0] == "expected parameter of shape (1,)"
