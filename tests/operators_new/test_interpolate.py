# operators/test_interpoloate.py
"""Tests for operators._interpolate."""

import os
import h5py
import pytest
import numpy as np
import scipy.linalg as la
import scipy.interpolate as interp

import opinf

from . import _get_operator_entries


_module = opinf.operators_new._interpolate

_d = 8
_Dblock = np.random.random((4, _d))


class _DummyOperator(opinf.operators_new._base._NonparametricOperator):
    """Instantiable version of _NonparametricOperator."""

    def set_entries(*args, **kwargs):
        opinf.operators_new._base._NonparametricOperator.set_entries(
            *args, **kwargs
        )

    def _str(*args, **kwargs):
        pass

    def apply(*args, **kwargs):
        return -1

    def galerkin(self, *args, **kwargs):
        return self

    def datablock(self, states, inputs=None):
        return _Dblock

    @staticmethod
    def operator_dimension(r, m):
        return _d


class _DummyInterpolator:
    """Dummy class for interpolator that obeys the following syntax.

       >>> interpolator_object = _InterpolatorDummy(data_points, data_values)
       >>> interpolator_evaluation = interpolator_object(new_data_point)

    Since this is a dummy, no actual interpolation or evaluation takes place.
    """

    def __init__(self, points, values, **kwargs):
        self.__values = values[0]

    def __call__(self, newpoint):
        return self.__values


class _DummyInterpolator2(_DummyInterpolator):
    pass


class TestInterpolatedOperator:
    """Test operators._interpolate._InterpolatedOperator."""

    class Dummy(_module._InterpolatedOperator):
        """Instantiable version of _InterpolatedOperator."""

        _OperatorClass = _DummyOperator

    def test_init(self, s=5, p=3, r=4):
        """Test _InterpolatedOperator.__init__()."""
        mu = np.random.random((s, p))
        self.Dummy(mu, _DummyInterpolator)
        entries = np.random.random((s, r, r))
        self.Dummy(mu, _DummyInterpolator, entries)
        model = self.Dummy(mu)
        assert model.InterpolatorClass is interp.LinearNDInterpolator
        model = self.Dummy(mu[:, 0])
        assert model.InterpolatorClass is interp.CubicSpline

    def test_properties(self, s=5, p=3, r=4):
        """Test _InterpolatedOperator.set_entries(),
        entries, __len__(), shape, state_dimension, training_parameters,
        interpolator(), set_InterpolatorClass(), and _clear().
        """
        mu = np.random.random((s, p))
        op = self.Dummy(mu, _DummyInterpolator)
        assert np.all(op.training_parameters == mu)
        assert len(op) == s
        assert op.entries is None
        assert op.shape is None
        assert op.state_dimension is None
        assert op.interpolator is None

        entries = np.random.random((s, r, r))
        op.set_entries(entries)
        assert np.all(op.entries == entries)
        assert op.shape == (r, r)
        assert op.state_dimension == r
        assert isinstance(op.interpolator, _DummyInterpolator)

        op.entries = np.hstack(entries)
        assert np.all(op.entries == entries)

        # Try with wrong number of entry arrays.
        with pytest.raises(ValueError) as ex:
            op.set_entries(entries[1:])
        assert (
            ex.value.args[0] == f"{s} = len(training_parameters) "
            f"!= len(entries) = {s-1}"
        )

        op.set_InterpolatorClass(_DummyInterpolator2)
        assert isinstance(op.interpolator, _DummyInterpolator2)
        op.InterpolatorClass = _DummyInterpolator
        assert isinstance(op.interpolator, _DummyInterpolator)
        assert op.InterpolatorClass == _DummyInterpolator

        del op.entries
        assert op.entries is None
        assert op.shape is None
        assert op.state_dimension is None
        assert op.interpolator is None

    def test_eq(self, s=4, p=3, r=2):
        """Test _InterpolatedOperator.__eq__()."""
        mu = np.random.random((s, p))
        entries = np.random.random((s, r, r + 1))

        op1 = self.Dummy(mu, _DummyInterpolator)
        assert op1 != 0

        op2 = self.Dummy(mu[1:, :-1], _DummyInterpolator)
        assert op1 != op2

        op2 = self.Dummy(mu + 1, _DummyInterpolator)
        assert op1 != op2

        op2 = self.Dummy(mu, int)
        assert op1 != op2

        op2 = self.Dummy(mu, _DummyInterpolator)
        assert op1 == op2

        op1.set_entries(entries)
        assert op1 != op2
        assert op2 != op1

        op2.set_entries([A.T for A in entries])
        assert op1 != op2

        op2.set_entries(entries)
        assert op1 == op2

    def test_evaluate(self, s=3, p=5, r=4):
        """Test _InterpolatedOperator.evaluate()."""
        mu = np.random.random((s, p))
        op = self.Dummy(mu, _DummyInterpolator)

        with pytest.raises(AttributeError):
            op.evaluate(mu[0])

        entries = np.random.random((s, r, r))
        op.set_entries(entries)
        op_evaluated = op.evaluate(mu[0])
        assert isinstance(op_evaluated, self.Dummy._OperatorClass)
        assert op_evaluated.entries.shape == (r, r)
        assert np.all(op_evaluated.entries == entries[0])

    def test_galerkin(self, s=5, p=2, n=10, r=4):
        """Test _InterpolatedOperator.galerkin()."""
        Vr = np.empty((n, r))
        mu = np.random.random((s, p))
        entries = np.random.random((s, n, n))
        op = self.Dummy(mu, _DummyInterpolator, entries)

        op_reduced = op.galerkin(Vr)
        assert isinstance(op_reduced, self.Dummy)
        assert np.all(op_reduced.entries == entries)

    def test_datablock(self, s=4, p=2, r=2, k=3):
        """Test _InterpolatedOperator.datablock()."""
        mu = np.random.random((s, p))
        states = np.random.random((s, r, k))
        op = self.Dummy(mu, _DummyInterpolator)
        block = op.datablock(states, states)
        assert block.shape == (s * _Dblock.shape[0], s * _Dblock.shape[1])
        assert np.all(block == la.block_diag(*[_Dblock for _ in range(s)]))

    def test_operator_dimension(self, s=3, p=6):
        """Test _InterpolatedOperator.operator_dimension()."""
        mu = np.random.random((s, p))
        op = self.Dummy(mu, _DummyInterpolator)
        assert op.operator_dimension(None, None) == _d * s

    def test_copy(self, s=4, p=2, r=5):
        """Test _InterpolatedOperator.copy()."""
        mu = np.random.random((s, p))
        entries = np.random.random((s, r, r))
        op1 = self.Dummy(mu, _DummyInterpolator)

        op2 = op1.copy()
        assert op2.training_parameters.shape == op1.training_parameters.shape
        assert np.all(op2.training_parameters == op1.training_parameters)
        assert op2.entries is None
        assert op2.interpolator is None

        op1.set_entries(entries)
        op1.set_InterpolatorClass(_DummyInterpolator2)
        op2 = op1.copy()
        assert op2.training_parameters.shape == op1.training_parameters.shape
        assert np.all(op2.training_parameters == op1.training_parameters)
        assert op2.shape == op1.shape
        assert np.all(op2.entries == op1.entries)
        assert isinstance(op2.interpolator, _DummyInterpolator2)

    def test_save(self, s=5, p=2, r=3, target="_interpolatedopsavetest.h5"):
        """Lightly test _InterpolatedOperator.save()."""
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        op = self.Dummy(np.random.random((s, p)), _DummyInterpolator)

        with pytest.warns(UserWarning) as wn:
            op.save(target)
        assert (
            wn[0].message.args[0] == "cannot serialize InterpolatorClass "
            "'_DummyInterpolator', must pass in the class when calling load()"
        )
        assert os.path.isfile(target)

        op = self.Dummy(np.sort(np.random.random(s)))
        op.save(target, overwrite=True)

        os.remove(target)

    def test_load(self, s=15, p=3, r=3, target="_interpolatedoploadtest.h5"):
        """Test _InterpolatedOperator.load()."""
        if os.path.isfile(target):
            os.remove(target)

        mu = np.random.random((s, p))
        entries = np.random.random((s, r, r))

        with h5py.File(target, "w") as hf:
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["class"] = "NotARealClass"

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.Dummy.load(target, None)
        assert ex.value.args[0] == (
            f"file '{target}' contains 'NotARealClass' object, "
            "use 'NotARealClass.load()'"
        )

        with pytest.warns(UserWarning) as wn:
            self.Dummy(mu, _DummyInterpolator).save(target, overwrite=True)
        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.Dummy.load(target)
        assert (
            ex.value.args[0] == "unknown InterpolatorClass "
            f"'_DummyInterpolator', "
            f"call Dummy.load({target}, _DummyInterpolator)"
        )
        self.Dummy.load(target, _DummyInterpolator)

        op1 = self.Dummy(mu)
        op1.save(target, overwrite=True)

        with pytest.warns(UserWarning) as wn:
            op2 = self.Dummy.load(target, _DummyInterpolator2)
        assert wn[0].message.args[0] == (
            "InterpolatorClass=_DummyInterpolator2 does not match loadfile "
            "InterpolatorClass 'LinearNDInterpolator'"
        )
        op2.set_InterpolatorClass(op1.InterpolatorClass)
        assert op2 == op1

        op1 = self.Dummy(np.sort(mu[:, 0]), entries=entries)
        op1.save(target, overwrite=True)
        op2 = self.Dummy.load(target)
        assert op2 == op1

        # Clean up.
        os.remove(target)


def test_1Doperators(r=10, m=3, s=5):
    """Test InterpolatedOperator classes with using all 1D interpolators
    from scipy.interpolate.
    """
    InterpolatorClass = interp.CubicSpline

    # Get nominal operators to play with.
    c, A, H, G, B, N = _get_operator_entries(r, m)

    # Get interpolation data for each type of operator.
    params = np.sort(np.linspace(0, 1, s) + np.random.standard_normal(s) / 40)
    mu_new = 0.314159

    for OpClass, Ohat in [
        (_module.InterpolatedConstantOperator, c),
        (_module.InterpolatedLinearOperator, A),
        (_module.InterpolatedQuadraticOperator, H),
        (_module.InterpolatedCubicOperator, G),
        (_module.InterpolatedInputOperator, B),
        (_module.InterpolatedStateInputOperator, N),
    ]:
        entries = [
            Ohat + p**2 + np.random.standard_normal(Ohat.shape) / 20
            for p in params
        ]
        for InterpolatorClass in [
            interp.Akima1DInterpolator,
            interp.BarycentricInterpolator,
            interp.CubicSpline,
            interp.KroghInterpolator,
            interp.PchipInterpolator,
        ]:
            op = OpClass(params, InterpolatorClass)
            if opinf.operators_new.has_inputs(op):
                assert op.input_dimension is None
            op.set_entries(np.column_stack(entries))
            if opinf.operators_new.has_inputs(op):
                assert op.input_dimension == m
            op_evaluated = op.evaluate(mu_new)
            assert isinstance(op_evaluated, OpClass._OperatorClass)
            assert op_evaluated.shape == op.shape
            for mu_i, Ohat_i in zip(params, entries):
                op_evaluated = op.evaluate(mu_i)
                assert isinstance(op_evaluated, OpClass._OperatorClass)
                assert op_evaluated.shape == op.shape
                assert np.allclose(op_evaluated.entries, Ohat_i)
