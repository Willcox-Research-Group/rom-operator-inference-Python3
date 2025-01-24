# operators/test_interpoloate.py
"""Tests for operators._interpolate."""

import os
import h5py
import pytest
import numpy as np
import scipy.linalg as la
import scipy.interpolate as interp

import opinf
import opinf.operators._utils as oputils


# try:
#     from .test_base import _TestParametricOpInfOperator
# except ImportError:
#     from test_base import _TestParametricOpInfOperator


_module = opinf.operators
_submodule = _module._interpolate

_d = 8
_Dblock = np.random.random((4, _d))


def _get_operator_entries(r=10, m=3, expanded=False):
    """Construct fake model operators."""
    c = np.random.random(r)
    A = np.eye(r)
    H = np.zeros((r, r**2 if expanded else r * (r + 1) // 2))
    G = np.zeros((r, r**3 if expanded else r * (r + 1) * (r + 2) // 6))
    B = np.random.random((r, m)) if m else None
    N = np.random.random((r, r * m)) if m else None
    return c, A, H, G, B, N


class _DummyOperator(opinf.operators.OpInfOperator):
    """Instantiable version of OpInfOperator."""

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


class TestInterpOperator:
    """Test operators._interpolate._InterpOperator."""

    class Dummy(_submodule._InterpOperator):
        """Instantiable version of _InterpOperator."""

        _OperatorClass = _DummyOperator

    def test_from_operators(self, s=7, p=2, r=5):
        """Test _InterpOperator._from_operators()."""
        mu = np.random.random((s, p))

        with pytest.raises(TypeError) as ex:
            self.Dummy._from_operators(mu, [10, "10"])
        assert ex.value.args[0] == (
            "can only interpolate operators of type '_DummyOperator'"
        )

        operators = [_DummyOperator() for _ in range(s)]
        with pytest.raises(ValueError) as ex:
            self.Dummy._from_operators(mu, operators)
        assert ex.value.args[0] == (
            "operators must have entries set in order to interpolate"
        )

        for op in operators:
            op.set_entries(np.random.random((r, r)))

        op = self.Dummy._from_operators(mu, operators)
        assert isinstance(op, self.Dummy)
        assert all(
            np.all(op.entries[i] == operators[i].entries) for i in range(s)
        )

        op = self.Dummy._from_operators(
            mu, operators, InterpolatorClass=_DummyInterpolator
        )
        assert isinstance(op.interpolator, _DummyInterpolator)

    def test_set_training_parameters(self, s=10, p=2, r=4):
        """Test _InterpOperator.set_training_parameters(),
        the training_parameter property, and __len__().
        """
        op = self.Dummy()
        assert op.training_parameters is None
        assert op.parameter_dimension is None
        assert op.state_dimension is None

        mu_bad = np.empty((s, p, p))
        with pytest.raises(ValueError) as ex:
            op.training_parameters = mu_bad
        assert ex.value.args[0] == (
            "parameter values must be scalars or 1D arrays"
        )

        mu = np.random.random((s, p))
        op.set_training_parameters(mu)
        assert np.all(op.training_parameters == mu)
        assert op.state_dimension is None
        assert op.interpolator is None
        assert op.parameter_dimension == p

        op.set_training_parameters(mu[:, 0])
        assert np.all(op.training_parameters == mu[:, 0])
        assert op.parameter_dimension == 1

        op.set_training_parameters(mu[:, 0].reshape((-1, 1)))
        assert np.all(op.training_parameters == mu[:, 0])
        assert op.parameter_dimension == 1

        entries = np.random.standard_normal((s, r, r))
        op = self.Dummy(mu, entries)

        with pytest.raises(AttributeError) as ex:
            op.set_training_parameters(mu)
        assert ex.value.args[0] == (
            "can't set attribute (entries already set, call _clear())"
        )

    def test_set_entries(self, s=5, p=3, r=4):
        """Test _InterpOperator.set_entries(), _clear(), and the
        the entries and shape properties.
        """
        mu = np.random.random((s, p))
        entries = np.random.random((s, r, r))

        # Try without training_parameters set.
        op = self.Dummy()
        with pytest.raises(AttributeError) as ex:
            op.set_entries(entries)
        assert ex.value.args[0] == (
            "training_parameters have not been set, "
            "call set_training_parameters() first"
        )

        op = self.Dummy(mu)

        # Try with wrong number of entry arrays.
        with pytest.raises(ValueError) as ex:
            op.set_entries(entries[1:])
        assert ex.value.args[0] == (
            f"{s} = len(training_parameters) != len(entries) = {s-1}"
        )

        # As a list of arrays.
        op.set_entries(entries, fromblock=False)
        assert op.shape == (r, r)
        assert op.state_dimension == r
        assert np.all(op.entries == entries)

        # Special case: r = 1, d > 1 (e.g., InputOperator with m > 1)
        # or r > 1, d = 1 (e.g., ConstantOperator)
        entries_1D = np.random.random((s, 5))
        op.set_entries(entries_1D, fromblock=False)
        assert op.shape == (5,)
        assert op.state_dimension == 5

        # As a horizontally concatenated array.
        entries_stacked = np.hstack(entries)
        op.set_entries(entries_stacked, fromblock=True)
        assert op.shape == (r, r)
        assert op.state_dimension == r
        assert np.all(op.entries == entries)

        with pytest.raises(ValueError) as ex:
            op.set_entries(entries, fromblock=True)
        assert ex.value.args[0] == (
            "entries must be a "
            "1- or 2-dimensional ndarray when fromblock=True"
        )

        # Test deletion.
        op._clear()
        assert op.entries is None
        assert op.interpolator is None
        assert op.shape is None
        assert op.state_dimension is None

    def test_set_interpolator(self, s=4, p=2, r=5):
        """Test _InterpOperator.set_interpolator() and the
        interpolator property.
        """
        op = self.Dummy()
        assert op.interpolator is None

        op.set_interpolator(_DummyInterpolator)
        assert op.interpolator is None

        mu = np.random.random((s, p))
        entries = np.random.random((s, r, r))
        op.set_training_parameters(mu)
        op.set_entries(entries)
        assert isinstance(op.interpolator, _DummyInterpolator)

        op.set_interpolator(_DummyInterpolator2)
        assert isinstance(op.interpolator, _DummyInterpolator2)

        assert isinstance(repr(op), str)

    def test_eq(self, s=4, p=3, r=2):
        """Test _InterpOperator.__eq__()."""
        op1 = self.Dummy()
        op2 = self.Dummy()
        assert op1 == op2

        mu = np.random.random((s, p))
        entries = np.random.random((s, r, r + 1))

        op1.set_training_parameters(mu)
        assert op1 != op2

        op1 = self.Dummy(mu, InterpolatorClass=_DummyInterpolator)
        assert op1 != 0

        op2 = self.Dummy(mu[1:, :-1], InterpolatorClass=_DummyInterpolator)
        assert op1 != op2

        op2 = self.Dummy(mu + 1, InterpolatorClass=_DummyInterpolator)
        assert op1 != op2

        op2 = self.Dummy(mu, InterpolatorClass=int)
        assert op1 != op2

        op2 = self.Dummy(mu, InterpolatorClass=_DummyInterpolator)
        assert op1 == op2

        op1.set_entries(entries)
        assert op1 != op2
        assert op2 != op1

        op2.set_entries([A.T for A in entries])
        assert op1 != op2

        op2.set_entries(entries)
        assert op1 == op2

    def test_evaluate(self, s=3, p=5, r=4):
        """Test _InterpOperator.evaluate()."""
        mu = np.random.random((s, p))
        op = self.Dummy(mu, InterpolatorClass=_DummyInterpolator)

        with pytest.raises(AttributeError):
            op.evaluate(mu[0])

        entries = np.random.random((s, r, r))
        op.set_entries(entries)
        op_evaluated = op.evaluate(mu[0])
        assert isinstance(op_evaluated, self.Dummy._OperatorClass)
        assert op_evaluated.entries.shape == (r, r)
        assert np.all(op_evaluated.entries == entries[0])

        # Scalar parameters.
        op = self.Dummy(
            mu[:, 0],
            entries=entries,
            InterpolatorClass=_DummyInterpolator,
            fromblock=False,
        )
        op_evaluated = op.evaluate(np.array([[mu[0, 0]]]))
        assert isinstance(op_evaluated, self.Dummy._OperatorClass)
        assert op_evaluated.entries.shape == (r, r)
        assert np.all(op_evaluated.entries == entries[0])

    def test_galerkin(self, s=5, p=2, n=10, r=4):
        """Test _InterpOperator.galerkin()."""
        Vr = np.empty((n, r))
        mu = np.random.random((s, p))
        entries = np.random.random((s, n, n))
        op = self.Dummy(mu, entries, _DummyInterpolator)

        op_reduced = op.galerkin(Vr)
        assert isinstance(op_reduced, self.Dummy)
        assert np.all(op_reduced.entries == entries)

    def test_datablock(self, s=4, p=2, r=2, k=3):
        """Test _InterpOperator.datablock()."""
        mu = np.random.random((s, p))
        states = np.random.random((s, r, k))
        op = self.Dummy(mu, InterpolatorClass=_DummyInterpolator)
        block = op.datablock(mu, states, states)
        assert block.shape == (s * _Dblock.shape[0], s * _Dblock.shape[1])
        assert np.all(block == la.block_diag(*[_Dblock for _ in range(s)]))

    def test_operator_dimension(self, s=3):
        """Test _InterpOperator.operator_dimension()."""
        assert self.Dummy.operator_dimension(s, None, None) == _d * s

    def test_copy(self, s=4, p=2, r=5):
        """Test _InterpOperator.copy()."""
        op1 = self.Dummy()
        op2 = op1.copy()
        assert op2 is not op1
        assert op2 == op1
        assert op2.training_parameters is None
        assert op2.entries is None
        assert op2.interpolator is None

        mu = np.random.random((s, p))
        entries = np.random.random((s, r, r))
        op1 = self.Dummy(mu, InterpolatorClass=_DummyInterpolator)

        op2 = op1.copy()
        assert op2.training_parameters.shape == op1.training_parameters.shape
        assert np.all(op2.training_parameters == op1.training_parameters)
        assert op2.entries is None
        assert op2.interpolator is None

        op1.set_entries(entries)
        op1.set_interpolator(_DummyInterpolator2)
        op2 = op1.copy()
        assert op2.training_parameters.shape == op1.training_parameters.shape
        assert np.all(op2.training_parameters == op1.training_parameters)
        assert op2.shape == op1.shape
        assert np.all(op2.entries == op1.entries)
        assert isinstance(op2.interpolator, _DummyInterpolator2)

    def test_save(self, s=5, p=2, r=3, target="_interpolatedopsavetest.h5"):
        """Lightly test _InterpOperator.save()."""
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        op = self.Dummy(
            np.random.random((s, p)),
            InterpolatorClass=_DummyInterpolator,
        )

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            op.save(target)
        assert wn[0].message.args[0] == (
            "cannot serialize InterpolatorClass "
            "'_DummyInterpolator', must pass in the class when calling load()"
        )
        assert os.path.isfile(target)

        op = self.Dummy(np.sort(np.random.random(s)))
        op.save(target, overwrite=True)

        os.remove(target)

    def test_load(self, s=15, p=3, r=3, target="_interpolatedoploadtest.h5"):
        """Test _InterpOperator.load()."""
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

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.Dummy(mu, InterpolatorClass=_DummyInterpolator).save(
                target, overwrite=True
            )
        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.Dummy.load(target)
        assert ex.value.args[0] == (
            "unknown InterpolatorClass "
            f"'_DummyInterpolator', "
            f"call Dummy.load({target}, _DummyInterpolator)"
        )
        self.Dummy.load(target, _DummyInterpolator)

        op1 = self.Dummy(mu, InterpolatorClass=_DummyInterpolator)
        with pytest.warns(opinf.errors.OpInfWarning):
            op1.save(target, overwrite=True)

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            op2 = self.Dummy.load(target, _DummyInterpolator2)
        assert wn[0].message.args[0] == (
            "InterpolatorClass=_DummyInterpolator2 does not match loadfile "
            "InterpolatorClass '_DummyInterpolator'"
        )
        op2.set_interpolator(_DummyInterpolator)
        assert op2 == op1

        op1 = self.Dummy(np.sort(mu[:, 0]), entries)
        op1.save(target, overwrite=True)
        op2 = self.Dummy.load(target)
        assert op2 == op1

        # Clean up.
        os.remove(target)


def test_publics():
    """Ensure all public InterpOperator classes can be instantiated
    without arguments.
    """
    for OpClassName in _submodule.__all__:
        if "Interpolated" in OpClassName:
            # Skip deprecations
            continue
        OpClass = getattr(_module, OpClassName)
        if not isinstance(OpClass, type) or not issubclass(
            OpClass, _submodule._InterpOperator
        ):
            continue
        op = OpClass()
        assert issubclass(
            op._OperatorClass,
            opinf.operators.OpInfOperator,
        )


def test_1Doperators(r=10, m=3, s=5):
    """Test InterpOperator classes with using all 1D interpolators
    from scipy.interpolate.
    """
    InterpolatorClass = interp.CubicSpline

    # Get nominal operators to play with.
    c, A, H, G, B, N = _get_operator_entries(r, m)

    # Get interpolation data for each type of operator.
    params = np.sort(np.linspace(0, 1, s) + np.random.standard_normal(s) / 40)
    mu_new = 0.314159

    for OpClass, Ohat in [
        (_module.InterpConstantOperator, c),
        (_module.InterpLinearOperator, A),
        (_module.InterpQuadraticOperator, H),
        (_module.InterpCubicOperator, G),
        (_module.InterpInputOperator, B),
        (_module.InterpStateInputOperator, N),
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
            op = OpClass(params, InterpolatorClass=InterpolatorClass)
            if oputils.has_inputs(op):
                assert op.input_dimension is None
            op.set_entries(entries)
            if oputils.has_inputs(op):
                assert op.input_dimension == m
            op_evaluated = op.evaluate(mu_new)
            assert isinstance(op_evaluated, OpClass._OperatorClass)
            assert op_evaluated.shape == op.shape
            op.set_entries(np.column_stack(entries), fromblock=True)
            op_evaluated2 = op.evaluate(mu_new)
            assert op_evaluated2.shape == op_evaluated.shape
            assert np.allclose(op_evaluated2.entries, op_evaluated.entries)

            for mu_i, Ohat_i in zip(params, entries):
                op_evaluated = op.evaluate(mu_i)
                assert isinstance(op_evaluated, OpClass._OperatorClass)
                assert op_evaluated.shape == op.shape
                assert np.allclose(op_evaluated.entries, Ohat_i)


def test_is_interpolated():
    """Test operators._interpolate.is_interpolated()."""
    op = TestInterpOperator.Dummy()
    assert _submodule.is_interpolated(op)
    assert not _submodule.is_interpolated(-1)


def test_nonparametric_to_interpolated():
    """Test operators._interpolate.nonparametric_to_interpolated()."""

    with pytest.raises(TypeError) as ex:
        _submodule.nonparametric_to_interpolated(float)
    assert ex.value.args[0] == ("_InterpOperator for class 'float' not found")

    OpClass = _submodule.nonparametric_to_interpolated(
        opinf.operators.QuadraticOperator
    )
    assert OpClass is opinf.operators.InterpQuadraticOperator


def test_deprecations():
    """Ensure deprecated classes still work."""
    for OpClass in [
        _module.InterpolatedConstantOperator,
        _module.InterpolatedLinearOperator,
        _module.InterpolatedQuadraticOperator,
        _module.InterpolatedCubicOperator,
        _module.InterpolatedInputOperator,
        _module.InterpolatedStateInputOperator,
    ]:
        with pytest.warns(DeprecationWarning) as wn:
            OpClass()
        assert len(wn) == 1


if __name__ == "__main__":
    pytest.main([__file__])
