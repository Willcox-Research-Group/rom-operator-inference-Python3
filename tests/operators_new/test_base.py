# operators/test_base.py
"""Tests for operators._base."""

import os
import h5py
import pytest
import numpy as np

import opinf


_module = opinf.operators_new._base


def test_requires_entries():
    """Test _requires_entries(), a decorator."""

    class Dummy:
        def __init__(self):
            self.entries = None

        @_module._requires_entries
        def required(self):
            return self.entries + 1

    dummy = Dummy()
    with pytest.raises(AttributeError) as ex:
        dummy.required()
    assert (
        ex.value.args[0] == "operator entries have not been set, "
        "call set_entries() first"
    )

    dummy.entries = 0
    assert dummy.required() == 1


# Nonparametric operators =====================================================
class TestNonparametricOperator:
    """Test operators_new._base._NonparametricOperator."""

    class Dummy(_module._NonparametricOperator):
        """Instantiable version of _NonparametricOperator."""

        def set_entries(*args, **kwargs):
            _module._NonparametricOperator.set_entries(*args, **kwargs)

        def input_dimension(*args, **kwargs):
            pass

        def _str(*args, **kwargs):
            pass

        def apply(*args, **kwargs):
            return -1

        def galerkin(*args, **kwargs):
            return _module._NonparametricOperator.galerkin(*args, **kwargs)

        def datablock(*args, **kwargs):
            pass

        def operator_dimension(*args, **kwargs):
            pass

    class Dummy2(Dummy):
        """A distinct instantiable version of _NonparametricOperator."""

        pass

    # Initialization ----------------------------------------------------------
    def test_init(self):
        """Test _NonparametricOperator.__init__()."""
        op = self.Dummy()
        assert op.entries is None

        with pytest.raises(AttributeError) as ex:
            op.jacobian(5)
        assert (
            ex.value.args[0] == "operator entries have not been set, "
            "call set_entries() first"
        )

        A = np.random.random((10, 11))
        op = self.Dummy(A)
        assert op.entries is A

        # This should be callable now.
        assert op.jacobian(None, None) == 0

    def test_validate_entries(self):
        """Test _NonparametricOperator._validate_entries()."""
        func = _module._NonparametricOperator._validate_entries
        with pytest.raises(TypeError) as ex:
            func([1, 2, 3, 4])
        assert (
            ex.value.args[0] == "operator entries must be "
            "NumPy or scipy.sparse array"
        )

        A = np.arange(12, dtype=float).reshape((4, 3)).T
        A[0, 0] = np.nan
        with pytest.raises(ValueError) as ex:
            func(A)
        assert ex.value.args[0] == "operator entries must not be NaN"

        A[0, 0] = np.inf
        with pytest.raises(ValueError) as ex:
            func(A)
        assert ex.value.args[0] == "operator entries must not be Inf"

        # Valid argument, no exceptions raised.
        A[0, 0] = 0
        func(A)

    # Properties --------------------------------------------------------------
    def test_entries(self):
        """Test _NonparametricOperator.entries and shape()."""
        op = self.Dummy()
        assert op.shape is None

        A = np.random.random((8, 6))
        op.set_entries(A)
        assert op.entries is A
        assert op.shape == A.shape
        del op.entries
        assert op.entries is None

        A2 = np.random.random(A.shape)
        op.entries = A2
        assert op.entries is A2

        op._clear()
        assert op.entries is None

    # Magic methods -----------------------------------------------------------
    def test_getitem(self):
        """Test _NonparametricOperator.__getitem__()."""
        op = self.Dummy()
        assert op[0, 1, 3:] is None

        A = np.random.random((8, 6))
        op.set_entries(A)
        for s in [slice(2), (slice(1), slice(1, 3)), slice(1, 4, 2)]:
            assert np.all(op[s] == A[s])

    def test_eq(self):
        """Test _NonparametricOperator.__eq__()."""
        op1 = self.Dummy()
        op2 = self.Dummy2()
        assert op1 != op2

        op2 = self.Dummy()
        assert op1 == op2

        A = np.arange(12).reshape((4, 3))
        op1.entries = A
        assert op1 != op2
        assert op2 != op1

        op2.entries = A.T
        assert op1 != op2

        op2.entries = A + 1
        assert op1 != op2

        op2.entries = A
        assert op1 == op2

    # Dimensionality reduction ------------------------------------------------
    def test_galerkin(self, n=10, r=3):
        """Test _NonparametricOperator.galerkin()."""
        Vr = np.random.random((n, r))
        A = np.random.random((n, n))
        op = self.Dummy(A)
        B = np.random.random(r)
        op_ = op.galerkin(Vr, Vr, lambda x, y, z: B)
        assert isinstance(op_, op.__class__)
        assert op_.shape == B.shape
        assert np.all(op_.entries == B)
        assert op is not op_

        A = np.random.random((n - 1, r + 1))
        op = self.Dummy(A)
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            op.galerkin(Vr, None, None)
        assert ex.value.args[0] == "basis and operator not aligned"

    # Model persistence -------------------------------------------------------
    def test_copy(self):
        """Test _NonparametricOperator.copy()."""
        op1 = self.Dummy()
        op1.set_entries(np.random.random((4, 4)))
        op2 = op1.copy()
        assert op2 is not op1
        assert op2.entries is not op1.entries
        assert op2 == op1

    def test_save(self, target="_baseoperatorsavetest.h5"):
        """Test _NonparametricOperator.save()."""
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        op = self.Dummy()
        op.save(target)

        with h5py.File(target, "r") as hf:
            assert "meta" in hf
            assert "class" in hf["meta"].attrs
            assert hf["meta"].attrs["class"] == "Dummy"

        A = np.random.random((4, 3))
        op.set_entries(A)
        op.save(target, overwrite=True)

        with h5py.File(target, "r") as hf:
            assert "meta" in hf
            assert "class" in hf["meta"].attrs
            assert hf["meta"].attrs["class"] == "Dummy"
            assert "entries" in hf
            entries = hf["entries"][:]
            assert entries.shape == A.shape
            assert np.all(entries == A)

        os.remove(target)

    def test_load(self, target="_baseoperatorloadtest.h5"):
        """Test _NonparametricOperator.load()."""
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        op1 = self.Dummy()
        op1.save(target)
        op2 = self.Dummy.load(target)
        assert op1 == op2

        op1.set_entries(np.random.random((6, 2)))
        op1.save(target, overwrite=True)
        op2 = self.Dummy.load(target)
        assert op1 == op2

        class Dummy2(self.Dummy):
            pass

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            Dummy2.load(target)
        assert (
            ex.value.args[0] == f"file '{target}' contains 'Dummy' object, "
            "use 'Dummy.load()"
        )

        os.remove(target)


# Parametric operators ========================================================
class TestParametricOperator:
    """Test operators_new._base._ParametricOperator."""

    class Dummy(_module._ParametricOperator):
        """Instantiable versino of _ParametricOperator."""

        _OperatorClass = TestNonparametricOperator.Dummy

        def __init__(self):
            _module._ParametricOperator.__init__(self)

        def _clear(self):
            pass

        def state_dimension(self):
            pass

        def input_dimension(self):
            pass

        def shape(self):
            pass

        def evaluate(self, parameter):
            op = self._OperatorClass()
            op.set_entries(np.random.random((2, 2)))
            return op

        def galerkin(self, *args, **kwargs):
            pass

        def datablock(self, *args, **kwargs):
            pass

        def operator_dimension(self, *args, **kwargs):
            pass

        def copy(self, *args, **kwargs):
            pass

        def save(self, *args, **kwargs):
            pass

        def load(self, *args, **kwargs):
            pass

    def test_init(self):
        """Test _ParametricOperator.__init__()"""
        temp = self.Dummy._OperatorClass
        self.Dummy._OperatorClass = int

        with pytest.raises(RuntimeError) as ex:
            self.Dummy()
        assert ex.value.args[0] == "invalid OperatorClass 'int'"

        self.Dummy._OperatorClass = temp
        op = self.Dummy()
        assert op.parameter_dimension is None

    def test_set_parameter_dimension_from_data(self):
        """Test _ParametricOperator._set_parameter_dimension_from_data()."""
        op = self.Dummy()

        # One-dimensional parameters.
        op._set_parameter_dimension_from_data(np.arange(10))
        assert op.parameter_dimension == 1
        op._set_parameter_dimension_from_data(np.arange(5).reshape((-1, 1)))
        assert op.parameter_dimension == 1

        # n-dimensional parameters.
        n = np.random.randint(2, 20)
        op._set_parameter_dimension_from_data(np.random.random((5, n)))
        assert op.parameter_dimension == n

        with pytest.raises(ValueError) as ex:
            op._set_parameter_dimension_from_data(np.random.random((2, 2, 2)))
        assert (
            ex.value.args[0] == "parameter values must be scalars or 1D arrays"
        )

    def test_check_shape_consistency(self):
        """Test _ParametricOperator._check_shape_consistency()."""
        arrays = [np.random.random((2, 3)), np.random.random((3, 2))]
        with pytest.raises(ValueError) as ex:
            self.Dummy._check_shape_consistency(arrays, "array")
        assert ex.value.args[0] == "array shapes inconsistent"

        arrays[1] = arrays[1].T
        self.Dummy._check_shape_consistency(arrays, "array")

    def test_check_parametervalue_dimension(self, p=3):
        """Test _ParametricOperator._check_parametervalue_dimension()."""
        op = self.Dummy()

        with pytest.raises(RuntimeError) as ex:
            op._check_parametervalue_dimension(10)
        assert ex.value.args[0] == "parameter_dimension not set"

        op._set_parameter_dimension_from_data(np.empty((5, p)))

        val = np.empty(p - 1)
        with pytest.raises(ValueError) as ex:
            op._check_parametervalue_dimension(val)
        assert ex.value.args[0] == f"expected parameter of shape ({p:d},)"

        op._check_parametervalue_dimension(np.empty(p))

    def test_apply(self):
        """Test _ParametricOperator.apply()."""
        assert self.Dummy().apply(None, None, None) == -1

    def test_jacobian(self):
        """Test _ParametricOperator.jacobian()."""
        assert self.Dummy().jacobian(None, None, None) == 0


# Utilities ===================================================================
def test_is_nonparametric():
    """Test operators._base.is_nonparametric()."""

    op = TestNonparametricOperator.Dummy()
    assert _module.is_nonparametric(op)
    assert not _module.is_nonparametric(10)


def test_has_inputs():
    """Test operators._base.has_inputs()."""

    op = _module._InputMixin()
    assert _module.has_inputs(op)
    assert not _module.has_inputs(5)


def test_is_parametric():
    """Test operators._base.is_parametric()."""
    op = TestParametricOperator.Dummy()
    assert _module.is_parametric(op)
    assert not _module.is_nonparametric(-1)
