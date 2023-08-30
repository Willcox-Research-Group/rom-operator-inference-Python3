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
    with pytest.raises(RuntimeError) as ex:
        dummy.required()
    assert ex.value.args[0] == \
        "operator entries have not been set, call set_entries() first"

    dummy.entries = 0
    assert dummy.required() == 1


class TestBaseNonparametricOperator:
    """Test operators._nonparametric._BaseNonparametricOperator."""
    class Dummy(_module._BaseNonparametricOperator):
        """Instantiable version of _BaseNonparametricOperator."""
        def _str(*args, **kwargs):
            pass

        def set_entries(*args, **kwargs):
            _module._BaseNonparametricOperator.set_entries(*args, **kwargs)

        @_module._requires_entries
        def __call__(*args, **kwargs):
            pass

        def evaluate(*args, **kwargs):
            pass

        @_module._requires_entries
        def jacobian(*args, **kwargs):
            pass

        def galerkin(*args, **kwargs):
            return _module._BaseNonparametricOperator.galerkin(*args, **kwargs)

        def datablock(*args, **kwargs):
            pass

        def column_dimension(*args, **kwargs):
            pass

    class Dummy2(Dummy):
        """A distinct instantiable version of _BaseNonparametricOperator."""
        pass

    def test_init(self):
        """Test _BaseNonparametricOperator.__init__()."""
        op = self.Dummy()
        assert op.entries is None

        # Check _requires_entries decorator working within these classes.
        with pytest.raises(RuntimeError) as ex:
            op.evaluate(5)
        assert ex.value.args[0] == \
            "operator entries have not been set, call set_entries() first"

        with pytest.raises(RuntimeError) as ex:
            op(5)
        assert ex.value.args[0] == \
            "operator entries have not been set, call set_entries() first"

        with pytest.raises(RuntimeError) as ex:
            op.jacobian(5)
        assert ex.value.args[0] == \
            "operator entries have not been set, call set_entries() first"

        A = np.random.random((10, 11))
        op = self.Dummy(A)
        assert op.entries is A

        # These should be callable now.
        op()
        op.evaluate()
        op.jacobian()

    def test_validate_entries(self):
        """Test _BaseNonparametricOperator._validate_entries()."""
        func = _module._BaseNonparametricOperator._validate_entries
        with pytest.raises(TypeError) as ex:
            func([1, 2, 3, 4])
        assert ex.value.args[0] == "operator entries must be NumPy array"

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
        """Test _BaseNonparametricOperator.entries and shape()."""
        op = self.Dummy()
        assert op.shape is None

        A = np.random.random((8, 6))
        op.set_entries(A)
        assert op.entries is A
        assert op.shape == A.shape

        A2 = np.random.random(A.shape)
        op.entries = A2
        assert op.entries is A2

        op._clear()
        assert op.entries is None

    def test_getitem(self):
        """Test _BaseNonparametricOperator.__getitem__()."""
        op = self.Dummy()
        assert op[0, 1, 3:] is None

        A = np.random.random((8, 6))
        op.set_entries(A)
        for s in [slice(2), (slice(1), slice(1, 3)), slice(1, 4, 2)]:
            assert np.all(op[s] == A[s])

    def test_eq(self):
        """Test _BaseNonparametricOperator.__eq__()."""
        A = np.arange(12).reshape((4, 3))
        opA = self.Dummy(A)
        opA2 = self.Dummy2(A)
        assert opA != opA2

        opAT = self.Dummy(A.T)
        assert opA != opAT

        opB = self.Dummy(A + 1)
        assert opA != opB

        opC = self.Dummy(A + 0)
        assert opA == opC

    # Dimensinoality reduction ------------------------------------------------
    def test_galerkin(self, n=10, r=3):
        """Test _BaseNonparametricOperator.galerkin()."""
        Vr = np.random.random((n, r))
        A_ = np.random.random((r, r))
        op = self.Dummy(A_)
        op_ = op.galerkin(Vr, None, None)
        assert op_ is op

        A = np.random.random((n, n))
        op = self.Dummy(A)
        B = np.random.random(r)
        op_ = op.galerkin(Vr, Vr, lambda x, y, z: B)
        assert isinstance(op_, op.__class__)
        assert op_.shape == B.shape
        assert np.all(op_.entries == B)

        A = np.random.random((n - 1, r + 1))
        op = self.Dummy(A)
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            op.galerkin(Vr, None, None)
        assert ex.value.args[0] == "basis and operator not aligned"

    # Model persistence -------------------------------------------------------
    def test_save(self, target="_baseoperatorsavetest.h5"):
        """Test _BaseNonparametricOperator.save()."""
        if os.path.isfile(target):              # pragma: no cover
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
        """Test _BaseNonparametricOperator.load()."""
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        op1 = self.Dummy()
        op1.save(target)
        op2 = self.Dummy.load(target)
        assert op1 == op2

        op1.set_entries(np.random.random((6, 2)))
        op1.save(target, overwrite=True)
        op2 = self.Dummy.load(target)
        assert op1 == op2

        with h5py.File(target, 'w'):
            pass

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.Dummy.load(target)
        assert ex.value.args[0] == "invalid save format (meta/ not found)"

        os.remove(target)
