# operators/test_base.py
"""Tests for operators._base."""

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

        def set_entries(self, entries):
            super().set_entries(entries)

        def datablock(*args, **kwargs):
            pass

        @_module._requires_entries
        def __call__(*args, **kwargs):
            pass

        def evaluate(*args, **kwargs):
            pass

        @_module._requires_entries
        def jacobian(*args, **kwargs):
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
        A = np.random.random((8, 6))
        op = self.Dummy()
        op.set_entries(A)
        assert op.entries is A
        assert op.shape == A.shape

        A2 = np.random.random(A.shape)
        op.entries = A2
        assert op.entries is A2

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
