# operators/test_base.py
"""Tests for operators._base."""

import abc
import pytest
import numpy as np
import scipy.sparse as sparse

import opinf


_module = opinf.operators._base


# Nonparametric operators =====================================================
class _TestOperatorTemplate(abc.ABC):
    """Tests for classes that inherit from operators._base.OperatorTemplate."""

    # Setup -------------------------------------------------------------------
    Operator = NotImplemented
    has_inputs = NotImplemented

    @abc.abstractmethod
    def get_operator(self, r: int, m: int = 0):
        """Return a valid operator to test.

        Parameters
        ----------
        r : int > 0
            State dimension.
        m : int or None
            Input dimension. Ignored if the operator does not act on inputs.

        Returns
        -------
        op : Operator
            Instantiated operator.
        """
        raise NotImplementedError

    # Properties --------------------------------------------------------------
    def test_dimensions(self, r=10, m=2):
        """Test state_dimension and input_dimension."""
        if self.has_inputs:
            op = self.get_operator(r, m)
            assert hasattr(op, "input_dimension")
            assert isinstance(op.input_dimension, int)
            assert op.input_dimension == m
        else:
            op = self.get_operator(r)
        assert isinstance(op.state_dimension, int)
        assert op.state_dimension == r

    def test_str(self, r=11, m=3):
        """Lightly test __str__() and _str()."""
        op = self.get_operator(r, m)
        repr(op)
        op._str("q", "u" if self.has_inputs else None)

    # Methods -----------------------------------------------------------------
    def test_apply_jacobian_galerkin_copy_save_load(self, r=9, m=3):
        """Use verify() to test apply(), jacobian(), and galerkin(), copy(),
        save(), and load().
        """
        self.get_operator(r, m).verify(plot=False)


class _TestOpInfOperator(_TestOperatorTemplate):
    """Tests for classes that inherit from operators._base.OpInfOperator."""

    @abc.abstractmethod
    def get_operator(self, r=None, m=None):
        """Return a valid operator to test.

        Parameters
        ----------
        r : int > 0 or None
            State dimension.
            If ``None`` (default), operator entries should not be populated.
            If a positive integer, operator entries should be populated.
        m : int > 0 or None
            Input dimension. Only required if ``r`` is a postive integer
            and the operator acts on inputs.

        Returns
        -------
        op : Operator
            Instantiated operator.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_entries(self, r: int, m=None):
        """Return a valid entries array with the appropriate dimensions.

        Parameters
        ----------
        r : int > 0
            State dimension.
        m : int or None
            Input dimension. Only required if the operator acts on inputs.

        Returns
        -------
        entries : (r, d) ndarray
            Entries array.
        """
        raise NotImplementedError

    def test_entries(self, r=11, m=3):
        """Test the entries property (including __init__() and setter)."""
        op = self.get_operator()
        assert op.entries is None
        assert op.state_dimension is None
        assert op.shape is None
        if self.has_inputs:
            assert op.input_dimension is None

        with pytest.raises(AttributeError) as ex:
            op.jacobian(5)
        assert ex.value.args[0] == "required attribute 'entries' not set"

        op = self.get_operator(r, m)
        assert op.entries is not None
        assert op.state_dimension == r
        if self.has_inputs:
            assert op.input_dimension == m

        # Test the setter.
        with pytest.raises(TypeError) as ex:
            op.entries = (1, 2, 3, 4)
        assert ex.value.args[0] == (
            "operator entries must be NumPy or scipy.sparse array"
        )

        A = self.get_entries(r - 1, m)
        sA = sparse.dok_array(A)
        op.set_entries(sA)
        assert op.state_dimension == r - 1

        first = (0, 0) if A.ndim == 2 else 0
        A[first] = np.nan
        with pytest.raises(ValueError) as ex:
            op.set_entries(A)
        assert ex.value.args[0] == "operator entries must not be NaN"

        A[first] = np.inf
        with pytest.raises(ValueError) as ex:
            op.entries = A
        assert ex.value.args[0] == "operator entries must not be Inf"

        # Valid argument, no exceptions raised.
        A[first] = 0
        op.entries = A

        del op.entries
        assert op.entries is None
        assert op.state_dimension is None

    def test_magics(self, r=10, m=2):
        """Test __getitem__(), __eq__(), and __add__()."""

        # Test __getitem__().
        op = self.get_operator()
        assert op[0, 1, 3:] is None

        A = self.get_entries(r, m)
        op.set_entries(A)
        for s in [slice(2), slice(1, 4, 2)]:
            assert np.all(np.ravel(op[s]) == np.ravel(A[s]))
        if op.entries.ndim > 1:
            s = (slice(1), slice(1, 3))
            assert np.all(op[s] == A[s])

        # Test __eq__().
        op1 = self.get_operator()
        assert op1 != 100

        op2 = self.get_operator()
        assert op1 == op2

        op1.set_entries(A)
        assert op1 != op2
        assert op2 != op1

        op2.entries = self.get_entries(r + 1, m)
        assert op1 != op2

        op2.entries = A + 1
        assert op1 != op2

        op2.entries = A
        assert op1 == op2

        # Test __add__().
        with pytest.raises(TypeError) as ex:
            op1 + 10
        assert ex.value.args[0].startswith(
            "can't add object of type 'int' to object of type"
        )

        op = op1 + op2
        assert isinstance(op, type(op1))
        assert np.all(op.entries == (op1.entries + op2.entries))


# Parametric operators ========================================================
class _TestParametricOperatorTemplate(abc.ABC):
    """Test operators._base.ParametricOperatorTemplate."""

    Operator = NotImplemented
    has_inputs = False

    @abc.abstractmethod
    def get_operator(self, p: int, r: int, m: int = 0):
        """Return a valid operator to test.

        Parameters
        ----------
        p : int > 0
            Parameter dimension.
        r : int > 0
            State dimension.
        m : int > 0
            Input dimension. Only required if the operator acts on inputs.

        Returns
        -------
        op : Operator
            Instantiated operator.
        """
        raise NotImplementedError

    def test_dimensions(self, p=3, r=10, m=2):
        """Test state_dimension and input_dimension."""
        if self.has_inputs:
            op = self.get_operator(p, r, m)
            assert hasattr(op, "input_dimension")
            assert isinstance(op.input_dimension, int)
            assert op.input_dimension == m
        else:
            op = self.get_operator(p, r)
        assert isinstance(op.state_dimension, int)
        assert op.state_dimension == r
        assert isinstance(op.parameter_dimension, int)
        assert op.parameter_dimension == p

    def test_str(self, p=2, r=11, m=3):
        """Lightly test __str__() and __repr__()."""
        op = (
            self.get_operator(p, r, m)
            if self.has_inputs
            else self.get_operator(p, r)
        )
        repr(op)

    def test_evaluate(self, p=3, r=11, m=3):
        """Use verify() to test evaluate()."""
        self.get_operator(p, r, m).verify()


class _TestParametricOpInfOperator(_TestParametricOperatorTemplate):
    """Test operators._base.ParametricOpInfOperator."""

    @abc.abstractmethod
    def get_operator(self, p: int, r=None, m=None):
        """Return a valid operator to test.

        Parameters
        ----------
        p : int > 0
            Parameter dimension.
        r : int > 0 or None
            State dimension.
            If ``None`` (default), operator entries should not be populated.
            If a positive integer, operator entries should be populated.
        m : int > 0 or None
            Input dimension. Only required if ``r`` is a postive integer
            and the operator acts on inputs.

        Returns
        -------
        op : Operator
            Instantiated operator.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_entries(self, p: int, r: int, m=None):
        """Return a valid entries array with the appropriate dimensions.

        Parameters
        ----------
        p : int > 0
            Parameter dimension.
        r : int > 0
            State dimension.
        m : int or None
            Input dimension. Only required if the operator acts on inputs.

        Returns
        -------
        entries : (r, d) ndarray
            Entries array.
        """
        raise NotImplementedError

    def test_parameter_dimension(self, p=4):
        """Test parameter_dimension and its setter."""
        op = self.get_operator(p)

        with pytest.raises(ValueError) as ex:
            op.parameter_dimension = -40
        assert ex.value.args[0] == (
            "parameter_dimension must be a positive integer"
        )

        op.parameter_dimension = 100

        with pytest.raises(AttributeError) as ex:
            op.parameter_dimension = 10
        assert ex.value.args[0] == (
            "can't set property 'parameter_dimension' twice"
        )

    def test_entries(self, p=2, r=8, m=3):
        """Test entries, shape, and set_entries()."""
        op = self.get_entries(p)
        assert op.entries is None
        assert op.shape is None

        op.set_entries(self.get_entries(p, r, m))
        assert op.entries is not None
        assert op.shape is not None


def test_utils():
    """Lightly test has_inputs(), is_nonparametric(), is_parametric(),
    is_opinf(), and is_uncalibrated().
    """
    op = opinf.operators.ConstantOperator()
    assert not _module.has_inputs(op)
    assert _module.is_nonparametric(op)
    assert not _module.is_parametric(op)
    assert _module.is_opinf(op)
    assert _module.is_uncalibrated(op)

    op = opinf.operators.AffineInputOperator(
        3,
        entries=np.random.random((3, 4, 2)),
    )
    assert _module.has_inputs(op)
    assert not _module.is_nonparametric(op)
    assert _module.is_parametric(op)
    assert _module.is_opinf(op)
    assert not _module.is_uncalibrated(op)


if __name__ == "__main__":
    pytest.main([__file__])
