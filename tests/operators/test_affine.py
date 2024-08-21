# operators/test_affine.py
"""Tests for operators._affine."""

import abc
import pytest
import numpy as np

import opinf

_module = opinf.operators._affine


class _TestAffineOperator:
    """Test operators._affine._AffineOperator."""

    OpClass = NotImplemented

    thetas = [
        (lambda mu: mu[0]),
        (lambda mu: mu[1]),
        (lambda mu: mu[2]),
        (lambda mu: mu[1] * mu[2] ** 2),
    ]

    p = 3

    @abc.abstractmethod
    def entries_shape(self, r, m):
        raise NotImplementedError

    def test_init(self, p=6):
        """Test __init__() and properties."""

        bad_thetas = [1, 2, 3]
        with pytest.raises(TypeError) as ex:
            self.OpClass(bad_thetas)
        assert ex.value.args[0] == (
            "coefficient_functions must be collection of callables"
        )
        ncoeffs = len(self.thetas)

        op = self.OpClass(self.thetas)
        assert op.parameter_dimension is None
        assert op.entries is None
        assert len(op.coefficient_functions) == ncoeffs
        assert op.nterms == ncoeffs
        mu = np.random.random(ncoeffs)
        for i in range(ncoeffs):
            opimu = op.coefficient_functions[i](mu)
            truth = self.thetas[i](mu)
            assert opimu == truth

        # Shortcut: coefficient_functions as an integer.
        op = self.OpClass(p)
        assert op.parameter_dimension == p
        assert op.entries is None
        assert op.nterms == p
        mu = np.random.random(p)
        for i in range(p):
            assert op.coefficient_functions[i](mu) == mu[i]

    def test_entries(self, r=10, m=3):
        """Test set_entries() and entries property."""
        ncoeffs = len(self.thetas)
        shape = self.entries_shape(r, m)
        arrays = [np.random.random(shape) for _ in range(ncoeffs)]

        op = self.OpClass(self.thetas)
        with pytest.raises(ValueError) as ex:
            op.set_entries(np.random.random((2, 3, 2)), fromblock=True)
        assert ex.value.args[0] == (
            "entries must be a 1- or 2-dimensional ndarray "
            "when fromblock=True"
        )
        with pytest.raises(ValueError) as ex:
            op.set_entries(arrays[:-1])
        assert ex.value.args[0] == (
            f"{ncoeffs} = len(coefficient_functions) "
            f"!= len(entries) = {ncoeffs - 1}"
        )

        op = self.OpClass(self.thetas)
        assert op.entries is None
        op.set_entries(arrays)
        for i in range(ncoeffs):
            assert np.all(op.entries[i] == arrays[i])

        op = self.OpClass(self.thetas, arrays)
        for i in range(ncoeffs):
            assert np.all(op.entries[i] == arrays[i])

        op = self.OpClass(self.thetas, np.hstack(arrays), fromblock=True)
        for i in range(ncoeffs):
            assert np.all(op.entries[i] == arrays[i])

    def test_evaluate(self, r=9, m=4):
        """Test evaluate()."""
        ncoeffs = len(self.thetas)
        shape = self.entries_shape(r, m)
        arrays = [np.random.random(shape) for _ in range(ncoeffs)]
        op = self.OpClass(self.thetas, arrays)

        mu = np.random.random(self.p)
        op_mu = op.evaluate(mu)
        assert isinstance(op_mu, op.OperatorClass)
        assert op_mu.entries.shape == arrays[0].shape
        Amu = np.sum(
            [theta(mu) * A for theta, A in zip(self.thetas, arrays)],
            axis=0,
        )
        assert np.allclose(op_mu.entries, Amu)


# Test public classes =========================================================
class TestAffineConstantOperator(_TestAffineOperator):
    """Test AffineConstantOperator."""

    OpClass = _module.AffineConstantOperator

    @staticmethod
    def entries_shape(r, m):
        return (r,)


class TestAffineLinearOperator(_TestAffineOperator):
    """Test AffineLinearOperator."""

    OpClass = _module.AffineLinearOperator

    @staticmethod
    def entries_shape(r, m):
        return (r, r)


class TestAffineQuadraticOperator(_TestAffineOperator):
    """Test AffineQuadraticOperator."""

    OpClass = _module.AffineQuadraticOperator

    @staticmethod
    def entries_shape(r, m):
        return (r, int(r * (r + 1) / 2))


class TestAffineCubicOperator(_TestAffineOperator):
    """Test AffineCubicOperator."""

    OpClass = _module.AffineCubicOperator

    @staticmethod
    def entries_shape(r, m):
        return (r, int(r * (r + 1) * (r + 2) / 6))


class TestAffineInputOperator(_TestAffineOperator):
    """Test AffineInputOperator."""

    OpClass = _module.AffineInputOperator

    @staticmethod
    def entries_shape(r, m):
        return (r, m)


class TestAffineStateInputOperator(_TestAffineOperator):
    OpClass = _module.AffineStateInputOperator

    @staticmethod
    def entries_shape(r, m):
        return (r, r * m)
