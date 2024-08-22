# operators/test_affine.py
"""Tests for operators._affine."""

import os
import abc
import pytest
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

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

    def test_galerkin(self, r=9, m=4):
        """Test galerkin()."""
        ncoeffs = len(self.thetas)
        shape = self.entries_shape(r, m)
        arrays = [np.random.random(shape) for _ in range(ncoeffs)]
        op = self.OpClass(self.thetas, arrays)

        Vr = la.qr(np.random.random((r, r // 2)), mode="economic")[0]
        Wr = la.qr(np.random.random((r, r // 2)), mode="economic")[0]
        for testbasis in (None, Wr):
            newop = op.galerkin(Vr, testbasis)
            assert isinstance(newop, self.OpClass)
            assert newop.state_dimension == r // 2

    def test_opinf(self, s=10, k=15, r=11, m=3):
        """Test operator_dimension() and datablock()."""
        ncoeffs = len(self.thetas)
        shape = self.entries_shape(r, m)
        arrays = [np.random.random(shape) for _ in range(ncoeffs)]
        op = self.OpClass(self.thetas, arrays)

        parameters = [np.random.random(self.p) for _ in range(s)]
        states = np.random.random((s, r, k))
        inputs = np.random.random((s, m, k))

        block = op.datablock(parameters, states, inputs)
        dim = op.operator_dimension(s, r, m)
        assert block.shape[0] == dim
        assert block.shape[1] == s * k

    def test_copysaveload(self, r=10, m=2, target="_affinesavetest.h5"):
        """Test copy(), save(), and load()."""
        ncoeffs = len(self.thetas)
        shape = self.entries_shape(r, m)
        arrays = [np.random.random(shape) for _ in range(ncoeffs)]

        def sparsearray(A):
            B = A.copy()
            B[B < 0.9] = 0
            B = np.atleast_2d(B)
            if B.shape[0] == 1:
                B = B.T
            return sparse.csr_array(B)

        sparrays = [sparsearray(A) for A in arrays]

        def _checksame(original, copied):
            assert copied is not original
            assert isinstance(copied, self.OpClass)
            if original.entries is None:
                assert copied.entries is None
            elif isinstance(original.entries[0], np.ndarray):
                for i, Ai in enumerate(copied.entries):
                    assert isinstance(Ai, np.ndarray)
                    assert np.all(Ai == original.entries[i])
            elif sparse.issparse(original.entries[0]):
                for i, Ai in enumerate(copied.entries):
                    assert sparse.issparse(Ai)
                    assert (Ai - original.entries[i]).sum() == 0
            if (p := original.parameter_dimension) is not None:
                assert copied.parameter_dimension == p

        # Test copy() without entries set.
        op = self.OpClass(self.thetas)
        _checksame(op, op.copy())

        op.parameter_dimension = self.p
        _checksame(op, op.copy())

        # Test copy() with entries set.
        op.set_entries(arrays)
        _checksame(op, op.copy())

        op.set_entries(sparrays)
        _checksame(op, op.copy())

        # Test save() and load() together.

        def _checkload(original):
            if os.path.isfile(target):
                os.remove(target)
            original.save(target)
            copied = self.OpClass.load(target, original.coefficient_functions)
            return _checksame(original, copied)

        # Test save()/load() without entries set.
        op = self.OpClass(self.thetas)
        _checkload(op)

        op.parameter_dimension = self.p
        _checkload(op)

        # Test save()/load() with entries set.
        op.set_entries(arrays)
        _checkload(op)

        op.set_entries(sparrays)
        _checkload(op)

        if os.path.isfile(target):
            os.remove(target)


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
