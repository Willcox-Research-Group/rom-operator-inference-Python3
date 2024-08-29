# operators/test_affine.py
"""Tests for operators._affine."""

import os
import abc
import pytest
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

import opinf


_module = opinf.operators
_submodule = _module._affine


class _TestAffineOperator:
    """Test operators._affine._AffineOperator."""

    OpClass = NotImplemented

    thetas1 = [
        (lambda mu: mu[0]),
        (lambda mu: mu[1]),
        (lambda mu: mu[2]),
        (lambda mu: mu[1] * mu[2] ** 2),
    ]

    @staticmethod
    def thetas2(mu):
        return np.array([mu[0], mu[1], mu[2], mu[1] * mu[2] ** 2])

    p = 3

    @abc.abstractmethod
    def entries_shape(self, r, m):
        raise NotImplementedError

    def test_init(self, p=6):
        """Test __init__() and properties."""

        # Bad input for coeffs.
        bad_thetas = 3.14159265358979
        with pytest.raises(TypeError) as ex:
            self.OpClass(bad_thetas)
        assert ex.value.args[0] == (
            "argument 'coeffs' must be callable, iterable, or a positive int"
        )
        bad_thetas = [1, 2, 3]
        with pytest.raises(TypeError) as ex:
            self.OpClass(bad_thetas)
        assert ex.value.args[0] == (
            "if 'coeffs' is iterable each entry must be callable"
        )
        ncoeffs = len(self.thetas1)

        # Bad input for nterms.
        with pytest.raises(TypeError) as ex:
            self.OpClass(None, -10)
        assert ex.value.args[0] == (
            "when provided, argument 'nterms' must be a positive integer"
        )

        # coeffs as an iterable of callables.
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            op = self.OpClass(self.thetas1, 100)
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            f"{ncoeffs} = len(coeffs) != nterms = 100, ignoring "
            f"argument 'nterms' and setting nterms = {ncoeffs}"
        )
        assert op.nterms == ncoeffs
        assert op.parameter_dimension is None
        assert op.entries is None
        mu = np.random.random(ncoeffs)
        opmu = op.coeffs(mu)
        assert all(opmu[i] == thta(mu) for i, thta in enumerate(self.thetas1))

        # coeffs as a single callable.
        with pytest.raises(ValueError) as ex:
            self.OpClass(self.thetas2)
        assert ex.value.args[0] == (
            "argument 'nterms' required when argument 'coeffs' is callable"
        )
        op = self.OpClass(self.thetas2, nterms=ncoeffs)
        assert op.parameter_dimension is None
        assert op.entries is None
        mu = np.random.random(ncoeffs)
        opmu = op.coeffs(mu)
        assert all(opmu[i] == thta(mu) for i, thta in enumerate(self.thetas1))

        # coeffs as an integer.
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            op = self.OpClass(p, p + 1)
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            f"{p} = coeffs != nterms = {p + 1}, ignoring "
            f"argument 'nterms' and setting nterms = {p}"
        )
        assert op.nterms == p
        assert op.parameter_dimension == p
        assert op.entries is None
        mu = np.random.random(p)
        assert np.array_equal(op.coeffs(mu), mu)

        assert repr(op).count(f"expansion terms:     {p}") == 1

    def test_entries(self, r=10, m=3):
        """Test set_entries() and entries property."""
        ncoeffs = len(self.thetas1)
        shape = self.entries_shape(r, m)
        arrays = [np.random.random(shape) for _ in range(ncoeffs)]

        op = self.OpClass(self.thetas1)
        with pytest.raises(ValueError) as ex:
            op.set_entries(np.random.random((2, 3, 2)), fromblock=True)
        assert ex.value.args[0] == (
            "entries must be a 1- or 2-dimensional ndarray "
            "when fromblock=True"
        )
        with pytest.raises(ValueError) as ex:
            op.set_entries(arrays[:-1])
        assert ex.value.args[0] == (
            f"{ncoeffs} = number of affine expansion terms "
            f"!= len(entries) = {ncoeffs - 1}"
        )

        op = self.OpClass(self.thetas2, ncoeffs)
        assert op.entries is None
        op.set_entries(arrays)
        for i in range(ncoeffs):
            assert np.all(op.entries[i] == arrays[i])

        op = self.OpClass(self.thetas1, entries=arrays)
        for i in range(ncoeffs):
            assert np.all(op.entries[i] == arrays[i])

        op = self.OpClass(
            self.thetas2,
            ncoeffs,
            entries=np.hstack(arrays),
            fromblock=True,
        )
        for i in range(ncoeffs):
            assert np.all(op.entries[i] == arrays[i])

    def test_evaluate(self, r=9, m=4):
        """Test evaluate()."""
        ncoeffs = len(self.thetas1)
        shape = self.entries_shape(r, m)
        arrays = [np.random.random(shape) for _ in range(ncoeffs)]
        op = self.OpClass(self.thetas1, entries=arrays)

        mu = np.random.random(self.p)
        op_mu = op.evaluate(mu)
        assert isinstance(op_mu, op._OperatorClass)
        assert op_mu.entries.shape == arrays[0].shape
        Amu = np.sum(
            [theta(mu) * A for theta, A in zip(self.thetas1, arrays)],
            axis=0,
        )
        assert np.allclose(op_mu.entries, Amu)

    def test_galerkin(self, r=9, m=4):
        """Test galerkin()."""
        ncoeffs = len(self.thetas1)
        shape = self.entries_shape(r, m)
        arrays = [np.random.random(shape) for _ in range(ncoeffs)]
        op = self.OpClass(self.thetas1, entries=arrays)

        Vr = la.qr(np.random.random((r, r // 2)), mode="economic")[0]
        Wr = la.qr(np.random.random((r, r // 2)), mode="economic")[0]
        for testbasis in (None, Wr):
            newop = op.galerkin(Vr, testbasis)
            assert isinstance(newop, self.OpClass)
            assert newop.state_dimension == r // 2

    def test_opinf(self, s=10, k=15, r=11, m=3):
        """Test operator_dimension() and datablock()."""
        ncoeffs = len(self.thetas1)
        shape = self.entries_shape(r, m)
        arrays = [np.random.random(shape) for _ in range(ncoeffs)]
        op = self.OpClass(self.thetas1, entries=arrays)

        parameters = [np.random.random(self.p) for _ in range(s)]
        states = np.random.random((s, r, k))
        inputs = np.random.random((s, m, k))

        block = op.datablock(parameters, states, inputs)
        dim = op.operator_dimension(s, r, m)
        assert block.shape[0] == dim
        assert block.shape[1] == s * k

    def test_copysaveload(self, r=10, m=2, target="_affinesavetest.h5"):
        """Test copy(), save(), and load()."""
        ncoeffs = len(self.thetas1)
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
        op = self.OpClass(self.thetas1)
        _checksame(op, op.copy())

        op.parameter_dimension = self.p
        _checksame(op, op.copy())

        # Test copy() with entries set.
        op.set_entries(arrays)
        _checksame(op, op.copy())

        op.set_entries(sparrays)
        _checksame(op, op.copy())

        # Test save() and load() together.

        class Dummy(self.OpClass):
            pass

        op = Dummy(self.thetas2, nterms=ncoeffs)
        op.save(target, overwrite=True)
        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.OpClass.load(target, self.thetas2)
        assert ex.value.args[0] == (
            f"file '{target}' contains 'Dummy' object, use 'Dummy.load()'"
        )

        def _checkload(original):
            if os.path.isfile(target):
                os.remove(target)
            original.save(target)
            copied = self.OpClass.load(target, original.coeffs)
            return _checksame(original, copied)

        # Test save()/load() without entries set.
        op = self.OpClass(self.thetas1)
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

    def test_input_dimension(self, r=8, m=3, p=3):
        """Test input_dimension."""
        Bs = [np.random.random((r, m)) for _ in range(p)]
        op = self.OpClass(p)
        assert op.input_dimension is None
        op.set_entries(Bs)
        assert op.input_dimension == m


class TestAffineStateInputOperator(_TestAffineOperator):
    OpClass = _module.AffineStateInputOperator

    @staticmethod
    def entries_shape(r, m):
        return (r, r * m)

    def test_input_dimension(self, r=7, m=4, p=5):
        """Test input_dimension."""
        Ns = [np.random.random((r, r * m)) for _ in range(p)]
        op = self.OpClass(p)
        assert op.input_dimension is None
        op.set_entries(Ns)
        assert op.input_dimension == m


def test_publics():
    """Ensure all public AffineOperator classes can be instantiated."""
    for OpClassName in _submodule.__all__:
        OpClass = getattr(_module, OpClassName)
        if not isinstance(OpClass, type) or not issubclass(
            OpClass, _submodule._AffineOperator
        ):
            continue
        op = OpClass(_TestAffineOperator.thetas1)
        assert issubclass(
            op._OperatorClass,
            opinf.operators.OpInfOperator,
        )


def test_is_affine():
    """Test operators._affine.is_interpolated()."""

    class Dummy(_submodule._AffineOperator):
        pass

    op = Dummy(_TestAffineOperator.thetas1)
    assert _submodule.is_affine(op)
    assert not _submodule.is_affine(-2)


def test_nonparametric_to_affine():
    """Test operators._affine.nonparametric_to_affine()."""

    with pytest.raises(TypeError) as ex:
        _submodule.nonparametric_to_affine(list)
    assert ex.value.args[0] == "_AffineOperator for class 'list' not found"

    OpClass = _submodule.nonparametric_to_affine(opinf.operators.CubicOperator)
    assert OpClass is opinf.operators.AffineCubicOperator
