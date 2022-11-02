# core/operators/test_nonparametric.py
"""Tests for core.operators._nonparametric."""

import pytest
import numpy as np

import opinf


_module = opinf.core.operators._nonparametric


class TestConstantOperator:
    """Test core.operators._nonparametric.ConstantOperator."""
    def test_init(self):
        """Test ConstantOperator.__init__()"""
        # Too many dimensions.
        cbad = np.arange(12).reshape((4, 3))
        with pytest.raises(ValueError) as ex:
            opinf.core.operators.ConstantOperator(cbad)
        assert ex.value.args[0] == "constant operator must be one-dimensional"

        # Case 1: one-dimensional array.
        c = np.arange(12)
        op = opinf.core.operators.ConstantOperator(c)
        assert op.entries is c

        # Case 2: two-dimensional array that can be flattened.
        op = opinf.core.operators.ConstantOperator(c.reshape((-1, 1)))
        assert op.shape == (12,)
        assert np.all(op.entries == c)

        op = opinf.core.operators.ConstantOperator(c.reshape((1, -1)))
        assert op.shape == (12,)
        assert np.all(op.entries == c)

    def test_evaluate(self, n=10):
        """Test ConstantOperator.evaluate()"""
        c = np.random.random(n)
        op = opinf.core.operators.ConstantOperator(c)
        assert op.evaluate() is c
        assert op.evaluate(1) is c
        assert op.evaluate([1, 2]) is c
        assert op.evaluate([1], 2) is c

    def test_jacobian(self, n=10):
        """Test ConstantOperator.jacobian()."""
        op = opinf.core.operators.ConstantOperator(np.random.random(n))
        Z = np.zeros((n, n))
        assert np.all(op.jacobian() == Z)
        assert np.all(op.jacobian(1) == Z)
        assert np.all(op.jacobian([1, 2]) == Z)
        assert np.all(op.jacobian([1], 2) == Z)


class TestLinearOperator:
    """Test core.operators._nonparametric.LinearOperator."""
    def test_init(self):
        """Test LinearOperator.__init__()"""

        # Too many dimensions.
        Abad = np.arange(12).reshape((2, 2, 3))
        with pytest.raises(ValueError) as ex:
            opinf.core.operators.LinearOperator(Abad)
        assert ex.value.args[0] == "linear operator must be two-dimensional"

        # No violation, nonsquare.
        A = Abad.reshape((4, 3))
        op = opinf.core.operators.LinearOperator(A)
        assert op.entries is A

        # Correct square usage.
        A = A[:3, :3]
        op = opinf.core.operators.LinearOperator(A)
        assert op.entries is A

        # Special case: "one-dimensional" operator.
        B = np.arange(5)
        op = opinf.core.operators.LinearOperator(B)
        assert op.shape == (5, 1)
        assert np.all(op.entries[:, 0] == B)

        # Special case: "scalar" operator.
        A = np.array([10])
        op = opinf.core.operators.LinearOperator(A)
        assert op.shape == (1, 1)
        assert op.entries[0, 0] == A[0]

    def test_evaluate(self):
        """Test LinearOperator.evaluate()"""

        # Special case: A is 1x1 (e.g., ROM state dimension = 1)
        A = np.random.random((1, 1))
        op = opinf.core.operators.LinearOperator(A)
        q = np.random.random()
        assert np.allclose(op.evaluate(q), A[0, 0] * q)

        # Scalar inputs (e.g., ROM state dimension > 1 but input dimension = 1)
        B = np.random.random(10)
        op = opinf.core.operators.LinearOperator(B)
        q = np.random.random()
        assert np.allclose(op.evaluate(q), B * q)

        # 1D inputs (usual case)
        def _check1D(A):
            q = np.random.random(A.shape[-1])
            op = opinf.core.operators.LinearOperator(A)
            assert np.allclose(op.evaluate(q), A @ q)

        _check1D(np.random.random((4, 3)))
        _check1D(np.random.random((4, 4)))
        _check1D(np.random.random((4, 1)))

        # 2D inputs (for applying to data residual)
        def _check2D(A):
            X = np.random.random((A.shape[-1], 20))
            op = opinf.core.operators.LinearOperator(A)
            assert np.allclose(op.evaluate(X), A @ X)

        _check2D(np.random.random((10, 3)))
        _check2D(np.random.random((6, 6)))

    def test_jacobian(self, n=9):
        """Test LinearOperator.jacobian()."""
        A = np.random.random((n, n))
        op = opinf.core.operators.LinearOperator(A)
        jac = op.jacobian(np.random.random(n))
        assert jac.shape == A.shape
        assert np.all(jac == A)


class TestQuadraticOperator:
    """Test core.operators._nonparametric.QuadraticOperator."""
    def test_init(self):
        """Test QuadraticOperator.__init__()"""
        # Too many dimensions.
        Hbad = np.arange(12).reshape((2, 2, 3))
        with pytest.raises(ValueError) as ex:
            opinf.core.operators.QuadraticOperator(Hbad)
        assert ex.value.args[0] == "quadratic operator must be two-dimensional"

        # Two-dimensional but invalid shape.
        Hbad = Hbad.reshape((4, 3))
        with pytest.raises(ValueError) as ex:
            opinf.core.operators.QuadraticOperator(Hbad)
        assert ex.value.args[0] == "invalid dimensions for quadratic operator"

        # Special case: r = 1
        H = np.random.random((1, 1))
        op = opinf.core.operators.QuadraticOperator(H)
        assert op.shape == (1, 1)
        assert np.allclose(op.entries, H)

        # Full operator, compressed internally.
        r = 4
        H = np.random.random((r, r**2))
        op = opinf.core.operators.QuadraticOperator(H)
        assert op.shape == (r, r*(r + 1)//2)
        assert np.allclose(op.entries, opinf.utils.compress_quadratic(H))

        # Compressed operator.
        r = 4
        H = np.random.random((r, r*(r + 1)//2))
        op = opinf.core.operators.QuadraticOperator(H)
        assert op.entries is H

    def test_evaluate(self, r=4, ntrials=10):
        """Test QuadraticOperator.evaluate()"""
        # Full operator, compressed internally.
        H = np.random.random((r, r**2))
        op = opinf.core.operators.QuadraticOperator(H)
        for _ in range(ntrials):
            q = np.random.random(r)
            evaltrue = H @ np.kron(q, q)
            assert np.allclose(op.evaluate(q), evaltrue)

        # Compressed operator.
        H = np.random.random((r, r*(r + 1)//2))
        op = opinf.core.operators.QuadraticOperator(H)
        for _ in range(ntrials):
            q = np.random.random(r)
            evaltrue = H @ opinf.utils.kron2c(q)
            assert np.allclose(op.evaluate(q), evaltrue)

        # Special case: r = 1
        H = np.random.random((1, 1))
        op = opinf.core.operators.QuadraticOperator(H)
        for _ in range(ntrials):
            q = np.random.random()
            evaltrue = H[0, 0] * q**2
            assert np.allclose(op.evaluate(q), evaltrue)

    def test_jacobian(self, r=5, ntrials=10):
        """Test QuadraticOperator.jacobian()."""
        H = np.random.random((r, r**2))
        op = opinf.core.operators.QuadraticOperator(H)

        Id = np.eye(r)
        for _ in range(ntrials):
            q = np.random.random(r)
            jac_true = H @ (np.kron(q, Id) + np.kron(Id, q)).T
            jac = op.jacobian(q)
            assert jac.shape == (r, r)
            assert np.allclose(jac, jac_true)

        # Special case: r = 1
        H = np.random.random((1, 1))
        op = opinf.core.operators.QuadraticOperator(H)
        for _ in range(ntrials):
            q = np.random.random()
            jac_true = 2 * H * q
            jac = op.jacobian(q)
            assert jac.shape == (1, 1)
            assert np.allclose(jac, jac_true)


# class TestCrossQuadraticOperator:
#     """Test core.operators._nonparametric.CrossQuadraticOperator."""
#     def test_init(self):
        # """Test CrossQuadraticOperator.__init__()"""
#         raise NotImplementedError
#
#     def test_evaluate(self):
#         """Test CrossQuadraticOperator.evaluate()"""
#         raise NotImplementedError


class TestCubicOperator:
    """Test core.operators._nonparametric.CubicOperator."""
    def test_init(self):
        """Test CubicOperator.__init__()"""
        # Too many dimensions.
        Gbad = np.arange(24).reshape((2, 4, 3))
        with pytest.raises(ValueError) as ex:
            opinf.core.operators.CubicOperator(Gbad)
        assert ex.value.args[0] == "cubic operator must be two-dimensional"

        # Two-dimensional but invalid shape.
        Gbad = Gbad.reshape((3, 8))
        with pytest.raises(ValueError) as ex:
            opinf.core.operators.CubicOperator(Gbad)
        assert ex.value.args[0] == "invalid dimensions for cubic operator"

        # Special case: r = 1
        G = np.random.random((1, 1))
        op = opinf.core.operators.CubicOperator(G)
        assert op.shape == (1, 1)
        assert np.allclose(op.entries, G)

        # Full operator, compressed internally.
        r = 4
        G = np.random.random((r, r**3))
        op = opinf.core.operators.CubicOperator(G)
        assert op.shape == (r, r*(r + 1)*(r + 2)//6)
        assert np.allclose(op.entries, opinf.utils.compress_cubic(G))

        # Compressed operator.
        r = 5
        G = np.random.random((r, r*(r + 1)*(r + 2)//6))
        op = opinf.core.operators.CubicOperator(G)
        assert op.entries is G

    def test_evaluate(self, r=4, ntrials=10):
        """Test CubicOperator.evaluate()"""
        # Full operator, compressed internally.
        G = np.random.random((r, r**3))
        op = opinf.core.operators.CubicOperator(G)
        for _ in range(ntrials):
            q = np.random.random(r)
            evaltrue = G @ np.kron(np.kron(q, q), q)
            Gq = op.evaluate(q)
            assert np.allclose(Gq, evaltrue)

        # Compressed operator.
        r += 1
        G = np.random.random((r, r*(r + 1)*(r + 2)//6))
        op = opinf.core.operators.CubicOperator(G)
        for _ in range(ntrials):
            q = np.random.random(r)
            evaltrue = G @ opinf.utils.kron3c(q)
            Gq = op.evaluate(q)
            assert np.allclose(Gq, evaltrue)

        # Special case: r = 1
        G = np.random.random((1, 1))
        op = opinf.core.operators.CubicOperator(G)
        for _ in range(ntrials):
            q = np.random.random()
            evaltrue = G[0, 0] * q**3
            Gq = op.evaluate(q)
            assert np.allclose(Gq, evaltrue)

    def test_jacobian(self, r=5, ntrials=10):
        """Test CubicOperator.jacobian()."""
        G = np.random.random((r, r**3))
        op = opinf.core.operators.CubicOperator(G)

        Id = np.eye(r)
        for _ in range(ntrials):
            q = np.random.random(r)
            qId = np.kron(q, Id)
            Idq = np.kron(Id, q)
            qqId = np.kron(q, qId)
            qIdq = np.kron(qId, q)
            Idqq = np.kron(Idq, q)
            jac_true = G @ (Idqq + qIdq + qqId).T
            jac = op.jacobian(q)
            assert jac.shape == (r, r)
            assert np.allclose(jac, jac_true)

        # Special case: r = 1
        G = np.random.random((1, 1))
        op = opinf.core.operators.CubicOperator(G)
        for _ in range(ntrials):
            q = np.random.random()
            jac_true = 3 * G * q**2
            jac = op.jacobian(q)
            assert jac.shape == (1, 1)
            assert np.allclose(jac, jac_true)
