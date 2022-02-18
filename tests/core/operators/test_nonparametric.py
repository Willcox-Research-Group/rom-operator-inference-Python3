# core/operators/test_nonparametric.py
"""Tests for rom_operator_inference.core.operators._nonparametric."""

import pytest
import numpy as np

import rom_operator_inference as opinf


_module = opinf.core.operators._nonparametric


class TestConstantOperator:
    """Test core.operators._nonparametric.ConstantOperator."""
    def test_init(self):
        """Test core.operators._nonparametric.ConstantOperator.__init__()"""
        # Too many dimensions.
        cbad = np.arange(12).reshape((4,3))
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

    def test_call(self):
        """Test core.operators._nonparametric.ConstantOperator.__call__()"""
        c = np.random.random(10)
        op = opinf.core.operators.ConstantOperator(c)
        assert op() is c
        assert op(1) is c
        assert op([1, 2]) is c
        assert op([1], 2) is c


class TestLinearOperator:
    """Test core.operators._nonparametric.LinearOperator."""
    def test_init(self):
        """Test core.operators._nonparametric.LinearOperator.__init__()"""

        # Too many dimensions.
        Abad = np.arange(12).reshape((2,2,3))
        with pytest.raises(ValueError) as ex:
            opinf.core.operators.LinearOperator(Abad)
        assert ex.value.args[0] == "linear operator must be two-dimensional"

        # No violation, nonsquare.
        A = Abad.reshape((4,3))
        op = opinf.core.operators.LinearOperator(A)
        assert op.entries is A

        # Correct square usage.
        A = A[:3,:3]
        op = opinf.core.operators.LinearOperator(A)
        assert op.entries is A

        # Special case: "one-dimensional" operator.
        B = np.arange(5)
        op = opinf.core.operators.LinearOperator(B)
        assert op.shape == (5,1)
        assert np.all(op.entries[:,0] == B)

        # Special case: "scalar" operator.
        A = np.array([10])
        op = opinf.core.operators.LinearOperator(A)
        assert op.shape == (1,1)
        assert op.entries[0,0] == A[0]

    def test_call(self):
        """Test core.operators._nonparametric.LinearOperator.__call__()"""

        # Special case: A is 1x1 (e.g., ROM state dimension = 1)
        A = np.random.random((1,1))
        op = opinf.core.operators.LinearOperator(A)
        x = np.random.random()
        assert np.allclose(op(x), A[0,0] * x)

        # Scalar inputs (e.g., ROM state dimension > 1 but input dimension = 1)
        B = np.random.random(10)
        op = opinf.core.operators.LinearOperator(B)
        x = np.random.random()
        assert np.allclose(op(x), B * x)

        # 1D inputs (usual case)
        def _check1D(A):
            x = np.random.random(A.shape[-1])
            op = opinf.core.operators.LinearOperator(A)
            assert np.allclose(op(x), A @ x)

        _check1D(np.random.random((4,3)))
        _check1D(np.random.random((4,4)))
        _check1D(np.random.random((4,1)))

        # 2D inputs (for applying to data residual)
        def _check2D(A):
            X = np.random.random((A.shape[-1], 20))
            op = opinf.core.operators.LinearOperator(A)
            assert np.allclose(op(X), A @ X)

        _check2D(np.random.random((10,3)))
        _check2D(np.random.random((6,6)))


class TestQuadraticOperator:
    """Test core.operators._nonparametric.QuadraticOperator."""
    def test_init(self):
        """Test core.operators._nonparametric.QuadraticOperator.__init__()"""
        # Too many dimensions.
        Hbad = np.arange(12).reshape((2,2,3))
        with pytest.raises(ValueError) as ex:
            opinf.core.operators.QuadraticOperator(Hbad)
        assert ex.value.args[0] == "quadratic operator must be two-dimensional"

        # Two-dimensional but invalid shape.
        Hbad = Hbad.reshape((4,3))
        with pytest.raises(ValueError) as ex:
            opinf.core.operators.QuadraticOperator(Hbad)
        assert ex.value.args[0] == "invalid dimensions for quadratic operator"

        # Special case: r = 1
        H = np.random.random((1,1))
        op = opinf.core.operators.QuadraticOperator(H)
        assert op.shape == (1,1)
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

    def test_call(self, ntrials=10):
        """Test core.operators._nonparametric.QuadraticOperator.__call__()"""
        # Full operator, compressed internally.
        r = 4
        H = np.random.random((r, r**2))
        op = opinf.core.operators.QuadraticOperator(H)
        for _ in range(ntrials):
            x = np.random.random(r)
            assert np.allclose(op(x), H @ np.kron(x, x))

        # Compressed operator.
        H = np.random.random((r, r*(r + 1)//2))
        op = opinf.core.operators.QuadraticOperator(H)
        for _ in range(ntrials):
            x = np.random.random(r)
            assert np.allclose(op(x), H @ opinf.utils.kron2c(x))

        # Special case: r = 1
        H = np.random.random((1,1))
        op = opinf.core.operators.QuadraticOperator(H)
        for _ in range(ntrials):
            x = np.random.random()
            assert np.allclose(op(x), H[0,0] * x**2)


# class TestCrossQuadraticOperator:
#     """Test core.operators._nonparametric.CrossQuadraticOperator."""
#     def test_init(self):
        # """Test CrossQuadraticOperator.__init__()"""
#         raise NotImplementedError
#
#     def test_call(self):
#         """Test CrossQuadraticOperator.__call__()"""
#         raise NotImplementedError


class TestCubicOperator:
    """Test core.operators._nonparametric.CubicOperator."""
    def test_init(self):
        """Test core.operators._nonparametric.CubicOperator.__init__()"""
        # Too many dimensions.
        Gbad = np.arange(24).reshape((2,4,3))
        with pytest.raises(ValueError) as ex:
            opinf.core.operators.CubicOperator(Gbad)
        assert ex.value.args[0] == "cubic operator must be two-dimensional"

        # Two-dimensional but invalid shape.
        Gbad = Gbad.reshape((3,8))
        with pytest.raises(ValueError) as ex:
            opinf.core.operators.CubicOperator(Gbad)
        assert ex.value.args[0] == "invalid dimensions for cubic operator"

        # Special case: r = 1
        G = np.random.random((1,1))
        op = opinf.core.operators.CubicOperator(G)
        assert op.shape == (1,1)
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

    def test_call(self, ntrials=10):
        """Test core.operators._nonparametric.CubicOperator.__call__()"""
        # Full operator, compressed internally.
        r = 4
        G = np.random.random((r, r**3))
        op = opinf.core.operators.CubicOperator(G)
        for _ in range(ntrials):
            x = np.random.random(r)
            assert np.allclose(op(x), G @ np.kron(np.kron(x, x), x))

        # Compressed operator.
        r = 5
        G = np.random.random((r, r*(r + 1)*(r + 2)//6))
        op = opinf.core.operators.CubicOperator(G)
        for _ in range(ntrials):
            x = np.random.random(r)
            assert np.allclose(op(x), G @ opinf.utils.kron3c(x))

        # Special case: r = 1
        G = np.random.random((1,1))
        op = opinf.core.operators.CubicOperator(G)
        for _ in range(ntrials):
            x = np.random.random()
            assert np.allclose(op(x), G[0,0] * x**3)
