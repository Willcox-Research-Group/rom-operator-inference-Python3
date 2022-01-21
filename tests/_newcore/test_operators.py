# _newcore/test_operators.py
"""Tests for rom_operator_inference._newcore.operators.py."""

import pytest
import numpy as np

import rom_operator_inference as opinf


# Base class ==================================================================
class TestBaseOperator:
    """Test _newcore.operators._BaseOperator."""
    class Dummy(opinf._newcore.operators._BaseOperator):
        """Instantiable version of _newcore.operators._BaseOperator."""
        def __init__(self, entries):
            super().__init__(entries)

        def __call__(self, *args, **kwargs):
            pass

    class Dummy2(opinf._newcore.operators._BaseOperator):
        """Instantiable version of _newcore.operators._BaseOperator."""
        def __init__(self, entries):
            super().__init__(entries)

        def __call__(self, *args, **kwargs):
            pass

    def test_init(self):
        """Test _newcore.operators._BaseOperator.__init__()."""
        A = np.random.random((10,11))
        op = self.Dummy(A)
        assert op.entries is A

    def test_validate_entries(self):
        """Test _newcore.operators._BaseOperator._validate_entries()."""
        func = opinf._newcore.operators._BaseOperator._validate_entries
        with pytest.raises(TypeError) as ex:
            func([1, 2, 3, 4])
        assert ex.value.args[0] == "operator entries must be NumPy array"

        A = np.arange(12, dtype=float).reshape((4,3)).T
        A[0,0] = np.nan
        with pytest.raises(ValueError) as ex:
            func(A)
        assert ex.value.args[0] == "operator entries must not be NaN"

        A[0,0] = np.inf
        with pytest.raises(ValueError) as ex:
            func(A)
        assert ex.value.args[0] == "operator entries must not be Inf"

        # Valid argument, no exceptions raised.
        A[0,0] = 0
        func(A)

    def test_eq(self):
        """Test _newcore.operators._BaseOperator.__eq__()."""
        A = np.arange(12).reshape((4,3))
        opA = self.Dummy(A)
        opA2 = self.Dummy2(A)
        assert opA != opA2

        opB = self.Dummy(A + 1)
        assert opA != opB

        opC = self.Dummy(A + 0)
        assert opA == opC


# Non-parametric operators ====================================================
class TestConstantOperator:
    """Test _newcore.operators.ConstantOperator."""
    def test_init(self):
        """Test _newcore.operators.ConstantOperator.__init__()"""
        # Too many dimensions.
        cbad = np.arange(12).reshape((4,3))
        with pytest.raises(ValueError) as ex:
            opinf._newcore.ConstantOperator(cbad)
        assert ex.value.args[0] == "constant operator must be one-dimensional"

        # Case 1: one-dimensional array.
        c = np.arange(12)
        op = opinf._newcore.ConstantOperator(c)
        assert op.entries is c

        # Case 2: two-dimensional array that can be flattened.
        op = opinf._newcore.ConstantOperator(c.reshape((-1, 1)))
        assert op.shape == (12,)
        assert np.all(op.entries == c)

        op = opinf._newcore.ConstantOperator(c.reshape((1, -1)))
        assert op.shape == (12,)
        assert np.all(op.entries == c)

    def test_call(self):
        """Test _newcore.operators.ConstantOperator.__call__()"""
        c = np.random.random(10)
        op = opinf._newcore.ConstantOperator(c)
        assert op() is c
        assert op(1) is c
        assert op([1, 2]) is c
        assert op([1], 2) is c


class TestLinearOperator:
    """Test _newcore.operators.LinearOperator."""
    def test_init(self):
        """Test _newcore.operators.LinearOperator.__init__()"""

        # Too many dimensions.
        Abad = np.arange(12).reshape((2,2,3))
        with pytest.raises(ValueError) as ex:
            opinf._newcore.LinearOperator(Abad)
        assert ex.value.args[0] == "linear operator must be two-dimensional"

        # Violate square requirement.
        A = Abad.reshape((4,3))
        with pytest.raises(ValueError) as ex:
            opinf._newcore.LinearOperator(A, square=True)
        assert ex.value.args[0] == "expected square array for linear operator"

        # No violation if square not a requirement.
        op = opinf._newcore.LinearOperator(A, square=False)
        assert op.entries is A

        # Correct square usage.
        A = A[:3,:3]
        op = opinf._newcore.LinearOperator(A, square=True)
        assert op.entries is A

        # Special case: "one-dimensional" operator.
        B = np.arange(5)
        op = opinf._newcore.LinearOperator(B, square=False)
        assert op.shape == (5,1)
        assert np.all(op.entries[:,0] == B)

        # Special case: "scalar" operator.
        A = np.array([10])
        op = opinf._newcore.LinearOperator(A, square=True)
        assert op.shape == (1,1)
        assert op.entries[0,0] == A[0]

    def test_call(self):
        """Test _newcore.operators.LinearOperator.__call__()"""

        # Special case: A is 1x1 (e.g., ROM state dimension = 1)
        A = np.random.random((1,1))
        op = opinf._newcore.LinearOperator(A, square=False)
        x = np.random.random()
        assert np.allclose(op(x), A[0,0] * x)

        # Scalar inputs (e.g., ROM state dimension > 1 but input dimension = 1)
        B = np.random.random(10)
        op = opinf._newcore.LinearOperator(B, square=False)
        x = np.random.random()
        assert np.allclose(op(x), B * x)

        # 1D inputs (usual case)
        def _check1D(A, square):
            x = np.random.random(A.shape[-1])
            op = opinf._newcore.LinearOperator(A, square=square)
            assert np.allclose(op(x), A @ x)

        _check1D(np.random.random((4,3)), False)
        _check1D(np.random.random((4,4)), True)
        _check1D(np.random.random((4,1)), False)

        # 2D inputs (for applying to data residual)
        def _check2D(A, square):
            X = np.random.random((A.shape[-1], 20))
            op = opinf._newcore.LinearOperator(A, square=square)
            assert np.allclose(op(X), A @ X)

        _check2D(np.random.random((10,3)), False)
        _check2D(np.random.random((6,6)), True)


class TestQuadraticOperator:
    """Test _newcore.operators.QuadraticOperator."""
    def test_init(self):
        """Test _newcore.operators.QuadraticOperator.__init__()"""
        # Too many dimensions.
        Hbad = np.arange(12).reshape((2,2,3))
        with pytest.raises(ValueError) as ex:
            opinf._newcore.QuadraticOperator(Hbad)
        assert ex.value.args[0] == "quadratic operator must be two-dimensional"

        # Two-dimensional but invalid shape.
        Hbad = Hbad.reshape((4,3))
        with pytest.raises(ValueError) as ex:
            opinf._newcore.QuadraticOperator(Hbad)
        assert ex.value.args[0] == "invalid dimensions for quadratic operator"

        # Special case: r = 1
        H = np.random.random((1,1))
        op = opinf._newcore.QuadraticOperator(H)
        assert op.shape == (1,1)
        assert np.allclose(op.entries, H)

        # Full operator, compressed internally.
        r = 4
        H = np.random.random((r, r**2))
        op = opinf._newcore.QuadraticOperator(H)
        assert op.shape == (r, r*(r + 1)//2)
        assert np.allclose(op.entries, opinf.utils.compress_quadratic(H))

        # Compressed operator.
        r = 4
        H = np.random.random((r, r*(r + 1)//2))
        op = opinf._newcore.QuadraticOperator(H)
        assert op.entries is H

    def test_call(self, ntrials=10):
        """Test _newcore.operators.QuadraticOperator.__call__()"""
        # Full operator, compressed internally.
        r = 4
        H = np.random.random((r, r**2))
        op = opinf._newcore.QuadraticOperator(H)
        for _ in range(ntrials):
            x = np.random.random(r)
            assert np.allclose(op(x), H @ np.kron(x, x))

        # Compressed operator.
        H = np.random.random((r, r*(r + 1)//2))
        op = opinf._newcore.QuadraticOperator(H)
        for _ in range(ntrials):
            x = np.random.random(r)
            assert np.allclose(op(x), H @ opinf.utils.kron2c(x))

        # Special case: r = 1
        H = np.random.random((1,1))
        op = opinf._newcore.QuadraticOperator(H)
        for _ in range(ntrials):
            x = np.random.random()
            assert np.allclose(op(x), H[0,0] * x**2)


# class TestCrossQuadraticOperator:
#     """Test _newcore.operators.CrossQuadraticOperator."""
#     def test_init(self):
#         """Test _newcore.operators.CrossQuadraticOperator.__init__()"""
#         raise NotImplementedError
#
#     def test_call(self):
#         """Test _newcore.operators.CrossQuadraticOperator.__call__()"""
#         raise NotImplementedError


class TestCubicOperator:
    """Test _newcore.operators.CubicOperator."""
    def test_init(self):
        """Test _newcore.operators.CubicOperator.__init__()"""
        # Too many dimensions.
        Gbad = np.arange(24).reshape((2,4,3))
        with pytest.raises(ValueError) as ex:
            opinf._newcore.CubicOperator(Gbad)
        assert ex.value.args[0] == "cubic operator must be two-dimensional"

        # Two-dimensional but invalid shape.
        Gbad = Gbad.reshape((3,8))
        with pytest.raises(ValueError) as ex:
            opinf._newcore.CubicOperator(Gbad)
        assert ex.value.args[0] == "invalid dimensions for cubic operator"

        # Special case: r = 1
        G = np.random.random((1,1))
        op = opinf._newcore.CubicOperator(G)
        assert op.shape == (1,1)
        assert np.allclose(op.entries, G)

        # Full operator, compressed internally.
        r = 4
        G = np.random.random((r, r**3))
        op = opinf._newcore.CubicOperator(G)
        assert op.shape == (r, r*(r + 1)*(r + 2)//6)
        assert np.allclose(op.entries, opinf.utils.compress_cubic(G))

        # Compressed operator.
        r = 5
        G = np.random.random((r, r*(r + 1)*(r + 2)//6))
        op = opinf._newcore.CubicOperator(G)
        assert op.entries is G

    def test_call(self, ntrials=10):
        """Test _newcore.operators.CubicOperator.__call__()"""
        # Full operator, compressed internally.
        r = 4
        G = np.random.random((r, r**3))
        op = opinf._newcore.CubicOperator(G)
        for _ in range(ntrials):
            x = np.random.random(r)
            assert np.allclose(op(x), G @ np.kron(np.kron(x, x), x))

        # Compressed operator.
        r = 5
        G = np.random.random((r, r*(r + 1)*(r + 2)//6))
        op = opinf._newcore.CubicOperator(G)
        for _ in range(ntrials):
            x = np.random.random(r)
            assert np.allclose(op(x), G @ opinf.utils.kron3c(x))

        # Special case: r = 1
        G = np.random.random((1,1))
        op = opinf._newcore.CubicOperator(G)
        for _ in range(ntrials):
            x = np.random.random()
            assert np.allclose(op(x), G[0,0] * x**3)


# # Affine-parametric base class ==============================================
# class TestBaseAffineOperator:
#     class AffineDummy(opinf._newcore.operators._BaseAffineOperator):
#         """Instantiable version of _newcore.operators._BaseAffineOperator."""
#         def __init__(self, coeffs, matrices, **kwargs):
#             super().__init__(self, TestBaseOperator.Dummy, coeffs, matrices)
#
#     def test_init(self):
#         """Test _newcore.operators._BaseAffineOperator.__init__()."""
#         pass
#
#
# # Affine-parametric operators ===============================================
# class TestAffineConstantOperator:
#     """Test _newcore.operators.ConstantOperator."""
#     def test_init(self):
#         """Test _newcore.operators.ConstantOperator.__init__()"""
#         raise NotImplementedError
#
#     def test_call(self):
#         """Test _newcore.operators.ConstantOperator.__call__()"""
#         raise NotImplementedError
#
#
# class TestAffineLinearOperator:
#     """Test _newcore.operators.LinearOperator."""
#     def test_init(self):
#         """Test _newcore.operators.LinearOperator.__init__()"""
#         raise NotImplementedError
#
#     def test_call(self):
#         """Test _newcore.operators.LinearOperator.__call__()"""
#         raise NotImplementedError
#
#
# class TestAffineQuadraticOperator:
#     """Test _newcore.operators.QuadraticOperator."""
#     def test_init(self):
#         """Test _newcore.operators.QuadraticOperator.__init__()"""
#         raise NotImplementedError
#
#     def test_call(self):
#         """Test _newcore.operators.QuadraticOperator.__call__()"""
#         raise NotImplementedError
#
#
# # class TestAffineCrossQuadraticOperator:
# #     """Test _newcore.operators.CrossQuadraticOperator."""
# #     def test_init(self):
# #         """Test _newcore.operators.CrossQuadraticOperator.__init__()"""
# #         raise NotImplementedError
# #
# #     def test_call(self):
# #         """Test _newcore.operators.CrossQuadraticOperator.__call__()"""
# #         raise NotImplementedError
#
#
# class TestAffineCubicOperator:
#     """Test _newcore.operators.CubicOperator."""
#     def test_init(self):
#         """Test _newcore.operators.CubicOperator.__init__()"""
#         raise NotImplementedError
#
#     def test_call(self):
#         """Test _newcore.operators.CubicOperator.__call__()"""
#         raise NotImplementedError
