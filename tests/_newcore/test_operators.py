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
        def __init__(self, entries, symbol=''):
            super().__init__(entries, symbol)

        def __call__(*args, **kwargs):
            pass

        def _str(self, *args, **kwargs):
            pass

    class Dummy2(Dummy):
        """Another instantiable version of _newcore.operators._BaseOperator."""
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

    def test_str(self):
        """Test _newcore.operators.ConstantOperator._str()."""
        c = opinf._newcore.ConstantOperator(np.random.random(10), symbol='c')
        assert c._str() == 'c'


class TestLinearOperator:
    """Test _newcore.operators.LinearOperator."""
    def test_init(self):
        """Test _newcore.operators.LinearOperator.__init__()"""

        # Too many dimensions.
        Abad = np.arange(12).reshape((2,2,3))
        with pytest.raises(ValueError) as ex:
            opinf._newcore.LinearOperator(Abad)
        assert ex.value.args[0] == "linear operator must be two-dimensional"

        # No violation, nonsquare.
        A = Abad.reshape((4,3))
        op = opinf._newcore.LinearOperator(A)
        assert op.entries is A

        # Correct square usage.
        A = A[:3,:3]
        op = opinf._newcore.LinearOperator(A)
        assert op.entries is A

        # Special case: "one-dimensional" operator.
        B = np.arange(5)
        op = opinf._newcore.LinearOperator(B)
        assert op.shape == (5,1)
        assert np.all(op.entries[:,0] == B)

        # Special case: "scalar" operator.
        A = np.array([10])
        op = opinf._newcore.LinearOperator(A)
        assert op.shape == (1,1)
        assert op.entries[0,0] == A[0]

    def test_call(self):
        """Test _newcore.operators.LinearOperator.__call__()"""

        # Special case: A is 1x1 (e.g., ROM state dimension = 1)
        A = np.random.random((1,1))
        op = opinf._newcore.LinearOperator(A)
        x = np.random.random()
        assert np.allclose(op(x), A[0,0] * x)

        # Scalar inputs (e.g., ROM state dimension > 1 but input dimension = 1)
        B = np.random.random(10)
        op = opinf._newcore.LinearOperator(B)
        x = np.random.random()
        assert np.allclose(op(x), B * x)

        # 1D inputs (usual case)
        def _check1D(A):
            x = np.random.random(A.shape[-1])
            op = opinf._newcore.LinearOperator(A)
            assert np.allclose(op(x), A @ x)

        _check1D(np.random.random((4,3)))
        _check1D(np.random.random((4,4)))
        _check1D(np.random.random((4,1)))

        # 2D inputs (for applying to data residual)
        def _check2D(A):
            X = np.random.random((A.shape[-1], 20))
            op = opinf._newcore.LinearOperator(A)
            assert np.allclose(op(X), A @ X)

        _check2D(np.random.random((10,3)))
        _check2D(np.random.random((6,6)))

    def test_str(self):
        """Test _newcore.operators.LinearOperator._str()."""
        A = opinf._newcore.LinearOperator(np.random.random((10, 10)),
                                          symbol='A')
        assert A._str("q(t)") == "Aq(t)"
        A.symbol = "B"
        assert A._str("u_{j}") == "Bu_{j}"


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

    def test_str(self):
        """Test _newcore.operators.QuadraticOperator._str()."""
        H = opinf._newcore.QuadraticOperator(np.random.random((10, 100)),
                                             symbol='H')
        assert H._str("q_{j}") == "H[q_{j} ⊗ q_{j}]"
        assert H._str("u(t)") == "H[u(t) ⊗ u(t)]"


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

    def test_str(self):
        """Test _newcore.operators.CubicOperator._str()."""
        G = opinf._newcore.CubicOperator(np.random.random((10, 1000)),
                                         symbol='G')
        assert G._str("q(t)") == "G[q(t) ⊗ q(t) ⊗ q(t)]"

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
