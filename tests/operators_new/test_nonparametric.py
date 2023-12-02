# operators/test_nonparametric.py
"""Tests for operators._nonparametric."""

import pytest
import numpy as np
import scipy.linalg as la

import opinf


_module = opinf.operators_new._nonparametric


# No dependence on state or input =============================================
class TestConstantOperator:
    """Test operators._nonparametric.ConstantOperator."""

    _OpClass = _module.ConstantOperator

    def test_str(self):
        """Test ConstantOperator._str()."""
        op = self._OpClass()
        assert op._str() == "c"
        assert op._str("q_j") == "c"
        assert op._str("q(t)", "u(t)") == "c"

    def test_set_entries(self):
        """Test ConstantOperator.set_entries()."""
        op = self._OpClass()

        # Too many dimensions.
        cbad = np.arange(12).reshape((4, 3))
        with pytest.raises(ValueError) as ex:
            op.set_entries(cbad)
        assert (
            ex.value.args[0] == "ConstantOperator entries must be "
            "one-dimensional"
        )

        # Case 1: one-dimensional array.
        c = np.arange(12)
        op.set_entries(c)
        assert op.entries is c

        # Case 2: two-dimensional array that can be flattened.
        op.set_entries(c.reshape((-1, 1)))
        assert op.shape == (12,)
        assert np.all(op.entries == c)

        op.set_entries(c.reshape((1, -1)))
        assert op.shape == (12,)
        assert np.all(op.entries == c)

        # Case 3: r = 1 and c is a scalar.
        c = np.random.random()
        op.set_entries(c)
        assert op.shape == (1,)
        assert op.entries[0] == c

    def test_apply(self, k=20):
        """Test ConstantOperator.apply()/__call__()."""
        op = self._OpClass()

        def _test_single(r):
            c = np.random.random(r)
            op.set_entries(c)
            assert np.allclose(op.apply(), c)
            # Evaluation for a single vector.
            q = np.random.random(r)
            assert np.allclose(op.apply(q), op.entries)
            # Vectorized evaluation.
            Q = np.random.random((r, k))
            ccc = np.column_stack([c for _ in range(k)])
            out = op.apply(Q)
            assert out.shape == (r, k)
            assert np.all(out == ccc)

        _test_single(10)
        _test_single(20)
        _test_single(1)

        # Special case: r = 1, scalar q.
        c = np.random.random()
        op.set_entries(c)
        # Evaluation for a single vector.
        q = np.random.random()
        out = op.apply(q)
        assert np.isscalar(out)
        assert out == c
        # Vectorized evaluation.
        Q = np.random.random(k)
        out = op.apply(Q)
        assert out.shape == (k,)
        assert np.all(out == c)

    def test_jacobian(self, r=10):
        """Test ConstantOperator.jacobian()."""
        op = self._OpClass()
        op.set_entries(np.random.random(r))
        Z = np.zeros((r, r))

        def _test_single(out):
            assert out.shape == Z.shape
            assert np.all(out == Z)

        _test_single(op.jacobian())
        _test_single(op.jacobian(1))
        _test_single(op.jacobian([1, 2]))
        _test_single(op.jacobian([1], 2))

    def test_galerkin(self, n=10, r=3, ntrials=10):
        """Test ConstantOperator.galerkin()."""
        Vr = np.random.random((n, r))
        Wr = np.random.random((n, r))
        c = np.random.random(n)

        op = self._OpClass(c)
        op_ = op.galerkin(Vr, Wr)
        assert isinstance(op_, self._OpClass)
        assert op_.shape == (r,)
        for _ in range(ntrials):
            q_ = np.random.random(r)
            full = Wr.T @ op.apply(Vr @ q_)
            reduced = op_.apply(q_)
            assert np.allclose(reduced, full)

    def test_datablock(self, k=20):
        """Test ConstantOperator.datablock()."""
        op = self._OpClass()
        ones = np.ones((1, k))

        def _test_single(out):
            assert out.shape == ones.shape
            assert np.all(out == ones)

        _test_single(op.datablock(np.random.random((5, k))))
        _test_single(op.datablock(np.random.random((3, k))))
        _test_single(op.datablock(np.random.random(k)))

    def test_column_dimension(self):
        """Test ConstantOperator.column_dimension()."""
        assert self._OpClass.column_dimension() == 1
        assert self._OpClass.column_dimension(4) == 1
        assert self._OpClass.column_dimension(1, 6) == 1


# Dependent on state but not on input =========================================
class TestLinearOperator:
    """Test operators._nonparametric.LinearOperator."""

    _OpClass = _module.LinearOperator

    def test_str(self):
        """Test LinearOperator._str()."""
        op = self._OpClass()
        assert op._str("q_j") == "Aq_j"
        assert op._str("q(t)", "u(t)") == "Aq(t)"

    def test_set_entries(self):
        """Test LinearOperator.set_entries()."""
        op = self._OpClass()

        # Too many dimensions.
        Abad = np.arange(12).reshape((2, 2, 3))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Abad)
        assert (
            ex.value.args[0] == "LinearOperator entries must be "
            "two-dimensional"
        )

        # Nonsquare.
        Abad = Abad.reshape((4, 3))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Abad)
        assert (
            ex.value.args[0] == "LinearOperator entries must be square (r x r)"
        )

        # Correct square usage.
        A = Abad[:3, :3]
        op.set_entries(A)
        assert op.entries is A

        # Special case: r = 1, scalar A.
        a = np.random.random()
        op.set_entries(a)
        assert op.shape == (1, 1)
        assert op[0, 0] == a

    def test_apply(self, k=20):
        """Test LinearOperator.apply()/__call__()."""
        op = self._OpClass()

        def _test_single(r):
            A = np.random.random((r, r))
            op.set_entries(A)
            # Evaluation for a single vector.
            q = np.random.random(r)
            assert np.allclose(op.apply(q), A @ q)
            # Vectorized evaluation.
            Q = np.random.random((r, k))
            assert np.allclose(op.apply(Q), A @ Q)

        _test_single(10)
        _test_single(4)
        _test_single(1)

        # Special case: A is 1x1 and q is a scalar.
        A = np.random.random()
        op.set_entries(A)
        # Evaluation for a single vector.
        q = np.random.random()
        out = op(q)
        assert np.isscalar(out)
        assert np.allclose(out, A * q)
        # Vectorized evaluation.
        Q = np.random.random(k)
        out = op(Q)
        assert out.shape == (k,)
        assert np.allclose(out, A * Q)

    def test_jacobian(self, r=9):
        """Test LinearOperator.jacobian()."""
        A = np.random.random((r, r))
        op = self._OpClass(A)
        jac = op.jacobian(np.random.random(r))
        assert jac.shape == A.shape
        assert np.all(jac == A)

    def test_galerkin(self, n=10, r=3, ntrials=10):
        """Test LinearOperator.galerkin()."""
        Vr = np.random.random((n, r))
        Wr = np.random.random((n, r))
        A = np.random.random((n, n))

        op = self._OpClass(A)
        op_ = op.galerkin(Vr, Wr)
        assert isinstance(op_, self._OpClass)
        assert op_.shape == (r, r)
        for _ in range(ntrials):
            q_ = np.random.random(r)
            full = Wr.T @ op.apply(Vr @ q_)
            reduced = op_.apply(q_)
            assert np.allclose(reduced, full)

    def test_datablock(self, m=3, k=20, r=10):
        """Test LinearOperator.datablock()."""
        op = self._OpClass()
        state_ = np.random.random((r, k))
        input_ = np.random.random((m, k))

        assert op.datablock(state_, input_) is state_
        assert op.datablock(state_, None) is state_

        # Special case: r = 1.
        state_ = np.random.random(k)
        block = op.datablock(state_)
        assert block.shape == (1, k)
        assert np.all(block[0] == state_)

    def test_column_dimension(self):
        """Test LinearOperator.column_dimension()."""
        assert self._OpClass.column_dimension(2) == 2
        assert self._OpClass.column_dimension(4, 6) == 4


class TestQuadraticOperator:
    """Test operators._nonparametric.QuadraticOperator."""

    _OpClass = _module.QuadraticOperator

    def test_str(self):
        """Test QuadraticOperator._str()."""
        op = self._OpClass()
        assert op._str("q_j") == "H[q_j ⊗ q_j]"
        assert op._str("q(t)", "u(t)") == "H[q(t) ⊗ q(t)]"

    def test_set_entries(self, r=4):
        """Test QuadraticOperator.set_entries()."""
        op = self._OpClass()

        # Too many dimensions.
        Hbad = np.arange(16).reshape((2, 2, 2, 2))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Hbad)
        assert (
            ex.value.args[0] == "QuadraticOperator entries must be "
            "two-dimensional"
        )

        # Two-dimensional but invalid shape.
        Hbad = Hbad.reshape((r, r))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Hbad)
        assert (
            ex.value.args[0] == "invalid QuadraticOperator entries dimensions"
        )

        # Special case: r = 1, H a scalar.
        H = np.random.random()
        op.set_entries(H)
        assert op.shape == (1, 1)
        assert op.entries[0, 0] == H

        # Full operator, compressed internally.
        # More thorough tests elsewhere for _kronecker.compress_quadratic().
        H = np.random.random((r, r**2))
        H_ = opinf.utils.compress_quadratic(H)
        op.set_entries(H)
        r2_ = r * (r + 1) // 2
        assert op.shape == (r, r2_)
        assert np.allclose(op.entries, H_)

        # Three-dimensional tensor.
        op.set_entries(H.reshape((r, r, r)))
        assert op.shape == (r, r2_)
        assert np.allclose(op.entries, H_)

        # Compressed operator.
        H = np.random.random((r, r2_))
        op.set_entries(H)
        assert op.entries is H

        # Test _clear().
        op._clear()
        assert op.entries is None
        assert op._mask is None
        assert op._jac is None

    def test_apply(self, k=10, ntrials=10):
        """Test QuadraticOperator.apply()/__call__()."""
        op = self._OpClass()

        def _test_single(r):
            H = np.random.random((r, r**2))
            op.set_entries(H)
            for _ in range(ntrials):
                # Evaluation for a single vector.
                q = np.random.random(r)
                evaltrue = H @ np.kron(q, q)
                evalgot = op.apply(q)
                assert evalgot.shape == (r,)
                assert np.allclose(evalgot, evaltrue)
                # Vectorized evaluation.
                Q = np.random.random((r, k))
                evaltrue = H @ la.khatri_rao(Q, Q)
                evalgot = op.apply(Q)
                assert evalgot.shape == (r, k)
                assert np.allclose(evalgot, evaltrue)

        _test_single(5)
        _test_single(2)
        _test_single(1)

        # Special case: r = 1 and q is a scalar.
        H = np.random.random()
        op.set_entries(H)
        for _ in range(ntrials):
            # Evaluation for a single vector.
            q = np.random.random()
            evaltrue = H * q**2
            evalgot = op.apply(q)
            assert np.isscalar(evalgot)
            assert np.isclose(evalgot, evaltrue)
            # Vectorized evaluation.
            Q = np.random.random(k)
            evaltrue = H * Q**2
            evalgot = op.apply(Q)
            assert evalgot.shape == (k,)
            assert np.allclose(evalgot, evaltrue)

    def test_jacobian(self, r=5, ntrials=10):
        """Test QuadraticOperator.jacobian()."""
        H = np.random.random((r, r**2))
        op = self._OpClass(H)

        # r > 1
        Id = np.eye(r)
        for _ in range(ntrials):
            q = np.random.random(r)
            jac_true = H @ (np.kron(Id, q) + np.kron(q, Id)).T
            jac = op.jacobian(q)
            assert jac.shape == (r, r)
            assert np.allclose(jac, jac_true)

        # Special case: r = 1
        H = np.random.random()
        op.set_entries(H)
        for _ in range(ntrials):
            q = np.random.random()
            jac_true = 2 * H * q
            jac = op.jacobian(q)
            assert jac.shape == (1, 1)
            assert np.isclose(jac[0, 0], jac_true)

    def test_galerkin(self, n=10, r=3, ntrials=10):
        """Test QuadraticOperator.galerkin()."""
        Vr = np.random.random((n, r))
        Wr = np.random.random((n, r))
        H = np.random.random((n, n**2))

        op = self._OpClass(H)
        op_ = op.galerkin(Vr, Wr)
        assert isinstance(op_, self._OpClass)
        assert op_.shape == (r, r * (r + 1) // 2)
        for _ in range(ntrials):
            q_ = np.random.random(r)
            full = Wr.T @ op.apply(Vr @ q_)
            reduced = op_.apply(q_)
            assert np.allclose(reduced, full)

    def test_datablock(self, k=20, r=10):
        """Test QuadraticOperator.datablock()."""
        op = self._OpClass()
        state_ = np.random.random((r, k))
        r2_ = r * (r + 1) // 2

        # More thorough tests elsewhere for _kronecker.kron2c().
        block = op.datablock(state_)
        assert block.shape == (r2_, k)
        op.entries = np.random.random((r, r2_))
        mult = op.entries @ block
        evald = op(state_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

        # Special case: r = 1.
        state_ = state_[0]
        block = op.datablock(state_)
        assert block.shape == (1, k)
        op.entries = np.random.random()
        mult = op.entries[0, 0] * block[0]
        evald = op(state_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

    def test_column_dimension(self):
        """Test QuadraticOperator.column_dimension()."""
        assert self._OpClass.column_dimension(1) == 1
        assert self._OpClass.column_dimension(3) == 6
        assert self._OpClass.column_dimension(5, 7) == 15


class TestCubicOperator:
    """Test operators._nonparametric.CubicOperator."""

    _OpClass = _module.CubicOperator

    def test_str(self):
        """Test CubicOperator._str()."""
        op = self._OpClass()
        assert op._str("q_j") == "G[q_j ⊗ q_j ⊗ q_j]"
        assert op._str("q(t)", "u(t)") == "G[q(t) ⊗ q(t) ⊗ q(t)]"

    def test_set_entries(self, r=4):
        """Test CubicOperator.set_entries()."""
        op = self._OpClass()

        # Too many dimensions.
        Gbad = np.arange(4).reshape((1, 2, 1, 2))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Gbad)
        assert (
            ex.value.args[0] == "CubicOperator entries must be two-dimensional"
        )

        # Two-dimensional but invalid shape.
        Gbad = np.random.random((3, 8))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Gbad)
        assert ex.value.args[0] == "invalid CubicOperator entries dimensions"

        # Special case: r = 1
        G = np.random.random((1, 1))
        op.set_entries(G)
        assert op.shape == (1, 1)
        assert np.allclose(op.entries, G)

        # Full operator, compressed internally.
        # More thorough tests elsewhere for _kronecker.compress_cubic().
        G = np.random.random((r, r**3))
        G_ = opinf.utils.compress_cubic(G)
        op.set_entries(G)
        r3_ = r * (r + 1) * (r + 2) // 6
        assert op.shape == (r, r3_)
        assert np.allclose(op.entries, G_)

        # Three-dimensional tensor.
        op.set_entries(G.reshape((r, r, r, r)))
        assert op.shape == (r, r3_)
        assert np.allclose(op.entries, G_)

        # Compressed operator.
        G = np.random.random((r, r3_))
        op.set_entries(G)
        assert op.entries is G

        # Test _clear().
        op._clear()
        assert op.entries is None
        assert op._mask is None
        assert op._jac is None

    def test_apply(self, k=20, ntrials=10):
        """Test CubicOperator.apply()/__call__()."""
        op = self._OpClass()

        def _test_single(r):
            G = np.random.random((r, r**3))
            op.set_entries(G)
            for _ in range(ntrials):
                # Evaluation for a single vector.
                q = np.random.random(r)
                evaltrue = G @ np.kron(np.kron(q, q), q)
                evalgot = op.apply(q)
                assert np.allclose(evalgot, evaltrue)
                # Vectorized evaluation.
                Q = np.random.random((r, k))
                evaltrue = G @ la.khatri_rao(Q, la.khatri_rao(Q, Q))
                evalgot = op.apply(Q)
                assert evalgot.shape == (r, k)
                assert np.allclose(evalgot, evaltrue)

        _test_single(5)
        _test_single(3)
        _test_single(1)

        # Special case: r = 1 and q is a scalar.
        G = np.random.random()
        op.set_entries(G)
        for _ in range(ntrials):
            # Evaluation for a single vector.
            q = np.random.random()
            evaltrue = G * q**3
            evalgot = op.apply(q)
            assert np.isscalar(evalgot)
            assert np.allclose(evalgot, evaltrue)
            # Vectorized evaluation.
            Q = np.random.random(k)
            evaltrue = G * Q**3
            evalgot = op.apply(Q)
            assert evalgot.shape == (k,)
            assert np.allclose(evalgot, evaltrue)

    def test_jacobian(self, r=5, ntrials=10):
        """Test CubicOperator.jacobian()."""
        G = np.random.random((r, r**3))
        op = self._OpClass(G)

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
        G = np.random.random()
        op.set_entries(G)
        for _ in range(ntrials):
            q = np.random.random()
            jac_true = 3 * G * q**2
            jac = op.jacobian(q)
            assert jac.shape == (1, 1)
            assert np.isclose(jac[0, 0], jac_true)

    def test_galerkin(self, n=5, r=2, ntrials=10):
        """Test CubicOperator.galerkin()."""
        Vr = np.random.random((n, r))
        Wr = np.random.random((n, r))
        H = np.random.random((n, n**3))

        op = self._OpClass(H)
        op_ = op.galerkin(Vr, Wr)
        assert isinstance(op_, self._OpClass)
        assert op_.shape == (r, r * (r + 1) * (r + 2) // 6)
        for _ in range(ntrials):
            q_ = np.random.random(r)
            full = Wr.T @ op.apply(Vr @ q_)
            reduced = op_.apply(q_)
            assert np.allclose(reduced, full)

    def test_datablock(self, k=20, r=10):
        """Test CubicOperator.datablock()."""
        op = self._OpClass()
        state_ = np.random.random((r, k))
        r3_ = r * (r + 1) * (r + 2) // 6

        # More thorough tests elsewhere for _kronecker.kron2c().
        block = op.datablock(state_)
        assert block.shape == (r3_, k)
        op.entries = np.random.random((r, r3_))
        mult = op.entries @ block
        evald = op(state_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

        # Special case: r = 1.
        state_ = state_[0]
        block = op.datablock(state_)
        assert block.shape == (1, k)
        op.entries = np.random.random()
        mult = op.entries[0, 0] * block[0]
        evald = op(state_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

    def test_column_dimension(self):
        """Test CubicOperator.column_dimension()."""
        assert self._OpClass.column_dimension(1) == 1
        assert self._OpClass.column_dimension(3) == 10
        assert self._OpClass.column_dimension(5, 2) == 35


# Dependent on input but not on state =========================================
class TestInputOperator:
    """Test operators._nonparametric.InputOperator."""

    _OpClass = _module.InputOperator

    def test_str(self):
        """Test InputOperator._str()."""
        op = self._OpClass()
        assert op._str("q_j", "u_j") == "Bu_j"
        assert op._str(None, "u(t)") == "Bu(t)"

    def test_set_entries(self):
        """Test InputOperator.set_entries()."""
        op = self._OpClass()

        # Too many dimensions.
        Bbad = np.arange(12).reshape((2, 2, 3))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Bbad)
        assert (
            ex.value.args[0] == "InputOperator entries must be two-dimensional"
        )

        # Nonsquare is OK.
        B = Bbad.reshape((4, 3))
        op.set_entries(B)
        assert op.entries is B
        assert op.m == 3

        # Special case: r > 1, m = 1
        B = np.random.random(5)
        op.set_entries(B)
        assert op.shape == (5, 1)
        assert op.m == 1
        assert np.allclose(op.entries[:, 0], B)

        # Special case: r = 1, m > 1
        B = np.random.random((1, 3))
        op.set_entries(B)
        assert op.shape == (1, 3)
        assert np.allclose(op.entries, B)

        # Special case: r = 1, m = 1 (scalar B).
        b = np.random.random()
        op.set_entries(b)
        assert op.shape == (1, 1)
        assert op[0, 0] == b

    def test_apply(self, k=20):
        """Test InputOperator.apply()/__call__()."""
        op = self._OpClass()

        def _test_single(r, m):
            B = np.random.random((r, m))
            op.set_entries(B)
            # Evaluation for a single vector.
            q = np.random.random(r)
            u = np.random.random(m)
            evaltrue = B @ u
            evalgot = op.apply(q, u)
            assert evalgot.shape == (r,)
            assert np.allclose(evalgot, evaltrue)
            # Vectorized evaluation.
            Q = np.random.random((r, k))
            U = np.random.random((m, k))
            evaltrue = B @ U
            evalgot = op.apply(Q, U)
            assert evalgot.shape == (r, k)
            assert np.allclose(op.apply(Q, U), B @ U)

        _test_single(10, 2)
        _test_single(2, 5)
        _test_single(3, 1)
        _test_single(1, 4)
        _test_single(1, 1)

        # Special case: B is 1x1 and u is a scalar.
        B = np.random.random()
        op.set_entries(B)
        # Evaluation for a single vector.
        q = np.random.random()
        u = np.random.random()
        out = op(q, u)
        assert np.isscalar(out)
        assert np.allclose(out, B * u)
        # Vectorized evaluation.
        U = np.random.random(k)
        out = op(None, U)
        assert out.shape == (k,)
        assert np.allclose(out, B * U)

        # Special case: B is rx1, r>1, and u is a scalar.
        r = 10
        B = np.random.random(r)
        op.set_entries(B)
        # Evaluation for a single vector.
        q = np.random.random(r)
        u = np.random.random()
        out = op(q, u)
        assert out.shape == (r,)
        assert np.allclose(out, B * u)
        # Vectorized evaluation.
        U = np.random.random(k)
        out = op(None, U)
        assert out.shape == (r, k)
        assert np.allclose(out, np.column_stack([B * u for u in U]))

    def test_jacobian(self, r=9):
        """Test InputOperator.jacobian()."""
        B = np.random.random((r, r - 2))
        op = self._OpClass(B)
        jac = op.jacobian(np.random.random(r))
        assert jac.shape == (r, r)
        assert np.all(jac == 0)

    def test_galerkin(self, n=10, r=4, m=3, ntrials=10):
        """Test InputOperator.galerkin()."""
        Vr = np.random.random((n, r))
        Wr = np.random.random((n, r))
        B = np.random.random((n, m))

        op = self._OpClass(B)
        op_ = op.galerkin(Vr, Wr)
        assert isinstance(op_, self._OpClass)
        assert op_.shape == (r, m)
        for _ in range(ntrials):
            q_ = np.random.random(r)
            u = np.random.random(m)
            full = Wr.T @ op.apply(Vr @ q_, u)
            reduced = op_.apply(q_, u)
            assert np.allclose(reduced, full)

    def test_datablock(self, m=3, k=20, r=10):
        """Test InputOperator.datablock()."""
        op = self._OpClass()
        state_ = np.random.random((r, k))
        input_ = np.random.random((m, k))

        assert op.datablock(state_, input_) is input_
        assert op.datablock(None, input_) is input_

        # Special case: m = 1.
        input_ = input_[0]
        block = op.datablock(state_, input_)
        assert block.shape == (1, k)
        assert np.all(block[0] == input_)

    def test_column_dimension(self):
        """Test InputOperator.column_dimension()."""
        assert self._OpClass.column_dimension(1, 3) == 3
        assert self._OpClass.column_dimension(3, 8) == 8
        assert self._OpClass.column_dimension(5, 2) == 2


# Dependent on state and input ================================================
class TestStateInputOperator:
    """Test operators._nonparametric.StateInputOperator."""

    _OpClass = _module.StateInputOperator

    def test_str(self):
        """Test StateInputOperator._str()."""
        op = self._OpClass()
        assert op._str("q_j", "u_j") == "N[u_j ⊗ q_j]"
        assert op._str("q(t)", "u(t)") == "N[u(t) ⊗ q(t)]"

    def test_set_entries(self):
        """Test StateInputOperator.set_entries()."""
        op = self._OpClass()

        # Too many dimensions.
        Nbad = np.arange(12).reshape((2, 2, 3))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Nbad)
        assert (
            ex.value.args[0] == "StateInputOperator entries must be "
            "two-dimensional"
        )

        # Two-dimensional but invalid shape.
        Nbad = np.random.random((3, 7))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Nbad)
        assert (
            ex.value.args[0] == "invalid StateInputOperator entries dimensions"
        )

        # Correct dimensions.
        r, m = 5, 3
        N = np.random.random((r, r * m))
        op.set_entries(N)
        assert op.entries is N
        assert op.m == m

        # Special case: r = 1, m = 1 (scalar B).
        n = np.random.random()
        op.set_entries(n)
        assert op.shape == (1, 1)
        assert op[0, 0] == n
        assert op.m == 1

    def test_apply(self, k=20):
        """Test StateInputOperator.apply()/__call__()."""
        op = self._OpClass()

        def _test_single(r, m):
            N = np.random.random((r, r * m))
            op.set_entries(N)
            # Evaluation for a single vector.
            q = np.random.random(r)
            u = np.random.random(m)
            evaltrue = N @ np.kron(u, q)
            evalgot = op.apply(q, u)
            assert evalgot.shape == (r,)
            assert np.allclose(evalgot, evaltrue)
            # Vectorized evaluation.
            Q = np.random.random((r, k))
            U = np.random.random((m, k))
            evaltrue = N @ la.khatri_rao(U, Q)
            evalgot = op.apply(Q, U)
            assert evalgot.shape == (r, k)
            assert np.allclose(evalgot, evaltrue)

        _test_single(10, 20)
        _test_single(2, 6)
        _test_single(3, 3)
        _test_single(2, 1)
        _test_single(1, 4)
        _test_single(1, 1)

        # Special case: N is 1x1 and q, u are scalars.
        N = np.random.random()
        op.set_entries(N)
        # Evaluation for a single vector.
        q = np.random.random()
        u = np.random.random()
        out = op(q, u)
        assert np.isscalar(out)
        assert np.allclose(out, N * u * q)
        # Vectorized evaluation.
        Q = np.random.random(k)
        U = np.random.random(k)
        out = op(Q, U)
        assert out.shape == (k,)
        assert np.allclose(out, N * U * Q)

        # Special case: N is rxr, r>1, and u is a scalar.
        r = 10
        N = np.random.random((r, r))
        op.set_entries(N)
        # Evaluation for a single vector.
        q = np.random.random(r)
        u = np.random.random()
        out = op(q, u)
        assert out.shape == (r,)
        assert np.allclose(out, (N @ q) * u)
        # Vectorized evaluation.
        Q = np.random.random((r, k))
        U = np.random.random(k)
        out = op(Q, U)
        assert out.shape == (r, k)
        assert np.allclose(
            out, np.column_stack([N @ q * u for q, u in zip(Q.T, U)])
        )

    def test_jacobian(self, r=9, m=4, ntrials=10):
        """Test StateInputOperator.jacobian()."""
        Ns = [np.random.random((r, r)) for _ in range(m)]
        N = np.hstack(Ns)
        op = self._OpClass(N)

        with pytest.raises(ValueError) as ex:
            op.jacobian(np.random.random(r), np.random.random(m - 1))
        assert ex.value.args[0] == "invalid input_ shape"

        for _ in range(ntrials):
            q = np.random.random(r)
            u = np.random.random(m)
            jac_true = sum([Ni * uu for Ni, uu in zip(Ns, u)])
            jac = op.jacobian(q, u)
            assert jac.shape == (r, r)
            assert np.allclose(jac, jac_true)

        # Special case: r = 1, m > 1, q is a scalar.
        N = np.random.random((1, m))
        op.set_entries(N)
        for _ in range(ntrials):
            q = np.random.random()
            u = np.random.random(m)
            jac_true = N @ u
            jac = op.jacobian(q, u)
            assert jac.shape == (1, 1)
            assert np.allclose(jac, jac_true)

        # Special case: r > 1, m = 1, u is a scalar.
        N = np.random.random((r, r))
        op.set_entries(N)
        for _ in range(ntrials):
            q = np.random.random(r)
            u = np.random.random()
            jac_true = N * u
            jac = op.jacobian(q, u)
            assert jac.shape == (r, r)
            assert np.allclose(jac, jac_true)

        # Special case: r = m = 1, q and u are scalars.
        N = np.random.random()
        op.set_entries(N)
        for _ in range(ntrials):
            q = np.random.random()
            u = np.random.random()
            jac_true = N * u
            jac = op.jacobian(q, u)
            assert jac.shape == (1, 1)
            assert np.isclose(jac[0, 0], jac_true)

    def test_galerkin(self, n=10, r=4, m=3, ntrials=10):
        """Test StateInputOperator.galerkin()."""
        Vr = np.random.random((n, r))
        Wr = np.random.random((n, r))
        N = np.random.random((n, n * m))

        op = self._OpClass(N)
        op_ = op.galerkin(Vr, Wr)
        assert isinstance(op_, self._OpClass)
        assert op_.shape == (r, r * m)
        for _ in range(ntrials):
            q_ = np.random.random(r)
            u = np.random.random(m)
            full = Wr.T @ op.apply(Vr @ q_, u)
            reduced = op_.apply(q_, u)
            assert np.allclose(reduced, full)

    def test_datablock(self, m=3, k=20, r=10):
        """Test StateInputOperator.datablock()."""
        op = self._OpClass()
        state_ = np.random.random((r, k))
        input_ = np.random.random((m, k))
        rm = r * m

        block = op.datablock(state_, input_)
        assert block.shape == (rm, k)
        op.entries = np.random.random((r, rm))
        mult = op.entries @ block
        evald = op(state_, input_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

        # Special case: r = 1, m > 1
        state_ = state_[0]
        block = op.datablock(state_, input_)
        assert block.shape == (m, k)
        op.entries = np.random.random((1, m))
        mult = op.entries @ block
        evald = op(state_, input_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

        # Special case: r > 1, m = 1
        state_ = np.random.random((r, k))
        input_ = input_[0]
        block = op.datablock(state_, input_)
        assert block.shape == (r, k)
        op.entries = np.random.random((r, r))
        mult = op.entries @ block
        evald = op(state_, input_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

        # Special case: r = m = 1.
        state_ = state_[0]
        block = op.datablock(state_, input_)
        assert block.shape == (1, k)
        op.entries = np.random.random()
        mult = op.entries[0, 0] * block[0]
        evald = op(state_, input_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

    def test_column_dimension(self):
        """Test StateInputOperator.column_dimension()."""
        assert self._OpClass.column_dimension(1, 2) == 2
        assert self._OpClass.column_dimension(3, 6) == 18
        assert self._OpClass.column_dimension(5, 2) == 10