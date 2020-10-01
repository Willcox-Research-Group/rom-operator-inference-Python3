# lstsq/test_tikhonov.py
"""Tests for rom_operator_inference.lstsq._tikhonov.py."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi


class TestBaseLstsqSolver:
    """Test lstsq._tikhonov._BaseLstsqSolver."""
    def test_check_shapes(self):
        """Test lstsq._tikhonov._BaseLstsqSolver._check_shapes()."""
        solver = roi.lstsq._tikhonov._BaseLstsqSolver()

        # Try with rhs with too many dimensions.
        b = np.empty((2,2,2))
        A = np.empty((2,2))
        with pytest.raises(ValueError) as ex:
            solver._check_shapes(A, b)
        assert ex.value.args[0] == "`b` must be one- or two-dimensional"

        # Try with misaligned inputs.
        b = np.empty(3)
        with pytest.raises(ValueError) as ex:
            solver._check_shapes(A, b)
        assert ex.value.args[0] == \
            "inputs not aligned: A.shape[0] != b.shape[0]"

        # Correct usage, underdetermined, ignoring warnings.
        b = np.empty(2)
        A = np.empty((2,4))
        solver._check_shapes(A, b)

        # Correct usage, not underdetermined.
        A = np.empty((2,2))
        solver._check_shapes(A, b)


class TestLstsqSolverL2:
    """Test lstsq._tikhonov.LstsqSolverL2."""
    def test_fit(self, m=20, n=10, k=5):
        """Test lstsq._tikhonov.LstsqSolverL2.fit()."""
        solver = roi.lstsq.LstsqSolverL2(compute_extras=True)

        def _test_shapes(A, b, shapes):
            solver.fit(A, b)
            for attr, shape in zip(["_V", "_s", "_Utb", "_A", "_b"], shapes):
                assert hasattr(solver, attr)
                obj = getattr(solver, attr)
                assert isinstance(obj, np.ndarray)
                assert obj.shape == shape

        # Test overdetermined, b.ndim = 1.
        A = np.random.random((m,n))
        b = np.random.random(m)
        _test_shapes(A, b, [(n,n), (n,), (n,), (m,n), (m,)])

        # Test overdetermined, b.ndim = 2.
        b = np.random.random((m,k))
        _test_shapes(A, b, [(n,n), (n,), (n,k), (m,n), (m,k)])

        # Test underdetermined, b.ndim = 1.
        m,n = n,m
        A = A.T
        b = np.random.random(m)
        _test_shapes(A, b, [(n,m), (m,), (m,), (m,n), (m,)])

        # Test underdetermined, b.ndim = 2.
        b = np.random.random((m,k))
        _test_shapes(A, b, [(n,m), (m,), (m,k), (m,n), (m,k)])

    def test_predict(self, m=20, n=10, k=5):
        """Test lstsq._tikhonov.LstsqSolverL2.predict()."""
        solver1D = roi.lstsq.LstsqSolverL2(compute_extras=True)
        solver2D = roi.lstsq.LstsqSolverL2(compute_extras=True)
        A = np.random.random((m,n))
        B = np.random.random((m,k))
        b = B[:,0]
        solver1D.fit(A, b)
        solver2D.fit(A, B)

        # Try with nonscalar regularizer.
        with pytest.raises(ValueError) as ex:
            solver1D.predict([1, 2, 3])
        assert ex.value.args[0] == "regularization parameter must be a scalar"

        # Negative regularization parameter not allowed.
        with pytest.raises(ValueError) as ex:
            solver2D.predict(-1)
        assert ex.value.args[0] == \
            "regularization parameter must be nonnegative"

        # Test without regularization, b.ndim = 1.
        x1 = la.lstsq(A, b)[0]
        x2, misfit, residual, cond, regcond = solver1D.predict(0)
        assert np.allclose(x1, x2)
        assert np.isclose(misfit, residual)
        assert np.allclose(misfit, la.norm(A @ x1 - b, ord=2)**2)
        assert np.isclose(cond, regcond)
        assert np.isclose(cond, np.linalg.cond(A))

        # Test without regularization, b.ndim = 2.
        X1 = la.lstsq(A, B)[0]
        X2, misfit, residual, cond, regcond = solver2D.predict(0)
        assert np.allclose(X1, X2)
        assert np.isclose(misfit, residual)
        assert np.allclose(misfit, la.norm(A @ X1 - B, ord='fro')**2)
        assert np.isclose(cond, regcond)
        assert np.isclose(cond, np.linalg.cond(A))

        # Test with regularization, b.ndim = 1.
        Apad = np.vstack((A, np.eye(n)))
        bpad = np.concatenate((b, np.zeros(n)))
        x1 = la.lstsq(Apad, bpad)[0]
        x2, misfit, residual, cond, regcond = solver1D.predict(1)
        assert np.allclose(x1, x2)
        assert misfit < residual
        assert np.allclose(misfit, la.norm(A @ x1 - b, ord=2)**2)
        assert np.allclose(residual, misfit + la.norm(x1, ord=2)**2)
        assert cond > regcond
        assert np.isclose(cond, np.linalg.cond(A))

        # Test with regularization, b.ndim = 2.
        Bpad = np.concatenate((B, np.zeros((n, k))))
        X1 = la.lstsq(Apad, Bpad)[0]
        X2, misfit, residual, cond, regcond = solver2D.predict(1)
        assert np.allclose(X1, X2)
        assert misfit < residual
        assert np.allclose(misfit, la.norm(A @ X1 - B, ord='fro')**2)
        assert np.allclose(residual, misfit + la.norm(X1, ord='fro')**2)
        assert cond > regcond
        assert np.isclose(cond, np.linalg.cond(A))

        # Test with underdetermined system, no regularization.
        m,n = n,m
        A = A.T
        b = np.random.random(m)
        x1 = la.lstsq(A, b)[0]
        solver1D.fit(A, b)
        with pytest.warns(la.LinAlgWarning) as wn:
            x2, misfit, residual, cond, regcond = solver1D.predict(0)
        assert len(wn) == 1
        assert wn[0].message.args[0] == \
            "least-squares system is underdetermined " \
            "(will compute minimum-norm solution)"
        assert np.allclose(x1, x2)
        assert np.isclose(misfit, residual)
        assert np.allclose(misfit, la.norm(A @ x1 - b, ord=2)**2)
        assert np.isclose(cond, regcond)
        assert np.isclose(cond, np.linalg.cond(A))

        # Test with underdetermined system and regularization.
        Apad = np.vstack((A, np.eye(n)))
        bpad = np.concatenate((b, np.zeros(n)))
        x1 = la.lstsq(Apad, bpad)[0]
        x2, misfit, residual, cond, regcond = solver1D.predict(1)
        assert np.allclose(x1, x2)
        assert misfit < residual
        assert np.allclose(misfit, la.norm(A @ x1 - b, ord=2)**2)
        assert np.isclose(cond, np.linalg.cond(A))

        solver1D.compute_extras = False
        x2 = solver1D.predict(1)


class TestLstsqSolverTikhonov:
    """Test lstsq._tikhonov.LstsqSolverTikhonov."""
    def test_fit(self, m=20, n=10, k=5):
        """Test lstsq._tikhonov.LstsqSolverTikhonov.fit()."""
        solver = roi.lstsq.LstsqSolverTikhonov(compute_extras=True)

        def _test_shapes(A, b, shapes):
            solver.fit(A, b)
            for attr, shape in zip(["_rhs", "_A", "_b"], shapes):
                assert hasattr(solver, attr)
                obj = getattr(solver, attr)
                assert isinstance(obj, np.ndarray)
                assert obj.shape == shape

        # Test overdetermined, b.ndim = 1.
        A = np.random.random((m,n))
        b = np.random.random(m)
        _test_shapes(A, b, [(m+n,), (m,n), (m,)])

        # Test overdetermined, b.ndim = 2.
        b = np.random.random((m,k))
        _test_shapes(A, b, [(m+n,k), (m,n), (m,k)])

        # Test underdetermined, b.ndim = 1.
        m,n = n,m
        A = A.T
        b = np.random.random(m)
        _test_shapes(A, b, [(m+n,), (m,n), (m,)])

        # Test underdetermined, b.ndim = 2.
        b = np.random.random((m,k))
        _test_shapes(A, b, [(m+n,k), (m,n), (m,k)])

    def test_predict(self, m=20, n=10, k=5):
        """Test lstsq._tikhonov.LstsqSolverTikhonov.predict()."""
        solver1D = roi.lstsq.LstsqSolverTikhonov(compute_extras=True)
        solver2D = roi.lstsq.LstsqSolverTikhonov(compute_extras=True)
        A = np.random.random((m,n))
        B = np.random.random((m,k))
        b = B[:,0]
        solver1D.fit(A, b)
        solver2D.fit(A, B)

        # Try with bad regularizer type.
        with pytest.raises(ValueError) as ex:
            solver1D.predict([1, 2, 3])
        assert ex.value.args[0] == \
            "regularization matrix must be a NumPy array"

        # Try with bad regularizer shape.
        P = np.empty((n-1,n+1))
        with pytest.raises(ValueError) as ex:
            solver2D.predict(P)
        assert ex.value.args[0] == "P.shape != (d,d) where d = A.shape[1]"

        # Test without regularization, b.ndim = 1.
        Z = np.zeros((n,n))
        x1 = la.lstsq(A, b)[0]
        x2, misfit, residual, cond, regcond = solver1D.predict(Z)
        assert np.allclose(x1, x2)
        assert np.isclose(misfit, residual)
        assert np.allclose(misfit, la.norm(A @ x1 - b, ord=2)**2)
        assert np.isclose(cond, regcond)
        assert np.isclose(cond, np.linalg.cond(A))

        # Test without regularization, b.ndim = 2.
        X1 = la.lstsq(A, B)[0]
        X2, misfit, residual, cond, regcond = solver2D.predict(Z)
        assert np.allclose(X1, X2)
        assert np.isclose(misfit, residual)
        assert np.allclose(misfit, la.norm(A @ X1 - B, ord='fro')**2)
        assert np.isclose(cond, regcond)
        assert np.isclose(cond, np.linalg.cond(A))

        # Test with regularization, b.ndim = 1.
        I = np.eye(n)
        Apad = np.vstack((A, I))
        bpad = np.concatenate((b, np.zeros(n)))
        x1 = la.lstsq(Apad, bpad)[0]
        x2, misfit, residual, cond, regcond = solver1D.predict(I)
        assert np.allclose(x1, x2)
        assert misfit < residual
        assert np.allclose(misfit, la.norm(A @ x1 - b, ord=2)**2)
        assert np.allclose(residual, misfit + la.norm(x1, ord=2)**2)
        assert cond > regcond
        assert np.isclose(cond, np.linalg.cond(A))

        # Test with regularization, b.ndim = 2.
        Bpad = np.concatenate((B, np.zeros((n, k))))
        X1 = la.lstsq(Apad, Bpad)[0]
        X2, misfit, residual, cond, regcond = solver2D.predict(I)
        assert np.allclose(X1, X2)
        assert misfit < residual
        assert np.allclose(misfit, la.norm(A @ X1 - B, ord='fro')**2)
        assert np.allclose(residual, misfit + la.norm(X1, ord='fro')**2)
        assert cond > regcond
        assert np.isclose(cond, np.linalg.cond(A))

        # Test with underdetermined system and regularization.
        I = np.eye(n)
        Apad = np.vstack((A, I))
        bpad = np.concatenate((b, np.zeros(n)))
        x1 = la.lstsq(Apad, bpad)[0]
        x2, misfit, residual, cond, regcond = solver1D.predict(I)
        assert np.allclose(x1, x2)
        assert misfit < residual
        assert np.allclose(misfit, la.norm(A @ x1 - b, ord=2)**2)
        assert np.isclose(cond, np.linalg.cond(A))

        solver1D.compute_extras = False
        x2 = solver1D.predict(I)


def test_lstsq_reg(m=20, n=10, k=5):
    """Test lstsq._tikhonov.lstsq_reg()."""
    A = np.random.random((m,n))
    B = np.random.random((m,k))
    Ps = [np.random.random((n,n)) for _ in range(k)]

    with pytest.raises(ValueError) as ex:
        roi.lstsq.lstsq_reg(A, B[:,0], Ps)
    assert ex.value.args[0] == "`b` must be two-dimensional with multiple P"

    # Bad number of regularization matrices (list).
    with pytest.raises(ValueError) as ex:
        roi.lstsq.lstsq_reg(A, B, Ps[:3])
    assert ex.value.args[0] == \
        "multiple P requires exactly r entries with r = number of columns of b"

    # Bad number of regularization matrices (generator).
    with pytest.raises(ValueError) as ex:
        roi.lstsq.lstsq_reg(A, B, (np.random.random((n,n)) for _ in range(3)))
    assert ex.value.args[0] == \
        "multiple P requires exactly r entries with r = number of columns of b"

    # Bad type for regularization.
    with pytest.raises(ValueError) as ex:
        roi.lstsq.lstsq_reg(A, B, {})
    assert ex.value.args[0] == "invalid input P of type 'dict'"

    # Correct usage.
    roi.lstsq.lstsq_reg(A, B, 0)
    roi.lstsq.lstsq_reg(A, B, 1)
    roi.lstsq.lstsq_reg(A, B, Ps[0])
    roi.lstsq.lstsq_reg(A, B, Ps)

    # TODO: output shape tests?
