# lstsq/test_tikhonov.py
"""Tests for lstsq._tikhonov.py."""

import pytest
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

import opinf


class TestBaseTikhonovSolver:
    """Test lstsq._tikhonov._BaseTikhonovSolver."""

    class Dummy(opinf.lstsq._tikhonov._BaseTikhonovSolver):
        """Instantiable version of _BaseTikhonovSolver."""

        def predict(*args, **kwargs):
            pass

        @property
        def regularizer(self):
            return None

        def regcond(self):
            return 0

        def residual(self):
            return 0

    # Validation --------------------------------------------------------------
    def test_fit(self, k=10, d=4, r=3):
        """Test _BaseTikhonovSolver.fit()."""
        solver = self.Dummy()
        A = np.empty((k, d))
        B = np.empty((k, r))

        # Correct usage but for an underdetermined system.
        Abad = np.empty((k, k + 1))
        with pytest.warns(la.LinAlgWarning) as wn:
            solver.fit(Abad, B)
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            "non-regularized least-squares system is underdetermined!"
        )
        assert solver.k == k
        assert solver.d == k + 1
        assert solver.r == r

        # Correct usage, not underdetermined.
        assert solver.fit(A, B) is solver
        assert solver.A is A
        assert solver.B is B
        assert solver.k == k
        assert solver.d == d
        assert solver.r == r


class TestL2Solver:
    """Test lstsq._tikhonov.L2Solver."""

    SolverClass = opinf.lstsq.L2Solver

    # Properties --------------------------------------------------------------
    def test_regularizer(self, k=20, d=11, r=3):
        """Test regularizer (property and setter)."""
        # Try with nonscalar regularizer.
        with pytest.raises(TypeError) as ex:
            self.SolverClass([1, 2, 3])
        assert ex.value.args[0] == (
            "regularization hyperparameter must be a scalar"
        )

        # Negative regularization parameter not allowed.
        with pytest.raises(ValueError) as ex:
            self.SolverClass(-1)
        assert ex.value.args[0] == (
            "regularization hyperparameter must be non-negative"
        )

        regularizer = np.random.uniform()
        solver = self.SolverClass(regularizer)
        assert solver.regularizer == regularizer

    # Main methods ------------------------------------------------------------
    def test_fit(self, k=20, d=11, r=3):
        """Test fit()."""

        solver = self.SolverClass(1)
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        solver.fit(A, B)

        for attr, shape in [
            ("_V", (d, d)),
            ("_svals", (d,)),
            ("_UtB", (d, r)),
            ("A", (k, d)),
            ("B", (k, r)),
        ]:
            assert hasattr(solver, attr)
            obj = getattr(solver, attr)
            assert isinstance(obj, np.ndarray)
            assert obj.shape == shape
        Xnoreg = solver._V @ np.diag(1 / solver._svals) @ solver._UtB
        assert np.allclose(Xnoreg, la.lstsq(A, B)[0])

    def test_predict(self, m=20, n=10, k=5):
        """Test predict()."""
        solver1D = self.SolverClass(0)
        solver2D = self.SolverClass(0)
        A = np.random.random((m, n))
        B = np.random.random((m, k))
        b = B[:, 0]

        # Try predicting before fitting.
        with pytest.raises(AttributeError) as ex:
            solver1D.predict()
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Fit the solvers.
        solver1D.fit(A, b)
        solver2D.fit(A, B)

        # Test without regularization, b.ndim = 1.
        x1 = la.lstsq(A, b)[0]
        x2 = solver1D.predict()
        assert np.allclose(x1, x2)

        # Test without regularization, b.ndim = 2.
        X1 = la.lstsq(A, B)[0]
        X2 = solver2D.predict()
        assert np.allclose(X1, X2)

        # Test with regularization, b.ndim = 1.
        Apad = np.vstack((A, np.eye(n)))
        bpad = np.concatenate((b, np.zeros(n)))
        x1 = la.lstsq(Apad, bpad)[0]
        solver1D.regularizer = 1
        x2 = solver1D.predict()
        assert np.allclose(x1, x2)

        # Test with regularization, b.ndim = 2.
        Bpad = np.concatenate((B, np.zeros((n, k))))
        X1 = la.lstsq(Apad, Bpad)[0]
        solver2D.regularizer = 1
        X2 = solver2D.predict()
        assert np.allclose(X1, X2)

    # Post-processing ---------------------------------------------------------
    def test_cond(self, k=20, d=11, r=3):
        """Test cond()."""
        solver = self.SolverClass(0)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.cond()
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Contrived test 1
        A = np.eye(d)
        B = np.zeros((d, r))
        solver.fit(A, B)
        assert np.isclose(solver.cond(), 1)

        # Contrived test 2
        A = np.diag(np.arange(1, d + 1))
        B = np.zeros((d, r))
        solver.fit(A, B)
        assert np.isclose(solver.cond(), d)

        # Random test
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        solver.fit(A, B)
        assert np.isclose(solver.cond(), np.linalg.cond(A))

    def test_regcond(self, k=20, d=11, r=3, ntests=5):
        """Test regcond()."""
        solver = self.SolverClass(0)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regcond()
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        def _singletest(reg, regcondtrue):
            solver.regularizer = reg
            regcond = solver.regcond()
            assert np.isclose(regcond, regcondtrue)

        # Square, diagonal tests.
        A = np.diag(np.arange(1, d + 1))
        B = np.zeros((d, r))
        solver.fit(A, B)
        assert np.isclose(solver.regcond(), d)
        for reg in np.random.uniform(1, 10, ntests):
            regcond_true = np.sqrt((d**2 + reg**2) / (1 + reg**2))
            _singletest(reg, regcond_true)

        # Rectangular, dense tests.
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        solver.fit(A, B)
        for reg in np.random.uniform(1, 10, ntests):
            regcond_true = np.linalg.cond(np.vstack((A, reg * np.eye(d))))
            _singletest(reg, regcond_true)

    def test_residual(self, k=20, d=11, r=3, ntests=5):
        """Test residual()."""
        solver = self.SolverClass(0)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.residual(0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        solver.fit(A, B)

        # Try with badly shaped X.
        X = np.random.standard_normal((d + 1, r - 1))
        with pytest.raises(ValueError) as ex:
            solver.residual(X)
        assert ex.value.args[0] == (
            f"X.shape = {(d+1, r-1)} != {(d, r)} = (d, r)"
        )

        # Two-dimensional tests.
        X = np.random.standard_normal((d, r))
        for reg in [0] + np.random.uniform(1, 10, ntests).tolist():
            solver.regularizer = reg
            residual = solver.residual(X)
            assert isinstance(residual, np.ndarray)
            assert residual.shape == (r,)
            ans = la.norm(A @ X - B, ord=2, axis=0) ** 2
            ans += la.norm(reg * np.eye(d) @ X, ord=2, axis=0) ** 2
            assert np.allclose(residual, ans)

        # One-dimensional tests.
        b = B[:, 0]
        solver.fit(A, b)
        assert solver.r == 1
        x = np.random.standard_normal(d)
        for reg in [0] + np.random.uniform(0, 10, ntests).tolist():
            solver.regularizer = reg
            residual = solver.residual(x)
            assert isinstance(residual, float)
            ans = np.linalg.norm(A @ x - b) ** 2 + np.linalg.norm(reg * x) ** 2
            assert np.isclose(residual, ans)


class TestL2SolverDecoupled:
    """Test lstsq._tikhonov.L2SolverDecoupled."""

    SolverClass = opinf.lstsq.L2SolverDecoupled

    # Properties --------------------------------------------------------------
    def test_regularizer(self, k=10, d=6, r=3):
        """Test _check_regularizer_shape(), fit(), and regularizer property."""
        solver = self.SolverClass(np.random.random(r + 1))
        A = np.empty((k, d))
        B = np.empty((k, r))

        with pytest.raises(ValueError) as ex:
            solver.fit(A, B)
        assert ex.value.args[0] == "len(regularizer) != number of columns of B"

        solver.regularizer = np.random.random(r)
        solver.fit(A, B)
        assert solver.r == r

    # Main methods ------------------------------------------------------------
    def test_predict(self, k=20, d=10):
        """Test predict()."""
        regularizers = np.array([0, 1, 3, 5])
        r = len(regularizers)
        A = np.random.random((k, d))
        B = np.random.random((k, r))
        solver = self.SolverClass(regularizers).fit(A, B)

        Id = np.eye(d)
        Apads = [np.vstack((A, reg * Id)) for reg in regularizers]
        Bpad = np.vstack((B, np.zeros((d, r))))
        X1 = np.column_stack(
            [la.lstsq(Apad, Bpad[:, j])[0] for j, Apad in enumerate(Apads)]
        )
        X2 = solver.predict()
        assert np.allclose(X1, X2)

    # Post-processing ---------------------------------------------------------
    def test_regcond(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.L2SolverDecoupled.regcond()."""
        solver = self.SolverClass([0] * r)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regcond()
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Square, diagonal tests.
        A = np.diag(np.arange(1, d + 1))
        B = np.zeros((d, r))
        solver.fit(A, B)
        assert np.allclose(solver.regcond(), [d] * r)
        reg = np.random.uniform(1, 10, r)
        solver.regularizer = reg
        assert np.allclose(
            solver.regcond(), np.sqrt((d**2 + reg**2) / (1 + reg**2))
        )

        # Rectangular, dense tests.
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        regs = np.random.uniform(1, 10, r)
        solver.regularizer = regs
        solver.fit(A, B)
        Id = np.eye(d)
        conds = [np.linalg.cond(np.vstack((A, reg * Id))) for reg in regs]
        assert np.allclose(solver.regcond(), conds)

    def test_residual(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.L2SolverDecoupled.residual()."""
        solver = self.SolverClass([0] * r)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.residual(0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        solver.fit(A, B)

        # Try with badly shaped X.
        X = np.random.standard_normal((d + 1, r - 1))
        with pytest.raises(ValueError) as ex:
            solver.residual(X)
        assert ex.value.args[0] == (
            f"X.shape = {(d+1, r-1)} != {(d, r)} = (d, r)"
        )

        # Correct usage.
        X = np.random.standard_normal((d, r))
        ls = np.array([0] + np.random.uniform(1, 10, r - 1).tolist())
        solver.regularizer = ls
        residual = solver.residual(X)
        assert isinstance(residual, np.ndarray)
        assert residual.shape == (r,)
        ans = la.norm(A @ X - B, ord=2, axis=0) ** 2
        ans += np.array(
            [la.norm(l * X[:, j], ord=2) ** 2 for j, l in enumerate(ls)]
        )
        assert np.allclose(residual, ans)


class TestTikhonovSolver:
    """Test lstsq._tikhonov.TikhonovSolver."""

    SolverClass = opinf.lstsq.TikhonovSolver

    # Properties --------------------------------------------------------------
    def test_regularizer(self, k=20, d=11, r=3):
        """Test _check_regularizer_shape(), regularizer, and method."""
        Z = np.random.random((d, d))

        with pytest.raises(ValueError) as ex:
            self.SolverClass(Z, method="invalidmethodoption")
        assert ex.value.args[0] == "method must be 'svd' or 'normal'"

        Zdiag = np.diag(Z)
        solver = self.SolverClass(Z, method="normal")
        solver.regularizer = sparse.diags(Zdiag)
        assert isinstance(solver.regularizer, np.ndarray)
        assert np.allclose(solver.regularizer, np.diag(Zdiag))
        solver.regularizer = Zdiag
        assert isinstance(solver.regularizer, np.ndarray)
        assert np.all(solver.regularizer == np.diag(Zdiag))

        P = [1] * d
        P[-1] = -1
        with pytest.raises(ValueError) as ex:
            solver.regularizer = P
        assert ex.value.args[0] == (
            "diagonal regularizer must be positive semi-definite"
        )

        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        solver.fit(A, B)

        # Try with bad regularizer shapes.
        P = np.empty((d - 1, d - 1))
        with pytest.raises(ValueError) as ex:
            solver.regularizer = P
        assert ex.value.args[0] == (
            "regularizer.shape != (d, d) (d = A.shape[1])"
        )

        # Correct usage
        solver.regularizer = np.full(d, 2)
        assert solver.regularizer.shape == (d, d)
        assert np.all(solver.regularizer == 2 * np.eye(d))

    # Main methods ------------------------------------------------------------
    def test_fit(self, k=20, d=10, r=5):
        """Test fit()."""
        Z = np.zeros((d, d))
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        solver = self.SolverClass(Z).fit(A, B)

        for attr, shape in [
            ("A", (k, d)),
            ("B", (k, r)),
            ("_Bpad", (k + d, r)),
            ("_AtA", (d, d)),
            ("_rhs", (d, r)),
        ]:
            assert hasattr(solver, attr)
            obj = getattr(solver, attr)
            assert isinstance(obj, np.ndarray)
            assert obj.shape == shape

        lstsq_noreg = la.lstsq(A, B)[0]
        lstsq_pred = la.solve(solver._AtA, solver._rhs, assume_a="sym")
        assert np.allclose(lstsq_pred, lstsq_noreg)

    def test_predict(self, k=40, d=15, r=5):
        """Test predict()."""
        A = np.random.random((k, d))
        B = np.random.random((k, r))
        Bpad = np.concatenate((B, np.zeros((d, r))))
        b = B[:, 0]
        bpad = Bpad[:, 0]
        Z = np.zeros((d, d))
        Id = np.eye(d)
        solver1D = self.SolverClass(Z).fit(A, b)
        solver2D = self.SolverClass(Z).fit(A, B)

        for method in ["normal", "svd"]:
            # Test without regularization, b.ndim = 1.
            solver1D.method = method
            solver1D.regularizer = Z
            x1 = la.lstsq(A, b)[0]
            x2 = solver1D.predict()
            assert np.allclose(x1, x2)

            # Test without regularization, b.ndim = 2.
            solver2D.method = method
            solver2D.regularizer = Z
            X1 = la.lstsq(A, B)[0]
            X2 = solver2D.predict()
            assert np.allclose(X1, X2)

            # Test with regularization, b.ndim = 1.
            solver1D.regularizer = Id
            Apad = np.vstack((A, Id))
            x1 = la.lstsq(Apad, bpad)[0]
            x2 = solver1D.predict()
            assert np.allclose(x1, x2)

            # Test with regularization, b.ndim = 2.
            solver2D.regularizer = Id
            X1 = la.lstsq(Apad, Bpad)[0]
            X2 = solver2D.predict()
            assert np.allclose(X1, X2)

        # Test SVD method with a severely ill-conditioned system.
        A = np.random.standard_normal((k, d))
        U, s, Vt = la.svd(A, full_matrices=False)
        s[-5:] = 1e-18
        s /= np.arange(1, s.size + 1) ** 2
        A = U @ np.diag(s) @ Vt
        B = np.random.standard_normal((k, r))
        assert np.linalg.cond(A) > 1e15

        # No regularization.
        solver2D = self.SolverClass(Z, method="svd").fit(A, B)
        X1 = la.lstsq(A, B)[0]
        X2 = solver2D.predict()
        assert np.allclose(X1, X2)

        # Some regularization.
        solver2D.regularizer = Id
        Apad = np.vstack((A, Id))
        Bpad = np.concatenate((B, np.zeros((d, r))))
        X1 = la.lstsq(Apad, Bpad)[0]
        X2 = solver2D.predict()
        assert np.allclose(X1, X2)

    # Post-processing ---------------------------------------------------------
    def test_regcond(self, k=20, d=11, r=3, ntests=5):
        """Test regcond()."""
        Z = np.zeros(d)
        Id = np.eye(d)
        solver = self.SolverClass(Z)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regcond()
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Square, diagonal tests.
        A = np.diag(np.arange(1, d + 1))
        B = np.zeros((d, r))
        solver.fit(A, B)
        assert np.isclose(solver.regcond(), d)
        regs = np.random.uniform(1, 10, d)
        regs.sort()
        solver.regularizer = regs
        assert np.isclose(
            solver.regcond(),
            np.sqrt((d**2 + regs[-1] ** 2) / (1 + regs[0] ** 2)),
        )

        # Rectangular, dense tests.
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        solver.fit(A, B)
        for reg in np.random.uniform(1, 10, ntests):
            solver.regularizer = reg * Id
            cond = np.linalg.cond(np.vstack((A, solver.regularizer)))
            assert np.isclose(solver.regcond(), cond)

            P = np.random.standard_normal((d, d))
            cond = np.linalg.cond(np.vstack((A, P)))
            solver.regularizer = P
            assert np.isclose(solver.regcond(), cond)

    def test_residual(self, k=20, d=11, r=3, ntests=5):
        """Test lstsq._tikhonov.TikhonovSolver.residual()."""
        Z = np.zeros(d)
        solver = self.SolverClass(Z)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.residual(0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        solver.fit(A, B)

        # Two-dimensional tests.
        X = np.random.standard_normal((d, r))
        for _ in range(ntests):
            P = np.random.uniform(1, 10, d)
            solver.regularizer = P
            residual = solver.residual(X)
            assert isinstance(residual, np.ndarray)
            assert residual.shape == (r,)
            ans = la.norm(A @ X - B, ord=2, axis=0) ** 2
            ans += la.norm(np.diag(P) @ X, ord=2, axis=0) ** 2
            assert np.allclose(residual, ans)

        # One-dimensional tests.
        b = B[:, 0]
        solver.fit(A, b)
        assert solver.r == 1
        x = np.random.standard_normal(d)
        for _ in range(ntests):
            P = np.random.uniform(1, 10, d)
            solver.regularizer = P
            residual = solver.residual(x)
            assert isinstance(residual, float)
            ans = np.linalg.norm(A @ x - b) ** 2 + np.linalg.norm(P * x) ** 2
            assert np.isclose(residual, ans)


class TestTikhonovSolverDecoupled:
    """Test lstsq._tikhonov.TikhonovSolverDecoupled."""

    SolverClass = opinf.lstsq.TikhonovSolverDecoupled

    # Properties --------------------------------------------------------------
    def test_check_Ps(self, k=10, d=6, r=3):
        """Test _check_regularizer_shape() and regularizer."""
        Z = np.random.random((d, d))
        solver = opinf.lstsq.TikhonovSolverDecoupled([Z] * r)
        A = np.empty((k, d))
        B = np.empty((k, r))
        solver.fit(A, B)
        assert solver.r == r

        Zs = [Z] * r
        with pytest.raises(ValueError) as ex:
            solver.regularizer = Zs[:-1]
        assert ex.value.args[0] == "len(regularizer) != r"

        Zs[-1] = np.random.random((d + 1, d + 1))
        with pytest.raises(ValueError) as ex:
            solver.regularizer = Zs
        assert ex.value.args[0] == f"regularizer[{r-1:d}].shape != (d, d)"

        with pytest.raises(ValueError) as ex:
            solver.regularizer = [[-1] * d] * r
        assert ex.value.args[0] == (
            "diagonal regularizer must be positive semi-definite"
        )

        Zs[-1] = sparse.diags(np.ones(d))
        solver.regularizer = Zs
        solver.regularizer = [[i] * d for i in range(1, r + 1)]
        assert np.all(solver.regularizer[0] == np.eye(d))

    # Main methods ------------------------------------------------------------
    def test_predict(self, k=20, d=10):
        """Test lstsq._tikhonov.TikhonovSolverDecoupled.predict()."""
        Z = np.zeros(d)
        Ps = [np.eye(d), np.full(d, 2)]
        r = len(Ps)
        A = np.random.random((k, d))
        B = np.random.random((k, r))
        solver = self.SolverClass(Ps)

        # Try predicting before fitting.
        with pytest.raises(AttributeError) as ex:
            solver.predict()
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"
        solver.fit(A, B)

        Apad1 = np.vstack((A, Ps[0]))
        Apad2 = np.vstack((A, np.diag(Ps[1])))
        Bpad = np.vstack((B, np.zeros((d, 2))))
        xx1 = la.lstsq(Apad1, Bpad[:, 0])[0]
        xx2 = la.lstsq(Apad2, Bpad[:, 1])[0]
        X1 = np.column_stack([xx1, xx2])
        X2 = solver.predict()
        assert np.allclose(X1, X2)

        # Test with a severely ill-conditioned system.
        A = np.random.standard_normal((k, d))
        U, s, Vt = la.svd(A, full_matrices=False)
        s[-5:] = 1e-18
        s /= np.arange(1, s.size + 1) ** 2
        A = U @ np.diag(s) @ Vt
        B = np.random.standard_normal((k, r))
        assert np.linalg.cond(A) > 1e15

        # No regularization.
        solver = self.SolverClass([Z] * r).fit(A, B)
        Z = np.zeros(d)
        X1 = la.lstsq(A, B)[0]
        X2 = solver.predict()
        assert np.allclose(X1, X2)

        # Some regularization.
        for method in "svd", "normal":
            solver.method = method
            solver.regularizer = Ps[:2]
            Apad1 = np.vstack((A, Ps[0]))
            Apad2 = np.vstack((A, np.diag(Ps[1])))
            Bpad = np.vstack((B, np.zeros((d, 2))))
            xx1 = la.lstsq(Apad1, Bpad[:, 0])[0]
            xx2 = la.lstsq(Apad2, Bpad[:, 1])[0]
            X1 = np.column_stack([xx1, xx2])
            X2 = solver.predict()
            assert np.allclose(X1, X2)

    # Post-processing ---------------------------------------------------------
    def test_regcond(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.TikhonovSolverDecoupled.regcond()."""

        # Square, diagonal tests.
        A = np.diag(np.arange(1, d + 1))
        B = np.zeros((d, r))
        z = np.zeros(d)
        solver = self.SolverClass([z] * r).fit(A, B)
        assert np.allclose(solver.regcond(), [d] * r)

        regularizer = np.random.uniform(1, 10, r)
        solver.regularizer = [lm * np.ones(d) for lm in regularizer]
        true_val = np.sqrt((d**2 + regularizer**2) / (1 + regularizer**2))
        assert np.allclose(solver.regcond(), true_val)

        # Rectangular, dense tests.
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        Ps = np.random.standard_normal((r, d, d))
        solver = self.SolverClass(Ps).fit(A, B)
        conds = [np.linalg.cond(np.vstack((A, P))) for P in Ps]
        assert np.allclose(solver.regcond(), conds)

    def test_residual(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.TikhonovSolverDecoupled.residual()."""
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        X = np.random.standard_normal((d, r))
        Ps = np.random.standard_normal((r, d, d))
        solver = self.SolverClass(Ps).fit(A, B)
        residual = solver.residual(X)
        assert isinstance(residual, np.ndarray)
        assert residual.shape == (r,)
        ans = la.norm(A @ X - B, ord=2, axis=0) ** 2
        ans += np.array(
            [la.norm(P @ X[:, j], ord=2) ** 2 for j, P in enumerate(Ps)]
        )
        assert np.allclose(residual, ans)
