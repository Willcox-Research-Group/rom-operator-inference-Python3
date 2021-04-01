# lstsq/test_tikhonov.py
"""Tests for rom_operator_inference.lstsq._tikhonov.py."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as opinf


class TestBaseSolver:
    """Test lstsq._tikhonov._BaseSolver."""
    # Properties --------------------------------------------------------------
    def test_properties(self, k=10, d=4, r=3):
        """Test lstsq._tikhonov._BaseSolver properties A, B, k, d, and r."""
        solver = opinf.lstsq._tikhonov._BaseSolver()
        for attr in "ABkdr":
            assert hasattr(solver, attr)
            assert getattr(solver, attr) is None

        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        solver._BaseSolver__A = A
        solver._BaseSolver__B = B
        assert solver.A is A
        assert solver.B is B
        assert solver.k == k
        assert solver.d == d
        assert solver.r == r
        solver._BaseSolver__B = np.empty((k,1))
        assert solver.r == 1

        for attr in "AB":
            with pytest.raises(AttributeError) as ex:
                setattr(solver, attr, A)
            assert ex.value.args[0] == "can't set attribute (call fit())"

        for attr in "kdr":
            with pytest.raises(AttributeError) as ex:
                setattr(solver, attr, 10)
            assert ex.value.args[0] == "can't set attribute"

    # Validation --------------------------------------------------------------
    def test_process_fit_arguments(self, k=10, d=4, r=3):
        """Test lstsq._tikhonov._BaseSolver._process_fit_arguments()."""
        solver = opinf.lstsq._tikhonov._BaseSolver()
        A = np.empty((k,d))
        B = np.empty((k,r))

        # Correct usage but for an underdetermined system.
        Abad = np.empty((k, k+1))
        with pytest.warns(la.LinAlgWarning) as wn:
            solver._process_fit_arguments(Abad, B)
        assert len(wn) == 1
        assert wn[0].message.args[0] == \
            "original least-squares system is underdetermined!"
        assert solver.k == k
        assert solver.d == k+1
        assert solver.r == r

        # Try with rhs with too many dimensions.
        Bbad = np.empty((k,r,r))
        with pytest.raises(ValueError) as ex:
            solver._process_fit_arguments(A, Bbad)
        assert ex.value.args[0] == "`B` must be one- or two-dimensional"

        # Try with misaligned inputs.
        Bbad = np.empty((k+1,r))
        with pytest.raises(ValueError) as ex:
            solver._process_fit_arguments(A, Bbad)
        assert ex.value.args[0] == \
            "inputs not aligned: A.shape[0] != B.shape[0]"

        # Correct usage, not underdetermined.
        solver._process_fit_arguments(A, B)
        assert solver.A is A
        assert solver.B is B
        assert solver.k == k
        assert solver.d == d
        assert solver.r == r

        # Check one-dimensional B edge case.
        solver._process_fit_arguments(A, B[:,0])
        assert solver.A is A
        assert solver.B.shape == (k,1)
        assert solver.k == k
        assert solver.d == d
        assert solver.r == 1

    def test_check_is_trained(self, k=10, d=4, r=3):
        """Test lstsq._tikhonov._BaseSolver._check_is_trained()."""
        solver = opinf.lstsq._tikhonov._BaseSolver()

        # Try before calling _process_fit_arguments().
        with pytest.raises(AttributeError) as ex:
            solver._check_is_trained()
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        A = np.empty((k,d))
        B = np.empty((k,r))
        solver._process_fit_arguments(A, B)

        # Try after calling _process_fit_arguments() but with a missing attr.
        with pytest.raises(AttributeError) as ex:
            solver._check_is_trained("_V")
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Correct usage.
        solver._check_is_trained()

    # Main methods ------------------------------------------------------------
    def test_fit(self):
        """Test lstsq._tikhonov._BaseSolver.fit()."""
        solver = opinf.lstsq._tikhonov._BaseSolver()
        with pytest.raises(NotImplementedError) as ex:
            solver.fit()
        assert ex.value.args[0] == "fit() implemented by child classes"

        with pytest.raises(NotImplementedError) as ex:
            solver.fit(1, 2, 3, 4, 5, 6, 7, a=8)
        assert ex.value.args[0] == "fit() implemented by child classes"

    def test_predict(self):
        """Test lstsq._tikhonov._BaseSolver.fit()."""
        solver = opinf.lstsq._tikhonov._BaseSolver()
        with pytest.raises(NotImplementedError) as ex:
            solver.predict()
        assert ex.value.args[0] == "predict() implemented by child classes"

        with pytest.raises(NotImplementedError) as ex:
            solver.predict(1, 2, 3, 4, 5, 6, 7, a=8)
        assert ex.value.args[0] == "predict() implemented by child classes"

    # Post-processing ---------------------------------------------------------
    def test_cond(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov._BaseSolver.cond()."""
        solver = opinf.lstsq._tikhonov._BaseSolver()

        # Try before calling _process_fit_arguments().
        with pytest.raises(AttributeError) as ex:
            solver.cond()
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Contrived test 1
        A = np.eye(d)
        B = np.zeros((d,r))
        solver._process_fit_arguments(A, B)
        assert np.isclose(solver.cond(), 1)

        # Contrived test 2
        A = np.diag(np.arange(1,d+1))
        B = np.zeros((d,r))
        solver._process_fit_arguments(A, B)
        assert np.isclose(solver.cond(), d)

        # Random test
        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        svals = la.svdvals(A)
        solver._process_fit_arguments(A, B)
        assert np.isclose(solver.cond(), svals[0] / svals[-1])

    def test_misfit(self, k=20, d=10, r=4):
        """Test lstsq._tikhonov._BaseSolver.misfit()."""
        solver = opinf.lstsq._tikhonov._BaseSolver()

        # Try before calling _process_fit_arguments().
        with pytest.raises(AttributeError) as ex:
            solver.misfit(0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        solver._process_fit_arguments(A, B)

        # Try with badly shaped X.
        X = np.random.standard_normal((d+1,r-1))
        with pytest.raises(ValueError) as ex:
            solver.misfit(X)
        assert ex.value.args[0] == f"X.shape = {(d+1,r-1)} != {(d,r)} = (d,r)"

        # Two-dimensional case.
        X = np.random.standard_normal((d,r))
        misfit = solver.misfit(X)
        assert isinstance(misfit, np.ndarray)
        assert misfit.shape == (r,)
        assert np.allclose(misfit, la.norm(A @ X - B, ord=2, axis=0)**2)

        # One-dimensional case.
        b = B[:,0]
        solver._process_fit_arguments(A, b)
        assert solver.r == 1
        x = np.random.standard_normal(d)
        misfit = solver.misfit(x)
        assert isinstance(misfit, float)
        assert np.isclose(misfit, np.linalg.norm(A @ x - b)**2)


class TestSolverL2:
    """Test lstsq._tikhonov.SolverL2."""
    # Validation --------------------------------------------------------------
    def test_process_regularizer(self, k=20, d=11, r=3):
        solver = opinf.lstsq.SolverL2()

        # Try with nonscalar regularizer.
        with pytest.raises(TypeError) as ex:
            solver._process_regularizer([1, 2, 3])
        assert ex.value.args[0] == \
            "regularization hyperparameter λ must be a scalar"

        # Negative regularization parameter not allowed.
        with pytest.raises(ValueError) as ex:
            solver._process_regularizer(-1)
        assert ex.value.args[0] == \
            "regularization hyperparameter λ must be non-negative"

        λ = np.random.uniform()
        assert solver._process_regularizer(λ) == λ**2

    # Helper methods ----------------------------------------------------------
    def test_Σinv(self, d=10, ntests=5):
        """Test lstsq._tikhonov.SolverL2._Σinv()"""
        solver = opinf.lstsq.SolverL2()
        Σ = np.random.standard_normal(d)
        solver._Σ = Σ
        for λ in [0] + np.random.uniform(1, 10, ntests).tolist():
            assert np.allclose(solver._Σinv(λ), Σ / (Σ**2 + λ**2))

    # Main methods ------------------------------------------------------------
    def test_fit(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.SolverL2.fit()."""
        solver = opinf.lstsq.SolverL2()
        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))

        solver.fit(A, B)
        for attr, shape in [("_V", (d,d)), ("_Σ", (d,)),
                            ("_UtB", (d,r)), ("A", (k,d)), ("B", (k,r))]:
            assert hasattr(solver, attr)
            obj = getattr(solver, attr)
            assert isinstance(obj, np.ndarray)
            assert obj.shape == shape

    def test_predict(self, m=20, n=10, k=5):
        """Test lstsq._tikhonov.SolverL2.predict()."""
        solver1D = opinf.lstsq.SolverL2()
        solver2D = opinf.lstsq.SolverL2()
        A = np.random.random((m,n))
        B = np.random.random((m,k))
        b = B[:,0]

        # Try predicting before fitting.
        with pytest.raises(AttributeError) as ex:
            solver1D.predict(0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Fit the solvers.
        solver1D.fit(A, b)
        solver2D.fit(A, B)

        # Test without regularization, b.ndim = 1.
        x1 = la.lstsq(A, b)[0]
        x2 = solver1D.predict(0)
        assert np.allclose(x1, x2)

        # Test without regularization, b.ndim = 2.
        X1 = la.lstsq(A, B)[0]
        X2 = solver2D.predict(0)
        assert np.allclose(X1, X2)

        # Test with regularization, b.ndim = 1.
        Apad = np.vstack((A, np.eye(n)))
        bpad = np.concatenate((b, np.zeros(n)))
        x1 = la.lstsq(Apad, bpad)[0]
        x2 = solver1D.predict(1)
        assert np.allclose(x1, x2)

        # Test with regularization, b.ndim = 2.
        Bpad = np.concatenate((B, np.zeros((n, k))))
        X1 = la.lstsq(Apad, Bpad)[0]
        X2 = solver2D.predict(1)
        assert np.allclose(X1, X2)

    # Post-processing ---------------------------------------------------------
    def test_cond(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.SolverL2.cond()."""
        solver = opinf.lstsq.SolverL2()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.cond()
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Contrived test 1
        A = np.eye(d)
        B = np.zeros((d,r))
        solver.fit(A, B)
        assert np.isclose(solver.cond(), 1)

        # Contrived test 2
        A = np.diag(np.arange(1,d+1))
        B = np.zeros((d,r))
        solver.fit(A, B)
        assert np.isclose(solver.cond(), d)

        # Random test
        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        solver.fit(A, B)
        assert np.isclose(solver.cond(), np.linalg.cond(A))

    def test_regcond(self, k=20, d=11, r=3, ntests=5):
        """Test lstsq._tikhonov.SolverL2.regcond()."""
        solver = opinf.lstsq.SolverL2()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regcond(0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Square, diagonal tests.
        A = np.diag(np.arange(1,d+1))
        B = np.zeros((d,r))
        solver.fit(A, B)
        assert np.isclose(solver.regcond(0), d)
        for λ in np.random.uniform(1, 10, ntests):
            assert np.isclose(solver.regcond(λ),
                              np.sqrt((d**2 + λ**2)/(1 + λ**2)))

        # Rectangular, dense tests.
        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        solver.fit(A, B)
        for λ in np.random.uniform(1, 10, ntests):
            Apad = np.vstack((A, λ*np.eye(d)))
            assert np.isclose(solver.regcond(λ), np.linalg.cond(Apad))

    def test_residual(self, k=20, d=11, r=3, ntests=5):
        """Test lstsq._tikhonov.SolverL2.residual()."""
        solver = opinf.lstsq.SolverL2()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.residual(0, 0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        solver.fit(A, B)

        # Try with badly shaped X.
        X = np.random.standard_normal((d+1,r-1))
        with pytest.raises(ValueError) as ex:
            solver.residual(X, 0)
        assert ex.value.args[0] == f"X.shape = {(d+1,r-1)} != {(d,r)} = (d,r)"

        # Two-dimensional tests.
        X = np.random.standard_normal((d,r))
        for λ in [0] + np.random.uniform(1, 10, ntests).tolist():
            residual = solver.residual(X, λ)
            assert isinstance(residual, np.ndarray)
            assert residual.shape == (r,)
            ans = la.norm(A @ X - B, ord=2, axis=0)**2
            ans += la.norm(λ*np.eye(d) @ X, ord=2, axis=0)**2
            assert np.allclose(residual, ans)

        # One-dimensional tests.
        b = B[:,0]
        solver.fit(A, b)
        assert solver.r == 1
        x = np.random.standard_normal(d)
        for λ in [0] + np.random.uniform(0, 10, ntests).tolist():
            residual = solver.residual(x, λ)
            assert isinstance(residual, float)
            ans = np.linalg.norm(A @ x - b)**2 + np.linalg.norm(λ*x)**2
            assert np.isclose(residual, ans)


class TestSolverL2Decoupled:
    """Test lstsq._tikhonov.SolverL2Decoupled."""
    # Validation --------------------------------------------------------------
    def test_check_λs(self, k=10, d=6, r=3):
        """Test lstsq._tikhonov.SolverL2Decoupled._check_λs()."""
        solver = opinf.lstsq.SolverL2Decoupled()
        A = np.empty((k,d))
        B = np.empty((k,r))
        solver._process_fit_arguments(A, B)
        assert solver.r == r

        with pytest.raises(TypeError) as ex:
            solver._check_λs(0)
        assert ex.value.args[0] == "object of type 'int' has no len()"

        with pytest.raises(ValueError) as ex:
            solver._check_λs([0]*(r-1))
        assert ex.value.args[0] == "len(λs) != number of columns of B"

        solver._check_λs([0]*r)

    # Main methods ------------------------------------------------------------
    def test_predict(self, k=20, d=10):
        λs = np.array([0, 1, 3, 5])
        r = len(λs)
        A = np.random.random((k,d))
        B = np.random.random((k,r))
        solver = opinf.lstsq.SolverL2Decoupled()

        # Try predicting before fitting.
        with pytest.raises(AttributeError) as ex:
            solver.predict(λs)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"
        solver.fit(A, B)

        # Try with the wrong number of regularization parameters.
        with pytest.raises(ValueError) as ex:
            solver.predict(λs[:-1])
        assert ex.value.args[0] == "len(λs) != number of columns of B"

        Id = np.eye(d)
        Apads = [np.vstack((A, λ*Id)) for λ in λs]
        Bpad = np.vstack((B, np.zeros((d,r))))
        X1 = np.column_stack([la.lstsq(Apad, Bpad[:,j])[0]
                              for j,Apad in enumerate(Apads)])
        X2 = solver.predict(λs)
        assert np.allclose(X1, X2)

    # Post-processing ---------------------------------------------------------
    def test_regcond(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.SolverL2Decoupled.regcond()."""
        solver = opinf.lstsq.SolverL2Decoupled()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regcond(0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Square, diagonal tests.
        A = np.diag(np.arange(1,d+1))
        B = np.zeros((d,r))
        solver.fit(A, B)
        assert np.allclose(solver.regcond([0]*r), [d]*r)
        λ = np.random.uniform(1, 10, r)
        assert np.allclose(solver.regcond(λ),
                           np.sqrt((d**2 + λ**2)/(1 + λ**2)))

        # Rectangular, dense tests.
        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        solver.fit(A, B)
        λ = np.random.uniform(1, 10, r)
        Id = np.eye(d)
        conds = [np.linalg.cond(np.vstack((A, λλ*Id))) for λλ in λ]
        assert np.allclose(solver.regcond(λ), conds)

    def test_residual(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.SolverL2Decoupled.residual()."""
        solver = opinf.lstsq.SolverL2Decoupled()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.residual(0, 0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        solver.fit(A, B)

        # Try with badly shaped X.
        X = np.random.standard_normal((d+1,r-1))
        with pytest.raises(ValueError) as ex:
            solver.residual(X, [0]*r)
        assert ex.value.args[0] == f"X.shape = {(d+1,r-1)} != {(d,r)} = (d,r)"

        # Correct usage.
        X = np.random.standard_normal((d,r))
        ls = np.array([0] + np.random.uniform(1, 10, r-1).tolist())
        residual = solver.residual(X, ls)
        assert isinstance(residual, np.ndarray)
        assert residual.shape == (r,)
        ans = la.norm(A @ X - B, ord=2, axis=0)**2
        ans += np.array([la.norm(l*X[:,j], ord=2)**2 for j,l in enumerate(ls)])
        assert np.allclose(residual, ans)


class TestSolverTikhonov:
    """Test lstsq._tikhonov.SolverTikhonov."""
    # Validation --------------------------------------------------------------
    def test_process_regularizer(self, k=20, d=11, r=3):
        solver = opinf.lstsq.SolverTikhonov()
        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        solver._process_fit_arguments(A, B)

        # Try with bad regularizer type.
        with pytest.raises(TypeError) as ex:
            solver._process_regularizer("not an array")
        assert ex.value.args[0] == \
            "regularization matrix must be a NumPy array"

        # Try with bad diagonal regularizer.
        P = np.ones(d)
        P[-1] = -1
        with pytest.raises(ValueError) as ex:
            solver._process_regularizer(P)
        assert ex.value.args[0] == "diagonal P must be positive semi-definite"

        # Try with bad regularizer shapes.
        P = np.empty(d-1)
        with pytest.raises(ValueError) as ex:
            solver._process_regularizer(P)
        assert ex.value.args[0] == \
            "P.shape != (d,d) or (d,) where d = A.shape[1]"

        P = np.empty((d,d-1))
        with pytest.raises(ValueError) as ex:
            solver._process_regularizer(P)
        assert ex.value.args[0] == \
            "P.shape != (d,d) or (d,) where d = A.shape[1]"

        # Correct usage
        P = np.full(d, 2)
        PP = solver._process_regularizer(P)
        assert PP.shape == (d,d)
        assert np.all(PP == 2*np.eye(d))

        P = np.eye(d) + np.diag(np.ones(d-1), -1)
        PP = solver._process_regularizer(P)
        assert P is PP

    # Helper methods ----------------------------------------------------------
    def test_lhs(self, d=11, ntests=5):
        """Test lstsq._tikhonov.SolverTikhonov._lhs()."""
        solver = opinf.lstsq.SolverTikhonov()
        A_or_B = np.empty((d,d))
        solver._process_fit_arguments(A_or_B, A_or_B)

        AtA = np.random.standard_normal((d,d))
        solver._AtA = AtA
        for _ in range(ntests):
            P = np.random.uniform(1, 10, d)
            PP, AtAPtP = solver._lhs(P)
            assert isinstance(PP, np.ndarray)
            assert PP.shape == (d,d)
            assert np.allclose(PP, np.diag(P))
            assert np.allclose(AtAPtP, AtA + np.diag(P**2))

            P = np.random.standard_normal((d,d))
            PP, AtAPtP = solver._lhs(P)
            assert PP is P
            assert np.allclose(AtAPtP, AtA + P.T @ P)

    # Main methods ------------------------------------------------------------
    def test_fit(self, k=20, d=10, r=5):
        """Test lstsq._tikhonov.SolverTikhonov.fit()."""
        solver = opinf.lstsq.SolverTikhonov()
        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))

        solver.fit(A, B)
        for attr, shape in [("A", (k,d)), ("B", (k,r)),
                            ("_AtA", (d,d)), ("_rhs", (d,r))]:
            assert hasattr(solver, attr)
            obj = getattr(solver, attr)
            assert isinstance(obj, np.ndarray)
            assert obj.shape == shape

    def test_predict(self, k=40, d=15, r=5):
        """Test lstsq._tikhonov.SolverTikhonov.predict()."""
        solver1D = opinf.lstsq.SolverTikhonov()
        solver2D = opinf.lstsq.SolverTikhonov()

        # Try predicting before fitting.
        with pytest.raises(AttributeError) as ex:
            solver1D.predict(0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        A = np.random.random((k,d))
        B = np.random.random((k,r))
        b = B[:,0]
        solver1D.fit(A, b)
        solver2D.fit(A, B)

        # Test without regularization, b.ndim = 1.
        Z = np.zeros((d,d))
        x1 = la.lstsq(A, b)[0]
        x2 = solver1D.predict(Z)
        assert np.allclose(x1, x2)

        # Test without regularization, b.ndim = 2.
        X1 = la.lstsq(A, B)[0]
        X2 = solver2D.predict(Z)
        assert np.allclose(X1, X2)

        # Test with regularization, b.ndim = 1.
        Id = np.eye(d)
        Apad = np.vstack((A, Id))
        bpad = np.concatenate((b, np.zeros(d)))
        x1 = la.lstsq(Apad, bpad)[0]
        x2 = solver1D.predict(Id)
        assert np.allclose(x1, x2)

        # Test with regularization, b.ndim = 2.
        Bpad = np.concatenate((B, np.zeros((d, r))))
        X1 = la.lstsq(Apad, Bpad)[0]
        X2 = solver2D.predict(Id)
        assert np.allclose(X1, X2)

        # Test with underdetermined system and regularization.
        Apad = np.vstack((A, Id))
        bpad = np.concatenate((b, np.zeros(d)))
        x1 = la.lstsq(Apad, bpad)[0]
        x2 = solver1D.predict(Id)

        assert np.allclose(x1, x2)
        x2 = solver1D.predict(np.ones(d))

        # Test with a severely ill-conditioned system.
        A = np.random.standard_normal((k,d))
        U,s,Vt = la.svd(A, full_matrices=False)
        s[-5:] = 1e-18
        s /= np.arange(1, s.size+1)**2
        A = U @ np.diag(s) @ Vt
        B = np.random.standard_normal((k,r))
        assert np.linalg.cond(A) > 1e15

        # No regularization.
        solver2D.fit(A, B)
        X1 = la.lstsq(A, B)[0]
        X2 = solver2D.predict(Z)
        assert np.allclose(X1, X2)

        # Some regularization.
        Apad = np.vstack((A, Id))
        Bpad = np.concatenate((B, np.zeros((d, r))))
        X1 = la.lstsq(Apad, Bpad)[0]
        X2 = solver2D.predict(Id)
        assert np.allclose(X1, X2)

    # Post-processing ---------------------------------------------------------
    def test_regcond(self, k=20, d=11, r=3, ntests=5):
        """Test lstsq._tikhonov.SolverTikhonov.regcond()."""
        solver = opinf.lstsq.SolverTikhonov()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regcond(0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Square, diagonal tests.
        A = np.diag(np.arange(1,d+1))
        B = np.zeros((d,r))
        Z = np.zeros(d)
        solver.fit(A, B)
        assert np.isclose(solver.regcond(Z), d)
        λ = np.random.uniform(1, 10, d)
        λ.sort()
        assert np.isclose(solver.regcond(λ),
                          np.sqrt((d**2 + λ[-1]**2)/(1 + λ[0]**2)))

        # Rectangular, dense tests.
        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        solver.fit(A, B)
        for λ in np.random.uniform(1, 10, ntests):
            cond = np.linalg.cond(np.vstack((A, λ*np.eye(d))))
            assert np.isclose(solver.regcond(np.array([λ]*d)), cond)

            P = np.random.standard_normal((d,d))
            cond = np.linalg.cond(np.vstack((A, P)))
            assert np.isclose(solver.regcond(P), cond)

    def test_residual(self, k=20, d=11, r=3, ntests=5):
        """Test lstsq._tikhonov.SolverTikhonov.residual()."""
        solver = opinf.lstsq.SolverTikhonov()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.residual(0, 0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        solver.fit(A, B)

        # Try with badly shaped X.
        X = np.random.standard_normal((d+1,r-1))
        with pytest.raises(ValueError) as ex:
            solver.residual(X, np.empty(d))
        assert ex.value.args[0] == f"X.shape = {(d+1,r-1)} != {(d,r)} = (d,r)"

        # Two-dimensional tests.
        X = np.random.standard_normal((d,r))
        for _ in range(ntests):
            P = np.random.uniform(1, 10, d)
            residual = solver.residual(X, P)
            assert isinstance(residual, np.ndarray)
            assert residual.shape == (r,)
            ans = la.norm(A @ X - B, ord=2, axis=0)**2
            ans += la.norm(np.diag(P) @ X, ord=2, axis=0)**2
            assert np.allclose(residual, ans)

        # One-dimensional tests.
        b = B[:,0]
        solver.fit(A, b)
        assert solver.r == 1
        x = np.random.standard_normal(d)
        for _ in range(ntests):
            P = np.random.uniform(1, 10, d)
            residual = solver.residual(x, P)
            assert isinstance(residual, float)
            ans = np.linalg.norm(A @ x - b)**2 + np.linalg.norm(P*x)**2
            assert np.isclose(residual, ans)


class TestSolverTikhonovDecoupled:
    """Test lstsq._tikhonov.SolverTikhonovDecoupled."""
    # Validation --------------------------------------------------------------
    def test_check_Ps(self, k=10, d=6, r=3):
        """Test lstsq._tikhonov.SolverTikhonovDecoupled._check_Ps()."""
        solver = opinf.lstsq.SolverTikhonovDecoupled()
        A = np.empty((k,d))
        B = np.empty((k,r))
        solver._process_fit_arguments(A, B)
        assert solver.r == r

        with pytest.raises(TypeError) as ex:
            solver._check_Ps(0)
        assert ex.value.args[0] == "object of type 'int' has no len()"

        with pytest.raises(ValueError) as ex:
            solver._check_Ps([0]*(r-1))
        assert ex.value.args[0] == "len(Ps) != number of columns of B"

        solver._check_Ps([0]*r)

    # Main methods ------------------------------------------------------------
    def test_predict(self, k=20, d=10):
        """Test lstsq._tikhonov.SolverTikhonovDecoupled.predict()."""
        solver = opinf.lstsq.SolverTikhonovDecoupled()
        Ps = [np.eye(d), np.full(d, 2)]
        r = len(Ps)
        A = np.random.random((k,d))
        B = np.random.random((k,r))

        # Try predicting before fitting.
        with pytest.raises(AttributeError) as ex:
            solver.predict(Ps)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"
        solver.fit(A, B)

        # Try with the wrong number of regularizers.
        with pytest.raises(ValueError) as ex:
            solver.predict(Ps[:1])
        assert ex.value.args[0] == "len(Ps) != number of columns of B"

        Apad1 = np.vstack((A, Ps[0]))
        Apad2 = np.vstack((A, np.diag(Ps[1])))
        Bpad = np.vstack((B, np.zeros((d,2))))
        xx1 = la.lstsq(Apad1, Bpad[:,0])[0]
        xx2 = la.lstsq(Apad2, Bpad[:,1])[0]
        X1 = np.column_stack([xx1, xx2])
        X2 = solver.predict(Ps)
        assert np.allclose(X1, X2)

        # Test with a severely ill-conditioned system.
        A = np.random.standard_normal((k,d))
        U,s,Vt = la.svd(A, full_matrices=False)
        s[-5:] = 1e-18
        s /= np.arange(1, s.size+1)**2
        A = U @ np.diag(s) @ Vt
        B = np.random.standard_normal((k,r))
        assert np.linalg.cond(A) > 1e15

        # No regularization.
        solver.fit(A, B)
        Z = np.zeros(d)
        X1 = la.lstsq(A, B)[0]
        X2 = solver.predict([Z, Z])
        assert np.allclose(X1, X2)

        # Some regularization.
        Apad1 = np.vstack((A, Ps[0]))
        Apad2 = np.vstack((A, np.diag(Ps[1])))
        Bpad = np.vstack((B, np.zeros((d,2))))
        xx1 = la.lstsq(Apad1, Bpad[:,0])[0]
        xx2 = la.lstsq(Apad2, Bpad[:,1])[0]
        X1 = np.column_stack([xx1, xx2])
        X2 = solver.predict(Ps)
        assert np.allclose(X1, X2)

    # Post-processing ---------------------------------------------------------
    def test_regcond(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.SolverTikhonovDecoupled.regcond()."""
        solver = opinf.lstsq.SolverTikhonovDecoupled()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regcond(0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Square, diagonal tests.
        A = np.diag(np.arange(1,d+1))
        B = np.zeros((d,r))
        z = np.zeros(d)
        solver.fit(A, B)
        assert np.allclose(solver.regcond([z]*r), [d]*r)
        λ = np.random.uniform(1, 10, r)
        Ps = [λλ*np.ones(d) for λλ in λ]
        assert np.allclose(solver.regcond(Ps),
                           np.sqrt((d**2 + λ**2)/(1 + λ**2)))

        # Rectangular, dense tests.
        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        solver.fit(A, B)
        Ps = np.random.standard_normal((r,d,d))
        conds = [np.linalg.cond(np.vstack((A, P))) for P in Ps]
        assert np.allclose(solver.regcond(Ps), conds)

    def test_residual(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.SolverTikhonovDecoupled.residual()."""
        solver = opinf.lstsq.SolverTikhonovDecoupled()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.residual(0, 0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))
        z = np.zeros(d)
        solver.fit(A, B)

        # Try with badly shaped X.
        X = np.random.standard_normal((d+1,r-1))
        with pytest.raises(ValueError) as ex:
            solver.residual(X, [z]*r)
        assert ex.value.args[0] == f"X.shape = {(d+1,r-1)} != {(d,r)} = (d,r)"

        # Correct usage.
        X = np.random.standard_normal((d,r))
        Ps = np.random.standard_normal((r,d,d))
        residual = solver.residual(X, Ps)
        assert isinstance(residual, np.ndarray)
        assert residual.shape == (r,)
        ans = la.norm(A @ X - B, ord=2, axis=0)**2
        ans += np.array([la.norm(P@X[:,j], ord=2)**2 for j,P in enumerate(Ps)])
        assert np.allclose(residual, ans)


def test_solver(m=20, n=10, k=5):
    """Test lstsq._tikhonov.solve()."""
    A = np.random.random((m,n))
    B = np.random.random((m,k))
    λs = 5 + np.random.random(k)
    Ps = [5 + np.random.random((n,n)) for _ in range(k)]
    Ps_diag = [5 + np.random.random(n) for _ in range(k)]

    # Bad number of regularization parameters.
    with pytest.raises(ValueError) as ex:
        opinf.lstsq.solver(A, B, λs[:k-2])
    assert ex.value.args[0] == "invalid or misaligned input P"

    # Bad number of regularization matrices.
    with pytest.raises(ValueError) as ex:
        opinf.lstsq.solver(A, B, Ps[:k-2])
    assert ex.value.args[0] == "invalid or misaligned input P"

    # Try to solve 1D problem with multiple regularizations.
    with pytest.raises(ValueError) as ex:
        opinf.lstsq.solver(A, B[:,0], Ps)
    assert ex.value.args[0] == "invalid or misaligned input P"

    # Bad type for regularization.
    with pytest.raises(ValueError) as ex:
        opinf.lstsq.solver(A, B, {})
    assert ex.value.args[0] == "invalid or misaligned input P"

    # Correct usage.
    solver = opinf.lstsq.solver(A, B, 0)
    assert isinstance(solver, opinf.lstsq.SolverL2)
    solver = opinf.lstsq.solver(A, B, λs[0])
    assert isinstance(solver, opinf.lstsq.SolverL2)
    solver = opinf.lstsq.solver(A, B, λs)
    assert isinstance(solver, opinf.lstsq.SolverL2Decoupled)
    solver = opinf.lstsq.solver(A, B, Ps[0])
    assert isinstance(solver, opinf.lstsq.SolverTikhonov)
    solver = opinf.lstsq.solver(A, B, Ps)
    assert isinstance(solver, opinf.lstsq.SolverTikhonovDecoupled)
    solver = opinf.lstsq.solver(A, B, Ps_diag)
    assert isinstance(solver, opinf.lstsq.SolverTikhonovDecoupled)
    opinf.lstsq.solve(A, B, 0)
