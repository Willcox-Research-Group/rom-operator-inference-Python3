# lstsq/test_tikhonov.py
"""Tests for lstsq._tikhonov.py."""

import pytest
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

import opinf

try:
    from .test_base import _TestSolverTemplate
except ImportError:
    from test_base import _TestSolverTemplate


class _TestBaseRegularizedSolver(_TestSolverTemplate):
    """Base class for classes that test Tikhonov-type solvers."""

    def test_fit_and_str(self, k=20, d=10, r=6):
        """Test fit() and lightly test __str__() and __repr__()."""
        # Underdetermined.
        k2 = d // 2
        D = np.random.standard_normal((k2, d))
        Z = np.random.random((r, k2))
        solver = next(self.get_solvers())

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            solver.fit(D, Z)
        assert wn[0].message.args[0] == (
            "non-regularized least-squares system is underdetermined"
        )

        return super().test_fit_and_str(k=k, d=d, r=r)

    def test_posterior(self, k=20, d=10, r=5):
        """Lightly test posterior()."""
        D = np.random.random((k, d))
        Z = np.random.random((r, k))

        for solver in self.get_solvers():
            solver.fit(D, Z)
            means, precisions = solver.posterior()
            assert len(means) == r
            assert len(precisions) == r
            for mean, prec in zip(means, precisions):
                assert isinstance(mean, np.ndarray)
                assert mean.shape == (d,)
                assert isinstance(prec, np.ndarray)
                assert prec.shape == (d, d)

        # Detect perfect regression.
        solver.fit(D, np.zeros_like(Z))
        with pytest.raises(RuntimeError) as ex:
            solver.posterior()
        assert ex.value.args[0] == (
            "zero residual --> posterior is deterministic"
        )


class TestL2Solver(_TestBaseRegularizedSolver):
    """Test lstsq._tikhonov.L2Solver."""

    Solver = opinf.lstsq.L2Solver

    def get_solvers(self):
        """Yield solvers to test."""
        yield self.Solver(0)
        yield self.Solver(1e-6)

    # Properties --------------------------------------------------------------
    def test_regularizer(self):
        """Test regularizer property and setter."""
        # Try with nonscalar regularizer.
        with pytest.raises(TypeError) as ex:
            self.Solver([1, 2, 3])
        assert ex.value.args[0] == "regularization constant must be a scalar"

        # Negative regularization parameter not allowed.
        with pytest.raises(ValueError) as ex:
            self.Solver(-1)
        assert ex.value.args[0] == (
            "regularization constant must be nonnegative"
        )

        regularizer = np.random.uniform()
        solver = self.Solver(regularizer)
        assert solver.regularizer == regularizer

    # Main methods ------------------------------------------------------------
    def test_solve(self, k=20, d=10, r=5):
        """Test solve()."""
        D = np.random.random((k, d))
        Z = np.random.random((r, k))

        def _check(o1, o2):
            assert isinstance(o2, np.ndarray)
            if o1.ndim == 1:
                o1 = o1.reshape((1, -1))
            assert o2.shape == o1.shape
            assert np.allclose(o2, o1)

        for solver in self.get_solvers():

            # Try solving before fitting.
            with pytest.raises(AttributeError) as ex:
                solver.solve()
            assert ex.value.args[0] == "solver not trained, call fit()"

            # Test without regularization.
            if solver.regularizer == 0:
                # Z.ndim = 2.
                Ohat1 = la.lstsq(D, Z.T)[0].T
                Ohat2 = solver.fit(D, Z).solve()
                _check(Ohat1, Ohat2)

                # Z.ndim = 1.
                ohat2 = solver.fit(D, Z[0]).solve()
                _check(Ohat1[0], ohat2)

            else:  # With regularization.
                # Z.ndim = 2.
                Dpad = np.vstack((D, solver.regularizer * np.eye(d)))
                Zpad = np.concatenate((Z.T, np.zeros((d, r))))
                Ohat1 = la.lstsq(Dpad, Zpad)[0].T
                Ohat2 = solver.fit(D, Z).solve()
                _check(Ohat1, Ohat2)

                # Z.ndim = 1.
                ohat2 = solver.fit(D, Z[0]).solve()
                _check(Ohat1[0], ohat2)

    # Post-processing ---------------------------------------------------------
    def test_regcond(self, k=20, d=11, r=3, ntests=5):
        """Test regcond()."""
        solver = self.Solver(0)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regcond()
        assert ex.value.args[0] == "solver not trained, call fit()"

        def _singletest(reg, regcondtrue):
            solver.regularizer = reg
            regcond = solver.regcond()
            assert np.isclose(regcond, regcondtrue)

        # Square, diagonal tests.
        A = np.diag(np.arange(1, d + 1))
        B = np.zeros((r, d))
        solver.fit(A, B)
        assert np.isclose(solver.regcond(), d)
        for reg in np.random.uniform(1, 10, ntests):
            regcond_true = np.sqrt((d**2 + reg**2) / (1 + reg**2))
            _singletest(reg, regcond_true)

        # Rectangular, dense tests.
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((r, k))
        solver.fit(A, B)
        for reg in np.random.uniform(1, 10, ntests):
            regcond_true = np.linalg.cond(np.vstack((A, reg * np.eye(d))))
            _singletest(reg, regcond_true)

    def test_regresidual(self, k=20, d=11, r=3, ntests=5):
        """Test regresidual()."""
        solver = self.Solver(0)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regresidual(0)
        assert ex.value.args[0] == "solver not trained, call fit()"

        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((r, k))
        solver.fit(A, B)

        # Try with badly shaped X.
        Ohat = np.random.standard_normal((r - 1, d + 1))
        with pytest.raises(ValueError) as ex:
            solver.regresidual(Ohat)
        assert ex.value.args[0] == (
            f"Ohat.shape = {(r - 1, d + 1)} != {(r, d)} = (r, d)"
        )

        # Two-dimensional tests.
        Ohat = np.random.standard_normal((r, d))
        for reg in [0] + np.random.uniform(1, 10, ntests).tolist():
            solver.regularizer = reg
            residual = solver.regresidual(Ohat)
            assert isinstance(residual, np.ndarray)
            assert residual.shape == (r,)
            ans = la.norm(A @ Ohat.T - B.T, ord=2, axis=0) ** 2
            ans += la.norm(reg * Ohat.T, ord=2, axis=0) ** 2
            assert np.allclose(residual, ans)

        # One-dimensional tests.
        b = B[0, :]
        solver.fit(A, b)
        assert solver.r == 1
        x = np.random.standard_normal(d)
        for reg in [0] + np.random.uniform(0, 10, ntests).tolist():
            solver.regularizer = reg
            residual = solver.regresidual(x)
            assert isinstance(residual, np.ndarray)
            assert residual.shape == (1,)
            ans = np.linalg.norm(A @ x - b) ** 2 + np.linalg.norm(reg * x) ** 2
            assert np.isclose(residual[0], ans)


class TestL2DecoupledSolver(_TestBaseRegularizedSolver):
    """Test lstsq._tikhonov.L2DecoupledSolver."""

    Solver = opinf.lstsq.L2DecoupledSolver
    test_1D_Z = False

    def get_solvers(self):
        """Yield solvers to test."""
        yield self.Solver(np.array([0, 0, 0, 0]))
        yield self.Solver(np.array([0, 1e-6, 1e-2, 10]))

    # Properties --------------------------------------------------------------
    def test_regularizer(self, k=10, d=6, r=3):
        """Test _check_regularizer_shape(), fit(), and regularizer property."""
        solver = self.Solver(np.random.random(r + 1))
        A = np.empty((k, d))
        B = np.empty((r, k))

        with pytest.raises(ValueError) as ex:
            solver.regularizer = np.random.random((r, r))
        assert ex.value.args[0] == "regularizer must be one-dimensional"

        with pytest.raises(ValueError) as ex:
            solver.regularizer = -np.ones(r)
        assert ex.value.args[0] == (
            "regularization constants must be nonnegative"
        )

        with pytest.raises(ValueError) as ex:
            solver.fit(A, B)
        assert ex.value.args[0] == (
            f"regularizer.shape = ({r + 1},) != ({r},) = (r,)"
        )

        solver.regularizer = np.random.random(r)
        solver.fit(A, B)
        assert solver.r == r

        repr(solver)

    # Main methods ------------------------------------------------------------
    def test_fit_and_str(self, k=20, d=10):
        return super().test_fit_and_str(k=k, d=d, r=4)

    def test_solve(self, k=20, d=10):
        """Test solve()."""
        r = 4
        D = np.random.random((k, d))
        Z = np.random.random((r, k))
        Id = np.eye(d)
        ZpadT = np.vstack((Z.T, np.zeros((d, r))))

        for solver in self.get_solvers():
            Ohat1 = []
            for i, reg in enumerate(solver.regularizer):
                if reg == 0:
                    ohat_i = la.lstsq(D, Z[i])[0]
                else:
                    Dpad = np.vstack((D, reg * Id))
                    ohat_i = la.lstsq(Dpad, ZpadT[:, i])[0]
                Ohat1.append(ohat_i)
            Ohat1 = np.array(Ohat1)

            Ohat2 = solver.fit(D, Z).solve()
            assert isinstance(Ohat2, np.ndarray)
            assert Ohat2.shape == Ohat1.shape
            assert np.allclose(Ohat2, Ohat1)

    def test_posterior(self, k=20, d=10):
        return super().test_posterior(k, d, 4)

    # Post-processing ---------------------------------------------------------
    def test_cond(self, k=20, d=11):
        return super().test_cond(k, d, 4)

    def test_residual(self, k=20, d=10):
        return super().test_residual(k, d, 4)

    def test_regcond(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.L2DecoupledSolver.regcond()."""
        solver = self.Solver([0] * r)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regcond()
        assert ex.value.args[0] == "solver not trained, call fit()"

        # Square, diagonal tests.
        A = np.diag(np.arange(1, d + 1))
        B = np.zeros((r, d))
        solver.fit(A, B)
        assert np.allclose(solver.regcond(), [d] * r)
        reg = np.random.uniform(1, 10, r)
        solver.regularizer = reg
        assert np.allclose(
            solver.regcond(), np.sqrt((d**2 + reg**2) / (1 + reg**2))
        )

        # Rectangular, dense tests.
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((r, k))
        regs = np.random.uniform(1, 10, r)
        solver.regularizer = regs
        solver.fit(A, B)
        Id = np.eye(d)
        conds = [np.linalg.cond(np.vstack((A, reg * Id))) for reg in regs]
        assert np.allclose(solver.regcond(), conds)

    def test_regresidual(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.L2DecoupledSolver.residual()."""
        solver = self.Solver([0] * r)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regresidual(0)
        assert ex.value.args[0] == "solver not trained, call fit()"

        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((r, k))
        solver.fit(A, B)

        # Try with badly shaped X.
        Ohat = np.random.standard_normal((r - 1, d + 1))
        with pytest.raises(ValueError) as ex:
            solver.regresidual(Ohat)
        assert ex.value.args[0] == (
            f"Ohat.shape = {(r - 1, d + 1)} != {(r, d)} = (r, d)"
        )

        # Correct usage.
        Ohat = np.random.standard_normal((r, d))
        ls = np.concatenate([[0], np.random.uniform(1, 10, r - 1)])
        solver.regularizer = ls
        residual = solver.regresidual(Ohat)
        assert isinstance(residual, np.ndarray)
        assert residual.shape == (r,)
        ans = la.norm(A @ Ohat.T - B.T, ord=2, axis=0) ** 2
        ans += np.array(
            [la.norm(l * Ohat[i], ord=2) ** 2 for i, l in enumerate(ls)]
        )
        assert np.allclose(residual, ans)

    def test_save_load_and_copy_via_verify(self, k=20, d=11):
        return super().test_save_load_and_copy_via_verify(k, d, 4)


class TestTikhonovSolver(_TestBaseRegularizedSolver):
    """Test lstsq._tikhonov.TikhonovSolver."""

    Solver = opinf.lstsq.TikhonovSolver

    def get_solvers(self):
        """Yield solvers to test."""
        d = 10
        yield self.Solver(np.zeros((d, d)))
        yield self.Solver(np.full(d, 1e-4))
        reg = np.full(d, 1e-6)
        reg[d // 2 :] = 1e-2
        yield self.Solver(np.diag(reg))

    # Properties --------------------------------------------------------------
    def test_regularizer(self, k=20, d=11, r=3):
        """Test _check_regularizer_shape(), regularizer, and method."""
        Z = np.random.random((d, d))

        with pytest.raises(ValueError) as ex:
            self.Solver(Z, method="invalidmethodoption")
        assert ex.value.args[0] == "method must be 'lstsq' or 'normal'"

        Zdiag = np.diag(Z)
        solver = self.Solver(Z, method="normal")
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
        B = np.random.standard_normal((r, k))
        solver.fit(A, B)

        # Try with bad regularizer shapes.
        P = np.empty((d - 1, d - 1))
        with pytest.raises(ValueError) as ex:
            solver.regularizer = P
        assert ex.value.args[0] == (
            f"regularizer.shape = ({d - 1}, {d - 1}) != ({d}, {d}) = (d, d)"
        )

        # Correct usage
        solver.regularizer = np.full(d, 2)
        assert solver.regularizer.shape == (d, d)
        assert np.all(solver.regularizer == 2 * np.eye(d))

        repr(solver)
        solver.method = "normal"
        repr(solver)

    @classmethod
    def test_get_operator_regularizer(cls, r=5, m=3):
        """Test get_operator_regularizer()."""
        c = opinf.operators.ConstantOperator()
        A = opinf.operators.LinearOperator()
        H = opinf.operators.QuadraticOperator()
        B = opinf.operators.InputOperator()

        creg = 1e-1
        Areg = 1e0
        Hreg = 1e1
        Breg = 5e-1

        _r2 = r * (r + 1) // 2

        with pytest.raises(ValueError) as ex:
            cls.Solver.get_operator_regularizer([c, A, H], [creg, Areg], r, m)
        assert ex.value.args[0] == (
            "len(operators) == 3 != 2 == len(regularization_parameters)"
        )

        with pytest.raises(TypeError) as ex:
            cls.Solver.get_operator_regularizer([c, 2], [creg, Areg], r)
        assert ex.value.args[0] == "unsupported operator type 'int'"

        # No inputs.
        reg = cls.Solver.get_operator_regularizer([c, A], [creg, Areg], r)
        assert reg.shape == (1 + r,)
        assert reg[0] == creg
        assert np.all(reg[1:] == Areg)

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            cls.Solver.get_operator_regularizer([c, H], [creg, Hreg], r, m)
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            "argument 'input_dimension' ignored, no operators act on inputs"
        )

        # Yes inputs.
        reg = cls.Solver.get_operator_regularizer([H, B], [Hreg, Breg], r, m)
        assert reg.shape == (_r2 + m,)
        assert np.all(reg[:-m] == Hreg)
        assert np.all(reg[-m:] == Breg)

        with pytest.raises(ValueError) as ex:
            cls.Solver.get_operator_regularizer([A, B], [Areg, Breg], r)
        assert ex.value.args[0] == (
            "argument 'input_dimension' required, operators[1] acts on inputs"
        )

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            cls.Solver.get_operator_regularizer([A], [Areg], r)
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            "consider using L2Solver for models with only one operator"
        )

        # Affine-parametric operators.
        cp = opinf.operators.AffineConstantOperator(3)
        Ap = opinf.operators.AffineLinearOperator(2)
        reg = cls.Solver.get_operator_regularizer(
            [cp, Ap, H], [creg, Areg, Hreg], r
        )
        assert reg.shape == (3 + 2 * r + _r2,)
        assert np.all(reg[:3] == creg)
        assert np.all(reg[3 : 3 + 2 * r] == Areg)
        assert np.all(reg[-_r2:] == Hreg)

    # Main methods ------------------------------------------------------------
    def test_fit_and_str(self, k=20, r=6):
        return super().test_fit_and_str(k, 10, r)

    def test_solve(self, k=40, r=5):
        """Test solve()."""
        d = 10
        A = np.random.random((k, d))
        B = np.random.random((r, k))
        Bpad = np.concatenate((B.T, np.zeros((d, r))))
        b = B[0, :]
        bpad = Bpad[:, 0]
        Z = np.zeros((d, d))
        Id = np.eye(d)
        solver1D = self.Solver(Z).fit(A, b)
        solver1D.verify()
        solver2D = self.Solver(Z).fit(A, B)
        solver2D.verify()

        for method in ["normal", "lstsq"]:
            # Test without regularization, b.ndim = 1.
            solver1D.method = method
            solver1D.regularizer = Z
            x1 = la.lstsq(A, b)[0]
            x2 = solver1D.solve()
            assert np.allclose(x1, x2)

            # Test without regularization, b.ndim = 2.
            solver2D.method = method
            solver2D.regularizer = Z
            X1 = la.lstsq(A, B.T)[0].T
            X2 = solver2D.solve()
            assert np.allclose(X1, X2)

            # Test with regularization, b.ndim = 1.
            solver1D.regularizer = Id
            Apad = np.vstack((A, Id))
            x1 = la.lstsq(Apad, bpad)[0]
            x2 = solver1D.solve()
            assert np.allclose(x1, x2)

            # Test with regularization, b.ndim = 2.
            solver2D.regularizer = Id
            X1 = la.lstsq(Apad, Bpad)[0].T
            X2 = solver2D.solve()
            assert np.allclose(X1, X2)

        # Test SVD method with a severely ill-conditioned system.
        A = np.random.standard_normal((k, d))
        U, s, Vt = la.svd(A, full_matrices=False)
        s[-5:] = 1e-18
        s /= np.arange(1, s.size + 1) ** 2
        A = U @ np.diag(s) @ Vt
        B = np.random.standard_normal((r, k))
        assert np.linalg.cond(A) > 1e15

        # No regularization.
        solver2D = self.Solver(Z, method="lstsq").fit(A, B)
        X1 = la.lstsq(A, B.T)[0].T
        X2 = solver2D.solve()
        assert np.allclose(X1, X2)

        # Some regularization.
        solver2D.regularizer = Id
        Apad = np.vstack((A, Id))
        Bpad = np.concatenate((B.T, np.zeros((d, r))))
        X1 = la.lstsq(Apad, Bpad)[0].T
        X2 = solver2D.solve()
        assert np.allclose(X1, X2)

    # Post-processing ---------------------------------------------------------
    def test_cond(self, k=20, r=3):
        return super().test_cond(k, d=10, r=r)

    def test_regcond(self, k=20, d=11, r=3, ntests=5):
        """Test regcond()."""
        Z = np.zeros(d)
        Id = np.eye(d)
        solver = self.Solver(Z)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regcond()
        assert ex.value.args[0] == "solver not trained, call fit()"

        # Square, diagonal tests.
        A = np.diag(np.arange(1, d + 1))
        B = np.zeros((r, d))
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
        B = np.random.standard_normal((r, k))
        solver.fit(A, B)
        for reg in np.random.uniform(1, 10, ntests):
            solver.regularizer = reg * Id
            cond = np.linalg.cond(np.vstack((A, solver.regularizer)))
            assert np.isclose(solver.regcond(), cond)

            P = np.random.standard_normal((d, d))
            cond = np.linalg.cond(np.vstack((A, P)))
            solver.regularizer = P
            assert np.isclose(solver.regcond(), cond)

    def test_regresidual(self, k=20, d=11, r=3, ntests=5):
        """Test lstsq._tikhonov.TikhonovSolver.residual()."""
        Z = np.zeros(d)
        solver = self.Solver(Z)

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.regresidual(0)
        assert ex.value.args[0] == "solver not trained, call fit()"

        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((r, k))
        solver.fit(A, B)

        # Two-dimensional tests.
        Ohat = np.random.standard_normal((r, d))
        for _ in range(ntests):
            P = np.random.uniform(1, 10, d)
            solver.regularizer = P
            residual = solver.regresidual(Ohat)
            assert isinstance(residual, np.ndarray)
            assert residual.shape == (r,)
            ans = la.norm(A @ Ohat.T - B.T, ord=2, axis=0) ** 2
            ans += la.norm(np.diag(P) @ Ohat.T, ord=2, axis=0) ** 2
            assert np.allclose(residual, ans)

        # One-dimensional tests.
        b = B[0, :]
        solver.fit(A, b)
        assert solver.r == 1
        x = np.random.standard_normal(d)
        for _ in range(ntests):
            P = np.random.uniform(1, 10, d)
            solver.regularizer = P
            residual = solver.regresidual(x)
            assert isinstance(residual, np.ndarray)
            assert residual.shape == (1,)
            ans = np.linalg.norm(A @ x - b) ** 2 + np.linalg.norm(P * x) ** 2
            assert np.isclose(residual[0], ans)

    def test_save_load_and_copy_via_verify(self, k=20, r=6):
        return super().test_save_load_and_copy_via_verify(k=k, d=10, r=r)


class TestTikhonovDecoupledSolver(_TestBaseRegularizedSolver):
    """Test lstsq._tikhonov.TikhonovDecoupledSolver."""

    Solver = opinf.lstsq.TikhonovDecoupledSolver
    test_1D_Z = False

    def get_solvers(self):
        """Yield solvers to test."""
        r, d = 5, 10
        yield self.Solver(np.zeros((r, d)))
        yield self.Solver([np.full(d, 10 ** (i - 8)) for i in range(r)])
        yield self.Solver(np.random.standard_normal((r, d, d)))

    # Properties --------------------------------------------------------------
    def test_regularizer(self, k=10, d=6, r=3):
        """Test _check_regularizer_shape() and regularizer."""
        Z = np.random.random((d, d))
        solver = opinf.lstsq.TikhonovDecoupledSolver([Z] * r)
        A = np.empty((k, d))
        B = np.empty((r, k))
        solver.fit(A, B)
        assert solver.r == r

        Zs = [Z] * r
        with pytest.raises(ValueError) as ex:
            solver.regularizer = Zs[:-1]
        assert ex.value.args[0] == "len(regularizer) != r"

        Zs[-1] = np.random.random((d + 1, d + 1))
        with pytest.raises(ValueError) as ex:
            solver.regularizer = Zs
        assert ex.value.args[0] == (
            f"regularizer[{r-1:d}].shape = ({d + 1}, {d + 1}) "
            f"!= ({d}, {d}) = (d, d)"
        )

        with pytest.raises(ValueError) as ex:
            solver.regularizer = [[-1] * d] * r
        assert ex.value.args[0] == (
            "diagonal regularizer must be positive semi-definite"
        )

        Zs[-1] = sparse.diags(np.ones(d))
        solver.regularizer = Zs
        solver.regularizer = [[i] * d for i in range(1, r + 1)]
        assert np.all(solver.regularizer[0] == np.eye(d))

        repr(solver)

    # Main methods ------------------------------------------------------------
    def test_fit_and_str(self, k=20):
        return super().test_fit_and_str(k, d=10, r=5)

    def test_solve(self, k=20, d=10):
        """Test lstsq._tikhonov.TikhonovDecoupledSolver.solve()."""
        Z = np.zeros(d)
        Ps = [np.eye(d), np.full(d, 2)]
        r = len(Ps)
        A = np.random.random((k, d))
        B = np.random.random((r, k))
        solver = self.Solver(Ps)

        # Try solving before fitting.
        with pytest.raises(AttributeError) as ex:
            solver.solve()
        assert ex.value.args[0] == "solver not trained, call fit()"
        solver.fit(A, B)

        Apad1 = np.vstack((A, Ps[0]))
        Apad2 = np.vstack((A, np.diag(Ps[1])))
        Bpad = np.vstack((B.T, np.zeros((d, 2))))
        xx1 = la.lstsq(Apad1, Bpad[:, 0])[0]
        xx2 = la.lstsq(Apad2, Bpad[:, 1])[0]
        X1 = np.array([xx1, xx2])
        X2 = solver.solve()
        assert np.allclose(X1, X2)

        # Test with a severely ill-conditioned system.
        A = np.random.standard_normal((k, d))
        U, s, Vt = la.svd(A, full_matrices=False)
        s[-5:] = 1e-18
        s /= np.arange(1, s.size + 1) ** 2
        A = U @ np.diag(s) @ Vt
        B = np.random.standard_normal((r, k))
        assert np.linalg.cond(A) > 1e15

        # No regularization.
        solver = self.Solver([Z] * r).fit(A, B)
        Z = np.zeros(d)
        X1 = la.lstsq(A, B.T)[0].T
        X2 = solver.solve()
        assert np.allclose(X1, X2)

        # Some regularization.
        for method in "lstsq", "normal":
            solver.method = method
            solver.regularizer = Ps[:2]
            Apad1 = np.vstack((A, Ps[0]))
            Apad2 = np.vstack((A, np.diag(Ps[1])))
            Bpad = np.vstack((B.T, np.zeros((d, 2))))
            xx1 = la.lstsq(Apad1, Bpad[:, 0])[0]
            xx2 = la.lstsq(Apad2, Bpad[:, 1])[0]
            X1 = np.array([xx1, xx2])
            X2 = solver.solve()
            assert np.allclose(X1, X2)

    # Post-processing ---------------------------------------------------------
    def test_cond(self, k=20):
        return super().test_cond(k, d=10, r=5)

    def test_regcond(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.TikhonovDecoupledSolver.regcond()."""

        # Square, diagonal tests.
        A = np.diag(np.arange(1, d + 1))
        B = np.zeros((r, d))
        z = np.zeros(d)
        solver = self.Solver([z] * r).fit(A, B)
        assert np.allclose(solver.regcond(), [d] * r)

        regularizer = np.random.uniform(1, 10, r)
        solver.regularizer = [lm * np.ones(d) for lm in regularizer]
        true_val = np.sqrt((d**2 + regularizer**2) / (1 + regularizer**2))
        assert np.allclose(solver.regcond(), true_val)

        # Rectangular, dense tests.
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((r, k))
        Ps = np.random.standard_normal((r, d, d))
        solver = self.Solver(Ps).fit(A, B)
        conds = [np.linalg.cond(np.vstack((A, P))) for P in Ps]
        assert np.allclose(solver.regcond(), conds)

    def test_residual(self, k=20):
        return super().test_residual(k, d=10, r=5)

    def test_regresidual(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.TikhonovDecoupledSolver.residual()."""
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((r, k))
        Ohat = np.random.standard_normal((r, d))
        Ps = np.random.standard_normal((r, d, d))
        solver = self.Solver(Ps).fit(A, B)
        residual = solver.regresidual(Ohat)
        for G, ohat in zip(Ps, Ohat):
            print(G.shape, G, ohat.shape, ohat, sep="\n")
        assert isinstance(residual, np.ndarray)
        assert residual.shape == (r,)
        ans = la.norm(A @ Ohat.T - B.T, ord=2, axis=0) ** 2
        ans += np.array(
            [la.norm(P @ Ohat[i], ord=2) ** 2 for i, P in enumerate(Ps)]
        )
        assert np.allclose(residual, ans)

    def test_save_load_and_copy_via_verify(self, k=20):
        return super().test_save_load_and_copy_via_verify(k=k, d=10, r=5)


if __name__ == "__main__":
    pytest.main([__file__])
