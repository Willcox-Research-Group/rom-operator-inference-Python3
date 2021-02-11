# lstsq/test_tikhonov.py
"""Tests for rom_operator_inference.lstsq._tikhonov.py."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi


class TestBaseSolver:
    """Test lstsq._tikhonov._BaseSolver."""
    def test_properties(self, k=10, d=4, r=3):
        """Test lstsq._tikhonov._BaseSolver properties k, d, and r."""
        solver = roi.lstsq._tikhonov._BaseSolver()
        for attr in "ABkdr":
            assert hasattr(solver, attr)
            assert getattr(solver, attr) is None
        solver.A = np.empty((k,d))
        solver.B = np.empty((k,r))
        assert solver.k == k
        assert solver.d == d
        assert solver.r == r
        solver.B = np.empty((k,1))
        assert solver.r == 1

    def test_process_fit_arguments(self, k=10, d=4, r=3):
        """Test lstsq._tikhonov._BaseSolver._process_fit_arguments()."""
        solver = roi.lstsq._tikhonov._BaseSolver()
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

    def test_fit(self):
        """Test lstsq._tikhonov._BaseSolver.fit()."""
        solver = roi.lstsq._tikhonov._BaseSolver()
        with pytest.raises(NotImplementedError) as ex:
            solver.fit()
        assert ex.value.args[0] == "fit() implemented by child classes"

        with pytest.raises(NotImplementedError) as ex:
            solver.fit(1, 2, 3, 4, 5, 6, 7, a=8)
        assert ex.value.args[0] == "fit() implemented by child classes"

    def test_predict(self):
        """Test lstsq._tikhonov._BaseSolver.fit()."""
        solver = roi.lstsq._tikhonov._BaseSolver()
        with pytest.raises(NotImplementedError) as ex:
            solver.predict()
        assert ex.value.args[0] == "predict() implemented by child classes"

        with pytest.raises(NotImplementedError) as ex:
            solver.predict(1, 2, 3, 4, 5, 6, 7, a=8)
        assert ex.value.args[0] == "predict() implemented by child classes"


class TestSolverL2:
    """Test lstsq._tikhonov.SolverL2."""
    def test_process_regularizer(self, k=20, d=11, r=3):
        solver = roi.lstsq.SolverL2()
        A = np.empty((k,d))
        B = np.empty((k,r))

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

        l = np.random.uniform()
        assert solver._process_regularizer(l) == l**2


    def test_fit(self, k=20, d=11, r=3):
        """Test lstsq._tikhonov.SolverL2.fit()."""
        solver = roi.lstsq.SolverL2(compute_extras=True)
        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))

        solver.fit(A, B)
        for attr, shape in zip(["_V",  "_Σ", "_UtB",   "A",   "B"],
                               [(d,d), (d,),  (d,r), (k,d), (k,r)]):
            assert hasattr(solver, attr)
            obj = getattr(solver, attr)
            assert isinstance(obj, np.ndarray)
            assert obj.shape == shape

        assert hasattr(solver, "cond_")
        assert np.isclose(solver.cond_, np.linalg.cond(A))

    def test_predict(self, m=20, n=10, k=5):
        """Test lstsq._tikhonov.SolverL2.predict()."""
        solver1D = roi.lstsq.SolverL2(compute_extras=True)
        solver2D = roi.lstsq.SolverL2(compute_extras=True)
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
        assert np.isclose(solver1D.misfit_, solver1D.residual_)
        assert np.allclose(solver1D.misfit_, la.norm(A @ x1 - b, ord=2)**2)
        assert np.isclose(solver1D.cond_, solver1D.regcond_)
        assert np.isclose(solver1D.cond_, np.linalg.cond(A))

        # Test without regularization, b.ndim = 2.
        X1 = la.lstsq(A, B)[0]
        X2 = solver2D.predict(0)
        assert np.allclose(X1, X2)
        assert np.isclose(solver2D.misfit_, solver2D.residual_)
        assert np.allclose(solver2D.misfit_, la.norm(A @ X1 - B, ord='fro')**2)
        assert np.isclose(solver2D.cond_, solver2D.regcond_)
        assert np.isclose(solver2D.cond_, np.linalg.cond(A))

        # Test with regularization, b.ndim = 1.
        Apad = np.vstack((A, np.eye(n)))
        bpad = np.concatenate((b, np.zeros(n)))
        x1 = la.lstsq(Apad, bpad)[0]
        x2 = solver1D.predict(1)
        assert np.allclose(x1, x2)
        assert solver1D.misfit_ < solver1D.residual_
        assert np.allclose(solver1D.misfit_, la.norm(A @ x1 - b, ord=2)**2)
        assert np.allclose(solver1D.residual_,
                           solver1D.misfit_ + la.norm(x1, ord=2)**2)
        assert solver1D.cond_ > solver1D.regcond_
        assert np.isclose(solver1D.cond_, np.linalg.cond(A))

        # Test with regularization, b.ndim = 2.
        Bpad = np.concatenate((B, np.zeros((n, k))))
        X1 = la.lstsq(Apad, Bpad)[0]
        X2 = solver2D.predict(1)
        assert np.allclose(X1, X2)
        assert solver2D.misfit_ < solver2D.residual_
        assert np.allclose(solver2D.misfit_, la.norm(A @ X1 - B, ord='fro')**2)
        assert np.allclose(solver2D.residual_, solver2D.misfit_ + la.norm(X1, ord='fro')**2)
        assert solver2D.cond_ > solver2D.regcond_
        assert np.isclose(solver2D.cond_, np.linalg.cond(A))


class TestSolverL2Decoupled:
    """Test lstsq._tikhonov.SolverL2Decoupled."""
    def test_predict(self, k=20, d=10):
        λs = np.array([0, 1, 3, 5])
        r = len(λs)
        A = np.random.random((k,d))
        B = np.random.random((k,r))
        solver = roi.lstsq.SolverL2Decoupled(compute_extras=True)

        # Try predicting before fitting.
        with pytest.raises(AttributeError) as ex:
            solver.predict(λs)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"
        solver.fit(A, B)

        # Try with the wrong number of regularization parameters.
        with pytest.raises(ValueError) as ex:
            solver.predict(λs[:-1])
        assert ex.value.args[0] == "len(λs) != number of columns of B"

        I = np.eye(d)
        Apads = [np.vstack((A, λ*I)) for λ in λs]
        Bpad = np.vstack((B, np.zeros((d,r))))
        X1 = np.column_stack([la.lstsq(Apad, Bpad[:,j])[0]
                              for j,Apad in enumerate(Apads)])
        X2 = solver.predict(λs)
        assert np.allclose(X1, X2)
        assert solver.misfit_.shape == (r,)
        assert solver.residual_.shape == (r,)
        assert np.all(solver.misfit_ <= solver.residual_)
        assert np.isclose(solver.misfit_.sum(),
                          la.norm(A @ X1 - B, ord='fro')**2)
        assert np.isclose(solver.cond_, np.linalg.cond(A))

        solver.compute_extras = False
        solver.predict(λs)


class TestSolverTikhonov:
    """Test lstsq._tikhonov.SolverTikhonov."""
    def test_process_regularizer(self, k=20, d=11, r=3):
        solver = roi.lstsq.SolverTikhonov(check_regularizer=True)
        solver.A = np.empty((k,d))

        # Try with bad regularizer type.
        with pytest.raises(TypeError) as ex:
            solver._process_regularizer("not an array")
        assert ex.value.args[0] == \
            "regularization matrix must be a NumPy array"

        # Try with bad regularizers.
        P = np.ones(d)
        P[-1] = -1
        with pytest.raises(ValueError) as ex:
            solver._process_regularizer(P)
        assert ex.value.args[0] == "diagonal P must be positive semi-definite"

        P = np.eye(d)
        P[-1,-1] = 0
        with pytest.raises(ValueError) as ex:
            solver._process_regularizer(P)
        assert ex.value.args[0] == "regularizer P is rank deficient"

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
        P2 = solver._process_regularizer(P)
        assert np.all(P2 == np.diag(np.full(d, 4)))

        P = np.eye(d) + np.diag(np.ones(d-1), -1)
        P2 = solver._process_regularizer(P)
        assert np.all(P2 == P.T @ P)

    def test_fit(self, k=20, d=10, r=5):
        """Test lstsq._tikhonov.SolverTikhonov.fit()."""
        solver = roi.lstsq.SolverTikhonov(compute_extras=True)
        A = np.random.standard_normal((k,d))
        B = np.random.standard_normal((k,r))

        solver.fit(A, B)
        for attr, shape in zip([  "A",   "B", "_AtA", "_rhs"],
                               [(k,d), (k,r),  (d,d),  (d,r)]):
            assert hasattr(solver, attr)
            obj = getattr(solver, attr)
            assert isinstance(obj, np.ndarray)
            assert obj.shape == shape

        assert hasattr(solver, "cond_")
        assert np.isclose(solver.cond_, np.linalg.cond(A))

    def test_predict(self, k=20, d=10, r=5):
        """Test lstsq._tikhonov.SolverTikhonov.predict()."""
        solver1D = roi.lstsq.SolverTikhonov(compute_extras=True,
                                            check_regularizer=False)
        solver2D = roi.lstsq.SolverTikhonov(compute_extras=True,
                                            check_regularizer=False)

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
        assert np.isclose(solver1D.misfit_, solver1D.residual_)
        assert np.allclose(solver1D.misfit_, la.norm(A @ x1 - b, ord=2)**2)
        assert np.isclose(solver1D.cond_, solver1D.regcond_)
        assert np.isclose(solver1D.cond_, np.linalg.cond(A))

        # Test without regularization, b.ndim = 2.
        X1 = la.lstsq(A, B)[0]
        X2 = solver2D.predict(Z)

        assert np.allclose(X1, X2)
        assert np.isclose(solver2D.misfit_, solver2D.residual_)
        assert np.allclose(solver2D.misfit_, la.norm(A @ X1 - B, ord='fro')**2)
        assert np.isclose(solver2D.cond_, solver2D.regcond_)
        assert np.isclose(solver2D.cond_, np.linalg.cond(A))

        # Test with regularization, b.ndim = 1.
        I = np.eye(d)
        Apad = np.vstack((A, I))
        bpad = np.concatenate((b, np.zeros(d)))
        x1 = la.lstsq(Apad, bpad)[0]
        x2 = solver1D.predict(I)

        assert np.allclose(x1, x2)
        assert solver1D.misfit_ < solver1D.residual_
        assert np.allclose(solver1D.misfit_, la.norm(A @ x1 - b, ord=2)**2)
        assert np.allclose(solver1D.residual_,
                           solver1D.misfit_ + la.norm(x1, ord=2)**2)
        assert solver1D.cond_ > solver1D.regcond_
        assert np.isclose(solver1D.cond_, np.linalg.cond(A))

        # Test with regularization, b.ndim = 2.
        Bpad = np.concatenate((B, np.zeros((d, r))))
        X1 = la.lstsq(Apad, Bpad)[0]
        X2 = solver2D.predict(I)

        assert np.allclose(X1, X2)
        assert solver2D.misfit_ < solver2D.residual_
        assert np.allclose(solver2D.misfit_, la.norm(A @ X1 - B, ord='fro')**2)
        assert np.allclose(solver2D.residual_, solver2D.misfit_ + la.norm(X1, ord='fro')**2)
        assert solver2D.cond_ > solver2D.regcond_
        assert np.isclose(solver2D.cond_, np.linalg.cond(A))

        # Test with underdetermined system and regularization.
        I = np.eye(d)
        Apad = np.vstack((A, I))
        bpad = np.concatenate((b, np.zeros(d)))
        x1 = la.lstsq(Apad, bpad)[0]
        x2 = solver1D.predict(I)

        assert np.allclose(x1, x2)
        assert solver1D.misfit_ < solver1D.residual_
        assert np.allclose(solver1D.misfit_, la.norm(A @ x1 - b, ord=2)**2)
        assert np.isclose(solver1D.cond_, np.linalg.cond(A))

        x2 = solver1D.predict(np.ones(d))


class TestSolverTikhonovDecoupled:
    """Test lstsq._tikhonov.SolverTikhonovDecoupled."""
    def test_predict(self, k=20, d=10):
        """Test lstsq._tikhonov.SolverTikhonovDecoupled.predict()."""
        solver = roi.lstsq.SolverTikhonovDecoupled(compute_extras=True,
                                                   check_regularizer=False)
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
        assert solver.misfit_.shape == (2,)
        assert solver.residual_.shape == (2,)
        assert np.all(solver.misfit_ <= solver.residual_)
        assert np.isclose(solver.misfit_.sum(),
                          la.norm(A @ X1 - B, ord='fro')**2)
        assert np.isclose(solver.cond_, np.linalg.cond(A))

        solver.compute_extras = False
        solver.predict(Ps)


def test_solver(m=20, n=10, k=5):
    """Test lstsq._tikhonov.solve()."""
    A = np.random.random((m,n))
    B = np.random.random((m,k))
    λs = 5 + np.random.random(k)
    Ps = [5 + np.random.random((n,n)) for _ in range(k)]
    Ps_diag = [5 + np.random.random(n) for _ in range(k)]

    # Bad number of regularization parameters.
    with pytest.raises(ValueError) as ex:
        roi.lstsq.solver(A, B, λs[:k-2])
    assert ex.value.args[0] == "invalid or misaligned input P"

    # Bad number of regularization matrices.
    with pytest.raises(ValueError) as ex:
        roi.lstsq.solver(A, B, Ps[:k-2])
    assert ex.value.args[0] == "invalid or misaligned input P"

    # Try to solve 1D problem with multiple regularizations.
    with pytest.raises(ValueError) as ex:
        roi.lstsq.solver(A, B[:,0], Ps)
    assert ex.value.args[0] == "invalid or misaligned input P"

    # Bad type for regularization.
    with pytest.raises(ValueError) as ex:
        roi.lstsq.solver(A, B, {})
    assert ex.value.args[0] == "invalid or misaligned input P"

    # Correct usage.
    solver = roi.lstsq.solver(A, B, 0)
    assert isinstance(solver, roi.lstsq.SolverL2)
    solver = roi.lstsq.solver(A, B, λs[0])
    assert isinstance(solver, roi.lstsq.SolverL2)
    solver = roi.lstsq.solver(A, B, λs)
    assert isinstance(solver, roi.lstsq.SolverL2Decoupled)
    solver = roi.lstsq.solver(A, B, Ps[0])
    assert isinstance(solver, roi.lstsq.SolverTikhonov)
    solver = roi.lstsq.solver(A, B, Ps)
    assert isinstance(solver, roi.lstsq.SolverTikhonovDecoupled)
    solver = roi.lstsq.solver(A, B, Ps_diag)
    assert isinstance(solver, roi.lstsq.SolverTikhonovDecoupled)
    roi.lstsq.solve(A, B, 0)
