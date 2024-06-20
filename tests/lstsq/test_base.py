# lstsq/_base.py
"""Tests for lstsq._base.py."""

import os
import pytest
import numpy as np
import scipy.linalg as la

import opinf


def test_lstsq_size():
    """Test lstsq.lstsq_size()."""
    m, r = 3, 7

    # Try with bad input combinations.
    with pytest.raises(ValueError) as ex:
        opinf.lstsq.lstsq_size("cAHB", r)
    assert ex.value.args[0] == "argument m > 0 required since 'B' in modelform"

    with pytest.raises(ValueError) as ex:
        opinf.lstsq.lstsq_size("cAH", r, m=10)
    assert ex.value.args[0] == "argument m=10 invalid since 'B' in modelform"

    # Test without inputs.
    assert opinf.lstsq.lstsq_size("c", r) == 1
    assert opinf.lstsq.lstsq_size("A", r) == r
    assert opinf.lstsq.lstsq_size("cA", r) == 1 + r
    assert opinf.lstsq.lstsq_size("cAH", r) == 1 + r + r * (r + 1) // 2
    assert opinf.lstsq.lstsq_size("cG", r) == 1 + r * (r + 1) * (r + 2) // 6

    # Test with inputs.
    assert opinf.lstsq.lstsq_size("cB", r, m) == 1 + m
    assert opinf.lstsq.lstsq_size("AB", r, m) == r + m
    assert opinf.lstsq.lstsq_size("cAB", r, m) == 1 + r + m
    assert opinf.lstsq.lstsq_size("AHB", r, m) == r + r * (r + 1) // 2 + m
    assert opinf.lstsq.lstsq_size("GB", r, m) == r * (r + 1) * (r + 2) // 6 + m

    # Test with affines.
    assert opinf.lstsq.lstsq_size("c", r, affines={"c": [0, 0]}) == 2
    assert opinf.lstsq.lstsq_size("A", r, affines={"A": [0, 0]}) == 2 * r


class TestSolverTemplate:
    """Test lstsq._base.SolverTemplate."""

    class Dummy(opinf.lstsq._base.SolverTemplate):
        """Instantiable version of SolverTemplate."""

        def predict(self):
            return np.ones((self.r, self.d))

    def test_properties(self):
        """Test data_matrix, lhs_matrix, k, d, r, properties."""
        solver = self.Dummy()
        for attr in ("data_matrix", "lhs_matrix", "k", "d", "r"):
            assert hasattr(solver, attr)
            assert getattr(solver, attr) is None

    def test_fit(self, k=30, d=20, r=5):
        """Test fit()."""
        solver = self.Dummy()

        # Bad dimensions.
        D = np.random.random(k)
        Z = np.random.random((r, k))
        with pytest.raises(ValueError) as ex:
            solver.fit(D, Z)
        assert ex.value.args[0] == "data_matrix must be two-dimensional"

        D = np.random.random((k, d))
        Z = np.random.random((r, k, d))
        with pytest.raises(ValueError) as ex:
            solver.fit(D, Z)
        assert ex.value.args[0] == "lhs_matrix must be one- or two-dimensional"

        # Mismatched shapes.
        D = np.random.random((k, d))
        Z = np.random.random((r, k - 1))
        with pytest.raises(ValueError) as ex:
            solver.fit(D, Z)
        assert ex.value.args[0] == (
            "data_matrix and lhs_matrix not aligned "
            f"(lhs_matrix.shape[-1] = {k - 1} != {k} = data_matrix.shape[0])"
        )

        # Correct usage, r > 1.
        D = np.random.random((k, d))
        Z = np.random.random((r, k))
        assert solver.fit(D, Z) is solver
        assert solver.data_matrix is D
        assert solver.lhs_matrix is Z
        assert solver.k == k
        assert solver.d == d
        assert solver.r == r

        # Correct usage, r = 1.
        Z = np.random.random(k)
        assert solver.fit(D, Z) is solver
        assert solver.data_matrix is D
        assert solver.lhs_matrix.shape == (1, k)
        assert solver.k == k
        assert solver.d == d
        assert solver.r == 1
        assert np.all(solver.lhs_matrix[0, :] == Z)

    # String representations --------------------------------------------------
    def test_str(self, k=20, d=6, r=3):
        """Test __str__() and __repr__()."""
        # Before fitting.
        solver = self.Dummy()
        assert str(solver) == "Dummy (not trained)"

        rep = repr(solver)
        assert rep.startswith("<Dummy object at ")
        assert len(rep.split("\n")) == 2

        D = np.empty((k, d))
        Z = np.empty((r, k))
        solver.fit(D, Z)
        strlines = str(solver).split("\n")
        assert len(strlines) == 5
        assert strlines[0] == "Dummy"
        assert strlines[1] == f"  Data matrix:     ({k:d}, {d:d})"
        assert strlines[2].startswith("    Condition number")
        assert strlines[3] == f"  LHS matrix:      ({r:d}, {k:d})"
        assert strlines[4] == f"  Operator matrix: ({r:d}, {d:d})"

        replines = repr(solver).split("\n")
        assert len(replines) == 6
        assert replines[1:] == strlines

    # Post-processing ---------------------------------------------------------
    def test_cond(self, k=20, d=11, r=3):
        """Test cond()."""
        solver = self.Dummy()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.cond()
        assert ex.value.args[0] == "solver not trained, call fit()"

        # Contrived test 1
        D = np.eye(d)
        Z = np.zeros((r, d))
        solver.fit(D, Z)
        assert np.isclose(solver.cond(), 1)

        # Contrived test 2
        D = np.diag(np.arange(1, d + 1))
        Z = np.zeros((r, d))
        solver.fit(D, Z)
        assert np.isclose(solver.cond(), d)

        # Random test
        D = np.random.standard_normal((k, d))
        Z = np.random.standard_normal((r, k))
        svals = la.svdvals(D)
        solver.fit(D, Z)
        assert np.isclose(solver.cond(), svals[0] / svals[-1])

    def test_residual(self, k=20, d=10, r=4):
        """Test residual()."""
        solver = self.Dummy()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.residual(0)
        assert ex.value.args[0] == "solver not trained, call fit()"

        D = np.random.standard_normal((k, d))
        Z = np.random.standard_normal((r, k))
        solver.fit(D, Z)

        # Try with badly shaped Ohat.
        Ohat = np.random.standard_normal((r - 1, d + 1))
        with pytest.raises(ValueError) as ex:
            solver.residual(Ohat)
        assert ex.value.args[0] == (
            f"Ohat.shape = {(r - 1, d + 1)} != {(r, d)} = (r, d)"
        )

        # Two-dimensional case.
        Ohat = np.random.standard_normal((r, d))
        residual = solver.residual(Ohat)
        assert isinstance(residual, np.ndarray)
        assert residual.shape == (r,)
        for i in range(r):
            assert np.isclose(residual[i], la.norm(D @ Ohat[i] - Z[i]) ** 2)

        # One-dimensional case.
        z = Z[0, :]
        solver.fit(D, z)
        assert solver.r == 1
        ohat = np.random.standard_normal(d)
        residual = solver.residual(ohat)
        assert isinstance(residual, float)
        assert np.isclose(residual, la.norm(D @ ohat - z) ** 2)

    def test_copy(self, k=18, d=10, r=4):
        """Test copy()."""
        solver = self.Dummy()
        solver2 = solver.copy()
        assert solver2 is not solver
        assert isinstance(solver2, self.Dummy)
        assert solver2.data_matrix is None
        assert solver2.lhs_matrix is None

        D = np.random.random((k, d))
        Z = np.random.random((r, k))
        solver.fit(D, Z)
        solver2 = solver.copy()
        assert solver2 is not solver
        assert isinstance(solver2, self.Dummy)
        assert solver2.r == r
        assert solver2.k == k
        assert solver2.d == d
        assert np.all(solver2.data_matrix == D)
        assert np.all(solver2.lhs_matrix == Z)

    # Verification ------------------------------------------------------------
    def test_verify(self):
        """Test verify()."""

        def _single(DClass, message):
            dummy = DClass()
            with pytest.raises(opinf.errors.VerificationError) as ex:
                dummy.verify()
            assert ex.value.args[0].startswith(message)

        class Dummy2(self.Dummy):
            def copy(self):
                return 10

        _single(Dummy2, "Dummy2.copy() returned object of type 'int'")

        class Dummy3(self.Dummy):
            def fit(self, D, Z):
                self.Ohat = D - Z

        _single(Dummy3, "fit() failed")

        class Dummy4(self.Dummy):
            def fit(self, D, Z):
                pass

        _single(Dummy4, "fit() should call SolverTemplate.fit()")

        class Dummy5(self.Dummy):
            def predict(self):
                return np.empty((1, 1))

        _single(Dummy5, "predict() did not return array of shape (r, d)")

        class Dummy6(self.Dummy):
            def copy(self):
                newsolver = Dummy6()
                if self.data_matrix is not None:
                    newsolver.fit(self.data_matrix[:, 1:], self.lhs_matrix)
                return newsolver

        _single(Dummy6, "copy() does not preserve problem dimensions")

        class Dummy7(self.Dummy):
            def predict(self):
                return np.random.random((self.r, self.d))

        _single(Dummy7, "copy() does not preserve the result of predict()")

        class Dummy8(self.Dummy):
            def save(self, savefile, overwrite=False):
                Dummy8.D = self.data_matrix
                Dummy8.Z = self.lhs_matrix

            @classmethod
            def load(cls, loadfile):
                newsolver = Dummy8()
                if cls.D is not None:
                    newsolver.fit(cls.D[:, 1:], cls.Z)
                return newsolver

        _single(Dummy8, "save()/load() does not preserve problem dimensions")

        self.Dummy().verify()


class TestPlainSolver:
    """Test lstsq._base.PlainSolver."""

    Solver = opinf.lstsq.PlainSolver

    def test_fit(self):
        """Test fit()."""
        solver = self.Solver(lapack_driver="gelsy")

        # Underdetermined.
        k = 5
        d = 10
        r = 6
        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            solver.fit(D, Z)
        assert wn[0].message.args[0] == (
            "least-squares system is underdetermined"
        )

        # Overdetermined.
        k = 15
        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))
        solver = self.Solver(lapack_driver="gelsy", cond=1e-4)
        out = solver.fit(D, Z)
        assert out is solver

        repr(solver)

    def test_predict(self, k=20, d=11, r=3):
        """Test predict()."""
        # Set up and manually solve a least-squares problem.
        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))
        U, s, Vt = la.svd(D, full_matrices=False)
        Ohat_true = Z @ U @ np.diag(1 / s) @ Vt

        # Check the least-squares solution.
        solver = self.Solver().fit(D, Z)
        Ohat = solver.predict()
        assert np.allclose(Ohat, Ohat_true)

    def test_save(self, k=6, d=4, r=2, outfile="_plainsolversavetest.h5"):
        """Lightly test save()."""
        if os.path.isfile(outfile):  # pragma: no cover
            os.remove(outfile)

        solver = self.Solver(lapack_driver="gelsy", cond=1e-14)
        solver.save(outfile)

        assert os.path.isfile(outfile)

        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))
        solver = self.Solver().fit(D, Z)
        solver.save(outfile, overwrite=True)

        os.remove(outfile)

    def test_load(self, k=10, d=6, r=3, outfile="_plainsolverloadtest.h5"):
        """Test that load() is the inverse of save()."""
        if os.path.isfile(outfile):  # pragma: no cover
            os.remove(outfile)

        solver = self.Solver()
        solver.save(outfile)
        solver2 = self.Solver.load(outfile)
        assert solver2.data_matrix is None

        solver = self.Solver(lapack_driver="gelsy", cond=1e-12)
        solver.save(outfile, overwrite=True)
        solver2 = self.Solver.load(outfile)
        assert solver2.data_matrix is None
        assert solver2.options["lapack_driver"] == "gelsy"
        assert solver2.options["cond"] == 1e-12

        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))
        solver = self.Solver().fit(D, Z)
        solver.save(outfile, overwrite=True)
        solver2 = self.Solver.load(outfile)
        assert solver2.r == r
        assert solver2.k == k
        assert solver2.d == d
        assert np.all(solver2.data_matrix == D)
        assert np.all(solver2.lhs_matrix == Z)

        os.remove(outfile)

    def test_copy(self, k=18, d=10, r=4):
        """Test copy()."""
        solver = self.Solver(lapack_driver="gelsy")
        solver2 = solver.copy()
        assert solver2 is not solver
        assert isinstance(solver2, self.Solver)
        assert solver2.data_matrix is None
        assert solver2.lhs_matrix is None
        assert solver2.options["lapack_driver"] == "gelsy"

        solver = self.Solver(cond=2e-3)
        D = np.random.random((k, d))
        Z = np.random.random((r, k))
        solver.fit(D, Z)
        solver2 = solver.copy()
        assert solver2 is not solver
        assert isinstance(solver2, self.Solver)
        assert solver2.options["cond"] == 2e-3
        assert solver2.r == r
        assert solver2.k == k
        assert solver2.d == d
        assert np.all(solver2.data_matrix == D)
        assert np.all(solver2.lhs_matrix == Z)
