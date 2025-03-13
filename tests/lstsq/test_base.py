# lstsq/_base.py
"""Tests for lstsq._base.py."""

import abc
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


class _TestSolverTemplate(abc.ABC):
    """Base class for tests of classes inheriting from SolverTemplate."""

    Solver = NotImplemented  # Class to test.
    test_1D_Z = True  # Whether to test one-dimensional rhs matrices.

    @abc.abstractmethod
    def get_solvers(self):
        """Yield (untrained) solvers to test."""
        raise NotImplementedError

    def test_fit_and_str(self, k=20, d=10, r=5):
        """Test fit() and lightly test __str__() and __repr__()."""
        D = np.random.random((k, d))
        Z = np.random.random((r, k))
        Zbad = np.random.random((r, k, d))

        for solver in self.get_solvers():

            repr(solver)

            # Bad shapes.
            with pytest.raises(ValueError) as ex:
                solver.fit(D[:, 0], Z)
            assert ex.value.args[0] == "data_matrix must be two-dimensional"

            with pytest.raises(ValueError) as ex:
                solver.fit(D, Zbad)
            assert ex.value.args[0] == (
                "lhs_matrix must be one- or two-dimensional"
            )

            # Mismatched shapes.
            with pytest.raises(opinf.errors.DimensionalityError) as ex:
                solver.fit(D, Z[:, :-1])
            assert ex.value.args[0] == (
                "data_matrix and lhs_matrix not aligned "
                f"(lhs_matrix.shape[-1] = {k - 1} "
                f"!= {k} = data_matrix.shape[0])"
            )

            # Correct usage, r > 1.
            assert solver.fit(D, Z) is solver
            assert solver.data_matrix is D
            assert solver.lhs_matrix is Z
            assert solver.k == k
            assert solver.d == d
            assert solver.r == r

            repr(solver)

            # Correct usage, r = 1.
            if self.test_1D_Z:
                assert solver.fit(D, Z[0]) is solver
                assert solver.data_matrix is D
                assert solver.lhs_matrix.shape == (1, k)
                assert solver.k == k
                assert solver.d == d
                assert solver.r == 1
                assert np.all(solver.lhs_matrix[0, :] == Z[0])

    @abc.abstractmethod
    def test_solve(self):
        """Test solve()."""
        raise NotImplementedError  # pragma: no cover

    # Post-processing ---------------------------------------------------------
    def test_cond(self, k=20, d=11, r=3):
        """Test cond()."""
        for solver in self.get_solvers():

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
        D = np.random.standard_normal((k, d))
        Z = np.random.standard_normal((r, k))

        for solver in self.get_solvers():

            # Try before calling fit().
            with pytest.raises(AttributeError) as ex:
                solver.residual(0)
            assert ex.value.args[0] == "solver not trained, call fit()"

            solver.fit(D, Z)

            # Try with badly shaped Ohat.
            Ohat = np.random.standard_normal((r - 1, d + 1))
            with pytest.raises(opinf.errors.DimensionalityError) as ex:
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
                assert np.isclose(
                    residual[i], la.norm(D @ Ohat[i] - Z[i]) ** 2
                )

            # One-dimensional case.
            if self.test_1D_Z:
                z = Z[0, :]
                solver.fit(D, z)
                assert solver.r == 1
                ohat = np.random.standard_normal(d)
                residual = solver.residual(ohat)
                assert isinstance(residual, np.ndarray)
                assert residual.shape == (1,)
                assert np.isclose(residual[0], la.norm(D @ ohat - z) ** 2)

    def test_save_load_and_copy_via_verify(self, k=20, d=11, r=6):
        """Use verify() to test save(), load(), and copy()."""
        D = np.random.random((k, d))
        Z = np.random.random((r, k))

        for solver in self.get_solvers():
            solver.fit(D, Z)
            solver.verify()


class TestPlainSolver(_TestSolverTemplate):
    """Test lstsq._base.PlainSolver."""

    Solver = opinf.lstsq.PlainSolver

    def get_solvers(self):
        """Yield solvers to test."""
        yield self.Solver(lapack_driver="gelsy")
        yield self.Solver(lapack_driver="gelsd", cond=1e-10)

    def test_fit_and_str(self):
        """Test fit() and lightly test __str__() and __repr__()."""

        # Underdetermined.
        k = 5
        d = 10
        r = 6
        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))

        solver = next(self.get_solvers())
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            solver.fit(D, Z)
        assert wn[0].message.args[0] == (
            "least-squares system is underdetermined"
        )

        return super().test_fit_and_str()

    def test_solve(self, k=20, d=11, r=3):
        """Test solve()."""
        # Set up and manually solve a least-squares problem.
        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))
        U, s, Vt = la.svd(D, full_matrices=False)
        Ohat_true = Z @ U @ np.diag(1 / s) @ Vt

        # Check the least-squares solution.
        for solver in self.get_solvers():
            Ohat = solver.fit(D, Z).solve()
            assert np.allclose(Ohat, Ohat_true)


if __name__ == "__main__":
    pytest.main([__file__])
