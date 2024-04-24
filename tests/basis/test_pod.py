# basis/test_pod.py
"""Tests for basis._pod."""

import os
import pytest
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

import opinf


def _spd(n):
    """Generate a random symmetric postive definite nxn matrix."""
    u, s, _ = la.svd(np.random.standard_normal((n, n)))
    return u @ np.diag(s + 1) @ u.T


def test_Wmult(n=10):
    """Test basis._pod._Wmult()."""
    func = opinf.basis._pod._Wmult

    w = np.random.random(n) + 0.1
    W = np.diag(w)

    # One-dimensional operand
    q = np.random.random(n)
    Wq = W @ q
    for ww in w, W:
        out = func(ww, q)
        assert out.shape == (n,)
        assert np.allclose(out, Wq)

    # Two-dimensional operand
    Q = np.random.random((n, n))
    WQ = W @ Q
    for ww in w, W:
        out = func(ww, Q)
        assert out.shape == (n, n)
        assert np.allclose(out, WQ)

    with pytest.raises(ValueError) as ex:
        func(w, np.random.random((2, 2, 2)))
    assert ex.value.args[0] == "expected one- or two-dimensional array"


class TestPODBasis:
    """Test basis._pod.PODBasis."""

    Basis = opinf.basis.PODBasis

    # Constructors ------------------------------------------------------------
    def test_init(self):
        """Test __init__() andd properties."""
        # Valid instantiation.
        basis = self.Basis(num_vectors=10, name="testbasis")
        for attr in (
            "reduced_state_dimension",
            "full_state_dimension",
            "entries",
            "weights",
            "leftvecs",
            "svdvals",
            "rightvecs",
            "residual_energy",
            "cumulative_energy",
        ):
            assert getattr(basis, attr) is None
        assert basis.name == "testbasis"
        assert isinstance(basis.svdsolver_options, dict)
        assert len(basis.svdsolver_options) == 0

        # Weights
        w = np.ones(10)
        self.Basis(num_vectors=2, weights=w)
        self.Basis(num_vectors=2, weights=np.diag(w))

        # Setter for svdsolver.
        with pytest.raises(AttributeError) as ex:
            basis.svdsolver = "smartly"
        assert ex.value.args[0].startswith(
            "invalid svdsolver 'smartly', options: "
        )
        basis.svdsolver = "randomized"
        assert basis.svdsolver == "randomized"

        # Setter for svdsolver_options.
        with pytest.raises(TypeError) as ex:
            basis.svdsolver_options = 10
        assert ex.value.args[0] == "svdsolver_options must be a dictionary"
        basis.svdsolver_options["full_matrices"] = False
        basis.svdsolver_options = None
        assert isinstance(basis.svdsolver_options, dict)

        with pytest.raises(ValueError) as ex:
            self.Basis(num_vectors=10, max_vectors=-3)
        assert ex.value.args[0] == "max_vectors must be a positive integer"

    def test_from_svd(self, n=50, k=20, r=6):
        """Test from_svd() pseudoconstructor."""
        Q = np.random.random((n, k))
        Phi, svals, PsiT = la.svd(Q, full_matrices=False)
        W = _spd(n)

        basis = self.Basis.from_svd(Phi, svals)
        assert basis.full_state_dimension == n
        assert basis.reduced_state_dimension == k
        assert basis.rightvecs is None
        assert np.all(basis.entries == Phi)

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            basis = self.Basis.from_svd(
                Phi,
                svals,
                PsiT.T,
                num_vectors=r,
                weights=W,
            )
        assert wn[0].message.args[0] == "basis not orthogonal"
        assert basis.full_state_dimension == n
        assert basis.reduced_state_dimension == r
        assert basis.rightvecs is not None
        assert np.all(basis.entries == Phi[:, :r])

    # Dimension management ----------------------------------------------------
    def test_set_dimension(self, n=40, k=11, r=9):
        basis = self.Basis(num_vectors=10)

        # Dimension selection criteria
        with pytest.warns(opinf.errors.UsageWarning) as wn:
            basis._set_dimension_selection_criterion(
                num_vectors=20,
                residual_energy=0.01,
                cumulative_energy=0.999,
            )
        assert wn[0].message.args[0] == (
            "received multiple dimension selection criteria, "
            "using num_vectors=20"
        )

        with pytest.raises(ValueError) as ex:
            basis._set_dimension_selection_criterion()
        assert ex.value.args[0] == (
            "exactly one dimension selection criterion must be provided"
        )

        basis._set_dimension_selection_criterion(residual_energy=1e-6)
        criterion = basis._PODBasis__criterion
        assert criterion[0] == "residual_energy"
        assert criterion[1] == 1e-6

        # Dimension setting (empty basis).
        basis.reduced_state_dimension = 5
        criterion = basis._PODBasis__criterion
        assert criterion[0] == "num_vectors"
        assert criterion[1] == 5
        assert basis.reduced_state_dimension == 5

        basis.set_dimension(cumulative_energy=0.999)
        criterion = basis._PODBasis__criterion
        assert criterion[0] == "cumulative_energy"
        assert criterion[1] == 0.999

        # Dimension setting (existing basis).
        Phi, svals, PsiT = la.svd(
            np.random.random((n, k)),
            full_matrices=False,
        )
        svals = np.sqrt(10 ** -np.arange(k, dtype=float))
        Q = Phi @ np.diag(svals) @ PsiT
        total = np.sum(svals**2)

        basis = self.Basis.from_svd(Phi, svals)
        assert basis.reduced_state_dimension == k
        assert basis.max_vectors == k

        basis.reduced_state_dimension = 3
        assert basis.reduced_state_dimension == 3
        assert basis.max_vectors == k
        assert basis.cumulative_energy == 1.11 / total

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            basis.reduced_state_dimension = k + 2
        assert wn[0].message.args[0] == (
            "selected reduced dimension exceeds number of stored vectors, "
            f"setting reduced_state_dimension = max_vectors = {k}"
        )
        assert basis.reduced_state_dimension == k

        basis.set_dimension(num_vectors=3)
        assert basis.reduced_state_dimension == 3

        basis.set_dimension(svdval_threshold=(svals[3] + svals[4]) / 2)
        assert basis.reduced_state_dimension == 4

        basis.set_dimension(cumulative_energy=0.99998)
        assert basis.reduced_state_dimension == 5

        basis.set_dimension(residual_energy=2e-2)
        assert basis.reduced_state_dimension == 2

        basis.set_dimension(projection_error=0.02)
        assert basis.projection_error(Q, relative=True) < 0.02

    def test_str(self, n=30, k=20, r=10):
        """Test __str__()."""
        basis = self.Basis(num_vectors=r)
        strbasis = str(basis)
        assert strbasis.count("\n") == 1
        assert strbasis.endswith("SVD solver: scipy.linalg.svd()")

        Q = np.random.random((n, k))
        basis.fit(Q)
        strbasis = str(basis)
        assert strbasis.count(f"Full state dimension    n = {n}") == 1
        assert strbasis.count(f"Reduced state dimension r = {r}") == 1
        assert strbasis.count(f"{k} basis vectors available") == 1
        assert strbasis.count("Cumulative energy:") == 1
        assert strbasis.count("Residual energy:") == 1
        assert strbasis.endswith("SVD solver: scipy.linalg.svd()")

        basis = self.Basis(
            num_vectors=r,
            max_vectors=r,
            svdsolver="randomized",
        ).fit(Q)
        strbasis = str(basis)
        assert strbasis.count(f"Full state dimension    n = {n}") == 1
        assert strbasis.count(f"Reduced state dimension r = {r}") == 1
        assert strbasis.count(f"{r} basis vectors available") == 1
        assert strbasis.count("Approximate cumulative energy:") == 1
        assert strbasis.count("Approximate residual energy:") == 1
        assert strbasis.endswith("sklearn.utils.extmath.randomized_svd()")

        basis = self.Basis(num_vectors=r, svdsolver=lambda s: s)
        strbasis = str(basis)
        assert strbasis.endswith("SVD solver: custom lambda function")

        def mysvdsolver(*args):
            pass

        basis = self.Basis(num_vectors=r, svdsolver=mysvdsolver)
        strbasis = str(basis)
        assert strbasis.endswith("SVD solver: mysvdsolver()")

    def test_fit(self, n=60, k=20, r=4):
        """Test fit()."""
        Q = np.random.random((n, k))

        # Dense, unweighted.
        basis = self.Basis(num_vectors=r, max_vectors=k + 2, svdsolver="dense")
        with pytest.warns(opinf.errors.UsageWarning) as wn:
            out = basis.fit(Q)
        assert wn[0].message.args[0] == (
            f"only {k} singular vectors can be extracted from ({n} x {k}) "
            f"snapshots, setting max_vectors={k}"
        )
        assert out is basis
        assert basis.full_state_dimension == n
        assert basis.reduced_state_dimension == r
        assert basis.max_vectors == k
        assert np.allclose(basis.entries.T @ basis.entries, np.eye(r))

        # Dense, weighted.
        w = np.random.random(n) + 0.1
        W = np.diag(w)
        Id = np.eye(r)
        basis = self.Basis(num_vectors=r, svdsolver="dense", weights=w)

        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            basis.fit(Q[:-1, :])
        assert ex.value.args[0] == (
            f"states not aligned with weights, should have {n} rows"
        )

        basis.fit(Q)
        assert np.allclose(basis.entries.T @ W @ basis.entries, Id)
        basis = self.Basis(num_vectors=r, svdsolver="dense", weights=W).fit(Q)
        assert np.allclose(basis.entries.T @ W @ basis.entries, Id)
        W = _spd(n)
        basis = self.Basis(num_vectors=r, svdsolver="dense", weights=W).fit(Q)
        assert np.allclose(basis.entries.T @ W @ basis.entries, Id)

        # Randomized.
        basis = self.Basis(
            num_vectors=r,
            svdsolver="randomized",
            max_vectors=r + 1,
        )
        Q = [np.random.random((n, k // 3)) for _ in range(3)]
        basis.fit(Q)
        assert basis.full_state_dimension == n
        assert basis.reduced_state_dimension == r
        assert basis.max_vectors == r + 1
        assert np.allclose(basis.entries.T @ basis.entries, Id)

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            basis.set_dimension(residual_energy=1e-2)
        assert wn[0].message.args[0] == (
            "residual energy is being estimated from only "
            f"{r + 1} singular values"
        )

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            basis.set_dimension(cumulative_energy=0.9)
        assert wn[0].message.args[0] == (
            "cumulative energy is being estimated from only "
            f"{r + 1} singular values"
        )

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            basis.set_dimension(projection_error=0.1)
        assert wn[0].message.args[0] == (
            "projection error is being estimated from only "
            f"{r + 1} singular values"
        )

    # Visualization -----------------------------------------------------------
    def test_plots(self, n=40, r=4):
        """Lightly test plot_*()."""
        basis = self.Basis(num_vectors=r)

        with pytest.raises(AttributeError) as ex:
            basis.plot_svdval_decay(threshold=1e-3)
        assert ex.value.args[0] == "no singular value data, call fit()"

        basis.fit(np.diag(10 ** np.arange(n)))

        # Turn interactive mode on.
        _pltio = plt.isinteractive()
        plt.ion()

        # Call each plotting routine.
        ax = basis.plot_svdval_decay(threshold=4e-1, right=(n - 2))
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        ax = basis.plot_residual_energy(threshold=1e-1, right=(n - 10))
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        ax = basis.plot_cumulative_energy(threshold=0.75, right=15.2)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        basis.plot_energy()
        plt.close("all")

        # Restore interactive mode setting.
        plt.interactive(_pltio)

    # Persistence -------------------------------------------------------------
    def test_save(self, n=20, k=14, r=6, target="_podbasissavetest.h5"):
        """Lightly test save()."""
        # Clean up after old tests.
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        # Just save a basis to a temporary file, don't interrogate the file.
        basis = self.Basis(num_vectors=r, name="testbasis")
        basis.save(target)
        assert os.path.isfile(target)

        basis.fit(np.random.random((n, k)))
        basis.save(target, overwrite=True)
        assert os.path.isfile(target)

        basis = self.Basis(
            cumulative_energy=0.99,
            max_vectors=10,
            weights=np.ones(10),
        )
        basis.save(target, overwrite=True)
        assert os.path.isfile(target)

        os.remove(target)

    def test_load(self, n=20, k=14, r=6):
        """Test load()."""
        # Clean up after old tests.
        target = "_podbasisloadtest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        # Test that save() and load() are inverses for an empty basis.
        w = np.random.random(n) + 1
        basis1 = self.Basis(
            num_vectors=r,
            weights=w,
            name="testbasis",
            max_vectors=2 * r,
        )
        basis1.save(target, overwrite=True)
        basis2 = self.Basis.load(target)
        assert basis1 == basis2

        # Test that save() and load() are inverses for a nonempty basis.
        basis1.fit(np.random.random((n, k)))
        basis1.save(target, overwrite=True)
        basis2 = self.Basis.load(target)
        assert basis1 == basis2

        # Test max_vectors gives a smaller basis.
        rnew = basis1.reduced_state_dimension - 2
        basis2 = self.Basis.load(target, max_vectors=rnew)
        assert basis2.full_state_dimension == n
        assert basis2.reduced_state_dimension == rnew
        assert basis2.entries.shape == (basis1.entries.shape[0], rnew)
        assert basis2.max_vectors == rnew
        assert basis2.svdvals.shape == (k,)
        assert np.allclose(basis2.svdvals, basis1.svdvals)
        assert np.allclose(basis1.entries[:, :-2], basis2.entries)

        # Clean up.
        os.remove(target)


def test_pod_basis(n=40, k=20, r=3):
    """Test basis._pod.pod_basis()."""
    Q = np.random.random((n, k))
    V, svals = opinf.basis.pod_basis(Q, num_vectors=r)
    assert V.shape == (n, r)
    assert svals.shape == (k,)
    assert np.allclose(V.T @ V, np.eye(r))

    V, svals, W = opinf.basis.pod_basis(
        Q - 1,
        num_vectors=r,
        return_rightvecs=True,
    )
    assert V.shape == (n, r)
    assert svals.shape == (k,)
    assert W.shape == (k, r)
    assert np.allclose(V.T @ V, np.eye(r))
