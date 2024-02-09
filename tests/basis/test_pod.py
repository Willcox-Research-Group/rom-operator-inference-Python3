# basis/test_pod.py
"""Tests for basis._pod."""

import os
import pytest
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

import opinf


class TestPODBasis:
    """Test basis._pod.PODBasis."""

    PODBasis = opinf.basis.PODBasis

    def test_init(self):
        """Test __init__()."""
        basis = self.PODBasis()
        assert basis.reduced_state_dimension is None
        assert basis.full_state_dimension is None
        assert basis.shape is None
        assert basis.entries is None
        assert basis.svdvals is None
        assert basis.dual is None
        assert not basis.economize
        assert basis.rmax is None

    def test_dimensions(self, n=20, r=5):
        """Test r, economize, and _shrink_stored_entries_to()."""
        basis = self.PODBasis(economize=False)

        # Try setting the basis dimension before setting the entries.
        with pytest.raises(AttributeError) as ex:
            basis.reduced_state_dimension = 10
        assert ex.value.args[0] == "empty basis (call fit() first)"

        # Test _store_svd() real quick.
        V, s, Wt = la.svd(np.random.standard_normal((n, n)))
        Vr, sr, Wtr = V[:, :r], s[:r], Wt[:r]
        basis._store_svd(Vr, sr, Wtr)
        assert np.all(basis.entries == Vr)
        assert np.allclose(basis.dual, Wtr.T)
        assert np.all(basis.svdvals == sr)
        assert basis.rmax == r

        # Try setting the dimension too high.
        with pytest.raises(ValueError) as ex:
            basis.reduced_state_dimension = r + 2
        assert ex.value.args[0] == f"only {r} basis vectors stored"

        # Shrink dimension and blow it back up (economize=False).
        basis.reduced_state_dimension = r - 1
        assert basis.shape == (n, r - 1)
        assert np.all(basis.entries == Vr[:, :-1])
        assert np.all(basis.dual == Wtr[:-1].T)
        assert basis.svdvals.shape == sr.shape
        assert np.all(basis.svdvals == sr)
        assert basis.rmax == r

        basis.reduced_state_dimension = r
        assert basis.shape == (n, r)
        assert np.all(basis.entries == Vr)
        assert np.all(basis.dual == Wtr.T)
        assert basis.svdvals.shape == sr.shape
        assert np.all(basis.svdvals == sr)
        assert basis.rmax == r

        # Shrink the dimension (economize=True).
        basis.economize = True
        basis.reduced_state_dimension = r - 1
        assert basis.shape == (n, r - 1)
        assert np.all(basis.entries == Vr[:, :-1])
        assert np.all(basis.dual == Wtr[:-1].T)
        assert basis.svdvals.shape == sr.shape
        assert np.all(basis.svdvals == sr)
        assert basis.rmax == r - 1

        # Try to recover forgotten columns.
        with pytest.raises(ValueError) as ex:
            basis.reduced_state_dimension = r
        assert ex.value.args[0] == f"only {r-1} basis vectors stored"

        # Ensure setting economize = True shrinks the dimension.
        basis.economize = False
        basis._store_svd(Vr, sr, Wtr)
        basis.reduced_state_dimension = r
        assert basis.rmax == r
        basis.reduced_state_dimension = r - 1
        assert basis.reduced_state_dimension == r - 1
        assert basis.rmax == r
        basis.economize = True
        assert basis.reduced_state_dimension == r - 1
        assert basis.rmax == r - 1
        with pytest.raises(ValueError) as ex:
            basis.reduced_state_dimension = r
        assert ex.value.args[0] == f"only {r-1} basis vectors stored"

        # Tests what happens when the dimension is set to 1.
        basis.economize = False
        basis._store_svd(Vr, sr, Wtr)
        basis.reduced_state_dimension = 1
        assert basis.shape == (n, 1)
        assert basis.dual.shape == (n, 1)
        assert basis.svdvals.shape == (r,)
        assert basis.compress(
            np.random.random(
                n,
            )
        ).shape == (1,)
        assert basis.compress(np.random.random((n, n))).shape == (1, n)

    def test_set_dimension(self, n=20):
        """Test set_dimension()."""
        basis = self.PODBasis(economize=False)

        # Try setting dimension without singular values.
        with pytest.raises(AttributeError) as ex:
            basis.set_dimension(r=None, cumulative_energy=0.9985)
        assert ex.value.args[0] == "no singular value data (call fit() first)"

        V, _, Wt = la.svd(np.random.standard_normal((n, n)))
        svdvals = np.sqrt(
            [0.9, 0.09, 0.009, 0.0009, 0.00009, 0.000009, 0.0000009]
        )

        # Default: use all basis vectors.
        basis._store_svd(V, svdvals, Wt)
        basis.set_dimension()
        assert basis.reduced_state_dimension == n

        # Set specified dimension.
        basis._store_svd(V, svdvals, Wt)
        basis.set_dimension(n - 1)
        assert basis.reduced_state_dimension == n - 1

        # Choose dimension based on an energy criteria.
        basis._store_svd(V, svdvals, Wt)
        basis.set_dimension(cumulative_energy=0.9999)
        assert basis.reduced_state_dimension == 4
        basis.set_dimension(residual_energy=0.01)
        assert basis.reduced_state_dimension == 2

    def test_str(self):
        """Lightly test __str__() and LinearBasis.__repr__()."""
        basis = self.PODBasis()
        assert str(basis) == "Empty PODBasis"
        assert repr(basis).startswith("<PODBasis object at ")

    def test_fit(self, n=20, k=15, r=5):
        """Test fit()."""
        # First test validate_rank().
        states = np.empty((n, n))
        with pytest.raises(ValueError) as ex:
            self.PODBasis._validate_rank(states, n + 1)
        assert (
            ex.value.args[0] == f"invalid POD rank r = {n + 1} "
            f"(need 1 ≤ r ≤ {n})"
        )

        self.PODBasis._validate_rank(states, n // 2)

        # Now test fit().
        states = np.random.standard_normal((n, k))
        U, vals, Wt = la.svd(states, full_matrices=False)
        basis = self.PODBasis().fit(states, r)
        assert basis.entries.shape == (n, r)
        assert basis.svdvals.shape == (min(n, k),)
        assert basis.dual.shape == (k, r)
        VrTVr = basis.entries.T @ basis.entries
        Ir = np.eye(r)
        assert np.allclose(VrTVr, Ir)
        WrTWr = basis.dual.T @ basis.dual
        assert np.allclose(WrTWr, Ir)
        assert basis.reduced_state_dimension == r
        assert np.allclose(basis.entries, U[:, :r])
        assert np.allclose(basis.svdvals, vals)
        assert np.allclose(basis.dual, Wt[:r, :].T)

        # TODO: weighted inner product matrix.

        # Repeat with list of state trajectories.
        states = [np.random.standard_normal((n, n)) for _ in range(4)]
        basis = self.PODBasis()
        basis.fit(states)
        assert basis.full_state_dimension == n

    def test_fit_randomized(self, n=20, k=14, r=5, tol=1e-6):
        """Test fit_randomized()."""
        options = dict(n_oversamples=30, n_iter=10, random_state=42)
        states = np.random.standard_normal((n, k))
        U, vals, Wt = la.svd(states, full_matrices=False)
        basis = self.PODBasis().fit_randomized(states, r, **options)
        assert basis.entries.shape == (n, r)
        assert basis.svdvals.shape == (r,)
        assert basis.dual.shape == (k, r)
        VrTVr = basis.entries.T @ basis.entries
        Ir = np.eye(r)
        assert np.allclose(VrTVr, Ir)
        WrTWr = basis.dual.T @ basis.dual
        assert np.allclose(WrTWr, Ir)
        assert basis.reduced_state_dimension == r
        # Flip the signs in U and W if needed so things will match.
        for i in range(r):
            if np.sign(U[0, i]) != np.sign(basis[0, i]):
                U[:, i] *= -1
            if np.sign(Wt[i, 0]) != np.sign(basis.dual[0, i]):
                Wt[i, :] *= -1
        assert la.norm(basis.entries - U[:, :r], ord=2) < tol
        assert la.norm(basis.svdvals - vals[:r]) / la.norm(basis.svdvals) < tol
        assert la.norm(basis.dual - Wt[:r, :].T, ord=2) < tol

        # Repeat with list of state trajectories.
        states = [np.random.standard_normal((n, n)) for _ in range(4)]
        basis = self.PODBasis()
        options.pop("random_state")
        basis.fit_randomized(states, r, **options)
        assert basis.full_state_dimension == n

    # Visualization -----------------------------------------------------------
    def test_plots(self, n=40, k=25, r=4):
        """Lightly test plot_*()."""
        basis = self.PODBasis().fit(np.random.standard_normal((n, k)))

        # Turn interactive mode on.
        _pltio = plt.isinteractive()
        plt.ion()

        # Call each plotting routine.
        ax = basis.plot_svdval_decay(threshold=1e-3, normalize=True)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        ax = basis.plot_residual_energy(threshold=1e-3)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        ax = basis.plot_cumulative_energy(threshold=0.999)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        fig, axes = basis.plot_energy()
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        for ax in axes.flat:
            assert isinstance(ax, plt.Axes)
        plt.close(fig)

        # Restore interactive mode setting.
        plt.interactive(_pltio)

    # Persistence -------------------------------------------------------------
    def test_save(self, n=20, k=14, r=6, target="_podbasissavetest.h5"):
        """Lightly test save()."""
        # Clean up after old tests.
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        # Just save a basis to a temporary file, don't interrogate the file.
        basis = self.PODBasis().fit(np.random.random((n, k)), r)
        basis.save(target)
        assert os.path.isfile(target)

        os.remove(target)

    def test_load(self, n=20, k=14, r=6):
        """Test load()."""
        # Clean up after old tests.
        target = "_podbasisloadtest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        # Test that save() and load() are inverses for an empty basis.
        basis1 = self.PODBasis(economize=True)
        basis1.save(target, overwrite=True)
        basis2 = self.PODBasis.load(target)
        assert isinstance(basis2, self.PODBasis)
        assert basis2.full_state_dimension is None
        assert basis2.r is None
        assert basis2.entries is None
        assert basis2.dual is None
        assert basis2.economize is True
        assert basis1 == basis2

        # Just save a basis to a temporary file, don't interrogate the file.
        basis1 = self.PODBasis().fit(np.random.random((n, k)), r)
        basis1.save(target, overwrite=True)

        # Test that save() and load() are inverses.
        basis2 = self.PODBasis.load(target)
        assert basis1.r == basis2.r
        assert basis1.entries.shape == basis2.entries.shape
        assert np.allclose(basis1.entries, basis2.entries)
        assert basis1.svdvals.shape == basis2.svdvals.shape
        assert np.allclose(basis1.svdvals, basis2.svdvals)
        assert basis1.dual.shape == basis2.dual.shape
        assert np.allclose(basis1.dual, basis2.dual)
        assert basis1 == basis2

        # Test rmax gives a smaller basis.
        rnew = basis1.r - 2
        basis2 = self.PODBasis.load(target, rmax=rnew)
        assert basis2.r == basis1.r - 2
        assert basis2.entries.shape == (basis1.entries.shape[0], rnew)
        assert basis2.svdvals.shape == (rnew,)
        assert basis2.dual.shape == (basis1.dual.shape[0], rnew)
        assert np.allclose(basis2.entries, basis1.entries[:, :rnew])
        assert np.allclose(basis2.svdvals, basis1.svdvals[:rnew])
        assert np.allclose(basis2.dual, basis1.dual[:, :rnew])

        # Clean up.
        os.remove(target)
