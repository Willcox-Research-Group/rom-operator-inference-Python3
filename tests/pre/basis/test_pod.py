# pre/basis/test_pod.py
"""Tests for rom_operator_inference.pre.basis._pod."""

import os
import h5py
import pytest
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

import rom_operator_inference as opinf


class TestPODBasis:
    """Test pre.basis._pod.PODBasis."""
    PODBasis = opinf.pre.PODBasis

    class DummyTransformer(opinf.pre.transform._base._BaseTransformer):
        """Instantiable version of _BaseTransformer."""
        def fit_transform(self, states):
            return states + 1

        def transform(self, states):
            return self.fit_transform(states)

        def inverse_transform(self, states):
            return states - 1

        def save(self, hf):
            pass

    def test_init(self):
        """Test PODBasis.init()."""
        basis = self.PODBasis()
        assert basis.transformer is None
        assert basis.r is None
        assert basis.n is None
        assert basis.shape is None
        assert basis.entries is None
        assert basis.svdvals is None
        assert basis.dual is None
        assert not basis.economize

    def test_dimensions(self, n=20, r=5):
        """Test PODBasis.r, economize, and _shrink_stored_entries_to()."""
        basis = self.PODBasis(economize=False)

        # Try setting the basis dimension before setting the entries.
        with pytest.raises(AttributeError) as ex:
            basis.r = 10
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
            basis.r = r + 2
        assert ex.value.args[0] == f"only {r} basis vectors stored"

        # Shrink dimension and blow it back up (economize=False).
        basis.r = r - 1
        assert basis.shape == (n, r - 1)
        assert np.all(basis.entries == Vr[:, :-1])
        assert np.all(basis.dual == Wtr[:-1].T)
        assert basis.svdvals.shape == sr.shape
        assert np.all(basis.svdvals == sr)
        assert basis.rmax == r

        basis.r = r
        assert basis.shape == (n, r)
        assert np.all(basis.entries == Vr)
        assert np.all(basis.dual == Wtr.T)
        assert basis.svdvals.shape == sr.shape
        assert np.all(basis.svdvals == sr)
        assert basis.rmax == r

        # Shrink the dimension (economize=True).
        basis.economize = True
        basis.r = r - 1
        assert basis.shape == (n, r - 1)
        assert np.all(basis.entries == Vr[:, :-1])
        assert np.all(basis.dual == Wtr[:-1].T)
        assert basis.svdvals.shape == sr.shape
        assert np.all(basis.svdvals == sr)
        assert basis.rmax == r - 1

        # Try to recover forgotten columns.
        with pytest.raises(ValueError) as ex:
            basis.r = r
        assert ex.value.args[0] == f"only {r-1} basis vectors stored"

        # Ensure setting economize = True shrinks the dimension.
        basis.economize = False
        basis._store_svd(Vr, sr, Wtr)
        basis.r = r
        assert basis.rmax == r
        basis.r = r - 1
        assert basis.r == r - 1
        assert basis.rmax == r
        basis.economize = True
        assert basis.r == r - 1
        assert basis.rmax == r - 1
        with pytest.raises(ValueError) as ex:
            basis.r = r
        assert ex.value.args[0] == f"only {r-1} basis vectors stored"

    def test_set_dimension(self, n=20, r=5):
        """Test PODBasis.set_dimension()."""
        basis = self.PODBasis(economize=False)

        # Try setting dimension without singular values.
        with pytest.raises(AttributeError) as ex:
            basis.set_dimension(r=None, cumulative_energy=.9985)
        assert ex.value.args[0] == "no singular value data (call fit() first)"

        V, _, Wt = la.svd(np.random.standard_normal((n, n)))
        svdvals = np.sqrt([.9, .09, .009, .0009, .00009, .000009, .0000009])
        Vr, Wtr = V[:, :r], Wt[:r]
        basis._store_svd(Vr, svdvals, Wtr)

        # Choose dimension based on an energy criteria.
        basis._store_svd(Vr, svdvals, Wtr)
        basis.set_dimension(cumulative_energy=.9999)
        assert basis.r == 4
        basis.set_dimension(residual_energy=.01)
        assert basis.r == 2

    def test_fit(self, n=20, k=15, r=5):
        """Test PODBasis.fit()."""
        # First test PODBasis.validate_rank().
        states = np.empty((n, n))
        with pytest.raises(ValueError) as ex:
            self.PODBasis._validate_rank(states, n + 1)
        assert ex.value.args[0] == f"invalid POD rank r = {n + 1} " \
                                   f"(need 1 ≤ r ≤ {n})"

        self.PODBasis._validate_rank(states, n // 2)

        # Now test PODBasis.fit().
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
        assert basis.r == r
        assert np.allclose(basis.entries, U[:, :r])
        assert np.allclose(basis.svdvals, vals)
        assert np.allclose(basis.dual, Wt[:r, :].T)

        # Test with a transformer.
        basis = self.PODBasis(transformer=self.DummyTransformer())
        basis.fit(states, r)
        U, vals, Wt = la.svd(states + 1, full_matrices=False)
        assert np.allclose(basis.entries, U[:, :r])
        assert np.allclose(basis.svdvals, vals)
        assert np.allclose(basis.dual, Wt[:r, :].T)

        # TODO: weighted inner product matrix.

    def test_fit_randomized(self, n=20, k=14, r=5, tol=1e-6):
        """Test PODBasis.fit_randomized()."""
        states = np.random.standard_normal((n, k))
        U, vals, Wt = la.svd(states, full_matrices=False)
        basis = self.PODBasis().fit_randomized(states, r)
        assert basis.entries.shape == (n, r)
        assert basis.svdvals.shape == (r,)
        assert basis.dual.shape == (k, r)
        VrTVr = basis.entries.T @ basis.entries
        Ir = np.eye(r)
        assert np.allclose(VrTVr, Ir)
        WrTWr = basis.dual.T @ basis.dual
        assert np.allclose(WrTWr, Ir)
        assert basis.r == r
        # Flip the signs in U and W if needed so things will match.
        for i in range(r):
            if np.sign(U[0, i]) != np.sign(basis[0, i]):
                U[:, i] *= -1
            if np.sign(Wt[i, 0]) != np.sign(basis.dual[0, i]):
                Wt[i, :] *= -1
        assert la.norm(basis.entries - U[:, :r], ord=2) < tol
        assert la.norm(basis.svdvals - vals[:r]) / la.norm(basis.svdvals) < tol
        assert la.norm(basis.dual - Wt[:r, :].T, ord=2) < tol

        # Test with a transformer.
        states = np.random.standard_normal((n, n))
        basis = self.PODBasis(transformer=self.DummyTransformer())
        basis.fit_randomized(states, None)
        U, vals, Wt = la.svd(states + 1, full_matrices=False)
        # Flip the signs in U and W if needed so things will match.
        for i in range(n):
            if np.sign(U[0, i]) != np.sign(basis[0, i]):
                U[:, i] *= -1
            if np.sign(Wt[i, 0]) != np.sign(basis.dual[0, i]):
                Wt[i, :] *= -1
        assert la.norm(basis.entries - U, ord=2) < tol
        assert la.norm(basis.svdvals - vals) / la.norm(basis.svdvals) < tol
        assert la.norm(basis.dual - Wt.T, ord=2) < tol

    # Visualization -----------------------------------------------------------
    def test_plots(self, n=40, k=25, r=4):
        """Lightly test PODBasis.plot_*()."""
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

        ax = basis.plot_cumulative_energy(threshold=.999)
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
    def test_save(self, n=20, k=14, r=6):
        """Lightly test PODBasis.save()."""
        # Clean up after old tests.
        target = "_podbasissavetest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Just save a basis to a temporary file, don't interrogate the file.
        basis = self.PODBasis().fit(np.random.random((n, k)), r)
        basis.save(target)
        assert os.path.isfile(target)

        # Repeat with a transformer.
        basis = self.PODBasis(transformer=self.DummyTransformer())
        basis.fit(np.random.random((n, k)), r)
        basis.save(target, overwrite=True)
        assert os.path.isfile(target)
        os.remove(target)

    def test_load(self, n=20, k=14, r=6):
        """Test PODBasis.load()."""
        # Clean up after old tests.
        target = "_podbasisloadtest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Try to load a bad file.
        with h5py.File(target, "w"):
            pass

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.PODBasis.load(target)
        assert ex.value.args[0] == "invalid save format (meta/ not found)"

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

        # Repeat with a transformer.
        st = opinf.pre.transform.SnapshotTransformer()
        basis1 = self.PODBasis(transformer=st).fit(np.random.random((n, k)), r)
        basis1.save(target, overwrite=True)
        basis2 = self.PODBasis.load(target)
        assert basis1 == basis2


class PODBasisMulti:
    """Test opinf.pre.basis._pod.PODBasisMulti."""
    # PODBasis = opinf.pre.PODBasisMulti
    pass


# Basis computation ===========================================================
def test_pod_basis(set_up_basis_data):
    """Test pre.basis._pod.pod_basis()."""
    Q = set_up_basis_data
    n, k = Q.shape

    # Try with an invalid rank.
    rmax = min(n, k)
    with pytest.raises(ValueError) as exc:
        opinf.pre.pod_basis(Q, rmax+1)
    assert exc.value.args[0] == \
        f"invalid POD rank r = {rmax+1} (need 1 ≤ r ≤ {rmax})"

    with pytest.raises(ValueError) as exc:
        opinf.pre.pod_basis(Q, -1)
    assert exc.value.args[0] == \
        f"invalid POD rank r = -1 (need 1 ≤ r ≤ {rmax})"

    # Try with an invalid mode.
    with pytest.raises(NotImplementedError) as exc:
        opinf.pre.pod_basis(Q, None, mode="full")
    assert exc.value.args[0] == "invalid mode 'full'"

    U, vals, Wt = la.svd(Q, full_matrices=False)
    for r in [2, 10, rmax]:
        Ur = U[:, :r]
        vals_r = vals[:r]
        Wr = Wt[:r, :].T
        Id = np.eye(r)

        for mode in ("dense", "sparse", "randomized"):

            print(r, mode)
            basis, svdvals = opinf.pre.pod_basis(Q, r, mode=mode)
            _, _, W = opinf.pre.pod_basis(Q, r, mode=mode, return_W=True)
            assert basis.shape == (n, r)
            assert np.allclose(basis.T @ basis, Id)
            assert W.shape == (k, r)
            assert np.allclose(W.T @ W, Id)

            if mode == "dense":
                assert svdvals.shape == (rmax,)
            if mode in ("sparse", "randomized"):
                assert svdvals.shape == (r,)
                # Make sure the basis vectors have the same sign.
                for j in range(r):
                    if not np.isclose(basis[0, j], Ur[0, j]):
                        basis[:, j] *= -1
                    if not np.isclose(W[0, j], Wr[0, j]):
                        W[:, j] *= -1

            if mode != "randomized":
                # Accuracy tests (none for randomized SVD).
                assert np.allclose(basis, Ur)
                assert np.allclose(svdvals[:r], vals_r)
                assert np.allclose(W, Wr)


# Reduced dimension selection =================================================
def test_svdval_decay(set_up_basis_data):
    """Test pre.basis._pod.svdval_decay()."""
    Q = set_up_basis_data
    svdvals = la.svdvals(Q)

    # Single cutoffs.
    r = opinf.pre.svdval_decay(svdvals, 1e-14, plot=False)
    assert isinstance(r, int) and r >= 1

    # Multiple cutoffss.
    rs = opinf.pre.svdval_decay(svdvals, [1e-10, 1e-12], plot=False)
    assert isinstance(rs, list)
    for r in rs:
        assert isinstance(r, int) and r >= 1
    assert rs == sorted(rs)

    # Plotting.
    status = plt.isinteractive()
    plt.ion()
    rs = opinf.pre.svdval_decay(svdvals, .0001, plot=True)
    assert len(plt.gcf().get_axes()) == 1
    rs = opinf.pre.svdval_decay(svdvals, [1e-4, 1e-8, 1e-12], plot=True)
    assert len(plt.gcf().get_axes()) == 1
    plt.interactive(status)
    plt.close("all")

    # Specific test.
    svdvals = [.9, .09, .009, .0009, .00009, .000009, .0000009]
    rs = opinf.pre.svdval_decay(svdvals, [.8, .1, .0004],
                                normalize=False, plot=False)
    assert len(rs) == 3
    assert rs == [1, 1, 4]

    svdvals = np.array([1e1, 1e2, 1e3, 1e0, 1e-2]) - 1e-3
    rs = opinf.pre.svdval_decay(svdvals, [9e-1, 9e-2, 5e-4, 0],
                                normalize=True, plot=False)
    assert len(rs) == 4
    assert rs == [1, 2, 4, 5]


def test_cumulative_energy(set_up_basis_data):
    """Test pre.basis._pod.cumulative_energy()."""
    Q = set_up_basis_data
    svdvals = la.svdvals(Q)
    energy = np.cumsum(svdvals**2)/np.sum(svdvals**2)

    def _test(r, thresh):
        assert isinstance(r, int)
        assert r >= 1
        assert energy[r-1] >= thresh
        assert np.all(energy[:r-2] < thresh)

    # Single threshold.
    thresh = .9
    r = opinf.pre.cumulative_energy(svdvals, thresh, plot=False)
    _test(r, thresh)

    # Multiple thresholds.
    thresh = [.9, .99, .999]
    rs = opinf.pre.cumulative_energy(svdvals, thresh, plot=False)
    assert isinstance(rs, list)
    for r, t in zip(rs, thresh):
        _test(r, t)
    assert rs == sorted(rs)

    # Plotting.
    status = plt.isinteractive()
    plt.ion()
    rs = opinf.pre.cumulative_energy(svdvals, .999, plot=True)
    assert len(plt.gcf().get_axes()) == 1
    rs = opinf.pre.cumulative_energy(svdvals, [.9, .99, .999], plot=True)
    assert len(plt.gcf().get_axes()) == 1
    plt.interactive(status)
    plt.close("all")

    # Specific test.
    svdvals = np.sqrt([.9, .09, .009, .0009, .00009, .000009, .0000009])
    rs = opinf.pre.cumulative_energy(svdvals, [.9, .99, .999], plot=False)
    assert len(rs) == 3
    assert rs == [1, 2, 3]


def test_residual_energy(set_up_basis_data):
    """Test pre.basis._pod.residual_energy()."""
    Q = set_up_basis_data
    svdvals = la.svdvals(Q)
    resid = 1 - np.cumsum(svdvals**2)/np.sum(svdvals**2)

    def _test(r, tol):
        assert isinstance(r, int)
        assert r >= 1
        assert resid[r-1] <= tol
        assert np.all(resid[:r-2] > tol)

    # Single tolerance.
    tol = 1e-2
    r = opinf.pre.residual_energy(svdvals, tol, plot=False)
    _test(r, tol)

    # Multiple tolerances.
    tols = [1e-2, 1e-4, 1e-6]
    rs = opinf.pre.residual_energy(svdvals, tols, plot=False)
    assert isinstance(rs, list)
    for r, t in zip(rs, tols):
        _test(r, t)
    assert rs == sorted(rs)

    # Plotting.
    status = plt.isinteractive()
    plt.ion()
    rs = opinf.pre.residual_energy(svdvals, 1e-3, plot=True)
    assert len(plt.gcf().get_axes()) == 1
    rs = opinf.pre.cumulative_energy(svdvals, [1e-2, 1e-4, 1e-6], plot=True)
    assert len(plt.gcf().get_axes()) == 1
    plt.interactive(status)
    plt.close("all")


def test_projection_error(set_up_basis_data):
    """Test pre.basis._pod.projection_error()."""
    Q = set_up_basis_data
    Vr = la.svd(Q, full_matrices=False)[0][:, :Q.shape[1]//3]

    abserr, relerr = opinf.pre.projection_error(Q, Vr)
    assert np.isscalar(abserr)
    assert abserr >= 0
    assert np.isscalar(relerr)
    assert relerr >= 0
    assert np.isclose(abserr, relerr * la.norm(Q))
