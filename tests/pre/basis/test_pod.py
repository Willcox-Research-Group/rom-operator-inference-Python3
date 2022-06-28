# pre/basis/test_pod.py
"""Tests for rom_operator_inference.pre.basis._pod."""

import pytest
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

import rom_operator_inference as opinf


class TestPODBasis:
    """Test pre.basis._pod.PODBasis."""
    PODBasis = opinf.pre.PODBasis

    # TODO: class DummyTransformer: ...

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

    def test_set_dimension(self, n=20, r=5):
        """Test PODBasis.set_dimension()."""
        basis = self.PODBasis(economize=False)

        # Try setting the basis dimension before setting the entries.
        with pytest.raises(AttributeError) as ex:
            basis.r = 10
        assert ex.value.args[0] == "empty basis (call fit() first)"

        with pytest.raises(AttributeError) as ex:
            basis.set_dimension(10)
        assert ex.value.args[0] == "empty basis (call fit() first)"

        # Try setting dimension without singular values.
        with pytest.raises(AttributeError) as ex:
            basis.set_dimension(r=None, cumulative_energy=.9985)
        assert ex.value.args[0] == "no singular value data (call fit() first)"

        # Test _store_svd() real quick.
        V, s, Wt = la.svd(np.random.standard_normal((n, n)))
        Vr, sr, Wtr = V[:, :r], s[:r], Wt[:r]
        basis._store_svd(Vr, sr, Wtr)
        assert np.all(basis.entries == Vr)
        assert np.allclose(basis.dual, Wtr.T)
        assert np.all(basis.svdvals == sr)

        # Try setting the dimension too high.
        with pytest.raises(ValueError) as ex:
            basis.set_dimension(r + 2)
        assert ex.value.args[0] == f"only {r} basis vectors stored"

        # Shrink dimension and blow it back up (economize=False).
        basis.set_dimension(r - 1)
        assert basis.shape == (n, r - 1)
        assert np.all(basis.entries == Vr[:, :-1])
        assert np.all(basis.dual == Wtr[:-1].T)
        assert basis.svdvals.shape == sr.shape
        assert np.all(basis.svdvals == sr)

        basis.set_dimension(r)
        assert basis.shape == (n, r)
        assert np.all(basis.entries == Vr)
        assert np.all(basis.dual == Wtr.T)
        assert basis.svdvals.shape == sr.shape
        assert np.all(basis.svdvals == sr)

        # Shrink the dimension (economize=True).
        basis.economize = True
        basis.set_dimension(r - 1)
        assert basis.shape == (n, r - 1)
        assert np.all(basis.entries == Vr[:, :-1])
        assert np.all(basis.dual == Wtr[:-1].T)
        assert basis.svdvals.shape == sr.shape
        assert np.all(basis.svdvals == sr)

        # Try to recover forgotten columns.
        with pytest.raises(ValueError) as ex:
            basis.set_dimension(r)
        assert ex.value.args[0] == f"only {r-1} basis vectors stored"

        # Choose dimension based on an energy criteria.
        basis.economize = False
        svdvals = np.sqrt([.9, .09, .009, .0009, .00009, .000009, .0000009])
        basis._store_svd(Vr, svdvals, Wtr)
        basis.set_dimension(cumulative_energy=.9999)
        assert basis.r == 4
        basis.set_dimension(residual_energy=.01)
        assert basis.r == 2

    def test_fit(self):
        """Test PODBasis.fit()."""
        pass

    def test_fit_randomized(self):
        """Test PODBasis.fit_randomized()."""
        pass

    def test_save(self):
        """Test PODBasis.save()."""
        pass

    def test_load(self):
        """Test PODBasis.load()."""
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
