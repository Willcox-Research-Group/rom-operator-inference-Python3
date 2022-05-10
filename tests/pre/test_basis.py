# pre/test_basis.py
"""Tests for rom_operator_inference.pre._basis.py"""

import pytest
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

import rom_operator_inference as opinf


# Basis computation ===========================================================
def test_pod_basis(set_up_basis_data):
    """Test pre._basis.pod_basis()."""
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
    """Test pre._basis.svdval_decay()."""
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
    """Test pre._basis.cumulative_energy()."""
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
    """Test pre._basis.residual_energy()."""
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
    """Test pre._basis.projection_error()."""
    Q = set_up_basis_data
    Vr = la.svd(Q, full_matrices=False)[0][:, :Q.shape[1]//3]

    abserr, relerr = opinf.pre.projection_error(Q, Vr)
    assert np.isscalar(abserr)
    assert abserr >= 0
    assert np.isscalar(relerr)
    assert relerr >= 0
    assert np.isclose(abserr, relerr * la.norm(Q))
