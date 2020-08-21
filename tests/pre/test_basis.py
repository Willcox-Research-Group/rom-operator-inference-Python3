# pre/test_basis.py
"""Tests for rom_operator_inference.pre._basis.py"""

import pytest
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

import rom_operator_inference as roi

from . import set_up_basis_data


# Basis computation ===========================================================
def test_pod_basis(set_up_basis_data):
    """Test pre._basis.pod_basis() on a small case with the ARPACK solver."""
    X = set_up_basis_data
    n,k = X.shape

    # Try with an invalid rank.
    rmax = min(n,k)
    with pytest.raises(ValueError) as exc:
        roi.pre.pod_basis(X, rmax+1)
    assert exc.value.args[0] == \
        f"invalid POD rank r = {rmax+1} (need 1 <= r <= {rmax})"

    with pytest.raises(ValueError) as exc:
        roi.pre.pod_basis(X, -1)
    assert exc.value.args[0] == \
        f"invalid POD rank r = -1 (need 1 <= r <= {rmax})"

    # Try with an invalid mode.
    with pytest.raises(NotImplementedError) as exc:
        roi.pre.pod_basis(X, None, mode="full")
    assert exc.value.args[0] == "invalid mode 'full'"

    U, vals, _ = la.svd(X, full_matrices=False)
    for r in [2, 10, rmax]:
        Ur = U[:,:r]
        vals_r = vals[:r]

        # Via scipy.linalg.svd().
        Vr, svdvals = roi.pre.pod_basis(X, r, mode="dense")
        assert Vr.shape == (n,r)
        assert np.allclose(Vr, Ur)
        assert svdvals.shape == (r,)
        assert np.allclose(svdvals, vals_r)

        # Via scipy.sparse.linalg.svds() (ARPACK).
        Vr, svdvals = roi.pre.pod_basis(X, r, mode="sparse")
        assert Vr.shape == (n,r)
        for j in range(r):      # Make sure the columns have the same sign.
            if not np.isclose(Ur[0,j], Vr[0,j]):
                Vr[:,j] = -Vr[:,j]
        assert np.allclose(Vr, Ur)
        assert svdvals.shape == (r,)
        assert np.allclose(svdvals, vals_r)

        # Via sklearn.utils.extmath.randomized_svd().
        Vr, svdvals = roi.pre.pod_basis(X, r, mode="randomized")
        assert Vr.shape == (n,r)
        # Light accuracy test (equality not guaranteed by randomized SVD).
        assert la.norm(np.abs(Vr) - np.abs(Ur)) < 5
        assert svdvals.shape == (r,)
        assert la.norm(svdvals - vals_r) < 3


# Reduced dimension selection =================================================
def test_svdval_decay(set_up_basis_data):
    """Test pre._basis.svdval_decay()."""
    X = set_up_basis_data
    svdvals = la.svdvals(X)

    # Single cutoffs.
    r = roi.pre.svdval_decay(svdvals, 1e-14, plot=False)
    assert isinstance(r, int) and r >= 1

    # Multiple cutoffss.
    rs = roi.pre.svdval_decay(svdvals, [1e-10,1e-12], plot=False)
    assert isinstance(rs, list)
    for r in rs:
        assert isinstance(r, int) and r >= 1
    assert rs == sorted(rs)

    # Plotting.
    status = plt.isinteractive()
    plt.ion()
    rs = roi.pre.svdval_decay(svdvals, .0001, plot=True)
    assert len(plt.gcf().get_axes()) == 1
    rs = roi.pre.svdval_decay(svdvals, [1e-4, 1e-8, 1e-12], plot=True)
    assert len(plt.gcf().get_axes()) == 1
    plt.interactive(status)
    plt.close("all")

    # Specific test.
    svdvals = [.9, .09, .009, .0009, .00009, .000009, .0000009]
    rs = roi.pre.svdval_decay(svdvals, [.8, .1, .0004], plot=False)
    assert len(rs) == 3
    assert rs == [1, 1, 4]


def test_cumulative_energy(set_up_basis_data):
    """Test pre._basis.cumulative_energy()."""
    X = set_up_basis_data
    svdvals = la.svdvals(X)

    # Single threshold.
    r = roi.pre.cumulative_energy(svdvals, .9, plot=False)
    assert isinstance(r, np.int64) and r >= 1

    # Multiple thresholds.
    rs = roi.pre.cumulative_energy(svdvals, [.9, .99, .999], plot=False)
    assert isinstance(rs, list)
    for r in rs:
        assert isinstance(r, np.int64) and r >= 1
    assert rs == sorted(rs)

    # Plotting.
    status = plt.isinteractive()
    plt.ion()
    rs = roi.pre.cumulative_energy(svdvals, .999, plot=True)
    assert len(plt.gcf().get_axes()) == 1
    rs = roi.pre.cumulative_energy(svdvals, [.9, .99, .999], plot=True)
    assert len(plt.gcf().get_axes()) == 1
    plt.interactive(status)
    plt.close("all")

    # Specific test.
    svdvals = np.sqrt([.9, .09, .009, .0009, .00009, .000009, .0000009])
    rs = roi.pre.cumulative_energy(svdvals, [.9, .99, .999], plot=False)
    assert len(rs) == 3
    assert rs == [1, 2, 3]


def test_projection_error(set_up_basis_data):
    """Test pre._basis.projection_error()."""
    X = set_up_basis_data
    Vr = la.svd(X, full_matrices=False)[0][:,:X.shape[1]//3]

    err = roi.pre.projection_error(X, Vr)
    assert np.isscalar(err) and err >= 0


def test_minimal_projection_error(set_up_basis_data):
    """Test pre._basis.minimal_projection_error()."""
    X = set_up_basis_data
    V = la.svd(X, full_matrices=False)[0][:,:X.shape[1]//3]

    # Try with bad data shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.minimal_projection_error(np.ravel(X), V, 1e-14, plot=False)
    assert exc.value.args[0] == "data X must be two-dimensional"

    # Try with bad basis shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.minimal_projection_error(X, V[0], 1e-14, plot=False)
    assert exc.value.args[0] == "basis V must be two-dimensional"

    # Single cutoffs.
    r = roi.pre.minimal_projection_error(X, V, 1e-14, plot=False)
    assert isinstance(r, int) and r >= 1

    # Multiple cutoffs.
    rs = roi.pre.minimal_projection_error(X, V, [1e-10, 1e-12], plot=False)
    assert isinstance(rs, list)
    for r in rs:
        assert isinstance(r, int) and r >= 1
    assert rs == sorted(rs)

    # Plotting
    status = plt.isinteractive()
    plt.ion()
    roi.pre.minimal_projection_error(X, V, .0001, plot=True)
    assert len(plt.gcf().get_axes()) == 1
    roi.pre.minimal_projection_error(X, V, [1e-4, 1e-6, 1e-10], plot=True)
    assert len(plt.gcf().get_axes()) == 1
    plt.interactive(status)
    plt.close("all")
