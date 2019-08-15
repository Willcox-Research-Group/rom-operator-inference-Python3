# pre.py
"""Tools for pre-processing data """

import numpy as np
import numba as _numba
from scipy.sparse.linalg import svds as _svds
from sklearn.utils.extmath import randomized_svd as _rsvd


def pod_basis(X, r, mode="arpack", **options):
    """Compute the POD basis of rank r corresponding to the data in X.
    This function does not shift or scale the data before computing the basis.

    Parameters
    ----------
    X : (n,k) ndarray
        A matrix of k snapshots. Each column is a single snapshot.

    r : int
        The number of POD basis vectors to compute.

    mode : str
        The strategy to use for computing the truncated SVD. Options:
        * "arpack": Use scipy.sparse.linalg.svds() to compute only the first r
            left singular vectors of X. This uses ARPACK for the eigensolver.
        * "randomized": Compute an approximate SVD with a randomized approach
            using sklearn.utils.extmath.randomized_svd(). This gives faster
            results at the cost of some accuracy.

    options
        Additional paramters for scipy.linalg.svds() if mode="arpack", or
        sklearn.utils.extmath.randomized_svd() if mode="randomized".

    Returns
    -------
    Vr : (n,r) ndarray
        The first r POD basis vectors of X. Each column is one basis vector.
    """
    if mode == "arpack":
        return _svds(X, r, which="LM", **options)[0][:,::-1]
    elif mode == "randomized":
        return _rsvd(X, r, **options)[0][:,::-1]
    else:
        raise ValueError(f"invalid mode '{mode}'")


@_numba.jit(nopython=True)
def _fwd4(Y, dt):
    """Compute the first column-wise derivative of a uniformly-spaced 2D array
    with a 4th order forward difference scheme.

    Parameters
    ----------
    Y : (n,5) ndarray
        The data to differentiate.

    Returns
    -------
    dy0 : (n,) ndarray
        Approximate derivative of Y at the first column, i.e., dY[:,0] / dt.
    """
    return (-25*Y[:,0] + 48*Y[:,1] - 36*Y[:,2] + 16*Y[:,3] - 3*Y[:,4])/(12*dt)


@_numba.jit(nopython=True)
def _fwd6(Y, dt):
    """Compute the first column-wise derivative of a uniformly-spaced 2D array
    with a 6th order forward difference scheme.

    Parameters
    ----------
    Y : (n,7) ndarray
        The data to differentiate.

    Returns
    -------
    dy0 : (n,) ndarray
        Approximate derivative of Y at the first column, i.e., dY[:,0] / dt.
    """
    return (-147*Y[:,0] + 360*Y[:,1] - 450*Y[:,2] + 400*Y[:,3] \
            - 225*Y[:,4] + 72*Y[:,5] - 10*Y[:,6]) / (60*dt)


def compute_xdot_uniform(X, dt, order=2):
    """Approximate the time derivatives for a chunk of snapshots that are
    uniformly spaced in time.

    Parameters
    ----------
    X : (n,k) ndarray
        The data to estimate the derivative of. The jth column is a snapshot
        that corresponds to the jth time step, i.e., x(t[j]) = X[:,j].

    dt : float
        The time step between the snapshots, i.e., t[j+1] - t[j] = dt.

    order : int {2, 4}
        The order of the derivative approximation.
        See https://en.wikipedia.org/wiki/Finite_difference_coefficient.

    Returns
    -------
    Xdot : (n,k) ndarray
        Approximate time derivative of the snapshot data. The jth column is
        the derivative dx / dt corresponding to the jth snapshot, X[:,j].
    """
    if order == 2:
        return np.gradient(X, dt, edge_order=2, axis=1)

    Xdot = np.empty_like(X)
    if order == 4:
        # Central difference on interior
        Xdot[:,2:-2] = (X[:,:-4] - 8*X[:,1:-3] + 8*X[:,3:-1] - X[:,4:])/(12*dt)

        # Forward difference on the front.
        Xdot[:,0] = _fwd4(X[:,:5], dt)
        Xdot[:,1] = _fwd4(X[:,1:6], dt)

        # Backward difference on the end.
        Xdot[:,-1] = -_fwd4(X[:,-5:][:,::-1], dt)
        Xdot[:,-2] = -_fwd4(X[:,-6:-1][:,::-1], dt)

    elif order == 6:
        # Central difference on interior
        Xdot[:,3:-3] = (-X[:,:-6] + 9*X[:,1:-5] - 45*X[:,2:-4] \
                        + 45*X[:,4:-2] - 9*X[:,5:-1] + X[:,6:]) / (60*dt)

        # Forward difference on the front.
        Xdot[:,0] = _fwd6(X[:,:7], dt)
        Xdot[:,1] = _fwd6(X[:,1:8], dt)
        Xdot[:,2] = _fwd6(X[:,2:9], dt)

        # Backward difference on the end.
        Xdot[:,-1] = -_fwd6(X[:,-7:][:,::-1], dt)
        Xdot[:,-2] = -_fwd6(X[:,-8:-1][:,::-1], dt)
        Xdot[:,-3] = -_fwd6(X[:,-9:-2][:,::-1], dt)

    else:
        raise ValueError(f"invalid order '{order}'")

    return Xdot


def compute_xdot(X, t):
    """Approximate the time derivatives for a chunk of snapshots with second-
    order finite differences.

    Parameters
    ----------
    X : (n,k) ndarray
        The data to estimate the derivative of. The jth column is a snapshot
        that corresponds to the jth time step, i.e., x(t[j]) = X[:,j].

    t : (k,) ndarray
        The times corresponding to the snapshots.

    Returns
    -------
    Xdot : (n,k) ndarray
        Approximate time derivative of the snapshot data. The jth column is
        the derivative dx / dt corresponding to the jth snapshot, X[:,j].
    """
    return np.gradient(X, t, edge_order=2, axis=1)
