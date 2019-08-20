# pre.py
"""Tools for pre-processing data """

import numpy as np
from scipy.linalg import svd as _svd
from scipy.sparse.linalg import svds as _svds
from sklearn.utils.extmath import randomized_svd as _rsvd

# Attempt to import numba (not included in the default requirements list).
try:
    import numba as _numba
except ImportError:                                         # pragma: no cover
    # If numba isn't installed, make an empty decorator in its place.
    def _numba(): pass
    def _jit(nopython=True): return lambda x: x
    _numba.jit = _jit


# Basis computation ===========================================================
def mean_shift(X):
    """Compute the mean of the columns of X, then use it to shift the columns
    so that they have mean zero.

    Parameters
    ----------
    X : (n,k) ndarray
        A matrix of k snapshots. Each column is a single snapshot.

    Returns
    -------
    xbar : (n,) ndarray
        The mean snapshot. Since this is a one-dimensional array, it must be
        reshaped to be applied to a matrix: Xshifted + xbar.reshape((-1,1)).

    Xshifted : (n,k) ndarray
        The matrix such that Xshifted[:,j] + xbar = X[:,j] for j=1,2,...,k.
    """
    # Check dimensions.
    if X.ndim != 2:
        raise ValueError("data X must be two-dimensional")

    xbar = np.mean(X, axis=1)               # Compute the mean column.
    Xshifted = X - xbar.reshape((-1,1))     # Shift the columns by the mean.
    return xbar, Xshifted


def pod_basis(X, r, mode="arpack", **options):
    """Compute the POD basis of rank r corresponding to the data in X.
    This function does NOT shift or scale the data before computing the basis.

    Parameters
    ----------
    X : (n,k) ndarray
        A matrix of k snapshots. Each column is a single snapshot.

    r : int
        The number of POD basis vectors to compute.

    mode : str
        The strategy to use for computing the truncated SVD. Options:
        * "simple": Use scipy.linalg.svd() to compute the entire SVD of X, then
            truncate it to get the first r left singular vectors of X. May be
            inefficient for very large matrices.
        * "arpack" (default): Use scipy.sparse.linalg.svds() to compute only
            the first r left singular vectors of X. This uses ARPACK for the
            eigensolver.
        * "randomized": Compute an approximate SVD with a randomized approach
            using sklearn.utils.extmath.randomized_svd(). This gives faster
            results at the cost of some accuracy.

    options
        Additional paramters for the SVD solver, which depends on `mode`:
        * "simple": scipy.linalg.svd()
        * "arpack": scipy.sparse.linalg.svds()
        * "randomized": sklearn.utils.extmath.randomized_svd()

    Returns
    -------
    Vr : (n,r) ndarray
        The first r POD basis vectors of X. Each column is one basis vector.
    """
    if mode == "simple":
        return _svd(X, full_matrices=False, **options)[0][:,:r]
    if mode == "arpack":
        return _svds(X, r, which="LM", **options)[0][:,::-1]
    elif mode == "randomized":
        return _rsvd(X, r, **options)[0][:,::-1]
    else:
        raise NotImplementedError(f"invalid mode '{mode}'")


# Derivative approximation ====================================================
@_numba.jit(nopython=True)
def _fwd4(y, dt):                                           # pragma: no cover
    """Compute the first derivative of a uniformly-spaced-in-time array with a
    fourth-order forward difference scheme.

    Parameters
    ----------
    y : (5,...) ndarray
        Data to differentiate. The derivative is taken along the first axis.

    Returns
    -------
    dy0 : float or (...) ndarray
        Approximate derivative of y at the first entry, i.e., dy[0] / dt.
    """
    return (-25*y[0] + 48*y[1] - 36*y[2] + 16*y[3] - 3*y[4]) / (12*dt)


@_numba.jit(nopython=True)
def _fwd6(y, dt):                                           # pragma: no cover
    """Compute the first derivative of a uniformly-spaced-in-time array with a
    sixth-order forward difference scheme.

    Parameters
    ----------
    y : (7,...) ndarray
        Data to differentiate. The derivative is taken along the first axis.

    Returns
    -------
    dy0 : float or (...) ndarray
        Approximate derivative of y at the first entry, i.e., dy[0] / dt.
    """
    return (-147*y[0] + 360*y[1] - 450*y[2] + 400*y[3] - 225*y[4] \
                                              + 72*y[5] - 10*y[6]) / (60*dt)


def xdot_uniform(X, dt, order=2):
    """Approximate the time derivatives for a chunk of snapshots that are
    uniformly spaced in time.

    Parameters
    ----------
    X : (n,k) ndarray
        The data to estimate the derivative of. The jth column is a snapshot
        that corresponds to the jth time step, i.e., X[:,j] = x(t[j]).

    dt : float
        The time step between the snapshots, i.e., t[j+1] - t[j] = dt.

    order : int {2, 4, 6}
        The order of the derivative approximation.
        See https://en.wikipedia.org/wiki/Finite_difference_coefficient.

    Returns
    -------
    Xdot : (n,k) ndarray
        Approximate time derivative of the snapshot data. The jth column is
        the derivative dx / dt corresponding to the jth snapshot, X[:,j].
    """
    # Check dimensions and input types.
    if X.ndim != 2:
        raise ValueError("data X must be two-dimensional")
    if not np.isscalar(dt):
        raise TypeError("time step dt must be a scalar (e.g., float)")

    if order == 2:
        return np.gradient(X, dt, edge_order=2, axis=1)

    Xdot = np.empty_like(X)
    n,k = X.shape
    if order == 4:
        # Central difference on interior
        Xdot[:,2:-2] = (X[:,:-4] - 8*X[:,1:-3] + 8*X[:,3:-1] - X[:,4:])/(12*dt)

        # Forward difference on the front.
        for j in range(2):
            Xdot[:,j] = _fwd4(X[:,j:j+5].T, dt)                 # Forward
            Xdot[:,-j-1] = -_fwd4(X[:,-j-5:k-j].T[::-1], dt)    # Backward

    elif order == 6:
        # Central difference on interior
        Xdot[:,3:-3] = (-X[:,:-6] + 9*X[:,1:-5] - 45*X[:,2:-4] \
                        + 45*X[:,4:-2] - 9*X[:,5:-1] + X[:,6:]) / (60*dt)

        # Forward / backward differences on the front / end.
        for j in range(3):
            Xdot[:,j] = _fwd6(X[:,j:j+7].T, dt)                 # Forward
            Xdot[:,-j-1] = -_fwd6(X[:,-j-7:k-j].T[::-1], dt)    # Backward

    else:
        raise NotImplementedError(f"invalid order '{order}'; "
                                  "valid options: {2, 4, 6}")

    return Xdot


def xdot_nonuniform(X, t):
    """Approximate the time derivatives for a chunk of snapshots with a
    second-order finite difference scheme.

    Parameters
    ----------
    X : (n,k) ndarray
        The data to estimate the derivative of. The jth column is a snapshot
        that corresponds to the jth time step, i.e., X[:,j] = x(t[j]).

    t : (k,) ndarray
        The times corresponding to the snapshots. May not be uniformly spaced.
        See xdot_uniform() for higher-order computation in the case of
        evenly-spaced-in-time snapshots.

    Returns
    -------
    Xdot : (n,k) ndarray
        Approximate time derivative of the snapshot data. The jth column is
        the derivative dx / dt corresponding to the jth snapshot, X[:,j].
    """
    # Check dimensions.
    if X.ndim != 2:
        raise ValueError("data X must be two-dimensional")
    if t.ndim != 1:
        raise ValueError("time t must be one-dimensional")
    if X.shape[-1] != t.shape[0]:
        raise ValueError("data X not aligned with time t")

    # Compute the derivative with a second-order difference scheme.
    return np.gradient(X, t, edge_order=2, axis=-1)


def xdot(X, *args, **kwargs):
    """Approximate the time derivatives for a chunk of snapshots with a finite
    difference scheme. Calls xdot_uniform() or xdot_nonuniform(), depending on
    the arguments.

    Parameters
    ----------
    X : (n,k) ndarray
        The data to estimate the derivative of. The jth column is a snapshot
        that corresponds to the jth time step, i.e., X[:,j] = x(t[j]).

    Additional parameters
    ---------------------
    dt : float
        The time step between the snapshots, i.e., t[j+1] - t[j] = dt.
    order : int {2, 4, 6} (optional)
        The order of the derivative approximation.
        See https://en.wikipedia.org/wiki/Finite_difference_coefficient.

    OR

    t : (k,) ndarray
        The times corresponding to the snapshots. May or may not be uniformly
        spaced.

    Returns
    -------
    Xdot : (n,k) ndarray
        Approximate time derivative of the snapshot data. The jth column is
        the derivative dx / dt corresponding to the jth snapshot, X[:,j].
    """
    n_args = len(args)          # Number of positional arguments (excluding X).
    n_kwargs = len(kwargs)      # Number of keyword arguments.
    n_total = n_args + n_kwargs # Total number of arguments (excluding X).

    if n_total == 0:
        raise TypeError("at least one other argument required (dt or t)")
    elif n_total == 1:              # There is only one other argument.
        if n_kwargs == 1:               # It is a keyword argument.
            arg_name = list(kwargs.keys())[0]
            if arg_name == "dt":
                func = xdot_uniform
            elif arg_name == "t":
                func = xdot_nonuniform
            elif arg_name == "order":
                raise TypeError("keyword argument 'order' requires float "
                                "argument dt")
            else:
                raise TypeError("xdot() got unexpected keyword argument "
                                f"'{arg_name}'")
        elif n_args == 1:               # It is a positional argument.
            arg = args[0]
            if isinstance(arg, float):          # arg = dt.
                func = xdot_uniform
            elif isinstance(arg, np.ndarray):   # arg = t; do uniformity test.
                func = xdot_nonuniform
            else:
                raise TypeError(f"invalid argument type '{type(arg)}'")
    elif n_total == 2:              # There are two other argumetns: dt, order.
        func = xdot_uniform
    else:
        raise TypeError("xdot() takes from 2 to 3 positional arguments "
                        f"but {n_total+1} were given")

    return func(X, *args, **kwargs)
