# pre.py
"""Tools for preprocessing data."""

import numpy as _np
from scipy import linalg as _la
from scipy.sparse import linalg as _spla
from sklearn.utils import extmath as _sklmath
from matplotlib import pyplot as _plt


# Shifting and MinMax scaling =================================================
def shift(X, shift_by=None):
    """Shift the columns of X by a vector.

    Parameters
    ----------
    X : (n,k) ndarray
        A matrix of k snapshots. Each column is a single snapshot.

    shift_by : (n,) or (n,1) ndarray
        A vector that is the same size as a single snapshot. If None,
        set to the mean of the columns of X.

    Returns
    -------
    Xshifted : (n,k) ndarray
        The matrix such that Xshifted[:,j] = X[:,j] - shift_by for j=0,...,k-1.

    xbar : (n,) ndarray
        The shift factor. Since this is a one-dimensional array, it must be
        reshaped to be applied to a matrix: Xshifted + xbar.reshape((-1,1)).
        Only returned if shift_by=None.

    For shift_by=None, only Xshifted is returned.

    Examples
    --------
    # Shift X by its mean, then shift Y by the same mean.
    >>> Xshifted, xbar = shift(X)
    >>> Yshifted = shift(Y, xbar)

    # Shift X by its mean, then undo the transformation by an inverse shift.
    >>> Xshifted, xbar = shift(X)
    >>> X_again = shift(Xshifted, -xbar)
    """
    # Check dimensions.
    if X.ndim != 2:
        raise ValueError("data X must be two-dimensional")

    # If not shift_by factor is provided, compute the mean column.
    learning = (shift_by is None)
    if learning:
        shift_by = _np.mean(X, axis=1)
    elif shift_by.ndim != 1:
        raise ValueError("shift_by must be one-dimensional")

    # Shift the columns by the mean.
    Xshifted = X - shift_by.reshape((-1,1))

    return (Xshifted, shift_by) if learning else Xshifted


def scale(X, scale_to, scale_from=None):
    """Scale the entries of the snapshot matrix X from the interval
    [scale_from[0], scale_from[1]] to [scale_to[0], scale_to[1]].
    Scaling algorithm follows sklearn.preprocessing.MinMaxScaler.

    Parameters
    ----------
    X : (n,k) ndarray
        A matrix of k snapshots to be scaled. Each column is a single snapshot.

    scale_to : (2,) tuple
        The desired minimum and maximum of the scaled data.

    scale_from : (2,) tuple
        The minimum and maximum of the snapshot data. If None, learn the
        scaling from X: scale_from[0] = min(X); scale_from[1] = max(X).

    Returns
    -------
    Xscaled : (n,k) ndarray
        The scaled snapshot matrix.

    scaled_to : (2,) tuple
        The bounds that the snapshot matrix was scaled to, i.e.,
        scaled_to[0] = min(Xscaled); scaled_to[1] = max(Xscaled).
        Only returned if scale_from = None.

    scaled_from : (2,) tuple
        The minimum and maximum of the snapshot data, i.e., the bounds that
        the data was scaled from. Only returned if scale_from = None.

    For scale_from=None, only Xscaled is returned.

    Examples
    --------
    # Scale X to [-1,1] and then scale Y with the same transformation.
    >>> Xscaled, scaled_to, scaled_from = scale(X, (-1,1))
    >>> Yscaled = scale(Y, scaled_to, scaled_from)

    # Scale X to [0,1], then undo the transformation by an inverse scaling.
    >>> Xscaled, scaled_to, scaled_from = scale(X, (0,1))
    >>> X_again = scale(Xscaled, scaled_from, scaled_to)
    """
    # If no scale_from bounds are provided, learn them.
    learning = (scale_from is None)
    if learning:
        scale_from = _np.min(X), _np.max(X)
        means = _np.mean(X)

    # Check scales.
    if len(scale_to) != 2:
        raise ValueError("scale_to must have exactly 2 elements")
    if len(scale_from) != 2:
        raise ValueError("scale_from must have exactly 2 elements")

    # Do the scaling.
    mini, maxi = scale_to
    xmin, xmax = scale_from
    scl = (maxi - mini)/(xmax - xmin)
    Xscaled = X*scl + (mini - xmin*scl)

    return (Xscaled, scale_to, scale_from) if learning else Xscaled


# Basis computation ===========================================================
def pod_basis(X, r=None, mode="dense", **options):
    """Compute the POD basis of rank r corresponding to the data in X.
    This function does NOT shift or scale data before computing the basis.
    This function is a simple wrapper for various SVD methods.

    Parameters
    ----------
    X : (n,k) ndarray
        A matrix of k snapshots. Each column is a single snapshot.

    r : int
        The number of POD basis vectors and singular values to compute.
        If None (default), compute the full SVD.

    mode : str
        The strategy to use for computing the truncated SVD of X. Options:
        * "dense" (default): Use scipy.linalg.svd() to compute the SVD of X.
            May be inefficient or intractable for very large matrices.
        * "sparse": Use scipy.sparse.linalg.svds() to compute the SVD of X.
            This uses ARPACK for the eigensolver. Inefficient for non-sparse
            matrices; requires separate computations for full SVD.
        * "randomized": Compute an approximate SVD with a randomized approach
            using sklearn.utils.extmath.randomized_svd(). This gives faster
            results at the cost of some accuracy.

    options
        Additional parameters for the SVD solver, which depends on `mode`:
        * "dense": scipy.linalg.svd()
        * "sparse": scipy.sparse.linalg.svds()
        * "randomized": sklearn.utils.extmath.randomized_svd()

    Returns
    -------
    Vr : (n,r) ndarray
        The first r POD basis vectors of X. Each column is one basis vector.

    svdvals : (r,) ndarray
        The first r singular values of X (highest magnitute first).
    """
    # Validate the rank.
    rmax = min(X.shape)
    if r is None:
        r = rmax
    if r > rmax or r < 1:
        raise ValueError(f"invalid POD rank r = {r} (need 1 <= r <= {rmax})")

    if mode == "dense" or mode == "simple":
        V, svdvals, _ = _la.svd(X, full_matrices=False, **options)

    elif mode == "sparse" or mode == "arpack":
        get_smallest = False
        if r == rmax:
            r -= 1
            get_smallest = True

        # Compute all but the last svd vectors / values (maximum allowed)
        V, svdvals, _ = _spla.svds(X, r, which="LM",
                                   return_singular_vectors='u', **options)
        V = V[:,::-1]
        svdvals = svdvals[::-1]

        # Get the smallest vector / value separately.
        if get_smallest:
            V1, smallest, _ = _spla.svds(X, 1, which="SM",
                                        return_singular_vectors='u', **options)
            V = _np.concatenate((V, V1), axis=1)
            svdvals = _np.concatenate((svdvals, smallest))
            r += 1

    elif mode == "randomized":
        V, svdvals, _ = _sklmath.randomized_svd(X, r, **options)

    else:
        raise NotImplementedError(f"invalid mode '{mode}'")

    # Return the first 'r' values.
    return V[:,:r], svdvals[:r]


# Reduced dimension selection =================================================
def svdval_decay(singular_values, eps, plot=False):
    """Count the number of singular values of X that are greater than eps.

    Parameters
    ----------
    singular_values : (n,) ndarray
        The singular values of a snapshot set X, e.g., scipy.linalg.svdvals(X).

    eps : float or list(float)
        Cutoff value(s) for the singular values of X.

    plot : bool
        If True, plot the singular values and the cutoff value(s) against the
        singular value index.

    Returns
    -------
    ranks : int or list(int)
        The number of singular values greater than the cutoff value(s).
    """
    # Calculate the number of singular values above the cutoff value(s).
    one_eps = _np.isscalar(eps)
    if one_eps:
        eps = [eps]
    singular_values = _np.array(singular_values)
    ranks = [_np.count_nonzero(singular_values > ep) for ep in eps]

    if plot:
        # Visualize singular values and cutoff value(s).
        fig, ax = _plt.subplots(1, 1, figsize=(9,3))
        j = _np.arange(1, singular_values.size + 1)
        ax.semilogy(j, singular_values, 'C0*', ms=10, mew=0, zorder=3)
        ax.set_xlim((0,j.size))
        ylim = ax.get_ylim()
        for ep,r in zip(eps, ranks):
            ax.hlines(ep, 0, r, color="black", linewidth=.5, alpha=.75)
            ax.vlines(r, ylim[0], singular_values[r-1] if r > 0 else ep,
                      color="black", linewidth=.5, alpha=.75)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"Singular value index $j$")
        ax.set_ylabel(r"Singular value $\sigma_j$")
        _plt.tight_layout()

    return ranks[0] if one_eps else ranks


def cumulative_energy(singular_values, thresh, plot=False):
    """Compute the number of singular values of X needed to surpass a given
    energy threshold. The energy of j singular values is defined by

        energy_j = sum(singular_values[:j]**2) / sum(singular_values**2).

    Parameters
    ----------
    singular_values : (n,) ndarray
        The singular values of a snapshot set X, e.g., scipy.linalg.svdvals(X).

    thresh : float or list(float)
        Energy capture threshold(s).

    plot : bool
        If True, plot the singular values and the energy capture against
        the singular value index (linear scale).

    Returns
    -------
    ranks : int or list(int)
        The number of singular values required to capture more than each
        energy capture threshold.
    """
    # Calculate the cumulative energy.
    svdvals2 = _np.array(singular_values)**2
    cum_energy = _np.cumsum(svdvals2) / _np.sum(svdvals2)

    # Determine the points at which the cumulative energy passes the threshold.
    one_thresh = _np.isscalar(thresh)
    if one_thresh:
        thresh = [thresh]
    ranks = [_np.searchsorted(cum_energy, th) + 1 for th in thresh]

    if plot:
        # Visualize cumulative energy and threshold value(s).
        fig, ax = _plt.subplots(1, 1, figsize=(9,3))
        j = _np.arange(1, singular_values.size + 1)
        ax.plot(j, cum_energy, 'C2.-', ms=10, lw=1, zorder=3)
        ax.set_xlim(0, j.size)
        ylim = ax.get_ylim()
        for th,r in zip(thresh, ranks):
            ax.hlines(th, 0, r, color="black", linewidth=.5, alpha=.5)
            ax.vlines(r, ylim[0], cum_energy[r-1] if r > 0 else th,
                      color="black", linewidth=.5, alpha=.5)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"Singular value index")
        ax.set_ylabel(r"Cumulative energy")
        _plt.tight_layout()

    return ranks[0] if one_thresh else ranks


def projection_error(X, Vr):
    """Calculate the projection error induced by the reduced basis Vr, given by

        err = ||X - Vr Vr^T X|| / ||X||,

    since (Vr Vr^T) is the orthogonal projector onto the range of Vr.

    Parameters
    ----------
    X : (n,k) or (k,) ndarray
        A 2D matrix of k snapshots where each column is a single snapshot, or a
        single 1D snapshot. If 2D, use the Frobenius norm; if 1D, the l2 norm.

    Vr : (n,r) ndarray
        The reduced basis of rank r. Each column is one basis vector.

    Returns
    -------
    error : float
        The projection error.
    """
    return _la.norm(X - Vr @ Vr.T @ X) / _la.norm(X)


def minimal_projection_error(X, V, eps, plot=False):
    """Compute the number of POD basis vectors required to obtain a projection
    error less than eps. The projection error is defined by

        err = ||X - Vr Vr^T X||_F / ||X||_F,

    since (Vr Vr^T) is the orthogonal projection onto the range of Vr.

    Parameters
    ----------
    X : (n,k) ndarray
        A matrix of k snapshots. Each column is a single snapshot.

    V : (n,rmax) ndarray
        The first rmax POD basis vectors of X. Each column is one basis vector.
        The projection error is calculated for each Vr = V[:,:r] for r <= rmax.

    eps : float or list(float)
        Cutoff value(s) for the projection error.

    plot : bool
        If True, plot the POD basis rank r against the projection error.

    Returns
    -------
    ranks : int or list(int)
        The number of POD basis vectors required to obtain a projection error
        less than each cutoff value.
    """
    # Check dimensions.
    if X.ndim != 2:
        raise ValueError("data X must be two-dimensional")
    if V.ndim != 2:
        raise ValueError("basis V must be two-dimensional")
    one_eps = _np.isscalar(eps)
    if one_eps:
        eps = [eps]

    # Calculate the projection errors.
    X_norm = _la.norm(X, ord="fro")
    rs = _np.arange(1, V.shape[1])
    errors = _np.empty_like(rs, dtype=_np.float)
    for r in rs:
        # Get the POD basis of rank r and calculate the projection error.
        Vr = V[:,:r]
        errors[r-1] = _la.norm(X - Vr @ Vr.T @ X, ord="fro") / X_norm
    # Calculate the ranks needed to get under each cutoff value.
    ranks = [_np.count_nonzero(errors > ep)+1 for ep in eps]

    if plot:
        fig, ax = _plt.subplots(1, 1, figsize=(12,4))
        ax.semilogy(rs, errors, 'C1.-', ms=4, zorder=3)
        ax.set_xlim((0,rs.size))
        ylim = ax.get_ylim()
        for ep,r in zip(eps, ranks):
            ax.hlines(ep, 0, r+1, color="black", linewidth=1)
            ax.vlines(r, ylim[0], ep, color="black", linewidth=1)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"POD basis rank $r$")
        ax.set_ylabel(r"Projection error")

    return ranks[0] if one_eps else ranks


# Reprojection schemes ========================================================
def reproject_discrete(f, Vr, x0, niters, U=None):
    """Sample re-projected trajectories of the discrete dynamical system

        x_{j+1} = f(x_{j}, u_{j}),  x_{0} = x0.

    Parameters
    ----------
    f : callable mapping (n,) ndarray (and (m,) ndarray) to (n,) ndarray
        Function defining the (full-order) discrete dynamical system. Accepts
        a full-order state vector and (optionally) an input vector and returns
        another full-order state vector.

    Vr : (n,r) ndarray
        Basis for the low-dimensional linear subspace (e.g., POD basis).

    x0 : (n,) ndarray
        Initial condition for the iteration in the high-dimensional space.

    niters : int
        The number of iterations to do.

    U : (m,niters-1) or (niters-1) ndarray
        Control inputs, one for each iteration beyond the initial condition.

    Returns
    -------
    X_reprojected : (r,niters) ndarray
        Re-projected state trajectories in the projected low-dimensional space.
    """
    # Validate and extract dimensions.
    n,r = Vr.shape
    if x0.shape != (n,):
        raise ValueError("basis Vr and initial condition x0 not aligned")

    # Create the solution array and fill in the initial condition.
    X_ = _np.empty((r,niters))
    X_[:,0] = Vr.T @ x0

    # Run the re-projection iteration.
    if U is None:
        for j in range(niters-1):
            X_[:,j+1] = Vr.T @ f(Vr @ X_[:,j])
    elif U.ndim == 1:
        for j in range(niters-1):
            X_[:,j+1] = Vr.T @ f(Vr @ X_[:,j], U[j])
    else:
        for j in range(niters-1):
            X_[:,j+1] = Vr.T @ f(Vr @ X_[:,j], U[:,j])

    return X_


def reproject_continuous(f, Vr, X, U=None):
    """Sample re-projected trajectories of the continuous system of ODEs

        dx / dt = f(t, x(t), u(t)),     x(0) = x0.

    Parameters
    ----------
    f : callable mapping (n,) ndarray (and (m,) ndarray) to (n,) ndarray
        Function defining the (full-order) differential equation. Accepts a
        full-order state vector and (optionally) an input vector and returns
        another full-order state vector.

    Vr : (n,r) ndarray
        Basis for the low-dimensional linear subspace.

    X : (n,k) ndarray
        State trajectories (training data).

    U : (m,k) or (k,) ndarray
        Control inputs corresponding to the state trajectories.

    Returns
    -------
    X_reprojected : (r,k) ndarray
        Re-projected state trajectories in the projected low-dimensional space.

    Xdot_reprojected : (r,k) ndarray
        Re-projected velocities in the projected low-dimensional space.
    """
    # Validate and extract dimensions.
    if X.shape[0] != Vr.shape[0]:
        raise ValueError("X and Vr not aligned, first dimension "
                         f"{X.shape[0]} != {Vr.shape[0]}")
    n,r = Vr.shape
    _,k = X.shape

    # Create the solution arrays.
    X_ = Vr.T @ X
    Xdot_ = _np.empty((r,k))

    # Run the re-projection iteration.
    if U is None:
        for j in range(k):
            Xdot_[:,j] = Vr.T @ f(Vr @ X_[:,j])
    elif U.ndim == 1:
        for j in range(k):
            Xdot_[:,j] = Vr.T @ f(Vr @ X_[:,j], U[j])
    else:
        for j in range(k):
            Xdot_[:,j] = Vr.T @ f(Vr @ X_[:,j], U[:,j])

    return X_, Xdot_


# Derivative approximation ====================================================
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
    if not _np.isscalar(dt):
        raise TypeError("time step dt must be a scalar (e.g., float)")

    if order == 2:
        return _np.gradient(X, dt, edge_order=2, axis=1)

    Xdot = _np.empty_like(X)
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
    return _np.gradient(X, t, edge_order=2, axis=-1)


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
            elif isinstance(arg, _np.ndarray):  # arg = t; do uniformity test.
                func = xdot_nonuniform
            else:
                raise TypeError(f"invalid argument type '{type(arg)}'")
    elif n_total == 2:              # There are two other argumetns: dt, order.
        func = xdot_uniform
    else:
        raise TypeError("xdot() takes from 2 to 3 positional arguments "
                        f"but {n_total+1} were given")

    return func(X, *args, **kwargs)


__all__ = [
            "shift",
            "scale",
            "pod_basis",
            "svdval_decay",
            "cumulative_energy",
            "projection_error",
            "minimal_projection_error",
            "reproject_discrete",
            "reproject_continuous",
            "xdot_uniform",
            "xdot_nonuniform",
            "xdot",
          ]


# Deprecations ================================================================

def mean_shift(X):                              # pragma nocover
    _np.warnings.warn("mean_shift() has been renamed shift()",
                   DeprecationWarning, stacklevel=1)
    a,b = shift(X)
    return b,a
mean_shift.__doc__ = "\nDEPRECATED! use shift().\n\n" + shift.__doc__

def significant_svdvals(*args, **kwargs):       # pragma nocover
    _np.warnings.warn("significant_svdvals() has been renamed svdval_decay()",
                   DeprecationWarning, stacklevel=1)
    return svdval_decay(*args, **kwargs)
significant_svdvals.__doc__ = "\nDEPRECATED! use svdval_decay().\n\n" \
                        + svdval_decay.__doc__

def energy_capture(*args, **kwargs):            # pragma nocover
    _np.warnings.warn("energy_capture() has been renamed cumulative_energy()",
                   DeprecationWarning, stacklevel=1)
    return cumulative_energy(*args, **kwargs)
energy_capture.__doc__ = "\nDEPRECATED! use cumulative_energy().\n\n" \
                        + cumulative_energy.__doc__
