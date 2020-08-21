# pre/_basis.py
"""Tools for basis computation and reduced-dimension selection."""

__all__ = [
            "pod_basis",
            "svdval_decay",
            "cumulative_energy",
            "projection_error",
            "minimal_projection_error",
            # DEPRECATIONS:
            "significant_svdvals",
            "energy_capture",
          ]

import numpy as np
from scipy import linalg as la
from scipy.sparse import linalg as spla
from sklearn.utils import extmath as sklmath
from matplotlib import pyplot as plt


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
        V, svdvals, _ = la.svd(X, full_matrices=False, **options)

    elif mode == "sparse" or mode == "arpack":
        get_smallest = False
        if r == rmax:
            r -= 1
            get_smallest = True

        # Compute all but the last svd vectors / values (maximum allowed)
        V, svdvals, _ = spla.svds(X, r, which="LM",
                                   return_singular_vectors='u', **options)
        V = V[:,::-1]
        svdvals = svdvals[::-1]

        # Get the smallest vector / value separately.
        if get_smallest:
            V1, smallest, _ = spla.svds(X, 1, which="SM",
                                        return_singular_vectors='u', **options)
            V = np.concatenate((V, V1), axis=1)
            svdvals = np.concatenate((svdvals, smallest))
            r += 1

    elif mode == "randomized":
        V, svdvals, _ = sklmath.randomized_svd(X, r, **options)

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
    one_eps = np.isscalar(eps)
    if one_eps:
        eps = [eps]
    singular_values = np.array(singular_values)
    ranks = [np.count_nonzero(singular_values > ep) for ep in eps]

    if plot:
        # Visualize singular values and cutoff value(s).
        ax = plt.gca()
        j = np.arange(1, singular_values.size + 1)
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
        If True, plot the singular values and the cumulative energy against
        the singular value index (linear scale).

    Returns
    -------
    ranks : int or list(int)
        The number of singular values required to capture more than each
        energy capture threshold.
    """
    # Calculate the cumulative energy.
    svdvals2 = np.array(singular_values)**2
    cum_energy = np.cumsum(svdvals2) / np.sum(svdvals2)

    # Determine the points at which the cumulative energy passes the threshold.
    one_thresh = np.isscalar(thresh)
    if one_thresh:
        thresh = [thresh]
    ranks = [np.searchsorted(cum_energy, th) + 1 for th in thresh]

    if plot:
        # Visualize cumulative energy and threshold value(s).
        ax = plt.gca()
        j = np.arange(1, singular_values.size + 1)
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
    return la.norm(X - Vr @ Vr.T @ X) / la.norm(X)


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
        If True, plot the POD basis rank r against the projection error on
        the current axis.

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
    one_eps = np.isscalar(eps)
    if one_eps:
        eps = [eps]

    # Calculate the projection errors.
    X_norm = la.norm(X, ord="fro")
    rs = np.arange(1, V.shape[1])
    errors = np.empty_like(rs, dtype=np.float)
    for r in rs:
        # Get the POD basis of rank r and calculate the projection error.
        Vr = V[:,:r]
        errors[r-1] = la.norm(X - Vr @ Vr.T @ X, ord="fro") / X_norm
    # Calculate the ranks needed to get under each cutoff value.
    ranks = [np.count_nonzero(errors > ep)+1 for ep in eps]

    if plot:
        ax = plt.gca()
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


# DEPRECATIONS ================================================================

def significant_svdvals(*args, **kwargs):       # pragma nocover
    np.warnings.warn("significant_svdvals() has been renamed svdval_decay()",
                   DeprecationWarning, stacklevel=1)
    return svdval_decay(*args, **kwargs)
significant_svdvals.__doc__ = "\nDEPRECATED! use svdval_decay().\n\n" \
                        + svdval_decay.__doc__

def energy_capture(*args, **kwargs):            # pragma nocover
    np.warnings.warn("energy_capture() has been renamed cumulative_energy()",
                   DeprecationWarning, stacklevel=1)
    return cumulative_energy(*args, **kwargs)
energy_capture.__doc__ = "\nDEPRECATED! use cumulative_energy().\n\n" \
                        + cumulative_energy.__doc__
