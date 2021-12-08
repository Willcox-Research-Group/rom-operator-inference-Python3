# pre/_basis.py
"""Tools for basis computation and reduced-dimension selection."""

__all__ = [
            "pod_basis",
            "svdval_decay",
            "cumulative_energy",
            "residual_energy",
            "projection_error",
            "minimal_projection_error",
          ]

import numpy as np
import scipy. linalg as la
import scipy.sparse.linalg as spla
import sklearn.utils.extmath as sklmath
import matplotlib.pyplot as plt


# Basis computation ===========================================================
def pod_basis(states, r=None, mode="dense", return_W=False, **options):
    """Compute the POD basis of rank r corresponding to the states.
    This function does NOT shift or scale data before computing the basis.
    This function is a simple wrapper for various SVD methods.

    Parameters
    ----------
    states : (n,k) ndarray
        A matrix of k snapshots. Each column is a single snapshot.
    r : int
        Number of POD basis vectors and singular values to compute.
        If None (default), compute the full SVD.
    mode : str
        Strategy to use for computing the truncated SVD of states. Options:
        * "dense" (default): Use scipy.linalg.svd() to compute the SVD.
            May be inefficient or intractable for very large matrices.
        * "sparse": Use scipy.sparse.linalg.svds() to compute the SVD.
            This uses ARPACK for the eigensolver. Inefficient for non-sparse
            matrices; requires separate computations for full SVD.
        * "randomized": Compute an approximate SVD with a randomized approach
            using sklearn.utils.extmath.randomized_svd(). This gives faster
            results at the cost of some accuracy.
    return_W : bool
        If True, also return the first r *right* singular vectors.
    options
        Additional parameters for the SVD solver, which depends on `mode`:
        * "dense": scipy.linalg.svd()
        * "sparse": scipy.sparse.linalg.svds()
        * "randomized": sklearn.utils.extmath.randomized_svd()

    Returns
    -------
    basis : (n,r) ndarray
        First r POD basis vectors. Each column is one basis vector.
    svdvals : (n,), (k,), or (r,) ndarray
        Singular values (highest magnitute first). Always return as many
        singular values as are calculated: r for mode="randomize", and min(n,k)
        otherwise.
    """
    # Validate the rank.
    rmax = min(states.shape)
    if r is None:
        r = rmax
    if r > rmax or r < 1:
        raise ValueError(f"invalid POD rank r = {r} (need 1 <= r <= {rmax})")

    if mode == "dense" or mode == "simple":
        V, svdvals, Wt = la.svd(states, full_matrices=False, **options)

    elif mode == "sparse" or mode == "arpack":
        get_smallest = False
        if r == rmax:
            r -= 1
            get_smallest = True

        # Compute all but the last svd vectors / values (maximum allowed)
        V, svdvals, Wt = spla.svds(states, r, which="LM",
                                   return_singular_vectors='u', **options)
        V = V[:,::-1]
        svdvals = svdvals[::-1]
        # Wt = TODO

        # Get the smallest vector / value separately.
        if get_smallest:
            V1, smallest, W = spla.svds(states, 1, which="SM",
                                        return_singular_vectors='u', **options)
            V = np.concatenate((V, V1), axis=1)
            svdvals = np.concatenate((svdvals, smallest))
            r += 1

    elif mode == "randomized":
        V, svdvals, Wt = sklmath.randomized_svd(states, r, **options)

    else:
        raise NotImplementedError(f"invalid mode '{mode}'")

    # Return the first 'r' basis vectors and all of the singular values.
    return V[:,:r], svdvals
    # TODO: if return_W is True: also return Wt[:r].T or W[:,:r].


# Reduced dimension selection =================================================
def svdval_decay(singular_values, tol, plot=False, ax=None):
    """Count the number of singular values that are greater than tol.

    Parameters
    ----------
    singular_values : (n,) ndarray
        Singular values of a snapshot set, e.g., scipy.linalg.svdvals(states).
    tol : float or list(float)
        Cutoff value(s) for the singular values.
    plot : bool
        If True, plot the singular values and the cutoff value(s) against the
        singular value index.
    ax : plt.Axes or None
        Matplotlib Axes to plot the results on if plot = True.
        If not given, a new single-axes figure is created.

    Returns
    -------
    ranks : int or list(int)
        The number of singular values greater than the cutoff value(s).
    """
    # Calculate the number of singular values above the cutoff value(s).
    one_tol = np.isscalar(tol)
    if one_tol:
        tol = [tol]
    singular_values = np.array(singular_values)
    ranks = [np.count_nonzero(singular_values > ε) for ε in tol]

    if plot:
        # Visualize singular values and cutoff value(s).
        if ax is None:
            ax = plt.figure().add_subplot(111)
        j = np.arange(1, singular_values.size + 1)
        ax.semilogy(j, singular_values, 'C0*', ms=10, mew=0, zorder=3)
        ax.set_xlim((0,j.size))
        ylim = ax.get_ylim()
        for ε,r in zip(tol, ranks):
            ax.axhline(ε, color="black", linewidth=.5, alpha=.75)
            ax.axvline(r, color="black", linewidth=.5, alpha=.75)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"Singular value index $j$")
        ax.set_ylabel(r"Singular value $\sigma_j$")

    return ranks[0] if one_tol else ranks


def cumulative_energy(singular_values, thresh, plot=False, ax=None):
    """Compute the number of singular values needed to surpass a given
    energy threshold. The energy of j singular values is defined by

        energy_j = sum(singular_values[:j]**2) / sum(singular_values**2).

    Parameters
    ----------
    singular_values : (n,) ndarray
        Singular values of a snapshot set, e.g., scipy.linalg.svdvals(states).
    thresh : float or list(float)
        Energy capture threshold(s).
    plot : bool
        If True, plot the singular values and the cumulative energy against
        the singular value index (linear scale).
    ax : plt.Axes or None
        Matplotlib Axes to plot the results on if plot = True.
        If not given, a new single-axes figure is created.

    Returns
    -------
    ranks : int or list(int)
        The number of singular values required to capture more than each
        energy capture threshold.
    """
    # Calculate the cumulative energy.
    svdvals2 = np.sort(singular_values)[::-1]**2
    cum_energy = np.cumsum(svdvals2) / np.sum(svdvals2)

    # Determine the points at which the cumulative energy passes the threshold.
    one_thresh = np.isscalar(thresh)
    if one_thresh:
        thresh = [thresh]
    ranks = [int(np.searchsorted(cum_energy, ξ)) + 1 for ξ in thresh]

    if plot:
        # Visualize cumulative energy and threshold value(s).
        if ax is None:
            ax = plt.figure().add_subplot(111)
        j = np.arange(1, singular_values.size + 1)
        ax.plot(j, cum_energy, 'C2.-', ms=10, lw=1, zorder=3)
        ax.set_xlim(0, j.size)
        for ξ,r in zip(thresh, ranks):
            ax.axhline(ξ, color="black", linewidth=.5, alpha=.5)
            ax.axvline(r, color="black", linewidth=.5, alpha=.5)
        ax.set_xlabel(r"Singular value index")
        ax.set_ylabel(r"Cumulative energy")

    return ranks[0] if one_thresh else ranks


def residual_energy(singular_values, tol, plot=False, ax=None):
    """Compute the number of singular values needed such that the residual
    energy drops beneath the given tolerance. The residual energy of j
    singular values is defined by

        residual_j = 1 - sum(singular_values[:j]**2) / sum(singular_values**2).

    Parameters
    ----------
    singular_values : (n,) ndarray
        Singular values of a snapshot set, e.g., scipy.linalg.svdvals(states).
    tol : float or list(float)
        Energy residual tolerance(s).
    plot : bool
        If True, plot the singular values and the residual energy against
        the singular value index (log scale).
    ax : plt.Axes or None
        Matplotlib Axes to plot the results on if plot = True.
        If not given, a new single-axes figure is created.

    Returns
    -------
    ranks : int or list(int)
        Number of singular values required to for the residual energy to drop
        beneath each tolerance.
    """
    # Calculate the cumulative energy.
    svdvals2 = np.sort(singular_values)[::-1]**2
    res_energy = 1 - (np.cumsum(svdvals2) / np.sum(svdvals2))

    # Determine the points when the residual energy dips under the tolerance.
    one_tol = np.isscalar(tol)
    if one_tol:
        tol = [tol]
    ranks = [np.count_nonzero(res_energy > ε) + 1 for ε in tol]

    if plot:
        # Visualize residual energy and tolerance value(s).
        if ax is None:
            ax = plt.figure().add_subplot(111)
        j = np.arange(1, singular_values.size + 1)
        ax.semilogy(j, res_energy, 'C1.-', ms=10, lw=1, zorder=3)
        ax.set_xlim(0, j.size)
        for ε,r in zip(tol, ranks):
            ax.axhline(ε, color="black", linewidth=.5, alpha=.5)
            ax.axvline(r, color="black", linewidth=.5, alpha=.5)
        ax.set_xlabel(r"Singular value index")
        ax.set_ylabel(r"Residual energy")

    return ranks[0] if one_tol else ranks


def projection_error(states, basis):
    """Calculate the projection error on the states X induced by the basis Vr:

        err = ||X - Vr Vr^T X|| / ||X||.

    Parameters
    ----------
    states : (n,k) or (k,) ndarray
        Matrix of k snapshots where each column is a single snapshot, or a
        single 1D snapshot. If 2D, use the Frobenius norm; if 1D, the l2 norm.
    basis : (n,r) ndarray
        Basis of rank r. Each column is one basis vector.

    Returns
    -------
    error : float
        Projection error.
    """
    return la.norm(states - basis @ basis.T @ states) / la.norm(states)


def minimal_projection_error(states, basis, tol, plot=False, ax=None):
    """Compute the number of POD basis vectors required to obtain a projection
    error less than tol. The projection error on the states X induced by the
    basis Vr is defined by

        err = ||X - Vr Vr^T X||_F / ||X||_F.

    Parameters
    ----------
    states : (n,k) ndarray
        Matrix of k snapshots. Each column is a single snapshot.
    basis : (n,rmax) ndarray
        First rmax POD basis vectors. Each column is one basis vector.
        The projection error is calculated with Vr = basis[:,:r] for r <= rmax.
    tol : float or list(float)
        Cutoff value(s) for the projection error.
    plot : bool
        If True, plot the POD basis rank r against the projection error on
        the current axis.

    Returns
    -------
    ranks : int or list(int)
        Number of POD basis vectors required to obtain a projection error
        less than each cutoff value.
    """
    # Check dimensions.
    if states.ndim != 2:
        raise ValueError("states must be two-dimensional")
    if basis.ndim != 2:
        raise ValueError("basis must be two-dimensional")
    one_tol = np.isscalar(tol)
    if one_tol:
        tol = [tol]

    # Calculate the projection errors.
    X_norm = la.norm(states, ord="fro")
    rs = np.arange(1, basis.shape[1])
    errors = np.empty(rs.shape, dtype=float)
    for r in rs:
        # Get the POD basis of rank r and calculate the projection error.
        Vr = basis[:,:r]
        errors[r-1] = la.norm(states - Vr @ Vr.T @ states, ord="fro") / X_norm
    # Calculate the ranks needed to get under each cutoff value.
    ranks = [np.count_nonzero(errors > ε)+1 for ε in tol]

    if plot:
        if ax is None:
            ax = plt.figure().add_subplot(111)
        ax.semilogy(rs, errors, 'C1.-', ms=4, zorder=3)
        ax.set_xlim((0,rs.size))
        ylim = ax.get_ylim()
        for ε,r in zip(tol, ranks):
            ax.axhline(ε, color="black", linewidth=.5, alpha=.5)
            ax.axvline(r, color="black", linewidth=.5, alpha=.5)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"POD basis rank $r$")
        ax.set_ylabel(r"Projection error")

    return ranks[0] if one_tol else ranks
