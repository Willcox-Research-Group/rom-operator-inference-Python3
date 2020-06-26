# post.py
"""Tools for accuracy and error evaluation."""

import numpy as _np
from scipy import linalg as _la


def _absolute_and_relative_error(X, Y, norm):
    """Compute the absolute and relative errors between X and Y where Y is an
    approximation to X,

        Abs_Err = ||X - Y||,
        Rel_Err = ||X - Y|| / ||X|| = Abs_Err / ||X||,

    with ||X|| defined by norm(X).
    """
    norm_of_data = norm(X)
    absolute_error = norm(X - Y)
    return absolute_error, absolute_error / norm_of_data


def frobenius_error(X, Y):
    """Compute the absolute and relative Frobenius-norm errors between two
    snapshot sets X and Y where Y is an approximation to X,

        Abs_Err = ||X - Y||_F,
        Rel_Err = ||X - Y||_F / ||X||_F = Abs_Err / ||X||_F.

    Parameters
    ----------
    X : (n,k)
        The "true" data. Each column is one snapshot, i.e., X[:,j] is the data
        at some time t[j].

    Y : (n,k)
        An approximation to X, i.e., Y[:,j] approximates X[:,j] and corresponds
        to some time t[j].

    Returns
    -------
    abs_err : float
        The absolute error ||X - Y||_F.

    rel_err : float
        The relative error ||X - Y||_F / ||X||_F.
    """
    # Check dimensions.
    if X.shape != Y.shape:
        raise ValueError("truth X and approximation Y not aligned")
    if X.ndim != 2:
        raise ValueError("X and Y must be two-dimensional")

    # Compute the errors.
    return _absolute_and_relative_error(X, Y, lambda Z: _la.norm(Z, ord="fro"))


def lp_error(X, Y, p=2, normalize=False):
    """Compute the absolute and relative lp-norm errors between two snapshot
    sets X and Y where Y is an approximation to X,

        Abs_Err_j = ||X_j - Y_j||_p,
        Rel_Err_j = ||X_j - Y_j||_p / ||X_j||_p = Abs_Err_j / ||X_j||_p.

    Parameters
    ----------
    X : (n,k) or (n,) ndarray
        The "true" data. Each column is one snapshot, i.e., X[:,j] is the data
        at some time t[j]. If one-dimensional, all of X is a single snapshot.

    Y : (n,k) or (n,) ndarray
        An approximation to X, i.e., Y[:,j] approximates X[:,j] and corresponds
        to some time t[j]. If one-dimensional, all of Y is a single snapshot
        approximation.

    p : float
        The order of the lp norm (default p=2 is the Euclidean norm). Used as
        the `ord` argument for scipy.linalg.norm(); see options at
        docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html.

    normalize : bool
        If true, compute the normalized absolute error instead of the relative
        error, defined by

            Normalized_Abs_Err_j = ||X_j - Y_j||_2 / max_j{||X_j||_2}.

    Returns
    -------
    abs_err : (k,) ndarray or float
        The absolute error of each pair of snapshots X[:,j] and Y[:,j]. If X
        and Y are one-dimensional, X and Y are treated as single snapshots, so
        the error is a float.

    rel_err : (k,) ndarray or float
        The relative or normed absolute error of each pair of snapshots X[:,j]
        and Y[:,j]. If X and Y are one-dimensional, X and Y are treated as
        single snapshots, so the error is a float.
    """
    # Check p.
    if not _np.isscalar(p) or p <= 0:
        raise ValueError("norm order p must be positive (np.inf ok)")

    # Check dimensions.
    if X.shape != Y.shape:
        raise ValueError("truth X and approximation Y not aligned")
    if X.ndim not in (1,2):
        raise ValueError("X and Y must be one- or two-dimensional")

    # Compute the error.
    norm_of_data = _la.norm(X, ord=p, axis=0)
    if normalize:
        norm_of_data = norm_of_data.max()
    absolute_error = _la.norm(X - Y, ord=p, axis=0)
    return absolute_error, absolute_error / norm_of_data


def Lp_error(X, Y, t=None, p=2):
    """Compute the absolute and relative Lp-norm error between two snapshot
    sets X and Y where Y is an approximation to X,

        Abs_Err = ||X - Y||_{L^p},
        Rel_Err = ||X - Y||_{L^p} / ||X||_{L^p} = Abs_Err / ||X||_{L^p},

    using the trapezoidal rule to approximate the integrals (for finite p).
    This error measure is only consistent for data sets where each snapshot
    represents function values, i.e., X[:,j] = [u(t1), u(t2), ..., u(tk)]^T.

    Parameters
    ----------
    X : (n,k) or (k,) ndarray
        The "true" data corresponding to time t. Each column is one snapshot,
        i.e., X[:,j] is the data at time t[j]. If one-dimensional, each entry
        is one snapshot.

    Y : (n,k) or (k,) ndarray
        An approximation to X, i.e., Y[:,j] approximates X[:,j] and corresponds
        to time t[j]. If one-dimensional, each entry is one snapshot.

    t : (k,) ndarray
        Time domain of the data X and the approximation Y.
        Required unless p == np.inf.

    p : float > 0
        The order of the Lp norm. May be infinite (np.inf).

    Returns
    -------
    abs_err : float
        The absolute error ||X - Y||_{L^p}.

    rel_err : float
        The relative error ||X - Y||_{L^p} / ||X||_{L^p}.
    """
    # Check p.
    if not _np.isscalar(p) or p <= 0:
        raise ValueError("norm order p must be positive (np.inf ok)")

    # Check dimensions.
    if X.shape != Y.shape:
        raise ValueError("truth X and approximation Y not aligned")
    if X.ndim == 1:
        X = _np.atleast_2d(X)
        Y = _np.atleast_2d(Y)
    elif X.ndim > 2:
        raise ValueError("X and Y must be one- or two-dimensional")

    # Pick the norm based on p.
    if p < _np.inf:
        if t is None:
            raise ValueError("time t required for p < infinty")
        if t.ndim != 1:
            raise ValueError("time t must be one-dimensional")
        if X.shape[-1] != t.shape[0]:
            raise ValueError("truth X not aligned with time t")
        pnorm = lambda Z: (_np.trapz(_np.sum(_np.abs(Z)**p, axis=0), t))**(1/p)
    elif p == _np.inf:
        pnorm = lambda Z: _np.max(_np.abs(Z), axis=0).max()

    # Compute the error.
    return _absolute_and_relative_error(X, Y, pnorm)
