# post.py
"""Tools for accuracy and error evaluation."""

import numpy as np
from scipy.linalg import norm as _norm


def discrete_error(X, Y):
    """Compute the absolute and relative l2-norm errors between two snapshot
    sets X and Y where Y is an approximation to X,

        Abs_Err_j = ||X_j - Y_j||_2,
        Rel_Err_j = ||X_j - Y_j||_2 / ||X_j||_2 = Abs_Err_j / ||X_j||_2.

    Parameters
    ----------
    X : (n,k) or (n,) ndarray
        The "true" data. Each column is one snapshot, i.e., X[:,j] is the data
        at some time t[j]. If one-dimensional, all of X is a single snapshot.

    Y : (n,k) or (n,) ndarray
        An approximation to X, i.e., Y[:,j] approximates X[:,j] and corresponds
        to some time t[j]. If one-dimensional, all of Y is a single snapshot
        approximation.

    Returns
    -------
    abs_err : (k,) ndarray or float
        The absolute error of each pair of snapshots X[:,j] and Y[:,j]. If X
        and Y are one-dimensional, X and Y are treated as single snapshots, so
        the error is a float.

    rel_err : (k,) ndarray or float
        The relative error of each pair of snapshots X[:,j] and Y[:,j]. If X
        and Y are one-dimensional, X and Y are treated as single snapshots, so
        the error is a float. Note that this error may be deceptively large
        when the norm of a true snapshot, ||X_j|| is small.
    """
    # Check dimensions.
    if X.shape != Y.shape:
        raise ValueError("truth X and approximation Y not aligned")
    if X.ndim not in (1,2):
        raise ValueError("X and Y must be one- or two-dimensional")

    # Compute the error.
    norm_of_data = _norm(X, axis=0)
    absolute_error = _norm(X - Y, axis=0)
    return absolute_error, absolute_error / norm_of_data


def continuous_error(X, Y, t, tol=1e-10):
    """Compute the absolute and relative L2-norm error between two snapshot
    sets X and Y where Y is an approximation to X,

        Abs_Err = ||X - Y||_{L^2},
        Rel_Err = ||X - Y||_{L^2} / ||X||_{L^2} = Abs_Err / ||X||_{L^2},

    using the trapezoidal rule to approximate the integrals.

    Parameters
    ----------
    X : (n,k) ndarray
        The "true" data corresponding to time t. Each column is one snapshot,
        i.e., X[:,j] is the data at time t[j].

    Y : (n,k) ndarray
        An approximation to X, i.e., Y[:,j] approximates X[:,j] and corresponds
        to time t[j].

    t : (k,) ndarray
        Time domain of the data X and the approximation Y.

    Returns
    -------
    err : float
        The relative error, or the absolute error if ||X|| < tol.
    """
    # Check dimensions.
    if X.shape != Y.shape:
        raise ValueError("truth X and approximation Y not aligned")
    if X.ndim != 2:
        raise ValueError("X and Y must be two-dimensional")
    if t.ndim != 1:
        raise ValueError("time t must be one-dimensional")
    k = t.shape[0]
    if X.shape[-1] != k:
        raise ValueError("truth X not aligned with time t")

    # Compute the error.
    norm_of_data = np.sqrt(np.trapz(np.sum(X**2, axis=0), t))
    absolute_error = np.sqrt(np.trapz(np.sum((X - Y)**2, axis=0), t))
    return absolute_error, absolute_error / norm_of_data
