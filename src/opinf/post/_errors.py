# post/_errors.py
"""Tools for accuracy and error evaluation."""

__all__ = [
    "projection_error",
    "frobenius_error",
    "lp_error",
    "Lp_error",
]

import numpy as np
import scipy.linalg as la
import scipy.integrate as spintegrate


def _absolute_and_relative_error(Qtrue, Qapprox, norm):
    """Compute the absolute and relative errors between Qtrue and Qapprox,
    where Qapprox approximates Qtrue:

        absolute_error = ||Qtrue - Qapprox||,
        relative_error = ||Qtrue - Qapprox|| / ||Qtrue||
                       = absolute_error / ||Qtrue||,

    with ||Q|| defined by norm(Q).
    """
    norm_of_data = norm(Qtrue)
    absolute_error = norm(Qtrue - Qapprox)
    return absolute_error, absolute_error / norm_of_data


def projection_error(states, basis):
    """Calculate the absolute and relative projection errors induced by
    projecting states to a low dimensional basis, i.e.,

        absolute_error = ||Q - Vr Vr^T Q||_F,
        relative_error = ||Q - Vr Vr^T Q||_F / ||Q||_F

    where Q = states and Vr = basis. Note that Vr Vr^T is the orthogonal
    projector onto subspace of R^n defined by the basis.

    Parameters
    ----------
    states : (n, k) or (k,) ndarray
        Matrix of k snapshots where each column is a single snapshot, or a
        single 1D snapshot. If 2D, use the Frobenius norm; if 1D, the l2 norm.
    Vr : (n, r) ndarray
        Low-dimensional basis of rank r. Each column is one basis vector.

    Returns
    -------
    absolute_error : float
        Absolute projection error ||Q - Vr Vr^T Q||_F.
    relative_error : float
        Relative projection error ||Q - Vr Vr^T Q||_F / ||Q||_F.
    """
    Qapprox = basis @ (basis.T @ states)
    return _absolute_and_relative_error(states, Qapprox, la.norm)


def frobenius_error(Qtrue, Qapprox):
    """Compute the absolute and relative Frobenius-norm errors between the
    snapshot sets Qtrue and Qapprox, where Qapprox approximates Qtrue:

        absolute_error = ||Qtrue - Qapprox||_F,
        relative_error = ||Qtrue - Qapprox||_F / ||Qtrue||_F.

    Parameters
    ----------
    Qtrue : (n, k)
        "True" data. Each column is one snapshot, i.e., Qtrue[:, j] is the data
        at some time t[j].
    Qapprox : (n, k)
        An approximation to Qtrue, i.e., Qapprox[:, j] approximates Qtrue[:, j]
        and corresponds to some time t[j].

    Returns
    -------
    abs_err : float
        Absolute error ||Qtrue - Qapprox||_F.
    rel_err : float
        Relative error ||Qtrue - Qapprox||_F / ||Qtrue||_F.
    """
    # Check dimensions.
    if Qtrue.shape != Qapprox.shape:
        raise ValueError("Qtrue and Qapprox not aligned")
    if Qtrue.ndim != 2:
        raise ValueError("Qtrue and Qapprox must be two-dimensional")

    # Compute the errors.
    return _absolute_and_relative_error(
        Qtrue, Qapprox, lambda Z: la.norm(Z, ord="fro")
    )


def lp_error(Qtrue, Qapprox, p=2, normalize=False):
    """Compute the absolute and relative lp-norm errors between the snapshot
    sets Qtrue and Qapprox, where Qapprox approximates to Qtrue:

        absolute_error_j = ||Qtrue_j - Qapprox_j||_p,
        relative_error_j = ||Qtrue_j - Qapprox_j||_p / ||Qtrue_j||_p.

    Parameters
    ----------
    Qtrue : (n, k) or (n,) ndarray
        "True" data. Each column is one snapshot, i.e., Qtrue[:, j] is the data
        at some time t[j]. If one-dimensional, all of Qtrue is a single
        snapshot.
    Qapprox : (n, k) or (n,) ndarray
        An approximation to Qtrue, i.e., Qapprox[:, j] approximates Qtrue[:, j]
        and corresponds to some time t[j]. If one-dimensional, all of Qapprox
        is a single snapshot approximation.
    p : float
        Order of the lp norm (default p=2 is the Euclidean norm). Used as
        the `ord` argument for scipy.linalg.norm(); see options at
        docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html.
    normalize : bool
        If true, compute the normalized absolute error instead of the relative
        error, defined by

            normalized_absolute_error_j
                = ||Qtrue_j - Qapprox_j||_2 / max_k{||Qtrue_k||_2}.

    Returns
    -------
    abs_err : (k,) ndarray or float
        Absolute error of each pair of snapshots Qtrue[:, j] and Qapprox[:, j].
        If Qtrue and Qapprox are one-dimensional, Qtrue and Qapprox are treated
        as single snapshots, so the error is a float.
    rel_err : (k,) ndarray or float
        Relative or normed absolute error of each pair of snapshots Qtrue[:, j]
        and Qapprox[:, j]. If Qtrue and Qapprox are one-dimensional, Qtrue and
        Qapprox are treated as single snapshots, so the error is a float.
    """
    # Check p.
    if not np.isscalar(p) or p <= 0:
        raise ValueError("norm order p must be positive (np.inf ok)")

    # Check dimensions.
    if Qtrue.shape != Qapprox.shape:
        raise ValueError("Qtrue and Qapprox not aligned")
    if Qtrue.ndim not in (1, 2):
        raise ValueError("Qtrue and Qapprox must be one- or two-dimensional")

    # Compute the error.
    norm_of_data = la.norm(Qtrue, ord=p, axis=0)
    if normalize:
        norm_of_data = norm_of_data.max()
    absolute_error = la.norm(Qtrue - Qapprox, ord=p, axis=0)
    return absolute_error, absolute_error / norm_of_data


def Lp_error(Qtrue, Qapprox, t=None, p=2):
    """Compute the absolute and relative Lp-norm error (with respect to time)
    between the snapshot sets Qtrue and Qapprox, where Qapprox approximates
    Qtrue:

        absolute_error = ||Qtrue - Qapprox||_{L^p},
        relative_error = ||Qtrue - Qapprox||_{L^p} / ||Qtrue||_{L^p},

    where

        ||Z||_{L^p} = (int_{t} ||z(t)||_{p} dt)^{1/p},          p < infinity,
        ||Z||_{L^p} = sup_{t}||z(t)||_{p},                      p = infinity.

    The trapezoidal rule is used to approximate the integrals (for finite p).
    This error measure is only consistent for data sets where each snapshot
    represents function values, i.e.,

        Qtrue[:, j] = [q(t1), q(t2), ..., q(tk)]^T.

    Parameters
    ----------
    Qtrue : (n, k) or (k,) ndarray
        "True" data corresponding to time t. Each column is one snapshot,
        i.e., Qtrue[:, j] is the data at time t[j]. If one-dimensional, each
        entry is one snapshot.
    Qapprox : (n, k) or (k,) ndarray
        An approximation to Qtrue, i.e., Qapprox[:, j] approximates Qtrue[:, j]
        and corresponds to time t[j]. If one-dimensional, each entry is one
        snapshot.
    t : (k,) ndarray
        Time domain of the data Qtrue and the Qapprox.
        Required unless p == np.inf.
    p : float > 0
        Order of the Lp norm. May be infinite (np.inf).

    Returns
    -------
    abs_err : float
        Absolute error ||Qtrue - Qapprox||_{L^p}.
    rel_err : float
        Relative error ||Qtrue - Qapprox||_{L^p} / ||Qtrue||_{L^p}.
    """
    # Check p.
    if not np.isscalar(p) or p <= 0:
        raise ValueError("norm order p must be positive (np.inf ok)")

    # Check dimensions.
    if Qtrue.shape != Qapprox.shape:
        raise ValueError("Qtrue and Qapprox not aligned")
    if Qtrue.ndim == 1:
        Qtrue = np.atleast_2d(Qtrue)
        Qapprox = np.atleast_2d(Qapprox)
    elif Qtrue.ndim > 2:
        raise ValueError("Qtrue and Qapprox must be one- or two-dimensional")

    # Pick the norm based on p.
    if 0 < p < np.inf:
        if t is None:
            raise ValueError("time t required for p < infinty")
        if t.ndim != 1:
            raise ValueError("time t must be one-dimensional")
        if Qtrue.shape[-1] != t.shape[0]:
            raise ValueError("Qtrue not aligned with time t")

        def pnorm(Z):
            Zpnorm = np.sum(np.abs(Z) ** p, axis=0)
            return spintegrate.trapezoid(Zpnorm, t) ** (1 / p)

    else:  # p == np.inf

        def pnorm(Z):
            return np.max(np.abs(Z), axis=0).max()

    # Compute the error.
    return _absolute_and_relative_error(Qtrue, Qapprox, pnorm)
