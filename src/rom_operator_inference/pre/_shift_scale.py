# pre/_shiftscale.py
"""Tools for preprocessing data."""

__all__ = [
            "shift",
            "scale",
          ]

import numpy as np


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
    >>> Xshifted, xbar = pre.shift(X)
    >>> Yshifted = pre.shift(Y, xbar)

    # Shift X by its mean, then undo the transformation by an inverse shift.
    >>> Xshifted, xbar = pre.shift(X)
    >>> X_again = pre.shift(Xshifted, -xbar)
    """
    # Check dimensions.
    if X.ndim != 2:
        raise ValueError("data X must be two-dimensional")

    # If not shift_by factor is provided, compute the mean column.
    learning = (shift_by is None)
    if learning:
        shift_by = np.mean(X, axis=1)
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
    >>> Xscaled, scaled_to, scaled_from = pre.scale(X, (-1,1))
    >>> Yscaled = pre.scale(Y, scaled_to, scaled_from)

    # Scale X to [0,1], then undo the transformation by an inverse scaling.
    >>> Xscaled, scaled_to, scaled_from = pre.scale(X, (0,1))
    >>> X_again = pre.scale(Xscaled, scaled_from, scaled_to)
    """
    # If no scale_from bounds are provided, learn them.
    learning = (scale_from is None)
    if learning:
        scale_from = np.min(X), np.max(X)
        means = np.mean(X)

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


# Deprecations ================================================================

def mean_shift(X):                              # pragma nocover
    np.warnings.warn("mean_shift() has been renamed shift()",
                   DeprecationWarning, stacklevel=1)
    a,b = shift(X)
    return b,a
mean_shift.__doc__ = "\nDEPRECATED! use shift().\n\n" + shift.__doc__
