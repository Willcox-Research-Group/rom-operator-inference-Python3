# utils/_kronecker.py
"""Utility functions for compact / full Kronecker products."""

__all__ = [
            "kron2c",
            "kron3c",
            "kron2c_indices",
            "kron3c_indices",
            "compress_quadratic",
            "expand_quadratic",
            "compress_cubic",
            "expand_cubic",
          ]

import itertools
import numpy as np
import scipy.special as special


# Compact Kronecker (Khatri-Rao) products =====================================
def kron2c(x, checkdim=False):
    """Calculate the unique terms of the quadratic Kronecker product x ⊗ x.

    Parameters
    ----------
    x : (n,) or (n, k) ndarray
        If two-dimensional, the product is computed column-wise (Khatri-Rao).
    checkdim : bool
        If true, check that the input `x` is one- or two-dimensional.

    Returns
    -------
    x ⊗ x : (n(n+1)/2,) or (n(n+1)/2, k) ndarray
        The "compact" Kronecker product of x with itself.
    """
    if checkdim and x.ndim not in (1, 2):
        raise ValueError("x must be one- or two-dimensional")
    return np.concatenate([x[i] * x[:i+1] for i in range(x.shape[0])], axis=0)


def kron3c(x, checkdim=False):
    """Calculate the unique terms of the cubic Kronecker product x ⊗ x ⊗ x.

    Parameters
    ----------
    x : (n,) or (n, k) ndarray
        If two-dimensional, the product is computed column-wise (Khatri-Rao).
    checkdim : bool
        If true, check that the input `x` is one- or two-dimensional.

    Returns
    -------
    x ⊗ x : (n(n+1)(n+2)/6,) or (n(n+1)(n+2)/6, k) ndarray
        The "compact" Kronecker product of x with itself three times.
    """
    if checkdim and x.ndim not in (1, 2):
        raise ValueError("x must be one- or two-dimensional")
    x2 = kron2c(x, False)
    lens = special.binom(np.arange(2, len(x)+2), 2).astype(int)
    return np.concatenate([x[i] * x2[:lens[i]]
                           for i in range(x.shape[0])], axis=0)


# Index generation for fast compact Kronecker product evaluation ==============
def kron2c_indices(r):
    """Construct masks for compact quadratic and cubic Kronecker."""
    mask = np.zeros((r*(r+1)//2, 2), dtype=int)
    count = 0
    for i in range(r):
        for j in range(i+1):
            mask[count, :] = (i, j)
            count += 1
    return mask


def kron3c_indices(r):
    """Construct masks for compact quadratic and cubic Kronecker."""
    mask = np.zeros((r*(r+1)*(r+2)//6, 3), dtype=int)
    count = 0
    for i in range(r):
        for j in range(i+1):
            for k in range(j+1):
                mask[count, :] = (i, j, k)
                count += 1
    return mask


# Matricized tensor management ================================================
def compress_quadratic(H):
    """Calculate the matricized quadratic operator that operates on the compact
    Kronecker product.

    Parameters
    ----------
    H : (r, r**2) ndarray
        The matricized quadratic tensor that operates on the full Kronecker
        product. This should be a symmetric operator in the sense that each
        layer of H.reshape((r, r, r)) is a symmetric (r, r) matrix, but it is
        not required.

    Returns
    -------
    Hc : (r, s) ndarray
        The matricized quadratic tensor that operates on the compact Kronecker
        product. Here s = r * (r+1) / 2.
    """
    r = H.shape[0]
    r2 = H.shape[1]
    if r2 != r**2:
        raise ValueError(f"invalid shape (r, a) = {(r, r2)} with a != r**2")
    s = r * (r+1) // 2
    Hc = np.empty((r, s))

    fj = 0
    for i in range(r):
        for j in range(i+1):
            if i == j:      # Place column for unique term.
                Hc[:, fj] = H[:, (i*r)+j]
            else:           # Combine columns for repeated terms.
                fill = H[:, (i*r)+j] + H[:, (j*r)+i]
                Hc[:, fj] = fill
            fj += 1

    return Hc


def expand_quadratic(Hc):
    """Calculate the matricized quadratic operator that operates on the full
    Kronecker product.

    Parameters
    ----------
    Hc : (r, s) ndarray
        The matricized quadratic tensor that operates on the compact Kronecker
        product. Here s = r * (r+1) / 2.

    Returns
    -------
    H : (r, r**2) ndarray
        The matricized quadratic tensor that operates on the full Kronecker
        product. This is a symmetric operator in the sense that each layer of
        H.reshape((r, r, r)) is a symmetric (r, r) matrix.
    """
    r, s = Hc.shape
    if s != r*(r+1)//2:
        raise ValueError(f"invalid shape (r, s) = {(r, s)} with s != r(r+1)/2")

    H = np.empty((r, r**2))
    fj = 0
    for i in range(r):
        for j in range(i+1):
            if i == j:      # Place column for unique term.
                H[:, (i*r)+j] = Hc[:, fj]
            else:           # Distribute columns for repeated terms.
                fill = Hc[:, fj] / 2
                H[:, (i*r)+j] = fill
                H[:, (j*r)+i] = fill
            fj += 1

    return H


def compress_cubic(G):
    """Calculate the matricized cubic operator that operates on the compact
    cubic Kronecker product.

    Parameters
    ----------
    G : (r, r**3) ndarray
        The matricized cubic tensor that operates on the full cubic Kronecker
        product. This should be a symmetric operator in the sense that each
        layer of G.reshape((r, r, r, r)) is a symmetric (r, r, r) tensor, but
        it is not required.

    Returns
    -------
    Gc : (r, s) ndarray
        The matricized cubic tensor that operates on the compact cubic
        Kronecker product. Here s = r * (r+1) * (r+2) / 6.
    """
    # TODO: only check that r3 is a perfect cube, not necessarily r**3
    # (may be useful for cubic interactions of input or for systems).
    r = G.shape[0]
    r3 = G.shape[1]
    if r3 != r**3:
        raise ValueError(f"invalid shape (r, a) = {(r, r3)} with a != r**3")
    s = r * (r+1) * (r+2) // 6
    Gc = np.empty((r, s))

    fj = 0
    for i in range(r):
        for j in range(i+1):
            for k in range(j+1):
                idxs = set(itertools.permutations((i, j, k), 3))
                Gc[:, fj] = np.sum([G[:, (a*r**2)+(b*r)+c]
                                    for a, b, c in idxs], axis=0)
                fj += 1

    # assert fj == s
    return Gc


def expand_cubic(Gc):
    """Calculate the matricized quadratic operator that operates on the full
    cubic Kronecker product.

    Parameters
    ----------
    Gc : (r, s) ndarray
        The matricized quadratic tensor that operates on the compact cubic
        Kronecker product. Here s = r * (r+1) * (r+2) / 6.

    Returns
    -------
    G : (r, r**3) ndarray
        The matricized quadratic tensor that operates on the full cubic
        Kronecker product. This is a symmetric operator in the sense that each
        layer of G.reshape((r, r, r, r)) is a symmetric (r, r, r) matrix.
    """
    r, s = Gc.shape
    if s != r * (r+1) * (r+2) // 6:
        raise ValueError(f"invalid shape (r, s) = {(r, s)}"
                         " with s != r(r+1)(r+2)/6")

    G = np.empty((r, r**3))
    fj = 0
    for i in range(r):
        for j in range(i+1):
            for k in range(j+1):
                idxs = set(itertools.permutations((i, j, k), 3))
                fill = Gc[:, fj] / len(idxs)
                for a, b, c in idxs:
                    G[:, (a*r**2)+(b*r)+c] = fill
                fj += 1

    # assert fj == s
    return G
