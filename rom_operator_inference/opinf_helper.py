# opinf_helper.py
"""Utility functions for the operator inference."""

import h5py
import numpy as np
from scipy import linalg as la
from scipy.sparse import csr_matrix


def normal_equations(D, r, reg, num):
    """Solve the normal equations corresponding to the regularized ordinary
    least squares problem

    minimize ||Do - r|| + k||Fo||

    Parameters
    ----------
    D : (K, n+1+s) ndarray
        Data matrix.
    r : (K,1) ndarray
        X dot data reduced.
    k : float
        Regularization parameter
    num : int
        Index of the OLS problem we are solving (0 ≤ num < r).
    offdiagonals : bool
        If True, minimize only off-diagonals of A; if False, minimize all
        values of A, F, and c.

    Returns
    -------
    o : (n+1+s,) ndarray
        The least squares solution.
    """
    K,rps = D.shape

    F = np.eye(rps)
    F[num,num] = 0

    pseudo = reg*F
    rhs = np.zeros((rps))

    Dplus = np.vstack((D,pseudo))
    Rplus = np.vstack((r.reshape((-1,1)),rhs.reshape((-1,1))))

    return np.linalg.lstsq(Dplus, Rplus, rcond=None)[0]


def get_x_sq(X):
    """
    TODO
    """
    N,w = X.shape
    S = w*(w+1)/2
    Xi = np.zeros((N,int(S)))
    c = 0
    for i in range(w):
        Xi[:,c:c+(w-i)] = np.multiply(np.tile(X[:,i].reshape(N,1),[1,w-i]),X[:,i:w])
        c = c+(w-i)
    return Xi


def F2H(F):
    """
    TODO
    """
    n,_ = F.shape

    jj,ii = np.nonzero(F.T)

    FT = F.T
    vv = np.ravel(FT[jj,ii])
    jj = jj+1
    ii = ii+1
    iH = []
    jH = []
    vH = []
    # print(type(iH))
    bb = 0
    for i in range(1,n+1):
        cc = bb+n+1-i
        # print(jj)
        sel = (jj>bb) & (jj <=cc)
        # print(sel)
        itemp = ii[sel]
        jtemp = jj[sel]
        vtemp = vv[sel]
        for j in range(1,len(jtemp)+1):
            sj = jtemp[j-1] - bb
            if sj == 1 :
                iH.append([itemp[j-1]])
                jH.append([(i-1)*n+i+(sj)-1])
                vH.append([vtemp[j-1]])
            else:
                iH.append([itemp[j-1]])
                iH.append([itemp[j-1]])
                jH.append([(i-1)*n+(i)+(sj)-1])
                jH.append([(i+(sj)-2)*n+i])
                vH.append([vtemp[j-1]/2])
                vH.append([vtemp[j-1]/2])
        bb = cc
    # print(np.array(vH).shape)
    # print(np.array(iH).shape)
    # print(np.array(jH).shape)
    vHa = np.ndarray.flatten(np.array(vH))

    iHa = np.ndarray.flatten(np.array(iH))
    iHa = iHa-1
    jHa = np.ndarray.flatten(np.array(jH))
    jHa = jHa-1
    # print(vH)
    # print(vHa)
    H = csr_matrix((vHa,(iHa,jHa)),shape=(n,n**2)).toarray()
    return H
