# pre/_reprojection.py
"""Re-projection of trajectories for recovering intrusive models with
Operator Inference.
"""

__all__ = [
            "reproject_discrete",
            "reproject_continuous",
          ]

import numpy as np


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
    X_ = np.empty((r,niters))
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
    Xdot_ = np.empty((r,k))

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
