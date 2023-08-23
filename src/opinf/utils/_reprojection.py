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
def reproject_discrete(f, basis, init, niters, inputs=None):
    """Sample re-projected trajectories of the discrete dynamical system

        q_{j+1} = f(q_{j}, u_{j}),  q_{0} = q0.

    Parameters
    ----------
    f : callable mapping (n,) ndarray (and (m,) ndarray) to (n,) ndarray
        Function defining the (full-order) discrete dynamical system. Accepts
        a full-order state vector and (optionally) an input vector and returns
        another full-order state vector.
    basis : (n, r) ndarray
        Basis for the low-dimensional linear subspace (e.g., POD basis).
    init : (n,) ndarray
        Initial condition for the iteration in the high-dimensional space.
    niters : int
        The number of iterations to do.
    inputs : (m, niters-1) or (niters-1) ndarray
        Control inputs, one for each iteration beyond the initial condition.

    Returns
    -------
    states_reprojected : (r, niters) ndarray
        Re-projected state trajectories in the projected low-dimensional space.
    """
    # Validate and extract dimensions.
    n, r = basis.shape
    if init.shape != (n,):
        raise ValueError("basis and initial condition not aligned")

    # Create the solution array and fill in the initial condition.
    states_ = np.empty((r, niters))
    states_[:, 0] = basis.T @ init

    # Run the re-projection iteration.
    if inputs is None:
        for j in range(niters-1):
            states_[:, j+1] = basis.T @ f(basis @ states_[:, j])
    elif inputs.ndim == 1:
        for j in range(niters-1):
            states_[:, j+1] = basis.T @ f(basis @ states_[:, j], inputs[j])
    else:
        for j in range(niters-1):
            states_[:, j+1] = basis.T @ f(basis @ states_[:, j], inputs[:, j])

    return states_


def reproject_continuous(f, basis, states, inputs=None):
    """Sample re-projected trajectories of the continuous system of ODEs

        dq / dt = f(t, q(t), u(t)),     q(0) = q0.

    Parameters
    ----------
    f : callable mapping (n,) ndarray (and (m,) ndarray) to (n,) ndarray
        Function defining the (full-order) differential equation. Accepts a
        full-order state vector and (optionally) an input vector and returns
        another full-order state vector.
    basis : (n, r) ndarray
        Basis for the low-dimensional linear subspace.
    states : (n, k) ndarray
        State trajectories (training data).
    inputs : (m, k) or (k,) ndarray
        Control inputs corresponding to the state trajectories.

    Returns
    -------
    states_reprojected : (r, k) ndarray
        Re-projected state trajectories in the projected low-dimensional space.
    ddts_reprojected : (r, k) ndarray
        Re-projected velocities in the projected low-dimensional space.
    """
    # Validate and extract dimensions.
    if states.shape[0] != basis.shape[0]:
        raise ValueError("states and basis not aligned, first dimension "
                         f"{states.shape[0]} != {basis.shape[0]}")
    n, r = basis.shape
    k = states.shape[1]

    # Create the solution arrays.
    states_ = basis.T @ states
    ddts_ = np.empty((r, k))

    # Run the re-projection iteration.
    if inputs is None:
        for j in range(k):
            ddts_[:, j] = basis.T @ f(basis @ states_[:, j])
    elif inputs.ndim == 1:
        for j in range(k):
            ddts_[:, j] = basis.T @ f(basis @ states_[:, j], inputs[j])
    else:
        for j in range(k):
            ddts_[:, j] = basis.T @ f(basis @ states_[:, j], inputs[:, j])

    return states_, ddts_
