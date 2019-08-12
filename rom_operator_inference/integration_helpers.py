# integration_helpers.py
"""Functions to integrate linear / quadratic / linear-quadratic systems."""

import numpy as np


def rk4advance_L(x, dt, A, B=0, u=0):
    """Do one step of 4th order Runge-Kutta integration of a system of the
    form x'(t) = Ax(t) + H(x ⊗ x)(t) + c.

    Parameters
    ----------
    x : (r,) ndarray
        State vector at the current timestep.
    dt : float
        Integration timestep size.
    A : (r,r) ndarray
        The linear operator.
    B : (r,p) ndarray, optional (default = 0, no input)
        The input operator.
    u : (p,) ndarray, optional (default = 0, no input)
        The input.

    Output
    -------------------
    x_next : (r,) ndarray
        The state vector integrated to the next timestep.
    """

    k1 = dt*(A@x + B@u)
    k2 = dt*(A@(x+k1/2) + B@u)
    k3 = dt*(A@(x+k2/2) + B@u)
    k4 = dt*(A@(x+k3) +B@u)

    r = (k1 + 2*k2 + 2*k3 + k4) / 6.
    return x + r


def rk4advance_Lc(x, dt, A, c, B=0, u=0):
    """One step of 4th order runge kutta integration of a system of the form
    x'(t) = Ax(t) + H(x ⊗ x)(t) + c

    Parameters
    ----------
    x : (r,) ndarray
        State vector at the current timestep.
    dt : float
        Integration timestep size.
    A : (r,r) ndarray
        The linear operator.
    c : (r,) ndarray
        The constant term.
    B : (r,p) ndarray, optional (default = 0, no input)
        The input operator.
    u : (p,) ndarray, optional (default = 0, no input)
        The input.

    Returns
    -------
    x_next : (r,) ndarray
        The state vector integrated to the next timestep.
    """
    k1 = dt*(A@x + c + B@u)
    k2 = dt*(A@(x+k1/2) + c + B@u)
    k3 = dt*(A@(x+k2/2) + c + B@u)
    k4 = dt*(A@(x+k3) + c +B@u)

    r = (k1 + 2*k2 + 2*k3 + k4) / 6.
    return x + r


def rk4advance_LQ(x, dt, A, H, B=0, u=0):
    """One step of 4th order runge kutta integration of a system of the form
    x'(t) = Ax(t) + H(x ⊗ x)(t) + c

    Parameters
    ----------
    x : (r,) ndarray
        State vector at the current timestep.
    dt : float
        Integration timestep size.
    A : (r,r) ndarray
        The linear operator.
    H : (r,r^2) ndarray
        The quadratic operator.
    B : (r,p) ndarray, optional (default = 0, no input)
        The input operator.
    u : (p,) ndarray, optional (default = 0, no input)
        The input.

    Returns
    -------
    x_next : (r,) ndarray
        The state vector integrated to the next timestep.
    """
    k1 = dt*(A@x + H@np.kron(x,x) + B@u)
    k2 = dt*(A@(x+k1/2) + H@np.kron(x+k1/2,x+k1/2) + B@u)
    k3 = dt*(A@(x+k2/2) + H@np.kron(x+k2/2,x+k2/2) + B@u)
    k4 = dt*(A@(x+k3) + H@np.kron(x+k3,x+k3) + B@u)

    r = (k1 + 2*k2 + 2*k3 + k4) / 6.
    return x + r


def rk4advance_LQ_addinput(x, dt, A, H, u):
    """One step of 4th order runge kutta integration of a system of the form
    x'(t) = Ax(t) + H(x ⊗ x)(t) + c

    Parameters
    ----------
    x : (r,) ndarray
        State vector at the current timestep.
    dt : float
        Integration timestep size.
    A : (r,r) ndarray
        The linear operator.
    H : (r,r^2) ndarray
        the quadratic operator
    B : (r,p) ndarray, optional (default = 0, no input)
        The input operator.
    u : (p,) ndarray, optional (default = 0, no input)
        The input.

    Returns
    -------
    x_next : (r,) ndarray
        The state vector integrated to the next timestep.
    """

    k1 = dt*(A@x + H@np.kron(x,x) + u)
    k2 = dt*(A@(x+k1/2) + H@np.kron(x+k1/2,x+k1/2) + u)
    k3 = dt*(A@(x+k2/2) + H@np.kron(x+k2/2,x+k2/2) + u)
    k4 = dt*(A@(x+k3) + H@np.kron(x+k3,x+k3) + u)

    r = (k1 + 2*k2 + 2*k3 + k4) / 6.
    return x + r


def rk4advance_LQc(x, dt, A, H, c, B=0, u=0):
    """One step of 4th order runge kutta integration of a system of the form
    x'(t) = Ax(t) + H(x ⊗ x)(t) + c

    Parameters
    ----------
    x : (r,) ndarray
        State vector at the current timestep.
    dt : float
        Integration timestep size.
    A : (r,r) ndarray
        The linear operator.
    H : (r,r^2) ndarray
        the quadratic operator
    c : (r,) ndarray
        The constant term.
    B : (r,p) ndarray, optional (default = 0, no input)
        The input operator.
    u : (p,) ndarray, optional (default = 0, no input)
        The input.

    Returns
    -------
    x_next : (r,) ndarray
        The state vector integrated to the next timestep.
    """

    k1 = dt*(A@x + H@np.kron(x,x) + c + B@u)
    k2 = dt*(A@(x+k1/2) + H@np.kron(x+k1/2,x+k1/2) + c + B@u)
    k3 = dt*(A@(x+k2/2) + H@np.kron(x+k2/2,x+k2/2) + c + B@u)
    k4 = dt*(A@(x+k3) + H@np.kron(x+k3,x+k3) + c + B@u)

    r = (k1 + 2*k2 + 2*k3 + k4) / 6.
    return x + r


def rk4advance_Q(x, dt, H, B=0, u=0):
    """One step of 4th order runge kutta integration of a system of the form
    x'(t) = Ax(t) + H(x ⊗ x)(t) + c

    Parameters
    ----------
    x : (r,) ndarray
        State vector at the current timestep.
    dt : float
        Integration timestep size.
    H : (r,r^2) ndarray
        the quadratic operator
    B : (r,p) ndarray, optional (default = 0, no input)
        The input operator.
    u : (p,) ndarray, optional (default = 0, no input)
        The input.

    Returns
    -------
    x_next : (r,) ndarray
        The state vector integrated to the next timestep.
    """

    k1 = dt*(H@np.kron(x,x) + B@u )
    k2 = dt*(H@np.kron(x+k1/2,x+k1/2) + B@u)
    k3 = dt*(H@np.kron(x+k2/2,x+k2/2) + B@u)
    k4 = dt*(H@np.kron(x+k3,x+k3) + B@u)

    r = (k1 + 2*k2 + 2*k3 + k4) / 6.
    return x + r


def rk4advance_Qc(x, dt, H, c, B=0, u=0):
    """One step of 4th order runge kutta integration of a system of the form
    x'(t) = Ax(t) + H(x ⊗ x)(t) + c

    Parameters
    ----------
    x : (r,) ndarray
        State vector at the current timestep.
    dt : float
        Integration timestep size.
    H : (r,r^2) ndarray
        the quadratic operator
    c : (r,) ndarray
        The constant term.
    B : (r,p) ndarray, optional (default = 0, no input)
        The input operator.
    u : (p,) ndarray, optional (default = 0, no input)
        The input.

    Returns
    -------
    x_next : (r,) ndarray
        The state vector integrated to the next timestep.
    """

    k1 = dt*(H@np.kron(x,x) + c + B@u)
    k2 = dt*(H@np.kron(x+k1/2,x+k1/2) + c + B@u)
    k3 = dt*(H@np.kron(x+k2/2,x+k2/2) + c + B@u)
    k4 = dt*(H@np.kron(x+k3,x+k3) + c +B@u)

    r = (k1 + 2*k2 + 2*k3 + k4) / 6.
    return x + r


def rk4_integrate(x0,dt,K,A, H, c):
    """Compute the full states as we integrate (for plotting a specific
    element).
    """

    r = len(x0)
    projected_state = np.zeros((r,K))
    projected_state[:,0] = x0

    for i in range(1,K):
        projected_state[:,i] = rk4advance_LQ(projected_state[:,i-1],
                                             dt, A, H, c)
        if np.any(np.isnan(projected_state[:,i])):
            print("NaNs enountered at simulation step ", i)
            break

    return projected_state,i
