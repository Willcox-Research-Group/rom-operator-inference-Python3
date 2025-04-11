# test_basics.py
"""Regression test: linear heat equation, extrapolation to new initial =
conditions.
"""
import os
import pytest
import numpy as np
import scipy.sparse
import scipy.integrate

import opinf


DATAFILE = f"{__file__[:-3]}-data.npz"


def generate_data():
    L = 1
    n = 2**10 - 1
    x_all = np.linspace(0, L, n + 2)
    x = x_all[1:-1]
    dx = x[1] - x[0]

    t0, tf = 0, 1
    k = 401
    t = np.linspace(t0, tf, k)

    diags = np.array([1, -2, 1]) / (dx**2)
    A = scipy.sparse.diags(diags, [-1, 0, 1], (n, n))

    # Initial conditions for the training data.
    q0s = [
        x * (1 - x),
        10 * x * (1 - x),
        5 * x**2 * (1 - x) ** 2,
        50 * x**4 * (1 - x) ** 4,
        0.5 * np.sqrt(x * (1 - x)),
        0.25 * np.sqrt(np.sqrt(x * (1 - x))),
        np.sin(np.pi * x) / 3 + np.sin(5 * np.pi * x) / 5,
    ]

    def full_order_solve(initial_condition, time_domain):
        """Solve the full-order model with SciPy."""
        return scipy.integrate.solve_ivp(
            fun=lambda t, q: A @ q,
            t_span=[time_domain[0], time_domain[-1]],
            y0=initial_condition,
            t_eval=time_domain,
            method="BDF",
        ).y

    Qs = np.array([full_order_solve(q0, t) for q0 in q0s])
    np.savez(DATAFILE, t=t, Qs=Qs)


@pytest.fixture
def data():
    if not os.path.isfile(DATAFILE):
        generate_data()
    return np.load(DATAFILE)


def test_01(data, whichinit: int = 0):
    """OpInf ROM, reproduce training data, no extrapolation."""
    t, Qs = data["t"], data["Qs"]
    Q = Qs[whichinit]

    rom = opinf.ROM(
        basis=opinf.basis.PODBasis(cumulative_energy=0.9999),
        ddt_estimator=opinf.ddt.UniformFiniteDifferencer(t, "ord6"),
        model=opinf.models.ContinuousModel(
            operators="A",
            solver=opinf.lstsq.L2Solver(regularizer=1e-8),
        ),
    ).fit(Q)

    Q_ROM = rom.predict(Q[:, 0], t, method="BDF")
    error = opinf.post.frobenius_error(Q, Q_ROM)[1]
    assert error < 0.00105


def test_02(data):
    """OpInf ROM, training data from one trajectory, basis enriched with a few
    initial conditions.
    """
    t, Qs = data["t"], data["Qs"]
    Qtrain = Qs[0]
    q0_new = [Q[:, 0] for Q in Qs[1:]]

    Q_and_new_q0s = np.column_stack((Qtrain, *q0_new))

    # Initialize a ROM with the new basis.
    rom = opinf.ROM(
        basis=opinf.basis.PODBasis(num_vectors=5).fit(Q_and_new_q0s),
        ddt_estimator=opinf.ddt.UniformFiniteDifferencer(t, "ord6"),
        model=opinf.models.ContinuousModel(
            operators="A",
            solver=opinf.lstsq.L2Solver(regularizer=1e-8),
        ),
    ).fit(Qtrain, fit_basis=False)

    maxerrors = [0.00041, 0.00085, 0.00335, 0.00340, 0.009500, 0.02610]
    for Q, maxerr in zip(Qs, maxerrors):
        Q_ROM = rom.predict(Q[:, 0], t, method="BDF")
        error = opinf.post.frobenius_error(Q, Q_ROM)[1]
        assert error < maxerr


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
