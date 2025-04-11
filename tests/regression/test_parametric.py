# test_parametric.py
"""Regression test: parametric problems."""

import os
import pytest
import numpy as np
import scipy.sparse

import opinf


DATAFILE = f"{__file__[:-3]}-data.npz"


def generate_data():
    s = 10
    training_parameters = np.logspace(-1, 1, s)
    testing_parameters = np.sqrt(
        training_parameters[:-1] * training_parameters[1:]
    )

    L = 1
    n = 2**10 - 1
    x_all = np.linspace(0, L, n + 2)
    x = x_all[1:-1]
    dx = x[1] - x[0]

    T = 1
    K = 401
    t_all = np.linspace(0, T, K)

    dx2inv = 1 / dx**2
    diags = np.array([1, -2, 1]) * dx2inv
    A0 = scipy.sparse.diags(diags, [-1, 0, 1], (n, n))

    c0 = np.zeros_like(x)
    c0[0], c0[-1] = dx2inv, dx2inv

    alpha = 100
    q0 = np.exp(alpha * (x - 1)) + np.exp(-alpha * x) - np.exp(-alpha)

    def full_order_solve(mu, time_domain):
        """Solve the full-order model with SciPy.
        Here, u is a callable function.
        """
        return scipy.integrate.solve_ivp(
            fun=lambda t, q: mu * (c0 + A0 @ q),
            y0=q0,
            t_span=[time_domain[0], time_domain[-1]],
            t_eval=time_domain,
            method="BDF",
        ).y

    Qtrain = [full_order_solve(mu, t_all) for mu in training_parameters]
    Qtest = [full_order_solve(mu, t_all) for mu in testing_parameters]

    np.savez(
        DATAFILE,
        t_all=t_all,
        training_parameters=training_parameters,
        testing_parameters=testing_parameters,
        Qtrain=Qtrain,
        Qtest=Qtest,
    )


@pytest.fixture
def data():
    if not os.path.isfile(DATAFILE):
        generate_data()
    return np.load(DATAFILE)


def test_01(data):
    t_all, training_parameters, testing_parameters, Qtrain, Qtest = (
        data["t_all"],
        data["training_parameters"],
        data["testing_parameters"],
        data["Qtrain"],
        data["Qtest"],
    )

    rom = opinf.ParametricROM(
        basis=opinf.basis.PODBasis(projection_error=1e-6),
        ddt_estimator=opinf.ddt.UniformFiniteDifferencer(t_all, "ord6"),
        model=opinf.models.ParametricContinuousModel(
            operators=[
                opinf.operators.AffineConstantOperator(1),
                opinf.operators.AffineLinearOperator(1),
            ],
            solver=opinf.lstsq.L2Solver(1e-6),
        ),
    ).fit(training_parameters, Qtrain)

    maxerrors = [
        0.0528,
        0.0344,
        0.0228,
        0.0156,
        0.0111,
        0.0078,
        0.0051,
        0.0028,
        0.0010,
        0.0002,
    ]
    for mu, Q, maxerr in zip(training_parameters, Qtrain, maxerrors):
        Q_ROM = rom.predict(mu, Q[:, 0], t_all, method="BDF")
        error = opinf.post.frobenius_error(Q, Q_ROM)[1]
        assert error < maxerr

    maxerrors = [
        0.0426,
        0.0279,
        0.0188,
        0.0131,
        0.0093,
        0.0064,
        0.0039,
        0.0018,
        0.0004,
    ]
    for mu, Q, maxerr in zip(testing_parameters, Qtest, maxerrors):
        Q_ROM = rom.predict(mu, Q[:, 0], t_all, method="BDF")
        error = opinf.post.frobenius_error(Q, Q_ROM)[1]
        assert error < maxerr


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
