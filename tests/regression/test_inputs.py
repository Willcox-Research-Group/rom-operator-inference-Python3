# test_inputs.py
"""Heat equation with inputs."""
import os
import pytest
import numpy as np
import scipy.sparse

import opinf


DATAFILE = f"{__file__[:-3]}-data.npz"


input_functions = [
    lambda t: np.ones_like(t) + np.sin(4 * np.pi * t) / 4,
    lambda t: np.exp(-t),
    lambda t: 1 + t**2 / 2,
    lambda t: 1 - np.sin(np.pi * t) / 2,
    lambda t: 1 - np.sin(3 * np.pi * t) / 3,
    lambda t: 1 + 25 * (t * (t - 1)) ** 3,
    lambda t: 1 + np.exp(-2 * t) * np.sin(np.pi * t),
]


def generate_data():

    # Construct the spatial domain.
    L = 1
    n = 2**10 - 1
    x_all = np.linspace(0, L, n + 2)
    x = x_all[1:-1]
    dx = x[1] - x[0]

    # Construct the temporal domain.
    T = 1
    K = 10**3 + 1
    t_all = np.linspace(0, T, K)

    # Construct the full-order state matrix A.
    dx2inv = 1 / dx**2
    diags = np.array([1, -2, 1]) * dx2inv
    A = scipy.sparse.diags(diags, [-1, 0, 1], (n, n))

    # Construct the full-order input matrix B.
    B = np.zeros_like(x)
    B[0], B[-1] = dx2inv, dx2inv

    # Define the full-order model with an opinf.models class.
    fom = opinf.models.ContinuousModel(
        operators=[
            opinf.operators.LinearOperator(A),
            opinf.operators.InputOperator(B),
        ]
    )

    # Construct the part of the initial condition not dependent on u(t).
    alpha = 100
    q0 = np.exp(alpha * (x - 1)) + np.exp(-alpha * x) - np.exp(-alpha)

    def full_order_solve(time_domain, u):
        """Solve the full-order model with SciPy.
        Here, u is a callable function.
        """
        return fom.predict(q0 * u(0), time_domain, u, method="BDF")

    # Solve the full-order model with the training input.
    Qs = np.array([full_order_solve(t_all, u) for u in input_functions])

    np.savez(DATAFILE, t_all=t_all, Qs=Qs, A=A.toarray(), B=B)


@pytest.fixture
def data():
    if not os.path.isfile(DATAFILE):
        generate_data()
    return np.load(DATAFILE)


def test_01(data, k: int = 200, whichinput: int = 0):
    """Intrusive ROM, single trajectory, prediction in time."""
    t_all, Qs, A, B = data["t_all"], data["Qs"], data["A"], data["B"]
    Q = Qs[whichinput]
    basis = opinf.basis.PODBasis(residual_energy=1e-6).fit(Q[:, :k])

    rom_intrusive = opinf.ROM(
        basis=basis,
        model=opinf.models.ContinuousModel(
            operators=[
                opinf.operators.LinearOperator(A),
                opinf.operators.InputOperator(B),
            ]
        ).galerkin(
            basis.entries
        ),  # Explicitly project FOM operators.
    )
    Q_ROM_intrusive = rom_intrusive.predict(
        Q[:, 0], t_all, input_func=input_functions[whichinput], method="BDF"
    )

    error = opinf.post.frobenius_error(Q, Q_ROM_intrusive)[1]
    assert error < 0.0009


def test_02(data, k: int = 200, whichinput: int = 0):
    """OpInf ROM, single trajectory, prediction in time."""
    t_all, Qs = data["t_all"], data["Qs"]
    t = t_all[:k]
    Q = Qs[whichinput]
    input_func = input_functions[whichinput]

    rom = opinf.ROM(
        basis=opinf.basis.PODBasis(residual_energy=1e-6),
        ddt_estimator=opinf.ddt.UniformFiniteDifferencer(t, "ord6"),
        model=opinf.models.ContinuousModel("AB"),
    ).fit(Q[:, :k], inputs=input_func(t))

    Q_ROM = rom.predict(Q[:, 0], t_all, input_func=input_func, method="BDF")

    error = opinf.post.frobenius_error(Q, Q_ROM)[1]
    assert error < 0.0009


def test_03(data, k: int = 200, ntrain: int = 4):
    """OpInf ROM, multiple trajectories, prediction in time"""
    t_all, Qs = data["t_all"], data["Qs"]
    t = t_all[:k]
    Q_train = [Q[:, :k] for Q in Qs[:ntrain]]
    Q_test = Qs[ntrain:]
    U_train = [u(t) for u in input_functions[:ntrain]]

    rom = opinf.ROM(
        basis=opinf.basis.PODBasis(residual_energy=1e-6),
        ddt_estimator=opinf.ddt.UniformFiniteDifferencer(t, "ord6"),
        model=opinf.models.ContinuousModel("AB"),
    ).fit(Q_train, inputs=U_train)

    for i, u in enumerate(input_functions[ntrain:]):
        Q_ROM = rom.predict(Q_test[i][:, 0], t_all, u, method="BDF")
        error = opinf.post.frobenius_error(Q_test[i], Q_ROM)[1]
        assert error < 0.001


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
