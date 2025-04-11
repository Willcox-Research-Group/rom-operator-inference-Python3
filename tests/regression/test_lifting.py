# test_lifting.py
"""Regression test: Lifting, quadratic ROM, regularization selection."""
import os
import pytest
import itertools
import numpy as np
import scipy.interpolate

import opinf


DATAFILE = f"{__file__[:-3]}-data.npz"


v_bar = 1e2
p_bar = 1e5
z_bar = 1e-1


class EulerFiniteDifferenceModel:
    n_var = 3
    gamma = 1.4

    def __init__(self, L: float = 2.0, n_x: int = 200):
        self.x = np.linspace(0, L, n_x + 1)[:-1]
        self.dx = self.x[1] - self.x[0]
        self.L = L

    def spline_initial_conditions(self, init_params):
        x, L = self.x, self.L
        rho0s, v0s = init_params[0:3], init_params[3:6]
        v0s = np.concatenate((v0s, [v0s[0]]))
        rho0s = np.concatenate((rho0s, [rho0s[0]]))

        nodes = np.array([0, L / 3, 2 * L / 3, L]) + x[0]
        rho = scipy.interpolate.CubicSpline(nodes, rho0s, bc_type="periodic")(
            x
        )
        v = scipy.interpolate.CubicSpline(nodes, v0s, bc_type="periodic")(x)
        p = 1e5 * np.ones_like(x)
        rho_e = p / (self.gamma - 1) + 0.5 * rho * v**2

        return np.concatenate((rho, rho * v, rho_e))

    def _ddx(self, variable):
        return (variable - np.roll(variable, 1, axis=0)) / self.dx

    def derivative(self, tt, state):
        rho, rho_v, rho_e = np.split(state, self.n_var)
        v = rho_v / rho
        p = (self.gamma - 1) * (rho_e - 0.5 * rho * v**2)

        return -np.concatenate(
            [
                self._ddx(rho_v),
                self._ddx(rho * v**2 + p),
                self._ddx((rho_e + p) * v),
            ]
        )

    def solve(self, q_init, time_domain):
        return scipy.integrate.solve_ivp(
            fun=self.derivative,
            t_span=[time_domain[0], time_domain[-1]],
            y0=q_init,
            method="RK45",
            t_eval=time_domain,
            rtol=1e-6,
            atol=1e-9,
        ).y


# Experiment data =============================================================
def generate_data():
    fom = EulerFiniteDifferenceModel(L=2, n_x=200)

    t_final = 0.15
    n_t = 501
    t_all = np.linspace(0, t_final, n_t)

    t_obs = 0.06
    t_train = t_all[t_all <= t_obs]

    q0s = [
        fom.spline_initial_conditions(icparams)
        for icparams in itertools.product(
            [20, 24],
            [22],
            [20, 24],
            [95, 105],
            [100],
            [95, 105],
        )
    ]

    Q_fom = np.array([fom.solve(q0, t_train) for q0 in q0s])
    test_init = fom.spline_initial_conditions([22, 21, 25, 100, 98, 102])
    test_solution = fom.solve(test_init, t_all)

    np.savez(
        DATAFILE,
        x=fom.x,
        t_train=t_train,
        Qs=Q_fom,
        t_all=t_all,
        Qtest=test_solution,
    )


@pytest.fixture
def data():
    if not os.path.isfile(DATAFILE):
        generate_data()
    return np.load(DATAFILE)


class EulerLifter(opinf.lift.LifterTemplate):
    num_original_variables = 3
    num_lifted_variables = 3

    def lift(self, original_state):
        rho, rho_v, rho_e = np.split(
            original_state,
            self.num_original_variables,
            axis=0,
        )

        v = rho_v / rho
        p = (EulerFiniteDifferenceModel.gamma - 1) * (
            rho_e - 0.5 * rho_v * v
        )  # From the ideal gas law.
        zeta = 1 / rho

        return np.concatenate((v, p, zeta))

    def unlift(self, lifted_state):
        v, p, zeta = np.split(
            lifted_state,
            self.num_lifted_variables,
            axis=0,
        )

        rho = 1 / zeta
        rho_v = rho * v
        rho_e = (
            p / (EulerFiniteDifferenceModel.gamma - 1) + 0.5 * rho_v * v
        )  # From the ideal gas law.

        return np.concatenate((rho, rho_v, rho_e))


def rom_test_error(rom, maxerr: float):
    thedata = np.load(DATAFILE)
    t, Qtest = thedata["t_all"], thedata["Qtest"]

    Qrom = rom.predict(Qtest[:, 0], t)
    error = opinf.post.Lp_error(Qtest, Qrom, t)[1]

    assert error < maxerr


def test_01(data):
    """Lifting, scaling, fixed regularization (no centering)."""
    t_train, Qs = data["t_train"], data["Qs"]

    rom = opinf.ROM(
        lifter=EulerLifter(),
        transformer=opinf.pre.TransformerMulti(
            [
                opinf.pre.ScaleTransformer(1 / v_bar),
                opinf.pre.ScaleTransformer(1 / p_bar),
                opinf.pre.ScaleTransformer(1 / z_bar),
            ]
        ),
        basis=opinf.basis.PODBasis(num_vectors=9),
        ddt_estimator=opinf.ddt.UniformFiniteDifferencer(
            t_train, scheme="fwd4"
        ),
        model=opinf.models.ContinuousModel(
            operators=[opinf.operators.QuadraticOperator()],
            solver=opinf.lstsq.L2Solver(regularizer=1e-8),
        ),
    ).fit(Qs)

    rom_test_error(rom, 0.0032)


def test_02(data):
    """Lifting, scaling, regularization selection (no centering)."""
    t_train, Qs = data["t_train"], data["Qs"]

    rom = opinf.ROM(
        lifter=EulerLifter(),
        transformer=opinf.pre.TransformerMulti(
            [
                opinf.pre.ScaleTransformer(1 / v_bar),
                opinf.pre.ScaleTransformer(1 / p_bar),
                opinf.pre.ScaleTransformer(1 / z_bar),
            ]
        ),
        basis=opinf.basis.PODBasis(num_vectors=9),
        ddt_estimator=opinf.ddt.UniformFiniteDifferencer(
            t_train, scheme="fwd4"
        ),
        model=opinf.models.ContinuousModel(
            operators=[opinf.operators.QuadraticOperator()],
            solver=opinf.lstsq.L2Solver(regularizer=1e-8),
        ),
    ).fit_regselect_continuous(
        candidates=np.logspace(-12, 2, 15),
        train_time_domains=t_train,
        states=Qs,
        verbose=False,
    )

    rom_test_error(rom, 0.0023)


def test_03(data):
    """Lifting, centering, data-driven scaling, regularization selection."""
    t_train, Qs = data["t_train"], data["Qs"]

    rom = opinf.ROM(
        lifter=EulerLifter(),
        transformer=opinf.pre.TransformerMulti(  # Variable-specific scaling
            [
                opinf.pre.ShiftScaleTransformer(
                    centering=True, scaling="minmax"
                ),
                opinf.pre.ShiftScaleTransformer(
                    centering=True, scaling="minmax"
                ),
                opinf.pre.ShiftScaleTransformer(
                    centering=True, scaling="minmax"
                ),
            ]
        ),
        basis=opinf.basis.PODBasis(num_vectors=9),
        ddt_estimator=opinf.ddt.UniformFiniteDifferencer(
            t_train, scheme="fwd4"
        ),
        model=opinf.models.ContinuousModel(
            operators=[
                opinf.operators.ConstantOperator(),  # c
                opinf.operators.LinearOperator(),  # Aq(t)
                opinf.operators.QuadraticOperator(),  # H[q(t) ⊗ q(t)]
            ],
            solver=opinf.lstsq.L2Solver(regularizer=1e-8),
        ),
    )

    rom.fit_regselect_continuous(
        candidates=np.logspace(-12, 2, 15),
        train_time_domains=t_train,
        states=Qs,
        verbose=False,
    )

    rom_test_error(rom, 0.0031)


def test_04(data):
    """Lifting, centering, exact scaling, regularization selection."""
    t_train, Qs = data["t_train"], data["Qs"]

    rom = opinf.ROM(
        lifter=EulerLifter(),
        transformer=opinf.pre.TransformerPipeline(
            [
                opinf.pre.ShiftScaleTransformer(centering=True),  # Center.
                opinf.pre.TransformerMulti(
                    transformers=[
                        opinf.pre.ScaleTransformer(1 / v_bar),
                        opinf.pre.ScaleTransformer(1 / p_bar),
                        opinf.pre.ScaleTransformer(1 / z_bar),
                    ]
                ),
            ],
        ),
        basis=opinf.basis.PODBasis(num_vectors=8),
        ddt_estimator=opinf.ddt.UniformFiniteDifferencer(
            t_train, scheme="fwd4"
        ),
        model=opinf.models.ContinuousModel(
            operators=[
                opinf.operators.ConstantOperator(),
                opinf.operators.LinearOperator(),
                opinf.operators.QuadraticOperator(),
            ],
            solver=opinf.lstsq.L2Solver(regularizer=1e-8),
        ),
    )

    rom.fit_regselect_continuous(
        candidates=np.logspace(-12, 2, 15),
        train_time_domains=t_train,
        states=Qs,
    )

    rom_test_error(rom, 0.0021)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
