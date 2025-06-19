# roms/test_nonparametric.py
"""Tests for roms._nonparametric.py."""

import pytest
import warnings
import numpy as np

import opinf

try:
    from .test_base import _TestBaseROM
except ImportError:
    from test_base import _TestBaseROM


_module = opinf.roms


class TestROM(_TestBaseROM):
    """Test roms.ROM."""

    ROM = _module.ROM
    ModelClasses = (
        opinf.models.ContinuousModel,
        opinf.models.DiscreteModel,
    )
    check_regselect_solver = True
    kwargs = dict()  # extra arguments for fit_regselect_*().

    def _get_models(self):
        """Return a list of valid model instantiations."""
        return [
            opinf.models.ContinuousModel(
                "A",
                solver=opinf.lstsq.L2Solver(1e-4),
            ),
            opinf.models.DiscreteModel(
                "AB",
                solver=opinf.lstsq.L2Solver(1e-2),
            ),
            opinf.models.DiscreteModel(
                "A",
                solver=opinf.lstsq.L2Solver(1e-3),
            ),
            opinf.models.ContinuousModel(
                "AB",
                solver=opinf.lstsq.L2Solver(1e-1),
            ),
        ]

    def test_init(self):
        """Test __init__() and properties."""

        # Model error.
        with pytest.raises(TypeError) as ex:
            self.ROM(10)
        assert ex.value.args[0] == (
            "'model' must be a nonparametric model instance"
        )

        # Other arguments.
        super().test_init()

    def test_fit(self, n=10, m=3, s=3, k0=50):
        """Test fit()."""
        states = [np.random.standard_normal((n, k0 + i)) for i in range(s)]
        lhs = [np.random.standard_normal(Q.shape) / 100 for Q in states]
        inputs = [np.ones((m, Q.shape[-1])) for Q in states]

        rom = self.ROM(
            model=opinf.models.ContinuousModel(
                "cBH",
                solver=opinf.lstsq.L2Solver(1e-2),
            )
        )
        with pytest.raises(ValueError) as ex:
            rom.fit(states, inputs)
        assert ex.value.args[0] == (
            "argument 'inputs' required (model depends on external inputs)"
        )

        with pytest.raises(ValueError) as ex:
            rom.fit(states, inputs=inputs)
        assert ex.value.args[0] == (
            "argument 'lhs' required when model is time-continuous"
            " and ddt_estimator=None"
        )

        def _fit(prom, withlhs=True, singletrajectory=False):
            kwargs = dict(states=states)
            if withlhs:
                kwargs["lhs"] = lhs
            if prom.model._has_inputs:
                kwargs["inputs"] = inputs
            if singletrajectory:
                kwargs = {key: val[0] for key, val in kwargs.items()}
            prom.fit(**kwargs)
            assert rom.model.operators[0].entries is not None
            if rom.basis is not None:
                assert (r := rom.basis.reduced_state_dimension) == 3
                assert model.state_dimension == r

        for model in self._get_models():
            # Model only.
            rom = self.ROM(model)
            _fit(rom)
            assert model.state_dimension == n

            # Model and basis.
            rom = self.ROM(model, basis=self._get("basis"))
            _fit(rom)
            assert rom.basis.full_state_dimension == n
            oldbasisentries = rom.basis.entries.copy()

            # Make sure fit_basis=False doesn't change the basis.
            rom.fit(
                states=[Q + 1 for Q in states],
                lhs=lhs,
                inputs=inputs if model._has_inputs else None,
                fit_basis=False,
            )
            assert np.array_equal(rom.basis.entries, oldbasisentries)

            # Model and basis and transformer.
            trans, base = self._get("transformer", "basis")
            rom = self.ROM(model, transformer=trans, basis=base)
            _fit(rom)
            assert rom.transformer.state_dimension == n

            # Make sure fit_transformer=False doesn't change the basis.
            z = np.random.random(n)
            ztrans = rom.transformer.transform(z)
            rom.fit(
                states=[Q + 1 for Q in states],
                lhs=lhs,
                inputs=inputs if model._has_inputs else None,
                fit_transformer=False,
            )
            ztrans2 = rom.transformer.transform(z)
            assert np.allclose(ztrans2, ztrans)

            # Model and lifter and basis and transformer.
            lift, trans, base = self._get("lifter", "transformer", "basis")
            rom = self.ROM(model, lifter=lift, transformer=trans, basis=base)
            _fit(rom)
            assert rom.transformer.state_dimension == 2 * n
            assert rom.basis.full_state_dimension == 2 * n

            # Without lhs.
            ddter = None
            if rom._iscontinuous:
                # Without ddt_estimator either.
                rom = self.ROM(model)
                with pytest.raises(ValueError) as ex:
                    _fit(rom, withlhs=False)
                assert ex.value.args[0] == (
                    "argument 'lhs' required when model is time-continuous "
                    "and ddt_estimator=None"
                )

                ddter = self._get("ddt_estimator")

            lift, trans, base = self._get("lifter", "transformer", "basis")
            rom = self.ROM(
                model,
                lifter=lift,
                transformer=trans,
                basis=base,
                ddt_estimator=ddter,
            )
            _fit(rom, withlhs=False)
            _fit(rom, singletrajectory=True)

    def test_predict(self, n=50, m=2, k=100):
        """Test predict()."""
        states = np.random.standard_normal((n, k))
        inputs = np.ones((m, k))
        t = np.linspace(0, 0.1, k)
        q0 = states[:, 0]

        cmodel, dmodel = self._get_models()[:2]

        # Continuous model.
        lift, trans, base, ddter = self._get(
            "lifter", "transformer", "basis", "ddt_estimator"
        )
        rom = self.ROM(
            cmodel,
            lifter=lift,
            transformer=trans,
            basis=base,
            ddt_estimator=ddter,
        )
        rom.fit(states)
        out = rom.predict(q0, t, input_func=None)
        assert out.shape == (n, t.size)

        # Discrete model.
        lift, trans, base = self._get("lifter", "transformer", "basis")
        rom = self.ROM(dmodel, lifter=lift, transformer=trans, basis=base)
        rom.fit(states, inputs=inputs)
        out = rom.predict(q0, k, inputs=inputs)
        assert out.shape == (n, k)

    def test_fit_regselect_continuous(self):
        """Lightly test fit_regselect_continuous()."""
        for model in self._get_models():
            if self.check_regselect_solver:
                rom = self.ROM(
                    model.__class__(
                        operators=model.operators,
                        solver=None,
                    )
                )
                if not rom._iscontinuous:
                    with pytest.raises(AttributeError) as ex:
                        rom.fit_regselect_continuous(None, None, None)
                    assert ex.value.args[0] == (
                        "this method is for time-continuous models only, "
                        "use fit_regselect_discrete()"
                    )
                    continue
                with pytest.raises(AttributeError) as ex:
                    rom.fit_regselect_continuous(None, None, None)
                assert ex.value.args[0] == (
                    "this method requires a model with a 'solver' attribute "
                    "which has a 'regularizer' attribute"
                )
            elif not opinf.models._utils.is_continuous(model):
                continue

            rom = self.ROM(model)

            def func(t):
                return np.ones(3)

            # Bad argument detection.
            t = np.linspace(0, 1, 100)
            Q = np.zeros((5, t.size // 2))
            with pytest.raises(opinf.errors.DimensionalityError) as ex:
                rom.fit_regselect_continuous(None, t, Q)
            assert ex.value.args[0] == (
                "train_time_domains and states not aligned"
            )

            Q = np.random.standard_normal((5, t.size))
            with pytest.raises(TypeError) as ex:
                rom.fit_regselect_continuous(None, t, Q, input_functions=10)
            assert ex.value.args[0] == (
                "argument 'input_functions' must be sequence of callables"
            )
            with pytest.raises(ValueError) as ex:
                rom.fit_regselect_continuous(None, t, Q, test_time_length=-10)
            assert ex.value.args[0] == (
                "argument 'test_time_length' must be nonnegative"
            )
            with pytest.raises(TypeError) as ex:
                rom.fit_regselect_continuous(
                    candidates=None,
                    train_time_domains=t,
                    states=Q,
                    ddts=Q if rom.ddt_estimator is None else None,
                    input_functions=func if rom.model._has_inputs else None,
                    test_cases=[10, 20],
                )
            assert ex.value.args[0] == (
                "test cases must be 'utils.ContinuousRegTest' objects"
            )

            # Tests.
            regs = np.logspace(-12, 2, 15)
            Q = np.ones((5, t.size))
            Q += np.random.standard_normal(Q.shape) / 10

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = rom.fit_regselect_continuous(
                    regs,
                    t,
                    Q,
                    ddts=np.random.standard_normal(Q.shape) / 100,
                    input_functions=func if rom.model._has_inputs else None,
                    test_time_length=(t[-1] - t[0]) / 10,
                    test_cases=opinf.utils.ContinuousRegTest(
                        np.ones(5),
                        t,
                        input_function=func if rom.model._has_inputs else None,
                    ),
                    **self.kwargs,
                )
            assert out is rom
            for op in rom.model.operators:
                assert op.entries is not None
            assert np.linalg.norm(rom.model.A_.entries) < 1

            def func(t):
                return 1

            with pytest.raises(RuntimeError) as ex:
                rom.fit_regselect_continuous(
                    regs,
                    t,
                    [Q, Q + 1],
                    ddts=[np.zeros_like(Q), np.ones_like(Q)],
                    input_functions=func if rom.model._has_inputs else None,
                    test_cases=opinf.utils.ContinuousRegTest(
                        np.ones(5) / 2,
                        t,
                        input_function=func if rom.model._has_inputs else None,
                        bound=1e-20,
                    ),
                    **self.kwargs,
                )
            assert ex.value.args[0] == "regularization grid search failed"

    def test_fit_regselect_discrete(self):
        """Lightly test fit_regselect_discrete()."""
        for model in self._get_models():
            if self.check_regselect_solver:
                rom = self.ROM(
                    model.__class__(
                        operators=model.operators,
                        solver=None,
                    )
                )
                if rom._iscontinuous:
                    with pytest.raises(AttributeError) as ex:
                        rom.fit_regselect_discrete(None, None)
                    assert ex.value.args[0] == (
                        "this method is for fully discrete models only, "
                        "use fit_regselect_continuous()"
                    )
                    continue

                with pytest.raises(AttributeError) as ex:
                    rom.fit_regselect_discrete(None, None)
                assert ex.value.args[0] == (
                    "this method requires a model with a 'solver' attribute "
                    "which has a 'regularizer' attribute"
                )
            elif opinf.models._utils.is_continuous(model):
                continue

            rom = self.ROM(model)

            # Bad argument detection.
            niters = 100
            Q = np.random.standard_normal((5, niters))
            with pytest.raises(ValueError) as ex:
                rom.fit_regselect_discrete(None, Q, num_test_iters=-10)
            assert ex.value.args[0] == (
                "argument 'num_test_iters' must be a nonnegative integer"
            )
            with pytest.raises(TypeError) as ex:
                rom.fit_regselect_discrete(
                    candidates=None,
                    states=Q,
                    inputs=np.ones(niters) if rom.model._has_inputs else None,
                    test_cases=[10, 20],
                )
            assert ex.value.args[0] == (
                "test cases must be 'utils.DiscreteRegTest' objects"
            )
            if rom.model._has_inputs:
                with pytest.raises(ValueError) as ex:
                    rom.fit_regselect_discrete(
                        None,
                        [Q, Q + 1],
                        inputs=np.ones(niters),
                    )
                assert ex.value.args[0] == (
                    f"2 state trajectories but "
                    f"{niters} input trajectories detected"
                )
                with pytest.raises(ValueError) as ex:
                    rom.fit_regselect_discrete(
                        None,
                        Q,
                        inputs=np.ones(niters),
                        num_test_iters=10,
                    )
                assert ex.value.args[0] == (
                    "argument 'inputs' must contain enough data for "
                    "10 iterations after the training data"
                )

            # Tests.
            regs = np.logspace(-12, 6, 19)
            Q = np.ones((5, niters))
            Q += np.random.standard_normal(Q.shape) / 10

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = rom.fit_regselect_discrete(
                    regs,
                    Q,
                    inputs=(
                        np.ones((3, niters + 20))
                        if rom.model._has_inputs
                        else None
                    ),
                    num_test_iters=20,
                    test_cases=opinf.utils.DiscreteRegTest(
                        np.ones(5),
                        niters=5,
                        inputs=(
                            np.ones((3, 5)) if rom.model._has_inputs else None
                        ),
                    ),
                    **self.kwargs,
                )
            assert out is rom
            for op in rom.model.operators:
                assert op.entries is not None

            def blowup(*args, **kwargs):
                return np.inf * np.ones_like(Q)

            rom.model.predict = blowup

            with pytest.raises(RuntimeError) as ex:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rom.fit_regselect_discrete(
                        regs,
                        [Q, Q + 1],
                        inputs=(
                            [np.ones(niters)] * 2
                            if rom.model._has_inputs
                            else None
                        ),
                        **self.kwargs,
                    )
            assert ex.value.args[0] == "regularization grid search failed"

            with pytest.raises(RuntimeError) as ex:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rom.fit_regselect_discrete(
                        regs,
                        Q,
                        inputs=(
                            np.ones(niters) if rom.model._has_inputs else None
                        ),
                        test_cases=opinf.utils.DiscreteRegTest(
                            Q[:, 0], niters=10
                        ),
                        **self.kwargs,
                    )
            assert ex.value.args[0] == "regularization grid search failed"


if __name__ == "__main__":
    pytest.main([__file__])
