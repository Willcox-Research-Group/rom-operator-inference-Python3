# roms/test_parametric.py
"""Tests for roms._parametric.py."""

import pytest
import numpy as np

import opinf

from .test_base import _TestBaseROM


_module = opinf.roms


class TestParametricROM(_TestBaseROM):
    """Test roms.ParametricROM."""

    ROM = _module.ParametricROM
    ModelClasses = (
        opinf.models.ParametricContinuousModel,
        opinf.models.ParametricDiscreteModel,
        opinf.models.InterpContinuousModel,
        opinf.models.InterpDiscreteModel,
    )

    def _get_models(self):
        """Return a list of valid model instantiations."""
        return [
            opinf.models.ParametricContinuousModel(
                [
                    opinf.operators.ConstantOperator(),
                    opinf.operators.AffineLinearOperator(3),
                ]
            ),
            opinf.models.ParametricDiscreteModel(
                [
                    opinf.operators.AffineLinearOperator(3),
                    opinf.operators.InterpInputOperator(),
                ]
            ),
            opinf.models.InterpContinuousModel("AB"),
            opinf.models.InterpDiscreteModel("A"),
        ]

    def test_init(self):
        """Test __init__() and properties."""

        # Model error.
        with pytest.raises(TypeError) as ex:
            self.ROM(10)
        assert ex.value.args[0] == (
            "'model' must be a parametric model instance"
        )

        # Other arguments.
        super().test_init()

    def test_fit(self, n=20, m=3, s=8, k0=50):
        """Test fit()."""
        parameters = [np.sort(np.random.random(3)) for _ in range(s)]
        states = [np.random.standard_normal((n, k0 + i)) for i in range(s)]
        lhs = [np.zeros_like(Q) for Q in states]
        inputs = [np.ones((m, Q.shape[-1])) for Q in states]

        rom = self.ROM(model=opinf.models.InterpContinuousModel("AB"))
        with pytest.raises(ValueError) as ex:
            rom.fit(parameters, states, inputs)
        assert ex.value.args[0] == (
            "argument 'inputs' required (model depends on external inputs)"
        )

        with pytest.raises(ValueError) as ex:
            rom.fit(parameters, states, inputs=inputs)
        assert ex.value.args[0] == (
            "argument 'lhs' required when model is time-continuous"
            " and ddt_estimator=None"
        )

        def _fit(prom, withlhs=True):
            kwargs = dict(parameters=parameters, states=states)
            if withlhs:
                kwargs["lhs"] = lhs
            if prom.model._has_inputs:
                kwargs["inputs"] = inputs
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
                parameters=parameters,
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
                parameters=parameters,
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

    def test_predict(self, n=50, m=2, s=10, k0=40):
        """Test predict()."""
        parameters = [np.sort(np.random.random(3)) for _ in range(s)]
        states = [np.random.standard_normal((n, k0 + i)) for i in range(s)]
        inputs = [np.ones((m, Q.shape[-1])) for Q in states]
        t = np.linspace(0, 0.1, k0)
        testparam = np.mean(parameters, axis=0)
        testinit = states[0][:, s // 2]

        cmodel, dmodel, _, _ = self._get_models()

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
        rom.fit(parameters, states)
        out = rom.predict(testparam, testinit, t, input_func=None)
        assert out.shape == (n, t.size)

        # Discrete model.
        lift, trans, base = self._get("lifter", "transformer", "basis")
        rom = self.ROM(dmodel, lifter=lift, transformer=trans, basis=base)
        rom.fit(parameters, states, inputs=inputs)
        out = rom.predict(testparam, testinit, k0, inputs=inputs[0])
        assert out.shape == (n, k0)

    def test_fit_regselect_continuous(self):
        """Very lightly test fit_regselect_continuous()."""
        for model in self._get_models():
            rom = self.ROM(model)
            if not rom._iscontinuous:
                continue

            # Give the model a compatible solver.
            rom = self.ROM(
                model.__class__(
                    operators=model.operators,
                    solver=opinf.lstsq.L2Solver(1e-2),
                )
            )

            def func(t):
                return np.ones(3)

            # Tests.
            t = np.linspace(0, 1, 100)
            regs = np.logspace(-12, 2, 15)
            Q = np.ones((20, t.size))

            def blowup(*args, **kwargs):
                return np.zeros((20, t.size // 2))

            rom.model.predict = blowup

            with (
                pytest.raises(RuntimeError) as ex,
                pytest.warns(opinf.errors.OpInfWarning) as wn,
            ):
                rom.fit_regselect_continuous(
                    regs,
                    t,
                    [np.random.random(3) for i in range(8)],
                    [Q + i for i in range(8)],
                    ddts=[np.zeros_like(Q) for _ in range(8)],
                    input_functions=func if rom.model._has_inputs else None,
                    test_time_length=(t[-1] - t[0]) / 10,
                )
            assert len(wn) == 8
            for w in wn:
                assert w.message.args[0].startswith("ignoring stability limit")
            assert ex.value.args[0] == "regularization grid search failed"

    def test_fit_regselect_discrete(self):
        """Very lightly test fit_regselect_discrete()."""
        for model in self._get_models():
            rom = self.ROM(model)
            if rom._iscontinuous:
                continue

            # Give the model a compatible solver.
            rom = self.ROM(
                model.__class__(
                    operators=model.operators,
                    solver=opinf.lstsq.L2Solver(1e-2),
                )
            )

            # Tests.
            niters = 100
            regs = np.logspace(-12, 6, 19)
            Q = np.ones((20, niters))

            with (
                # pytest.raises(RuntimeError) as ex,
                pytest.warns(opinf.errors.OpInfWarning) as wn,
            ):
                rom.fit_regselect_discrete(
                    regs,
                    [np.random.random(3) for _ in range(8)],
                    [Q + i for i in range(8)],
                    inputs=(
                        [np.ones((3, niters)) for _ in range(8)]
                        if rom.model._has_inputs
                        else None
                    ),
                )
            assert len(wn) >= 8
            for w in wn[:8]:
                assert w.message.args[0].startswith("ignoring stability limit")
            # assert ex.value.args[0] == "regularization grid search failed"


if __name__ == "__main__":
    pytest.main([__file__])
