# roms/test_nonparametric.py
"""Tests for roms._nonparametric.py."""

import pytest
import numpy as np

import opinf

from .test_base import _TestBaseROM


_module = opinf.roms


class TestROM(_TestBaseROM):
    """Test roms.ROM."""

    ROM = _module.ROM
    ModelClasses = (
        opinf.models.ContinuousModel,
        opinf.models.DiscreteModel,
    )

    def _get_models(self):
        """Return a list of valid model instantiations."""
        return [
            opinf.models.ContinuousModel("A"),
            opinf.models.DiscreteModel("AB"),
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
        lhs = [np.zeros_like(Q) for Q in states]
        inputs = [np.ones((m, Q.shape[-1])) for Q in states]

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

        cmodel, dmodel = self._get_models()

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
