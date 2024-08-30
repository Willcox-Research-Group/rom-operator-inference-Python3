# roms/test_base.py
"""Tests for roms._base."""

import abc
import pytest
import numpy as np

import opinf
from opinf.models import _utils as modutils


class _TestBaseROM(abc.ABC):
    """Test opinf.roms._base._BaseROM."""

    ROM = NotImplemented
    ModelClasses = NotImplemented

    @abc.abstractmethod
    def _get_models(self):
        """Return a list of valid model instantiations."""
        pass

    @staticmethod
    def _get(*keys):
        args = dict(
            lifter=opinf.lift.QuadraticLifter(),
            transformer=opinf.pre.ShiftScaleTransformer(centering=True),
            transformer2=opinf.pre.ShiftScaleTransformer(scaling="standard"),
            basis=opinf.basis.PODBasis(num_vectors=3),
            basis2=opinf.basis.PODBasis(num_vectors=4),
            ddt_estimator=opinf.ddt.UniformFiniteDifferencer(
                np.linspace(0, 1, 100)
            ),
        )
        args["multi_transformer"] = opinf.pre.TransformerMulti(
            [args["transformer"], args["transformer2"]]
        )
        args["multi_basis"] = opinf.basis.BasisMulti(
            [args["basis"], args["basis2"]]
        )
        if len(keys) == 1:
            return args[keys[0]]
        return [args[k] for k in keys]

    def test_init(self):
        """Test __init__() and properties."""

        for model in self._get_models():
            # Warnings for non-model arguments.
            with pytest.warns(opinf.errors.OpInfWarning) as wn:
                self.ROM(
                    model,
                    lifter=10,
                    transformer=8,
                    basis=6,
                    ddt_estimator=4,
                )
            assert len(wn) == 4
            assert wn[0].message.args[0] == (
                "lifter not derived from LifterTemplate, "
                "unexpected behavior may occur"
            )
            assert wn[1].message.args[0] == (
                "transformer not derived from TransformerTemplate "
                "or TransformerMulti, unexpected behavior may occur"
            )
            assert wn[2].message.args[0] == (
                "basis not derived from BasisTemplate or BasisMulti, "
                "unexpected behavior may occur"
            )
            assert wn[3].message.args[0] == (
                "ddt_estimator not derived from DerivativeEstimatorTemplate, "
                "unexpected behavior may occur"
            )

            # Given ddt_estimator with non-continuous model.
            if modutils.is_discrete(model):
                with pytest.warns(opinf.errors.OpInfWarning) as wn:
                    rom = self.ROM(
                        model,
                        ddt_estimator=self._get_args("ddt_estimator"),
                    )
                assert len(wn) == 1
                assert wn[0].message.args[0] == (
                    "ddt_estimator ignored for discrete models"
                )
                assert rom.ddt_estimator is None
                assert not rom._iscontinuous

            # Correct usage.
            lifter, ddt_estimator = self._get("lifter", "ddt_estimator")
            rom = self.ROM(
                model,
                lifter=lifter,
                ddt_estimator=ddt_estimator,
            )
            assert rom.lifter is lifter
            assert rom.transformer is None
            assert rom.basis is None
            assert rom.ddt_estimator is ddt_estimator

            transformer, basis = self._get("multi_transformer", "multi_basis")
            rom = self.ROM(
                model,
                transformer=transformer,
                basis=basis,
            )
            assert rom.lifter is None
            assert rom.transformer is transformer
            assert rom.basis is basis
            assert rom.ddt_estimator is None

    def test_str(self):
        """Lightly test __str__() and __repr__()."""
        for model in self._get_models():
            a1, a2, a3 = self._get("lifter", "transformer", "basis")
            repr(self.ROM(model, lifter=a1, transformer=a2, basis=a3))

    def test_encode(self, n=40, k=20):
        """Test encode()."""
        states = np.random.random((n, k))
        lhs = np.random.random((n, k))

        def _check(arr, shape):
            assert isinstance(arr, np.ndarray)
            assert arr.shape == shape

        for model in self._get_models():
            # Lifter only.
            rom = self.ROM(model, lifter=self._get("lifter"))
            _check(rom.encode(states), (2 * n, k))
            out1, out2 = rom.encode(states, lhs)
            for out in out1, out2:
                _check(out, (2 * n, k))
            _check(rom.encode(states[:, 0]), (2 * n,))
            out1, out2 = rom.encode(states[:, 0], lhs[:, 0])
            for out in out1, out2:
                _check(out, (2 * n,))

            # Transformer only.
            rom = self.ROM(model, transformer=self._get("transformer"))
            with pytest.raises(AttributeError) as ex:
                rom.encode(states)
            assert ex.value.args[0] == (
                "transformer not trained (call fit() or fit_transform())"
            )

            out = rom.encode(states, fit_transformer=True, inplace=False)
            _check(out, (n, k))
            out1, out2 = rom.encode(states, lhs)
            for out in out1, out2:
                _check(out, (n, k))
            out = rom.encode(states[:, 0])
            _check(out, (n,))
            out1, out2 = rom.encode(states[:, 0], lhs[:, 0])
            for out in out1, out2:
                _check(out, (n,))

            # Basis only.
            rom = self.ROM(model, basis=self._get("basis"))
            with pytest.raises(AttributeError) as ex:
                rom.encode(states)
            assert ex.value.args[0] == "basis entries not initialized"

            out = rom.encode(states, fit_basis=True)
            r = rom.basis.reduced_state_dimension
            _check(out, (r, k))
            out1, out2 = rom.encode(states, lhs)
            for out in out1, out2:
                _check(out, (r, k))
            _check(rom.encode(states[:, 0]), (r,))
            out1, out2 = rom.encode(states[:, 0], lhs[:, 0])
            for out in out1, out2:
                _check(out, (r,))

            # Lifter, transformer, and basis.
            a1, a2, a3 = self._get("lifter", "transformer", "basis")
            rom = self.ROM(model, lifter=a1, transformer=a2, basis=a3)
            out = rom.encode(states, fit_transformer=True, fit_basis=True)
            r = rom.basis.reduced_state_dimension
            _check(out, (r, k))
            out1, out2 = rom.encode(states, lhs)
            for out in out1, out2:
                _check(out, (r, k))
            _check(rom.encode(states[:, 0]), (r,))
            out1, out2 = rom.encode(states[:, 0], lhs[:, 0])
            for out in out1, out2:
                _check(out, (r,))

    def test_decode(self):
        """Test decode()."""
        raise NotImplementedError

    def test_project(self):
        """Test project()."""
        raise NotImplementedError

    def test_fit(self):
        """Test fit()."""
        raise NotImplementedError

    def test_predict(self):
        """Test predict()."""
        raise NotImplementedError
