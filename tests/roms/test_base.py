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
            if modutils.is_continuous(model):
                assert wn[3].message.args[0] == (
                    "ddt_estimator not derived from "
                    "DerivativeEstimatorTemplate, "
                    "unexpected behavior may occur"
                )
            else:
                assert wn[3].message.args[0] == (
                    "ddt_estimator ignored for discrete models"
                )

            # Given ddt_estimator with non-continuous model.
            if modutils.is_discrete(model):
                with pytest.warns(opinf.errors.OpInfWarning) as wn:
                    rom = self.ROM(
                        model,
                        ddt_estimator=self._get("ddt_estimator"),
                    )
                assert len(wn) == 1
                assert wn[0].message.args[0] == (
                    "ddt_estimator ignored for discrete models"
                )
                assert rom.ddt_estimator is None
                assert not rom._iscontinuous

            # Correct usage.
            if modutils.is_continuous(model):
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

    def test_decode(self, n=22, k=18):
        """Test decode()."""

        def _check(arr, shape):
            assert isinstance(arr, np.ndarray)
            assert arr.shape == shape

        for model in self._get_models():
            # Lifter only.
            rom = self.ROM(model, lifter=self._get("lifter"))
            states = np.random.random((2 * n, k))
            _check(rom.decode(states), (n, k))
            _check(rom.decode(states[:, 0]), (n,))

            # Transformer only.
            rom = self.ROM(model, transformer=self._get("transformer"))
            with pytest.raises(AttributeError) as ex:
                rom.decode(states)
            assert ex.value.args[0] == (
                "transformer not trained (call fit() or fit_transform())"
            )
            states = np.random.random((n, k))
            states_ = rom.encode(states, fit_transformer=True)
            out = rom.decode(states_)
            _check(out, (n, k))
            assert np.allclose(out, states)
            out = rom.decode(states_[:, 0])
            _check(out, (n,))
            assert np.allclose(out, states[:, 0])

            # Basis only.
            rom = self.ROM(model, basis=self._get("basis"))
            with pytest.raises(AttributeError) as ex:
                rom.decode(states)
            assert ex.value.args[0] == "basis entries not initialized"
            states_ = rom.encode(states, fit_basis=True)
            _check(rom.decode(states_), (n, k))
            _check(rom.decode(states_[:, 0]), (n,))

            # Lifter, transformer, and basis.
            a1, a2, a3 = self._get("lifter", "transformer", "basis")
            rom = self.ROM(model, lifter=a1, transformer=a2, basis=a3)
            states_ = rom.encode(states, fit_transformer=True, fit_basis=True)
            out1 = rom.decode(states_)
            _check(out1, (n, k))
            out2 = rom.decode(states_[:, 0])
            _check(out2, (n,))
            assert np.allclose(out2, out1[:, 0])

            # With the locs argument.
            a2, a3 = self._get("transformer", "basis")
            rom = self.ROM(model, transformer=a2, basis=a3)
            states_ = rom.encode(states, fit_transformer=True, fit_basis=True)
            out1 = rom.decode(states_)
            locs = np.sort(np.random.choice(n, n // 3))
            out2 = rom.decode(states_, locs=locs)
            _check(out2, (n // 3, k))
            assert np.allclose(out2, out1[locs])

    def test_project(self, n=30, k=19):
        """Test project()."""
        states = np.random.random((n, k))

        def _check(rom, preserved=False):
            rom.encode(states, fit_transformer=True, fit_basis=True)
            out = rom.project(states)
            assert isinstance(out, np.ndarray)
            assert out.shape == (n, k)
            if preserved:
                assert np.allclose(out, states)
            out0 = rom.project(states[:, 0])
            assert isinstance(out0, np.ndarray)
            assert out0.shape == (n,)
            assert np.allclose(out0, out[:, 0])

        for model in self._get_models():
            # Lifter only.
            _check(self.ROM(model, lifter=self._get("lifter")), preserved=True)

            # Transformer only.
            rom = self.ROM(model, transformer=self._get("transformer"))
            with pytest.raises(AttributeError) as ex:
                rom.project(states)
            assert ex.value.args[0] == (
                "transformer not trained (call fit() or fit_transform())"
            )
            _check(rom, preserved=True)

            # Basis only.
            rom = self.ROM(model, basis=self._get("basis"))
            with pytest.raises(AttributeError) as ex:
                rom.project(states)
            assert ex.value.args[0] == "basis entries not initialized"
            _check(rom, preserved=False)

            # Lifter, transformer, and basis.
            a1, a2, a3 = self._get("lifter", "transformer", "basis")
            _check(self.ROM(model, lifter=a1, transformer=a2, basis=a3))
