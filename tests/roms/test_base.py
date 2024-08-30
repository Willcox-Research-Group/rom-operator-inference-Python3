# roms/test_base.py
"""Tests for roms._base."""

import abc
import pytest
import numpy as np

import opinf
from opinf.models import _utils as modutils


args = dict(
    lifter=opinf.lift.QuadraticLifter(),
    transformer=opinf.pre.ShiftScaleTransformer(centering=True),
    transformer2=opinf.pre.ShiftScaleTransformer(scaling="standard"),
    basis=opinf.basis.PODBasis(num_vectors=3),
    basis2=opinf.basis.PODBasis(num_vectors=4),
    ddt_estimator=opinf.ddt.UniformFiniteDifferencer(np.linspace(0, 1, 100)),
)
args["multi_transformer"] = opinf.pre.TransformerMulti(
    [args["transformer"], args["transformer2"]]
)
args["multi_basis"] = opinf.basis.BasisMulti([args["basis"], args["basis2"]])
basics = {
    key: val
    for key, val in args.items()
    if key in ("lifter", "transformer", "basis")
}


class _TestBaseROM(abc.ABC):
    """Test opinf.roms._base._BaseROM."""

    ROM = NotImplemented
    ModelClasses = NotImplemented

    @abc.abstractmethod
    def _get_models(self):
        """Return a list of valid model instantiations."""
        pass

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
                        ddt_estimator=args["ddt_estimator"],
                    )
                assert len(wn) == 1
                assert wn[0].message.args[0] == (
                    "ddt_estimator ignored for discrete models"
                )
                assert rom.ddt_estimator is None
                assert not rom._iscontinuous

            # Correct usage.
            rom = self.ROM(
                model,
                lifter=args["lifter"],
                ddt_estimator=args["ddt_estimator"],
            )
            assert rom.lifter is args["lifter"]
            assert rom.transformer is None
            assert rom.basis is None
            assert rom.ddt_estimator is args["ddt_estimator"]

            rom = self.ROM(
                args["model2"],
                transformer=args["multi_transformer"],
                basis=args["multi_basis"],
            )
            assert rom.lifter is None
            assert rom.transformer is args["multi_transformer"]
            assert rom.basis is args["multi_basis"]
            assert rom.ddt_estimator is None

    def test_str(self):
        """Lightly test __str__() and __repr__()."""
        for model in self._get_models():
            repr(self.ROM(model, **basics))

    def test_encode(self):
        """Test encode()."""
        raise NotImplementedError

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
