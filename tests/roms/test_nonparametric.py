# roms/test_nonparametric.py
"""Tests for roms._nonparametric.py."""

import pytest
import numpy as np

import opinf


module = opinf.roms


args = dict(
    model=opinf.models.ContinuousModel("A"),
    model2=opinf.models.DiscreteModel("AB"),
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
    k: v
    for k, v in args.items()
    if k in ("model", "lifter", "transformer", "basis", "ddt_estimator")
}


class TestROM:
    """Test roms.ROM."""

    ROM = module.ROM

    def test_init(self):
        """Test __init__() and properties."""

        # Model error.
        with pytest.raises(TypeError) as ex:
            self.ROM(10)
        assert ex.value.args[0] == "invalid model type"

        # Warnings for other arguments.
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.ROM(
                args["model"],
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
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.ROM(args["model2"], ddt_estimator=args["ddt_estimator"])
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            "ddt_estimator ignored for discrete models"
        )

        # Correct usage.
        rom = self.ROM(
            args["model"],
            lifter=args["lifter"],
            ddt_estimator=args["ddt_estimator"],
        )
        assert rom.iscontinuous
        assert rom.transformer is None
        assert rom.basis is None

        rom = self.ROM(
            args["model2"],
            transformer=args["multi_transformer"],
            basis=args["multi_basis"],
        )
        assert rom.lifter is None
        assert rom.ddt_estimator is None
        assert not rom.iscontinuous

    def test_str(self):
        """Test __str__() and __repr__()."""
        print(repr(self.ROM(**basics)))

    def test_econde(self):
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
