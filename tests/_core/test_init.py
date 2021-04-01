# _core/test_init.py
"""Tests for rom_operator_inference._core.__init__.py."""

import os
import h5py
import pytest
import numpy as np

import rom_operator_inference as opinf

from . import _get_operators


def test_select_model_class():
    """Test select_model_class()."""
    # Try with bad `time` argument.
    with pytest.raises(ValueError) as ex:
        opinf.select_model_class("semidiscrete", "inferred", False)
    assert "input `time` must be one of " in ex.value.args[0]

    # Try with bad `rom_strategy` argument.
    with pytest.raises(ValueError) as ex:
        opinf.select_model_class("discrete", "opinf", False)
    assert "input `rom_strategy` must be one of " in ex.value.args[0]

    # Try with bad `parametric` argument.
    with pytest.raises(ValueError) as ex:
        opinf.select_model_class("discrete", "inferred", True)
    assert "input `parametric` must be one of " in ex.value.args[0]

    # Try with bad combination.
    with pytest.raises(NotImplementedError) as ex:
        opinf.select_model_class("discrete", "intrusive", "interpolated")
    assert ex.value.args[0] == "model type invalid or not implemented"

    # Valid cases.
    assert opinf.select_model_class("discrete", "inferred") \
        is opinf.InferredDiscreteROM
    assert opinf.select_model_class("continuous", "inferred") \
        is opinf.InferredContinuousROM
    assert opinf.select_model_class("discrete", "intrusive") \
        is opinf.IntrusiveDiscreteROM
    assert opinf.select_model_class("continuous", "intrusive") \
        is opinf.IntrusiveContinuousROM
    assert opinf.select_model_class("discrete", "intrusive", "affine") \
        is opinf.AffineIntrusiveDiscreteROM
    assert opinf.select_model_class("continuous", "intrusive", "affine") \
        is opinf.AffineIntrusiveContinuousROM
    assert opinf.select_model_class("discrete", "inferred", "affine") \
        is opinf.AffineInferredDiscreteROM
    assert opinf.select_model_class("continuous", "inferred", "affine") \
        is opinf.AffineInferredContinuousROM
    assert opinf.select_model_class("discrete", "inferred", "interpolated") \
        is opinf.InterpolatedInferredDiscreteROM
    assert opinf.select_model_class("continuous", "inferred", "interpolated") \
        is opinf.InterpolatedInferredContinuousROM


def test_load_model():
    """Test load_model()."""
    # Get test operators.
    n, m, r = 20, 2, 5
    Vr = np.random.random((n,r))
    c_, A_, H_, G_, B_ = _get_operators(n=r, m=m)

    # Try loading a file that does not exist.
    target = "loadmodeltest.h5"
    if os.path.isfile(target):                  # pragma: no cover
        os.remove(target)
    with pytest.raises(FileNotFoundError) as ex:
        rom = opinf.load_model(target)
    assert ex.value.args[0] == target

    # Make an empty HDF5 file to start with.
    with h5py.File(target, 'w') as f:
        pass

    with pytest.raises(ValueError) as ex:
        rom = opinf.load_model(target)
    assert ex.value.args[0] == "invalid save format (meta/ not found)"

    # Make a (mostly) compatible HDF5 file to start with.
    with h5py.File(target, 'a') as f:
        # Store metadata.
        meta = f.create_dataset("meta", shape=(0,))
        meta.attrs["modelclass"] = "InferredDiscreteROOM"
        meta.attrs["modelform"] = "cAB"

    with pytest.raises(ValueError) as ex:
        rom = opinf.load_model(target)
    assert ex.value.args[0] == "invalid save format (operators/ not found)"

    # Store the arrays.
    with h5py.File(target, 'a') as f:
        f.create_dataset("operators/c_", data=c_)
        f.create_dataset("operators/A_", data=A_)
        f.create_dataset("operators/B_", data=B_)

    # Try to load the file, which has a bad modelclass attribute.
    with pytest.raises(ValueError) as ex:
        rom = opinf.load_model(target)
    assert ex.value.args[0] == \
        "invalid modelclass 'InferredDiscreteROOM' (meta.attrs)"

    # Fix the file.
    with h5py.File(target, 'a') as f:
        f["meta"].attrs["modelclass"] = "InferredDiscreteROM"

    def _check_model(mdl):
        assert isinstance(mdl, opinf.InferredDiscreteROM)
        for attr in ["modelform",
                     "n", "r", "m",
                     "c_", "A_", "H_", "G_", "B_", "Vr"]:
            assert hasattr(mdl, attr)
        assert mdl.modelform == "cAB"
        assert mdl.r == r
        assert mdl.m == m
        assert np.allclose(mdl.c_, c_)
        assert np.allclose(mdl.A_, A_)
        assert mdl.H_ is None
        assert mdl.G_ is None
        assert np.allclose(mdl.B_, B_)

    # Load the file correctly.
    rom = opinf.load_model(target)
    _check_model(rom)
    assert rom.Vr is None
    assert rom.n is None

    # Add the basis and then load the file correctly.
    with h5py.File(target, 'a') as f:
        f.create_dataset("Vr", data=Vr)
    rom = opinf.load_model(target)
    _check_model(rom)
    assert np.allclose(rom.Vr, Vr)
    assert rom.n == n

    # One additional test to cover other cases.
    with h5py.File(target, 'a') as f:
        f["meta"].attrs["modelclass"] = "InferredContinuousROM"
        f["meta"].attrs["modelform"] = "HG"
        f.create_dataset("operators/H_", data=H_)
        f.create_dataset("operators/G_", data=G_)

    rom = opinf.load_model(target)
    assert isinstance(rom, opinf.InferredContinuousROM)
    for attr in ["modelform",
                 "n", "r", "m",
                 "c_", "A_", "H_", "G_", "B_", "Vr"]:
        assert hasattr(rom, attr)
    assert rom.modelform == "HG"
    assert rom.r == r
    assert rom.m == 0
    assert rom.c_ is None
    assert rom.A_ is None
    assert np.allclose(rom.H_, H_)
    assert np.allclose(rom.G_, G_)
    assert rom.B_ is None
    assert np.allclose(rom.Vr, Vr)
    assert rom.n == n

    # Clean up.
    os.remove(target)
