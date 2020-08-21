# test_init.py
"""Tests for rom_operator_inference.__init__.py."""

import os
import h5py
import pytest
import numpy as np

import rom_operator_inference as roi

from . import _get_operators


def test_select_model_class():
    """Test select_model_class()."""
    # Try with bad `time` argument.
    with pytest.raises(ValueError) as ex:
        roi.select_model_class("semidiscrete", "inferred", False)
    assert "input `time` must be one of " in ex.value.args[0]

    # Try with bad `rom_strategy` argument.
    with pytest.raises(ValueError) as ex:
        roi.select_model_class("discrete", "opinf", False)
    assert "input `rom_strategy` must be one of " in ex.value.args[0]

    # Try with bad `parametric` argument.
    with pytest.raises(ValueError) as ex:
        roi.select_model_class("discrete", "inferred", True)
    assert "input `parametric` must be one of " in ex.value.args[0]

    # Try with bad combination.
    with pytest.raises(NotImplementedError) as ex:
        roi.select_model_class("discrete", "intrusive", "interpolated")
    assert ex.value.args[0] == "model type invalid or not implemented"

    # Valid cases.
    assert roi.select_model_class("discrete", "inferred") is  \
                                        roi.InferredDiscreteROM
    assert roi.select_model_class("continuous", "inferred") is \
                                        roi.InferredContinuousROM
    assert roi.select_model_class("discrete", "intrusive") is \
                                        roi.IntrusiveDiscreteROM
    assert roi.select_model_class("continuous", "intrusive") is \
                                        roi.IntrusiveContinuousROM
    assert roi.select_model_class("discrete", "intrusive", "affine") is \
                                        roi.AffineIntrusiveDiscreteROM
    assert roi.select_model_class("continuous", "intrusive", "affine") is \
                                        roi.AffineIntrusiveContinuousROM
    assert roi.select_model_class("discrete", "inferred", "affine") is \
                                        roi.AffineInferredDiscreteROM
    assert roi.select_model_class("continuous", "inferred", "affine") is \
                                        roi.AffineInferredContinuousROM
    assert roi.select_model_class("discrete", "inferred", "interpolated") is \
                                        roi.InterpolatedInferredDiscreteROM
    assert roi.select_model_class("continuous", "inferred", "interpolated") is \
                                        roi.InterpolatedInferredContinuousROM


def test_load_model():
    """Test load_model()."""
    # Get test operators.
    n, m, r = 20, 2, 5
    Vr = np.random.random((n,r))
    c_, A_, H_, Hc_, G_, Gc_, B_ = _get_operators(n=r, m=m)

    # Try loading a file that does not exist.
    target = "loadmodeltest.h5"
    if os.path.isfile(target):                  # pragma: no cover
        os.remove(target)
    with pytest.raises(FileNotFoundError) as ex:
        model = roi.load_model(target)
    assert ex.value.args[0] == target

    # Make an empty HDF5 file to start with.
    with h5py.File(target, 'w') as f:
        pass

    with pytest.raises(ValueError) as ex:
        model = roi.load_model(target)
    assert ex.value.args[0] == "invalid save format (meta/ not found)"

    # Make a (mostly) compatible HDF5 file to start with.
    with h5py.File(target, 'a') as f:
        # Store metadata.
        meta = f.create_dataset("meta", shape=(0,))
        meta.attrs["modelclass"] = "InferredDiscreteROOM"
        meta.attrs["modelform"] = "cAB"

    with pytest.raises(ValueError) as ex:
        model = roi.load_model(target)
    assert ex.value.args[0] == "invalid save format (operators/ not found)"

    # Store the arrays.
    with h5py.File(target, 'a') as f:
        f.create_dataset("operators/c_", data=c_)
        f.create_dataset("operators/A_", data=A_)
        f.create_dataset("operators/B_", data=B_)

    # Try to load the file, which has a bad modelclass attribute.
    with pytest.raises(ValueError) as ex:
        model = roi.load_model(target)
    assert ex.value.args[0] == \
        "invalid modelclass 'InferredDiscreteROOM' (meta.attrs)"

    # Fix the file.
    with h5py.File(target, 'a') as f:
        f["meta"].attrs["modelclass"] = "InferredDiscreteROM"

    def _check_model(mdl):
        assert isinstance(mdl, roi.InferredDiscreteROM)
        for attr in ["modelform",
                     "n", "r", "m",
                     "c_", "A_", "Hc_", "Gc_", "B_", "Vr"]:
            assert hasattr(mdl, attr)
        assert mdl.modelform == "cAB"
        assert model.r == r
        assert model.m == m
        assert np.allclose(mdl.c_, c_)
        assert np.allclose(mdl.A_, A_)
        assert mdl.Hc_ is None
        assert mdl.Gc_ is None
        assert np.allclose(mdl.B_, B_)

    # Load the file correctly.
    model = roi.load_model(target)
    _check_model(model)
    assert model.Vr is None
    assert model.n is None

    # Add the basis and then load the file correctly.
    with h5py.File(target, 'a') as f:
        f.create_dataset("Vr", data=Vr)
    model = roi.load_model(target)
    _check_model(model)
    assert np.allclose(model.Vr, Vr)
    assert model.n == n

    # One additional test to cover other cases.
    with h5py.File(target, 'a') as f:
        f["meta"].attrs["modelclass"] = "InferredContinuousROM"
        f["meta"].attrs["modelform"] = "HG"
        f.create_dataset("operators/Hc_", data=Hc_)
        f.create_dataset("operators/Gc_", data=Gc_)

    model = roi.load_model(target)
    assert isinstance(model, roi.InferredContinuousROM)
    for attr in ["modelform",
                 "n", "r", "m",
                 "c_", "A_", "Hc_", "Gc_", "B_", "Vr"]:
        assert hasattr(model, attr)
    assert model.modelform == "HG"
    assert model.r == r
    assert model.m is None
    assert model.c_ is None
    assert model.A_ is None
    assert np.allclose(model.Hc_, Hc_)
    assert np.allclose(model.Gc_, Gc_)
    assert model.B_ is None
    assert np.allclose(model.Vr, Vr)
    assert model.n == n

    # Clean up.
    os.remove(target)


# TODO: move check_docs.py here or test_docs.py.
