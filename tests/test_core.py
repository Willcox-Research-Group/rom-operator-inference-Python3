# test_core.py
"""Tests for rom_operator_inference.core.ReducedModel."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi


def test_init():
    # roi.ReducedModel.__init__() requires a single parameter.
    with pytest.raises(TypeError) as exc:
        roi.ReducedModel()
    assert exc.value.args[0] == \
        "__init__() missing 1 required positional argument: 'modelform'"

    with pytest.raises(TypeError) as exc:
        roi.ReducedModel("LQc", False, None)
    assert exc.value.args[0] == \
        "__init__() takes from 2 to 3 positional arguments but 4 were given"

    model = roi.ReducedModel("LQc")
    assert hasattr(model, "modelform")


@pytest.fixture
def set_up_fresh_model():
    return roi.ReducedModel("LQc", has_inputs=False)

@pytest.fixture
def set_up_trained_model(n=200, k=100, m=20, r=10):
    X = np.random.random((n,k))
    Xdot = np.zeros((n,k))
    Vr = la.svd(X)[0][:,:r]
    U = np.ones((m,k))
    return roi.ReducedModel("LQc", has_inputs=True).fit(X, Xdot, Vr, U)

def test_fit(set_up_fresh_model):
    model = set_up_fresh_model
    assert isinstance(model, roi.ReducedModel)

    # Get test data.
    n, k, m, r = 200, 100, 20, 10
    X = np.random.random((n,k))
    Xdot = np.zeros((n,k))
    Vr = la.svd(X)[0][:,:r]
    U = np.ones(m)

    # Try to use an invalid modelform.
    model.modelform = "LLL"
    with pytest.raises(ValueError) as exc:
        model.fit(X, Xdot, Vr)
    assert exc.value.args[0] == \
        f"invalid modelform 'LLL'. Options are {model._VALID_MODEL_FORMS}."
    model.modelform = "LQc"

    model.has_inputs = True
    with pytest.raises(ValueError) as exc:
        model.fit(X, Xdot, Vr)
    assert exc.value.args[0] == "argument 'U' required since has_inputs=True"

    model.has_inputs = False
    with pytest.raises(ValueError) as exc:
        model.fit(X, Xdot, Vr, U=U)
    assert exc.value.args[0] == "argument 'U' invalid since has_inputs=False"

    # Try to fit the model with misaligned X and Xdot.
    with pytest.raises(ValueError) as exc:
        model.fit(X, Xdot[:,1:-1], Vr)
    assert exc.value.args[0] == \
        f"X and Xdot different shapes ({(n,k)} != {(n,k-2)})"

    # Try to fit the model with misaligned X and Vr.
    with pytest.raises(ValueError) as exc:
        model.fit(X, Xdot, Vr[1:-1,:])
    assert exc.value.args[0] == \
        f"X and Vr not aligned, first dimension {n} != {n-2}"

def test_fit_outputs(set_up_trained_model):
    model = set_up_trained_model
    n, r, m = model.n, model.r, model.m

    assert model.A_.shape == (r,r)
    assert model.F_.shape == (r,r*(r+1)//2)
    assert model.H_.shape == (r,r**2)
    assert model.c_.shape == (r,)
    assert model.B_.shape == (r,m)

    assert hasattr(model, "residual_")
