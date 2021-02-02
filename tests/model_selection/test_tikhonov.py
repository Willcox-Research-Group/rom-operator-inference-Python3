# model_selection/test_tikhonov.py
"""Tests for rom_operator_inference.model_selection._tikhonov.py."""

import pytest
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import rom_operator_inference as roi

from .._core import _get_data


def test_Lcurve():
    """Test model_selection._tikhonov.Lcurve()."""

    # Bad inputs: not an Operator Inference ROM
    badrom = roi.IntrusiveDiscreteROM("cAH")
    with pytest.raises(TypeError) as ex:
        roi.model_selection.Lcurve(badrom, 1, 1, 1)
    assert ex.value.args[0] == "rom must be Operator Inference ROM instance"

    # Correct usage.
    # rom, regs, fit_args, discrete=False
    X, Xdot, U = _get_data(n=60, k=30, m=2)
    Vr = la.svd(X)[0][:,:10]
    X_ = Vr.T @ X
    Xdot_ = Vr.T @ Xdot

    rom = roi.InferredContinuousROM("cAB")
    regs = [1e0, 1e1, 1e2, 1e3, 1e4]
    fit_args = [None, X_, Xdot_, U]
    plt.ion()
    roi.model_selection.Lcurve(rom, regs, fit_args, discrete=True)
    plt.close("all")
    roi.model_selection.Lcurve(rom, regs, fit_args, discrete=False)
    plt.close("all")
    plt.ioff()


def test_best_bounded_reg():
    """Test model_selection._tikhonov.best_bounded_reg()."""

    # Bad inputs: not an Operator Inference ROM
    badrom = roi.IntrusiveDiscreteROM("cAH")
    with pytest.raises(TypeError) as ex:
        roi.model_selection.best_bounded_reg(badrom, 1, 1, 1, 1, 1, 1)
    assert ex.value.args[0] == "rom must be Operator Inference ROM instance"

    # Bad inputs: not a callable error norm
    rom = roi.InferredDiscreteROM("cAB")
    with pytest.raises(TypeError) as ex:
        roi.model_selection.best_bounded_reg(rom, 1, 1, 1, 1, 1, 1)
    assert ex.value.args[0] == "errornorm must be callable"

    # Correct usage.
    X, Xdot, U = _get_data(n=60, k=30, m=2)
    Xdot = np.zeros_like(Xdot)
    Vr = la.svd(X)[0][:,:10]
    X_ = Vr.T @ X
    Xdot_ = Vr.T @ Xdot
    t = np.linspace(0, 1, 101)

    # Test 1: with basis, no inputs.
    fit_args = [Vr, X, Xdot]
    predict_args = [X[:,0], t]
    Xnorm = la.norm(X)
    def norm(Y):
        return la.norm(X - Y[:,:X_.shape[1]]) / Xnorm
    rom = roi.InferredContinuousROM("cA")
    B = 1.5 * np.abs(X_).max()
    bds = (1e-6, 1)
    best_reg, rom = roi.model_selection.best_bounded_reg(rom, B, norm, bds,
                                                         fit_args,
                                                         predict_args, True)
    assert isinstance(best_reg, float)
    assert best_reg > 0
    assert isinstance(rom, roi.InferredContinuousROM)
    rom._check_is_trained()

    # Test 2: no basis, with inputs.
    u = lambda t: np.random.random(size=U.shape[0])
    fit_args = [None, X_, Xdot_, U]
    predict_args = [X_[:,0], t, u]
    k = X_.shape[1]
    def norm(Y_):
        return roi.post.Lp_error(X_, Y_[:,:k], t[:k])[1]
    rom = roi.InferredContinuousROM("cAB")
    B = 1.5 * np.abs(X_).max()
    bds = (1e-6, 1)
    best_reg, rom = roi.model_selection.best_bounded_reg(rom, B, norm, bds,
                                                         fit_args,
                                                         predict_args, False)
    assert isinstance(best_reg, float)
    assert best_reg > 0
    assert isinstance(rom, roi.InferredContinuousROM)
    rom._check_is_trained()
