# _core/_interpolate/test_inferred.py
"""Tests for rom_operator_inference._core._interpolate._inferred."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as opinf

from .. import _get_data


# Interpolated inferred mixin (private) =======================================
class TestInterpolatedInferredMixin:
    """Test _core._interpolate._inferred._InterpolatedInferredMixin."""
    pass


# Interpolated inferred models (public) =======================================
class TestInterpolatedInferredDiscreteROM:
    """Test _core._interpolate._inferred.InterpolatedInferredDiscreteROM."""
    def test_fit(self, n=20, m=4, k=500, r=3):
        """Test
        _core._interpolate._inferred.InterpolatedInferredDiscreteROM.fit().
        """
        rom = opinf.InterpolatedInferredDiscreteROM("cAH")

        # Get data for fitting.
        X1, _, U1 = _get_data(n, k, m)
        X2, U2 = X1+1, U1+1
        Xs = [X1, X2]
        Us = [U1, U2]
        ps = [1, 2]
        Vr = la.svd(np.hstack(Xs))[0][:,:r]

        # Try with non-scalar parameters.
        # with pytest.raises(ValueError) as ex:
        #     rom.fit(Vr, [np.array([1,1]), np.array([2,2])], Xs)
        # assert ex.value.args[0] == "only scalar parameter values
        #     are supported"

        # Try with bad number of Xs.
        with pytest.raises(ValueError) as ex:
            rom.fit(Vr, ps, [X1, X2, X2+1])
        assert ex.value.args[0] == "num parameter samples != num state " \
                                   "snapshot training sets (2 != 3)"

        # Fit correctly with no inputs.
        rom.modelform = "cAH"
        rom.fit(Vr, ps, Xs)
        for attr in ["models_", "fs_"]:
            assert hasattr(rom, attr)
            assert len(getattr(rom, attr)) == len(rom.models_)

        # Fit correctly with inputs.
        rom.modelform = "cAHGB"
        rom.fit(Vr, ps, Xs, Us)

        assert len(rom) == len(ps)

        # Test again with Vr = None and projected inputs.
        Xs_ = [Vr.T @ X for X in Xs]
        rom.fit(None, ps, Xs_, Us)
        assert len(rom) == len(ps)
        assert rom.Vr is None
        assert rom.n is None

    def test_predict(self):
        """Test
        _core._interpolate._inferred.InterpolatedInferredDiscreteROM.predict().
        """
        rom = opinf.InterpolatedInferredDiscreteROM("cAH")

        # Get data for fitting.
        n, m, k, r = 50, 10, 100, 5
        X1, _, U1 = _get_data(n, k, m)
        X2, U2 = X1+1, U1+1
        Xs = [X1, X2]
        Us = [U1, U2]
        ps = [1, 2]
        Vr = la.svd(np.hstack(Xs))[0][:,:r]

        # Parameters for predicting.
        x0 = np.zeros(n)
        niters = 5
        U = np.zeros((m,niters))

        # Fit / predict with no inputs.
        rom.fit(Vr, ps, Xs)
        rom.predict(1, x0, niters)
        rom.predict(1.5, x0, niters)

        # Fit / predict with inputs.
        rom.modelform = "cAHB"
        rom.fit(Vr, ps, Xs, Us)
        rom.predict(1, x0, niters, U)
        rom.predict(1.5, x0, niters, U)


class TestInterpolatedInferredContinuousROM:
    """Test _core._interpolate._inferred.InterpolatedInferredContinuousROM."""
    def test_fit(self):
        """Test
        _core._interpolate._inferred.InterpolatedInferredContinuousROM.fit().
        """
        rom = opinf.InterpolatedInferredContinuousROM("cAH")

        # Get data for fitting.
        n, m, k, r = 50, 10, 100, 5
        X1, Xdot1, U1 = _get_data(n, k, m)
        X2, Xdot2, U2 = X1+1, Xdot1.copy(), U1+1
        Xs = [X1, X2]
        Xdots = [Xdot1, Xdot2]
        Us = [U1, U2]
        ps = [1, 2]
        Vr = la.svd(np.hstack(Xs))[0][:,:r]

        # Try with non-scalar parameters.
        # with pytest.raises(ValueError) as ex:
        #     rom.fit(Vr, [np.array([1,1]), np.array([2,2])], Xs, Xdots)
        # assert ex.value.args[0] == "only scalar parameter values
        #     are supported"

        # Try with bad number of Xs.
        with pytest.raises(ValueError) as ex:
            rom.fit(Vr, ps, [X1, X2, X2+1], Xdots)
        assert ex.value.args[0] == "num parameter samples != num state " \
                                   "snapshot training sets (2 != 3)"

        # Try with bad number of Xdots.
        with pytest.raises(ValueError) as ex:
            rom.fit(Vr, ps, Xs, Xdots + [Xdot1])
        assert ex.value.args[0] == "num parameter samples != num time " \
                                   "derivative training sets (2 != 3)"

        # Fit correctly with no inputs.
        rom.modelform = "cAH"
        rom.fit(Vr, ps, Xs, Xdots)
        for attr in ["models_", "fs_"]:
            assert hasattr(rom, attr)
            assert len(getattr(rom, attr)) == len(rom.models_)

        # Fit correctly with inputs.
        rom.modelform = "cAHB"
        rom.fit(Vr, ps, Xs, Xdots, Us)
        assert len(rom) == len(ps)

        # Test again with Vr = None and projected inputs.
        Xs_ = [Vr.T @ X for X in Xs]
        Xdots_ = [Vr.T @ Xdot for Xdot in Xdots]
        rom.fit(None, ps, Xs_, Xdots_, Us)
        assert len(rom) == len(ps)
        assert rom.Vr is None
        assert rom.n is None

    def test_predict(self, n=50, m=10, k=100, r=3):
        """Test
        _core._interpolate._inferred.InterpolatedInferredContinuousROM.predict().
        """
        # Get data for fitting.
        X1, Xdot1, U1 = _get_data(n, k, m)
        X2, Xdot2, U2 = X1+1, Xdot1.copy(), U1+1
        Xs = [X1, X2]
        Xdots = [Xdot1, Xdot2]
        Us = [U1, U2]
        ps = [1, 2]
        Vr = la.svd(np.hstack(Xs))[0][:,:r]

        # Parameters for predicting.
        x0 = np.zeros(n)
        nt = 5
        t = np.linspace(0, .01*nt, nt)
        def u(t):
            return np.zeros(m)

        # Fit / predict with no inputs.
        rom = opinf.InterpolatedInferredContinuousROM("AH")
        rom.fit(Vr, ps, Xs, Xdots)
        rom.predict(1, x0, t)
        rom.predict(1.5, x0, t)

        # Fit / predict with inputs.
        rom = opinf.InterpolatedInferredContinuousROM("AHB")
        rom.fit(Vr, ps, Xs, Xdots, Us)
        rom.predict(1, x0, t, u)
        rom.predict(1.5, x0, t, u)
