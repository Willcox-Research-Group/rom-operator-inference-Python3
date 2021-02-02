# _core/_interpolate/test_inferred.py
"""Tests for rom_operator_inference._core._interpolate._inferred."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi

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
        model = roi.InterpolatedInferredDiscreteROM("cAH")

        # Get data for fitting.
        X1, _, U1 = _get_data(n, k, m)
        X2, U2 = X1+1, U1+1
        Xs = [X1, X2]
        Us = [U1, U2]
        ps = [1, 2]
        Vr = la.svd(np.hstack(Xs))[0][:,:r]

        # Try with non-scalar parameters.
        # with pytest.raises(ValueError) as ex:
        #     model.fit(Vr, [np.array([1,1]), np.array([2,2])], Xs)
        # assert ex.value.args[0] == "only scalar parameter values are supported"

        # Try with bad number of Xs.
        with pytest.raises(ValueError) as ex:
            model.fit(Vr, ps, [X1, X2, X2+1])
        assert ex.value.args[0] == "num parameter samples != num state " \
                                   "snapshot training sets (2 != 3)"

        # Fit correctly with no inputs.
        model.modelform = "cAH"
        model.fit(Vr, ps, Xs)
        for attr in ["models_", "fs_"]:
            assert hasattr(model, attr)
            assert len(getattr(model, attr)) == len(model.models_)

        # Fit correctly with inputs.
        model.modelform = "cAHGB"
        model.fit(Vr, ps, Xs, Us)

        assert len(model) == len(ps)

        # Test again with Vr = None and projected inputs.
        Xs_ = [Vr.T @ X for X in Xs]
        model.fit(None, ps, Xs_, Us)
        assert len(model) == len(ps)
        assert model.Vr is None
        assert model.n is None

    def test_predict(self):
        """Test
        _core._interpolate._inferred.InterpolatedInferredDiscreteROM.predict().
        """
        model = roi.InterpolatedInferredDiscreteROM("cAH")

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
        model.fit(Vr, ps, Xs)
        model.predict(1, x0, niters)
        model.predict(1.5, x0, niters)

        # Fit / predict with inputs.
        model.modelform = "cAHB"
        model.fit(Vr, ps, Xs, Us)
        model.predict(1, x0, niters, U)
        model.predict(1.5, x0, niters, U)


class TestInterpolatedInferredContinuousROM:
    """Test _core._interpolate._inferred.InterpolatedInferredContinuousROM."""
    def test_fit(self):
        """Test
        _core._interpolate._inferred.InterpolatedInferredContinuousROM.fit().
        """
        model = roi.InterpolatedInferredContinuousROM("cAH")

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
        #     model.fit(Vr, [np.array([1,1]), np.array([2,2])], Xs, Xdots)
        # assert ex.value.args[0] == "only scalar parameter values are supported"

        # Try with bad number of Xs.
        with pytest.raises(ValueError) as ex:
            model.fit(Vr, ps, [X1, X2, X2+1], Xdots)
        assert ex.value.args[0] == "num parameter samples != num state " \
                                   "snapshot training sets (2 != 3)"

        # Try with bad number of Xdots.
        with pytest.raises(ValueError) as ex:
            model.fit(Vr, ps, Xs, Xdots + [Xdot1])
        assert ex.value.args[0] == "num parameter samples != num time " \
                                   "derivative training sets (2 != 3)"

        # Fit correctly with no inputs.
        model.modelform = "cAH"
        model.fit(Vr, ps, Xs, Xdots)
        for attr in ["models_", "fs_"]:
            assert hasattr(model, attr)
            assert len(getattr(model, attr)) == len(model.models_)

        # Fit correctly with inputs.
        model.modelform = "cAHB"
        model.fit(Vr, ps, Xs, Xdots, Us)
        assert len(model) == len(ps)

        # Test again with Vr = None and projected inputs.
        Xs_ = [Vr.T @ X for X in Xs]
        Xdots_ = [Vr.T @ Xdot for Xdot in Xdots]
        model.fit(None, ps, Xs_, Xdots_, Us)
        assert len(model) == len(ps)
        assert model.Vr is None
        assert model.n is None

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
        u = lambda t: np.zeros(m)

        # Fit / predict with no inputs.
        model = roi.InterpolatedInferredContinuousROM("AH")
        model.fit(Vr, ps, Xs, Xdots)
        model.predict(1, x0, t)
        model.predict(1.5, x0, t)

        # Fit / predict with inputs.
        model = roi.InterpolatedInferredContinuousROM("AHB")
        model.fit(Vr, ps, Xs, Xdots, Us)
        model.predict(1, x0, t, u)
        model.predict(1.5, x0, t, u)
