# test_interpolate.py
"""Tests for rom_operator_inference.interpolate."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi

from _common import _LSTSQ_REPORTS, _get_data


# Mixins (private) ============================================================
class TestInterpolatedMixin:
    """Test _interpolate._base._InterpolatedMixin."""
    pass


class TestInterpolatedInferredMixin:
    """Test _interpolate._inferred._InterpolatedInferredMixin."""
    pass


# Useable classes (public) ====================================================
class TestInterpolatedInferredDiscreteROM:
    """Test _interpolate._inferred.InterpolatedInferredDiscreteROM."""
    def test_fit(self):
        """Test
        _interpolate._inferred.InterpolatedInferredDiscreteROM.fit().
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

        # Try with non-scalar parameters.
        with pytest.raises(ValueError) as ex:
            model.fit(Vr, [np.array([1,1]), np.array([2,2])], Xs)
        assert ex.value.args[0] == "only scalar parameter values are supported"

        # Try with bad number of Xs.
        with pytest.raises(ValueError) as ex:
            model.fit(Vr, ps, [X1, X2, X2+1])
        assert ex.value.args[0] == \
            "num parameter samples != num state snapshot sets (2 != 3)"

        # Try with varying input sizes.
        model.modelform = "cAHB"
        with pytest.raises(ValueError) as ex:
            model.fit(Vr, ps, Xs, [U1, U2[:-1]])
        assert ex.value.args[0] == "control inputs not aligned"

        # Fit correctly with no inputs.
        model.modelform = "cAH"
        model.fit(Vr, ps, Xs)
        for attr in ["models_", "fs_"] + [s[:-1]+"s_" for s in _LSTSQ_REPORTS]:
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
        _interpolate._inferred.InterpolatedInferredDiscreteROM.predict().
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
        x0 = np.random.random(n)
        niters = 5
        U = np.ones((m,niters))

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
    """Test _interpolate._inferred.InterpolatedInferredContinuousROM."""
    def test_fit(self):
        """Test
        _interpolate._inferred.InterpolatedInferredContinuousROM.fit().
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
        with pytest.raises(ValueError) as ex:
            model.fit(Vr, [np.array([1,1]), np.array([2,2])], Xs, Xdots)
        assert ex.value.args[0] == "only scalar parameter values are supported"

        # Try with bad number of Xs.
        with pytest.raises(ValueError) as ex:
            model.fit(Vr, ps, [X1, X2, X2+1], Xdots)
        assert ex.value.args[0] == \
            "num parameter samples != num state snapshot sets (2 != 3)"

        # Try with bad number of Xdots.
        with pytest.raises(ValueError) as ex:
            model.fit(Vr, ps, Xs, Xdots + [Xdot1])
        assert ex.value.args[0] == \
            "num parameter samples != num velocity snapshot sets (2 != 3)"

        # Try with varying input sizes.
        model.modelform = "cAHB"
        with pytest.raises(ValueError) as ex:
            model.fit(Vr, ps, Xs, Xdots, [U1, U2[:-1]])
        assert ex.value.args[0] == "control inputs not aligned"

        # Fit correctly with no inputs.
        model.modelform = "cAH"
        model.fit(Vr, ps, Xs, Xdots)
        for attr in ["models_", "fs_"] + [s[:-1]+"s_" for s in _LSTSQ_REPORTS]:
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

    def test_predict(self):
        """Test
        _interpolate._inferred.InterpolatedInferredContinuousROM.predict().
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

        # Parameters for predicting.
        x0 = np.random.random(n)
        nt = 5
        t = np.linspace(0, .01*nt, nt)
        u = lambda t: np.ones(10)

        # Fit / predict with no inputs.
        model.fit(Vr, ps, Xs, Xdots)
        model.predict(1, x0, t)
        model.predict(1.5, x0, t)

        # Fit / predict with inputs.
        model.modelform = "cAHB"
        model.fit(Vr, ps, Xs, Xdots, Us)
        model.predict(1, x0, t, u)
        model.predict(1.5, x0, t, u)
