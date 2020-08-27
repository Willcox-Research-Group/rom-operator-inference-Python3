# _core/_affine/test_inferred.py
"""Tests for rom_operator_inference._core._affine._inferred."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi

from .test_base import TestAffineMixin
from .. import _MODEL_FORMS, _get_data, _get_operators


# Affine inferred mixin (private) =============================================
class TestAffineInferredMixin:
    """Test _core._affine._inferred._AffineInferredMixin."""
    def _test_fit(self, ModelClass):
        """Test _core._affine._inferred._AffineInferredMixin.fit(),
        parent method of
        _core._affine._inferred.AffineInferredDiscreteROM.fit() and
        _core._affine._inferred.AffineInferredContinuousROM.fit().
        """
        model = ModelClass("cAHB")
        is_continuous = issubclass(ModelClass, roi._core._base._ContinuousROM)

        # Get test data.
        n, k, m, r, s = 50, 200, 5, 10, 3
        X, Xdot, U = _get_data(n, k, m)
        Vr = la.svd(X)[0][:,:r]
        θs = [lambda µ: µ, lambda µ: µ**2, lambda µ: µ**3, lambda µ: µ**4]
        µs = np.arange(1, s+1)
        affines = {"c": θs[:2],
                   "A": θs,
                   "H": θs[:1],
                   "G": θs[:1],
                   "B": θs[:3]}
        Xs, Xdots, Us = [X]*s, [Xdot]*s, [U]*s
        args = [Vr, µs, affines, Xs]
        if is_continuous:
            args.insert(3, Xdots)

        # Try with bad number of parameters.
        model.modelform = "cAHGB"
        with pytest.raises(ValueError) as ex:
            model.fit(Vr, µs[:-1], *args[2:], Us)
        assert ex.value.args[0] == \
            f"num parameter samples != num state snapshot training sets ({s-1} != {s})"

        if is_continuous:
            with pytest.raises(ValueError) as ex:
                model.fit(Vr, µs, affines, Xs, Xdots[:-1], Us)
            assert ex.value.args[0] == "num parameter samples != num rhs " \
                                       f"training sets ({s} != {s-1})"

        # Try with varying input sizes.
        if is_continuous:
            with pytest.raises(ValueError) as ex:
                model.fit(Vr, µs, affines, Xs, Xdots, [U, U[:-1], U])
            assert ex.value.args[0] == "control inputs not aligned"
        else:
            with pytest.raises(ValueError) as ex:
                model.fit(Vr, µs, affines, Xs, [U, U[:-1], U])
            assert ex.value.args[0] == "control inputs not aligned"

        for form in _MODEL_FORMS:
            args[2] = {key:val for key,val in affines.items() if key in form}
            model.modelform = form
            model.fit(*args, Us=Us if "B" in form else None)

            args[2] = {} # Non-affine case.
            model.fit(*args, Us=Us if "B" in form else None)

        def _test_output_shapes(model):
            """Test shapes of output operators for modelform="cAHB"."""
            assert model.n == n
            assert model.r == r
            assert model.m == m
            assert model.A_.shape == (r,r)
            assert model.Hc_.shape == (r,r*(r+1)//2)
            assert model.H_.shape == (r,r**2)
            assert model.c_.shape == (r,)
            assert model.B_.shape == (r,m)
            assert hasattr(model, "residual_")

        model.modelform = "cAHB"
        model.fit(*args, Us=Us)
        _test_output_shapes(model)

        # Fit the model with 1D inputs (1D array for B)
        model.modelform = "cAHB"
        model.fit(*args, Us=np.ones((s,k)))
        m = 1
        _test_output_shapes(model)


# Affine inferred models (public) =============================================
class TestAffineInferredDiscreteROM:
    """Test _core._affine._inferred.AffineInferredDiscreteROM."""
    def test_fit(self):
        """Test _core._affine._inferred.AffineInferredDiscreteROM.fit()."""
        TestAffineInferredMixin()._test_fit(roi.AffineInferredDiscreteROM)

    def test_predict(self):
        """Test _core._affine._inferred.AffineInferredDiscreteROM.predict()."""
        TestAffineMixin()._test_predict(roi.AffineInferredDiscreteROM)


class TestAffineInferredContinuousROM:
    """Test _core._affine._inferred.AffineInferredContinuousROM."""
    def test_fit(self):
        """Test _core._affine._inferred.AffineInferredContinuousROM.fit()."""
        TestAffineInferredMixin()._test_fit(roi.AffineInferredContinuousROM)

    def test_predict(self):
        """Test _core._affine._inferred.AffineInferredContinuousROM.predict()."""
        TestAffineMixin()._test_predict(roi.AffineInferredContinuousROM)
