# test_inferred.py
"""Tests for rom_operator_inference._core._inferred.py."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi

from . import _MODEL_FORMS, _get_data


# Mixins (private) ============================================================
class TestInferredMixin:
    """Test _core._inferred._InferredMixin."""
    def test_check_training_data_shapes(self):
        """Test _core._inferred._InferredMixin._check_training_data_shapes()."""
        # Get test data.
        n, k, m, r = 60, 50, 20, 10
        X, Xdot, U = _get_data(n, k, m)
        model = roi._core._inferred._InferredMixin()

        # Try to fit the model with misaligned X and Xdot.
        with pytest.raises(ValueError) as ex:
            model._check_training_data_shapes([X, Xdot[:,1:-1]])
        assert ex.value.args[0] == "data sets not aligned, dimension 1"

        # Try to fit the model with misaligned X and U.
        with pytest.raises(ValueError) as ex:
            model._check_training_data_shapes([X, Xdot, U[:,:-1]])
        assert ex.value.args[0] == "data sets not aligned, dimension 1"

        model._check_training_data_shapes([X, Xdot])
        model._check_training_data_shapes([X, Xdot, U])

    def _test_fit(self, ModelClass):
        """Test _core._inferred._InferredMixin.fit(), the parent method for
        _core._inferred.InferredDiscreteROM.fit() and
        _core._inferred.InferredContinuousROM.fit().
        """
        model = ModelClass("cAH")

        # Get test data.
        n, k, m, r = 60, 500, 20, 10
        X, Xdot, U = _get_data(n, k, m)
        Vr = la.svd(X)[0][:,:r]
        args = [Vr, X]
        if issubclass(ModelClass, roi._core._inferred._ContinuousROM):
            args.insert(1, Xdot)

        # Fit the model with each possible non-input modelform.
        for form in _MODEL_FORMS:
            if "B" not in form:
                model.modelform = form
                model.fit(*args)

        def _test_output_shapes(model):
            """Test shapes of output operators for modelform="cAHB"."""
            assert model.r == r
            assert model.m == m
            assert model.c_.shape == (r,)
            assert model.A_.shape == (r,r)
            assert model.Hc_.shape == (r,r*(r+1)//2)
            assert model.H_.shape == (r,r**2)
            assert model.Gc_.shape == (r,r*(r+1)*(r+2)//6)
            assert model.G_.shape == (r,r**3)
            assert model.B_.shape == (r,m)
            assert hasattr(model, "datacond_")
            assert hasattr(model, "dataregcond_")
            assert round(model.dataregcond_, 6) <= round(model.datacond_, 6)
            assert hasattr(model, "residual_")
            assert hasattr(model, "misfit_")
            assert round(model.misfit_, 6) <= round(model.residual_, 6)

        # Test with high-dimensional inputs.
        model.modelform = "cAHGB"
        model.fit(*args, U=U)
        _test_output_shapes(model)
        assert model.n == n
        assert np.allclose(model.Vr, Vr)

        # Test again with one-dimensional inputs.
        m = 1
        model.fit(*args, U=np.random.random(k))
        _test_output_shapes(model)
        assert model.n == n
        assert np.allclose(model.Vr, Vr)

        # Test again with Vr = None and projected data.
        args[0] = None
        for i in range(1,len(args)):
            args[i] = Vr.T @ args[i]
        model.fit(*args, U=np.random.random(k))
        _test_output_shapes(model)
        assert model.n is None
        assert model.Vr is None


# Useable classes (public) ====================================================
class TestInferredDiscreteROM:
    """Test _core._inferred.InferredDiscreteROM."""
    def test_fit(self):
        """Test _core._inferred.InferredDiscreteROM.fit()."""
        TestInferredMixin()._test_fit(roi.InferredDiscreteROM)


class TestInferredContinuousROM:
    """Test _core._inferred.InferredContinuousROM."""
    def test_fit(self):
        """Test _core._inferred.InferredContinuousROM.fit()."""
        TestInferredMixin()._test_fit(roi.InferredContinuousROM)
