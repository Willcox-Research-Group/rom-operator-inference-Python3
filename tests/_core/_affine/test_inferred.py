# _core/_affine/test_inferred.py
"""Tests for rom_operator_inference._core._affine._inferred."""

import pytest
import numpy as np
import scipy.linalg as la

import rom_operator_inference as roi

from .test_base import TestAffineMixin
from .. import MODEL_FORMS, _get_data, _get_operators


# Affine inferred mixin (private) =============================================
class TestAffineInferredMixin:
    """Test _core._affine._inferred._AffineInferredMixin."""

    class Dummy(roi._core._base._BaseROM,
                roi._core._affine._inferred._AffineInferredMixin):
        def __init__(self, modelform):
            self.modelform = modelform

    def test_check_affines(self):
        """Test _core._affine._inferred._AffineInferredMixin._check_affines().
        """
        # Try affines with linearly dependent parameter samples.
        model = self.Dummy("c")
        affines = {"c": [lambda µ: µ[0], lambda µ: µ[1]]}
        with pytest.warns(la.LinAlgWarning) as wn:
            model._check_affines(affines, [[1, 1], [2, 2]])
        assert len(wn) == 1
        assert wn[0].message.args[0] == \
            "rank-deficient data matrix due to 'c' affine structure " \
            "and parameter samples"

        affines = {"c": [lambda µ: µ[0], lambda µ: µ[1], lambda µ: µ[2]],
                   "A": [lambda µ: µ[0], lambda µ: µ[2]**2]}
        model.modelform = "cA"
        with pytest.warns(la.LinAlgWarning) as wn:
            model._check_affines(affines, [[1, 3, 1], [1, 2, 1]])
        assert len(wn) == 2
        assert wn[0].message.args[0] == \
            "rank-deficient data matrix due to 'c' affine structure " \
            "and parameter samples"
        assert wn[1].message.args[0] == \
            "rank-deficient data matrix due to 'A' affine structure " \
            "and parameter samples"

    def test_process_fit_arguments(self):
        """Test _core._affine._inferred.
                _AffineInferredMixin._process_fit_arguments().
        """
        # Get test data.
        n, k, m, r = 60, 500, 20, 10
        s = 5
        X, rhs, U = _get_data(n, k, m)
        Xs, rhss, Us = [X]*s, [rhs]*s, [U]*s
        Us1d = [U[0,:]]*s
        Vr = np.random.random((n,r))

        # Get test parameters and affine functions.
        θs = [lambda µ: µ, lambda µ: µ**2, lambda µ: µ**3, lambda µ: µ**4]
        µs = list(range(1, s+1))
        affines = {"c": θs[:2],
                   "A": θs,
                   "H": θs[:1],
                   "G": θs[:1],
                   "B": θs[:3]}

        # Try with mismatched number of parameters and data sets.
        model = self.Dummy("cAHGB")
        with pytest.raises(ValueError) as ex:
            model._process_fit_arguments(Vr, µs, affines, Xs[:-1], rhss, Us)
        assert ex.value.args[0] == \
            "num parameter samples != num state snapshot training sets " \
            f"({s} != {s-1})"

        with pytest.raises(ValueError) as ex:
            model._process_fit_arguments(Vr, µs, affines, Xs, rhss[:-2], Us)
        assert ex.value.args[0] == \
            f"num parameter samples != num rhs training sets ({s} != {s-2})"

        with pytest.raises(ValueError) as ex:
            model._process_fit_arguments(Vr, µs, affines, Xs, rhss, Us[:-3])
        assert ex.value.args[0] == \
            f"num parameter samples != num input training sets ({s} != {s-3})"

        # With basis and input.
        Xs_, rhss_, Us_ = model._process_fit_arguments(Vr, µs, affines,
                                                       Xs, rhss, Us)
        assert model.n == n
        assert model.r == r
        assert model.Vr is Vr
        assert model.m == m
        # assert np.allclose(X_, Vr.T @ X)
        # assert np.allclose(rhs_, Vr.T @ rhs)
        # assert U_ is U

        # Without basis and with a one-dimensional input.
        Xs_, rhss_, Us_ = model._process_fit_arguments(None, µs, affines,
                                                       Xs, rhss, Us1d)
        assert model.n is None
        assert model.r == n
        assert model.Vr is None
        assert model.m == 1
        # assert Xs_ is Xs
        # assert rhss_ is rhss
        # assert U_.shape == (1,k)
        # assert np.allclose(U_.reshape(-1), U)

        # With basis and no input.
        model.modelform = "cAHG"
        affs = {key:val for key,val in affines.items() if key != "B"}
        Xs_, rhss_, Us_ = model._process_fit_arguments(Vr, µs, affs,
                                                       Xs, rhss, None)
        assert model.n == n
        assert model.r == r
        assert model.Vr is Vr
        assert model.m is None
        # assert np.allclose(X_, Vr.T @ X)
        # assert np.allclose(rhs_, Vr.T @ rhs)
        # assert U_ is None

    def test_construct_data_matrix(self):
        """Test _core._affine._inferred.
                _AffineInferredMixin._construct_data_matrix().
        """
        pass

    def test_extract_operators(self):
        """Test _core._affine._inferred.
                _AffineInferredMixin._extract_operators().
        """
        pass

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
            assert ex.value.args[0] == \
                f"invalid training input (Us[1].shape[0] != m={m})"
        else:
            with pytest.raises(ValueError) as ex:
                model.fit(Vr, µs, affines, Xs, [U, U, U[:-1]])
            assert ex.value.args[0] == \
                f"invalid training input (Us[2].shape[0] != m={m})"

        for form in MODEL_FORMS:
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
