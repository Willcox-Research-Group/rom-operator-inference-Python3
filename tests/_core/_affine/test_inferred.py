# _core/_affine/test_inferred.py
"""Tests for rom_operator_inference._core._affine._inferred."""

import pytest
import numpy as np
import scipy.linalg as la

import rom_operator_inference as opinf

from .test_base import TestAffineMixin
from .. import MODEL_KEYS, MODEL_FORMS, _get_data


# Affine inferred mixin (private) =============================================
class TestAffineInferredMixin:
    """Test _core._affine._inferred._AffineInferredMixin."""

    class Dummy(opinf._core._affine._inferred._AffineInferredMixin,
                opinf._core._base._BaseROM):
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
        model = self.Dummy("cA")
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
        for X, X_ in zip(Xs, Xs_):
            assert np.allclose(X_, Vr.T @ X)
        for rhs, rhs_ in zip(rhss, rhss_):
            assert np.allclose(rhs_, Vr.T @ rhs)
        assert Us_ is Us

        # Without basis and with a one-dimensional input.
        Xs_, rhss_, Us_ = model._process_fit_arguments(None, µs, affines,
                                                       Xs, rhss, Us1d)
        assert model.n is None
        assert model.r == n
        assert model.Vr is None
        assert model.m == 1
        for X, X_ in zip(Xs, Xs_):
            assert np.all(X_ == X)
        for rhs, rhs_ in zip(rhss, rhss_):
            assert np.all(rhs_ == rhs)
        for U, U_ in zip(Us1d, Us_):
            assert U_.shape == (1,k)
            assert np.all(U_.reshape(-1) == U)

        # With basis and no input.
        model = self.Dummy("cAHG")
        affs = {key:val for key,val in affines.items() if key != "B"}
        Xs_, rhss_, Us_ = model._process_fit_arguments(Vr, µs, affs,
                                                       Xs, rhss, None)
        assert model.n == n
        assert model.r == r
        assert model.Vr is Vr
        assert model.m == 0
        for X, X_ in zip(Xs, Xs_):
            assert np.allclose(X_, Vr.T @ X)
        for rhs, rhs_ in zip(rhss, rhss_):
            assert np.allclose(rhs_, Vr.T @ rhs)
        assert Us_ is None

    def test_assemble_data_matrix(self):
        """Test _core._affine._inferred.
                _AffineInferredMixin._assemble_data_matrix().
        """
        # Get test data.
        k, m, r, s = 200, 5, 10, 10
        X_, _, U = _get_data(r, k, m)
        θs = [lambda µ: np.sin(µ[0]), lambda µ: np.cos(µ[0]),
              lambda µ: np.sin(µ[1]), lambda µ: np.cos(µ[1])]
        µs = np.arange(1, 2*s+1).reshape((s,2))
        affines = {"c": θs[:2],
                   "A": θs,
                   "H": θs[:1],
                   "G": θs[:1],
                   "B": θs[:3]}
        Xs_, Us = [X_]*s, [U]*s
        Us1d = [U[0,:]]*s

        # Test with each possible modelform.
        for form in MODEL_FORMS:
            model = self.Dummy(form)
            model.r = r
            if 'B' in form:
                model.m = m
            D = model._assemble_data_matrix(µs, affines, Xs_, Us)
            d = opinf.lstsq.lstsq_size(form, r, m if 'B' in form else 0,
                                       affines)
            assert D.shape == (k*s,d)

            # Spot check.
            if form == "c":
                θc = np.array([[θ(µ) for θ in affines["c"]] for µ in µs])
                assert np.allclose(D, np.kron(θc, np.ones((k,1))))
            elif form == "H":
                θH = np.array([[θ(µ) for θ in affines["H"]] for µ in µs])
                assert np.allclose(D, np.kron(θH, opinf.utils.kron2c(X_).T))
            elif form == "G":
                θG = np.array([[θ(µ) for θ in affines["G"]] for µ in µs])
                assert np.allclose(D, np.kron(θG, opinf.utils.kron3c(X_).T))
            elif form == "AB":
                θA = np.array([[θ(µ) for θ in affines["A"]] for µ in µs])
                θB = np.array([[θ(µ) for θ in affines["B"]] for µ in µs])
                rr = r*len(affines["A"])
                assert np.allclose(D[:,:rr], np.kron(θA, X_.T))
                assert np.allclose(D[:,rr:], np.kron(θB, U.T))

        # Try with one-dimensional inputs as a 1D array.
        model = self.Dummy("B")
        model.m = 1
        D = model._assemble_data_matrix(µs, affines, Xs_, Us1d)
        d = opinf.lstsq.lstsq_size(model.modelform, r, model.m, affines)
        assert D.shape == (k*s, d)
        θB = np.array([[θ(µ) for θ in affines["B"]] for µ in µs])
        assert np.allclose(D, np.kron(θB, U[0].reshape((-1,1))))

    def test_extract_operators(self, k=200, m=5, r=10, s=10):
        """Test _core._affine._inferred.
                _AffineInferredMixin._extract_operators().
        """
        X_, _, U = _get_data(r, k, m)
        θs = [lambda µ: np.sin(µ[0]), lambda µ: µ[0] + µ[1],
              lambda µ: np.cos(µ[1]), lambda µ: np.sin(µ[1])]
        affines = {"c": θs[:2],
                   "A": θs,
                   "H": θs[:1],
                   "G": θs[:1],
                   "B": θs[:3]}
        shapes = {
                    "c_": (r,),
                    "A_": (r,r),
                    "H_": (r,r*(r+1)//2),
                    "G_": (r,r*(r+1)*(r+2)//6),
                    "B_": (r,m),
                 }

        for form in MODEL_FORMS:
            model = self.Dummy(form)
            model.r = r
            if 'B' in form:
                model.m = m
            d = opinf.lstsq.lstsq_size(form, r, model.m, affines)
            Ohat = np.random.random((r,d))
            model._extract_operators(affines, Ohat)
            for prefix in MODEL_KEYS:
                attr = prefix+'_'
                assert hasattr(model, attr)
                value = getattr(model, attr)
                if prefix in form:
                    assert isinstance(value,
                                      opinf._core._affine.AffineOperator)
                    assert value.shape == shapes[attr]
                else:
                    assert value is None

    def _test_fit(self, ModelClass):
        """Test _core._affine._inferred._AffineInferredMixin.fit(),
        parent method of
        _core._affine._inferred.AffineInferredDiscreteROM.fit() and
        _core._affine._inferred.AffineInferredContinuousROM.fit().
        """
        model = ModelClass("cAHB")
        is_continuous = issubclass(ModelClass,
                                   opinf._core._base._ContinuousROM)

        # Get test data.
        n, k, m, r, s = 50, 200, 5, 10, 10
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

        # Run fit() for each possible model form.
        for form in MODEL_FORMS:
            args[2] = {key:val for key,val in affines.items() if key in form}
            model = ModelClass(form)
            model.fit(*args, Us=Us if "B" in form else None)

            args[2] = {}    # Non-affine case.
            model.fit(*args, Us=Us if "B" in form else None)

        def _test_output_shapes(model):
            """Test shapes of output operators for modelform="cAHGB"."""
            assert model.n == n
            assert model.r == r
            assert model.m == m
            assert model.A_.shape == (r,r)
            assert model.H_.shape == (r,r*(r+1)//2)
            assert model.G_.shape == (r,r*(r+1)*(r+2)//6)
            assert model.c_.shape == (r,)
            assert model.B_.shape == (r,m)

        model = ModelClass("cAHGB")
        model.fit(*args, Us=Us)
        _test_output_shapes(model)

        # Fit the model with 1D inputs (1D array for B)
        model = ModelClass("cAHGB")
        model.fit(*args, Us=np.ones((s,k)))
        m = 1
        _test_output_shapes(model)


# Affine inferred models (public) =============================================
class TestAffineInferredDiscreteROM:
    """Test _core._affine._inferred.AffineInferredDiscreteROM."""
    def test_fit(self):
        """Test _core._affine._inferred.AffineInferredDiscreteROM.fit()."""
        TestAffineInferredMixin()._test_fit(opinf.AffineInferredDiscreteROM)

    def test_predict(self):
        """Test _core._affine._inferred.AffineInferredDiscreteROM.predict()."""
        TestAffineMixin()._test_predict(opinf.AffineInferredDiscreteROM)


class TestAffineInferredContinuousROM:
    """Test _core._affine._inferred.AffineInferredContinuousROM."""
    def test_fit(self):
        """Test _core._affine._inferred.AffineInferredContinuousROM.fit()."""
        TestAffineInferredMixin()._test_fit(opinf.AffineInferredContinuousROM)

    def test_predict(self):
        """Test _core._affine._inferred.
                AffineInferredContinuousROM.predict().
        """
        TestAffineMixin()._test_predict(opinf.AffineInferredContinuousROM)
