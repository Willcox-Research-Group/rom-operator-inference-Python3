# test_affine.py
"""Tests for rom_operator_inference.affine."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi

from _common import _MODEL_FORMS, _get_data, _get_operators


class TestAffineOperator:
    """Test _affine._base.AffineOperator."""
    @staticmethod
    def _set_up_affine_attributes(n=5):
        fs = [np.sin, np.cos, np.exp]
        As = list(np.random.random((3,n,n)))
        return fs, As

    def test_init(self):
        """Test _affine._base.AffineOperator.__init__()."""
        fs, As = self._set_up_affine_attributes()

        # Try with different number of functions and matrices.
        with pytest.raises(ValueError) as ex:
            roi.AffineOperator(fs, As[:-1])
        assert ex.value.args[0] == "expected 3 matrices, got 2"

        # Try with matrices of different shapes.
        with pytest.raises(ValueError) as ex:
            roi.AffineOperator(fs, As[:-1] + [np.random.random((4,4))])
        assert ex.value.args[0] == \
            "affine operator matrix shapes do not match ((4, 4) != (5, 5))"

        # Correct usage.
        affop = roi.AffineOperator(fs, As)
        affop = roi.AffineOperator(fs)
        affop.matrices = As

    def test_validate_coeffs(self):
        """Test _affine._base.AffineOperator.validate_coeffs()."""
        fs, As = self._set_up_affine_attributes()

        # Try with non-callables.
        affop = roi.AffineOperator(As)
        with pytest.raises(ValueError) as ex:
            affop.validate_coeffs(10)
        assert ex.value.args[0] == \
            "coefficients of affine operator must be callable functions"

        # Try with vector-valued functions.
        f1 = lambda t: np.array([t, t**2])
        affop = roi.AffineOperator([f1, f1])
        with pytest.raises(ValueError) as ex:
            affop.validate_coeffs(10)
        assert ex.value.args[0] == \
            "coefficient functions of affine operator must return a scalar"

        # Correct usage.
        affop = roi.AffineOperator(fs, As)
        affop.validate_coeffs(0)

    def test_call(self):
        """Test _affine._base.AffineOperator.__call__()."""
        fs, As = self._set_up_affine_attributes()

        # Try without matrices set.
        affop = roi.AffineOperator(fs)
        with pytest.raises(RuntimeError) as ex:
            affop(10)
        assert ex.value.args[0] == "component matrices not initialized!"

        # Correct usage.
        affop.matrices = As
        Ap = affop(10)
        assert Ap.shape == (5,5)
        assert np.allclose(Ap, np.sin(10)*As[0] + \
                               np.cos(10)*As[1] + np.exp(10)*As[2])

    def test_eq(self):
        """Test _affine._base.AffineOperator.__eq__()."""
        fs, As = self._set_up_affine_attributes()
        affop1 = roi.AffineOperator(fs[:-1])
        affop2 = roi.AffineOperator(fs, As)

        assert affop1 != 1
        assert affop1 != affop2
        affop1 = roi.AffineOperator(fs)
        assert affop1 != affop2
        affop1.matrices = As
        assert affop1 == affop2


# Mixins (private) ============================================================
class TestAffineMixin:
    """Test _affine._base._AffineMixin."""
    def test_check_affines(self):
        """Test _affine._base._AffineMixin._check_affines()."""
        model = roi._affine._base._AffineMixin()
        model.modelform = "cAHB"
        v = [lambda s: 0, lambda s: 0]

        # Try with surplus affine keys.
        with pytest.raises(KeyError) as ex:
            model._check_affines({'CC':v, "c":v, "A":v, "H":v, "B":v}, 0)
        assert ex.value.args[0] == "invalid affine key 'CC'"

        with pytest.raises(KeyError) as ex:
            model._check_affines({"c":v, "A":v, "H":v, "B":v,
                                    'CC':v, 'LL':v}, 0)
        assert ex.value.args[0] == "invalid affine keys 'CC', 'LL'"

        # Correct usage.
        model._check_affines({"c":v, "H":v}, 0)     # OK to be missing some.
        model._check_affines({"c":v, "A":v, "H":v, "B":v}, 0)

    def _test_predict(self, ModelClass):
        """Test predict() methods for Affine classes:
        * _affine._inferred.AffineInferredDiscreteROM.predict()
        * _affine._inferred.AffineInferredContinuousROM.predict()
        * _affine._intrusive.AffineIntrusiveDiscreteROM.predict()
        * _affine._intrusive.AffineIntrusiveContinuousROM.predict()
        """
        model = ModelClass("cAHG")

        # Get test data.
        n, k, m, r = 60, 50, 20, 10
        X = _get_data(n, k, m)[0]
        Vr = la.svd(X)[0][:,:r]

        # Get test operators.
        ident = lambda a: a
        c, A, H, Hc, G, Gc, B = _get_operators(r, m)
        model.Vr = Vr
        model.c_ = roi.AffineOperator([ident, ident], [c,c])
        model.A_ = roi.AffineOperator([ident, ident, ident], [A,A,A])
        model.Hc_ = roi.AffineOperator([ident], [Hc])
        model.Gc_ = roi.AffineOperator([ident, ident], [Gc, Gc])
        model.B_ = None

        # Predict.
        if issubclass(ModelClass, roi._base._ContinuousROM):
            model.predict(1, X[:,0], np.linspace(0, 1, 100))
        else:
            model.predict(1, X[:,0], 100)


class TestAffineInferredMixin:
    """Test _affine._inferred._AffineInferredMixin."""
    def _test_fit(self, ModelClass):
        """Test _affine._inferred._AffineInferredMixin.fit(), parent method of
        _affine._inferred.AffineInferredDiscreteROM.fit() and
        _affine._inferred.AffineInferredContinuousROM.fit().
        """
        model = ModelClass("cAHB")
        is_continuous = issubclass(ModelClass, roi._base._ContinuousROM)

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
            f"num parameter samples != num state snapshot sets ({s-1} != {s})"

        if is_continuous:
            with pytest.raises(ValueError) as ex:
                model.fit(Vr, µs, affines, Xs, Xdots[:-1], Us)
            assert ex.value.args[0] == \
                f"num parameter samples != num rhs sets ({s} != {s-1})"

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


class TestAffineIntrusiveMixin:
    """Test _affine._intrusive._AffineIntrusiveMixin."""
    def _test_fit(self, ModelClass):
        """Test _affine._intrusive._AffineIntrusiveMixin.fit(), parent method of
        _affine._intrusive.AffineIntrusiveDiscreteROM.fit() and
        _affine._intrusive.AffineIntrusiveContinuousROM.fit().
        """
        model = ModelClass("cAHGB")

        # Get test data.
        n, k, m, r = 30, 1000, 10, 5
        X = _get_data(n, k, m)[0]
        Vr = la.svd(X)[0][:,:r]

        # Get test operators.
        c, A, H, Hc, G, Gc, B = _get_operators(n, m)
        B1d = B[:,0]
        ident = lambda a: a
        affines = {"c": [ident, ident],
                   "A": [ident, ident, ident],
                   "H": [ident],
                   "G": [ident],
                   "B": [ident, ident]}
        operators = {"c": [c, c],
                     "A": [A, A, A],
                     "H": [H],
                     "G": [G],
                     "B": [B, B]}

        # Try to fit the model with misaligned operators and Vr.
        Abad = A[:,:-2]
        Hbad = H[:,1:]
        Gbad = G[:,:-1]
        cbad = c[::2]
        Bbad = B[1:,:]

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, affines,
                      {"c":[cbad, cbad],
                       "A": [A, A, A],
                       "H": [H],
                       "G": [G],
                       "B": [B, B]})
        assert ex.value.args[0] == "basis Vr and FOM operator c not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, affines,
                      {"c":[c, c],
                       "A": [Abad, Abad, Abad],
                       "H": [H],
                       "G": [G],
                       "B": [B, B]})
        assert ex.value.args[0] == "basis Vr and FOM operator A not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, affines,
                      {"c":[c, c],
                       "A": [A, A, A],
                       "H": [Hbad],
                       "G": [G],
                       "B": [B, B]})
        assert ex.value.args[0] == \
            "basis Vr and FOM operator H not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, affines,
                      {"c":[c, c],
                       "A": [A, A, A],
                       "H": [H],
                       "G": [Gbad],
                       "B": [B, B]})
        assert ex.value.args[0] == \
            "basis Vr and FOM operator G not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, affines,
                      {"c":[c, c],
                       "A": [A, A, A],
                       "H": [H],
                       "G": [G],
                       "B": [Bbad, Bbad]})
        assert ex.value.args[0] == "basis Vr and FOM operator B not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, {}, {"c":cbad, "A":A, "H":H, "G":G, "B":B})
        assert ex.value.args[0] == "basis Vr and FOM operator c not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, {}, {"c":c, "A":Abad, "H":H, "G":G, "B":B})
        assert ex.value.args[0] == "basis Vr and FOM operator A not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, {}, {"c":c, "A":A, "H":Hbad, "G":G, "B":B})
        assert ex.value.args[0] == "basis Vr and FOM operator H not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, {}, {"c":c, "A":A, "H":H, "G":Gbad, "B":B})
        assert ex.value.args[0] == "basis Vr and FOM operator G not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, {}, {"c":c, "A":A, "H":H, "G":G, "B":Bbad})
        assert ex.value.args[0] == "basis Vr and FOM operator B not aligned"

        # Fit the model correctly with each possible modelform.
        for form in ["A", "cA", "H", "cH", "AG", "cAH", "cAHB"]:
            model.modelform = form
            afs = {key:val for key,val in affines.items() if key in form}
            ops = {key:val for key,val in operators.items() if key in form}
            model.fit(Vr, afs, ops)

        model.modelform = "cAHGB"
        model.fit(Vr, {}, {"c":c, "A":A, "H":H, "G":Gc, "B":B})
        model.fit(Vr, {}, {"c":c, "A":A, "H":Hc, "G":G, "B":B})
        model.fit(Vr, {}, {"c":c, "A":A, "H":H, "G":Gc, "B":B1d})
        model.fit(Vr, affines,
                  {"c":[c, c],
                   "A": [A, A, A],
                   "H": [Hc],
                   "G": [Gc],
                   "B": [B, B]})
        model.fit(Vr, affines, operators)

        # Test fit output sizes.
        assert model.n == n
        assert model.r == r
        assert model.m == m
        assert model.A.shape == (n,n)
        assert model.Hc.shape == (n,n*(n+1)//2)
        assert model.H.shape == (n,n**2)
        assert model.Gc.shape == (n,n*(n+1)*(n+2)//6)
        assert model.G.shape == (n,n**3)
        assert model.c.shape == (n,)
        assert model.B.shape == (n,m)
        assert model.A_.shape == (r,r)
        assert model.Hc_.shape == (r,r*(r+1)//2)
        assert model.H_.shape == (r,r**2)
        assert model.Gc_.shape == (r,r*(r+1)*(r+2)//6)
        assert model.G_.shape == (r,r**3)
        assert model.c_.shape == (r,)
        assert model.B_.shape == (r,m)

        # Fit the model with 1D inputs (1D array for B)
        model.modelform = "cAHGB"
        model.fit(Vr, affines,
                  {"c":[c, c],
                   "A": [A, A, A],
                   "H": [Hc],
                   "G": [Gc],
                   "B": [B1d, B1d]})
        assert model.B.shape == (n,1)
        assert model.B_.shape == (r,1)


# Useable classes (public) ====================================================

# Affine inferred models ------------------------------------------------------
class TestAffineInferredDiscreteROM:
    """Test _affine._inferred.AffineInferredDiscreteROM."""
    def test_fit(self):
        """Test _affine._inferred.AffineInferredDiscreteROM.fit()."""
        TestAffineInferredMixin()._test_fit(roi.AffineInferredDiscreteROM)

    def test_predict(self):
        """Test _affine._inferred.AffineInferredDiscreteROM.predict()."""
        TestAffineMixin()._test_predict(roi.AffineInferredDiscreteROM)


class TestAffineInferredContinuousROM:
    """Test _affine._inferred.AffineInferredContinuousROM."""
    def test_fit(self):
        """Test _affine._inferred.AffineInferredContinuousROM.fit()."""
        TestAffineInferredMixin()._test_fit(roi.AffineInferredContinuousROM)

    def test_predict(self):
        """Test _affine._inferred.AffineInferredContinuousROM.predict()."""
        TestAffineMixin()._test_predict(roi.AffineInferredContinuousROM)


# Affine intrusive models -----------------------------------------------------
class TestAffineIntrusiveDiscreteROM:
    """Test _affine._intrusive.AffineIntrusiveDiscreteROM."""
    def test_fit(self):
        """Test _affine._intrusive.AffineIntrusiveDiscreteROM.fit()."""
        TestAffineIntrusiveMixin()._test_fit(roi.AffineIntrusiveDiscreteROM)

    def test_predict(self):
        """Test _affine._intrusive.AffineIntrusiveDiscreteROM.predict()."""
        TestAffineMixin()._test_predict(roi.AffineIntrusiveDiscreteROM)


class TestAffineIntrusiveContinuousROM:
    """Test _affine._intrusive.AffineIntrusiveContinuousROM."""
    def test_fit(self):
        """Test _affine._intrusive.AffineIntrusiveContinuousROM.fit()."""
        TestAffineIntrusiveMixin()._test_fit(roi.AffineIntrusiveContinuousROM)

    def test_predict(self):
        """Test _affine._intrusive.AffineIntrusiveContinuousROM.predict()."""
        TestAffineMixin()._test_predict(roi.AffineIntrusiveContinuousROM)
