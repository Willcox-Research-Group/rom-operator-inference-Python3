# _core/_affine/test_base.py
"""Tests for rom_operator_inference._core._affine._base."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi

from .. import _get_data, _get_operators


# Affine operator (public) ====================================================
class TestAffineOperator:
    """Test _core._affine._base.AffineOperator."""
    @staticmethod
    def _set_up_affine_attributes(n=5):
        fs = [np.sin, np.cos, np.exp]
        As = list(np.random.random((3,n,n)))
        return fs, As

    def test_init(self):
        """Test _core._affine._base.AffineOperator.__init__()."""
        fs, As = self._set_up_affine_attributes()

        # Try with non-callables.
        with pytest.raises(TypeError) as ex:
            roi.AffineOperator(As, As)
        assert ex.value.args[0] == \
            "coefficients of affine operator must be callable"

        # Try with different number of functions and matrices.
        with pytest.raises(ValueError) as ex:
            roi.AffineOperator(fs, As[:-1])
        assert ex.value.args[0] == \
            f"{len(fs)} = len(coeffs) != len(matrices) = {len(As[:-1])}"

        # Try with matrices of different shapes.
        with pytest.raises(ValueError) as ex:
            roi.AffineOperator(fs, As[:-1] + [np.random.random((4,4))])
        assert ex.value.args[0] == \
            "affine component matrix shapes do not match"

        # Correct usage.
        affop = roi.AffineOperator(fs, As)
        for A in As:
            assert affop.shape == A.shape

    def test_validate_coeffs(self):
        """Test _core._affine._base.AffineOperator.validate_coeffs()."""
        fs, As = self._set_up_affine_attributes()

        # Try with non-callables.
        with pytest.raises(TypeError) as ex:
            roi.AffineOperator.validate_coeffs(As, 10)
        assert ex.value.args[0] == \
            "coefficient functions of affine operator must be callable"

        # Try with vector-valued functions.
        f1 = lambda t: np.array([t, t**2])
        with pytest.raises(ValueError) as ex:
            roi.AffineOperator.validate_coeffs([f1, f1], 10)
        assert ex.value.args[0] == \
            "coefficient functions of affine operator must return a scalar"

        # Correct usage.
        roi.AffineOperator.validate_coeffs(fs, 10)

    def test_call(self):
        """Test _core._affine._base.AffineOperator.__call__()."""
        fs, As = self._set_up_affine_attributes()

        # Try without matrices set.
        affop = roi.AffineOperator(fs, As)
        Ap = affop(10)
        assert Ap.shape == (5,5)
        assert np.allclose(Ap, np.sin(10)*As[0] + \
                               np.cos(10)*As[1] + np.exp(10)*As[2])

    def test_eq(self):
        """Test _core._affine._base.AffineOperator.__eq__()."""
        fs, As = self._set_up_affine_attributes()
        affop1 = roi.AffineOperator(fs[:-1], As[:-1])
        affop2 = roi.AffineOperator(fs, As)

        assert affop1 != 1
        assert affop1 != affop2
        affop1 = roi.AffineOperator(fs, As)
        assert affop1 == affop2


# Affine base mixin (private) =================================================
class TestAffineMixin:
    """Test _core._affine._base._AffineMixin."""
    def test_check_affines_keys(self):
        """Test _core._affine._base._AffineMixin._check_affines_keys()."""
        model = roi._core._affine._base._AffineMixin()
        model.modelform = "cAHB"
        v = [lambda s: 0, lambda s: 0]

        # Try with surplus affine keys.
        with pytest.raises(KeyError) as ex:
            model._check_affines_keys({'CC':v, "c":v, "A":v, "H":v, "B":v})
        assert ex.value.args[0] == "invalid affine key 'CC'"

        with pytest.raises(KeyError) as ex:
            model._check_affines_keys({"c":v, "A":v, "H":v, "B":v,
                                    'CC':v, 'LL':v})
        assert ex.value.args[0] == "invalid affine keys 'CC', 'LL'"

        # Correct usage.
        model._check_affines_keys({"c":v, "H":v})   # OK to be missing some.
        model._check_affines_keys({"c":v, "A":v, "H":v, "B":v})

    def _test_predict(self, ModelClass):
        """Test predict() methods for Affine classes:
        * _core._affine._inferred.AffineInferredDiscreteROM.predict()
        * _core._affine._inferred.AffineInferredContinuousROM.predict()
        * _core._affine._intrusive.AffineIntrusiveDiscreteROM.predict()
        * _core._affine._intrusive.AffineIntrusiveContinuousROM.predict()
        """
        model = ModelClass("cAHG")

        # Get test data.
        n, k, m, r = 60, 50, 20, 10
        X = _get_data(n, k, m)[0]
        Vr = la.svd(X)[0][:,:r]

        # Get test operators.
        ident = lambda a: a
        c, A, H, G, B = _get_operators(r, m)
        model.Vr = Vr
        model.c_ = roi.AffineOperator([ident, ident], [c,c])
        model.A_ = roi.AffineOperator([ident, ident, ident], [A,A,A])
        model.H_ = roi.AffineOperator([ident], [H])
        model.G_ = roi.AffineOperator([ident, ident], [G, G])
        model.B_ = None

        # Predict.
        if issubclass(ModelClass, roi._core._base._ContinuousROM):
            model.predict(1, X[:,0], np.linspace(0, 1, 100))
        else:
            model.predict(1, X[:,0], 100)
