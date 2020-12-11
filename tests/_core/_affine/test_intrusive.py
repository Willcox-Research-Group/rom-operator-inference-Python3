# _core/_affine/test_intrusive.py
"""Tests for rom_operator_inference._core._affine._intrusive."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi

from .test_base import TestAffineMixin
from .. import _get_data, _get_operators


# Affine intrusive mixin (private) ============================================
class TestAffineIntrusiveMixin:
    """Test _core._affine._intrusive._AffineIntrusiveMixin."""
    def _test_fit(self, ModelClass):
        """Test _core._affine._intrusive._AffineIntrusiveMixin.fit(),
        parent method of
        _core._affine._intrusive.AffineIntrusiveDiscreteROM.fit() and
        _core._affine._intrusive.AffineIntrusiveContinuousROM.fit().
        """
        model = ModelClass("cAHGB")

        # Get test data.
        n, k, m, r = 30, 1000, 10, 5
        X = _get_data(n, k, m)[0]
        Vr = la.svd(X)[0][:,:r]

        # Get test operators.
        c, A, H, G, B = _get_operators(n, m, expanded=True)
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
        Hc = roi.utils.compress_H(H)
        Gc = roi.utils.compress_G(G)
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
        assert model.c.shape == (n,)
        assert model.A.shape == (n,n)
        assert model.H.shape == (n,n**2)
        assert model.G.shape == (n,n**3)
        assert model.B.shape == (n,m)
        assert model.c_.shape == (r,)
        assert model.A_.shape == (r,r)
        assert model.H_.shape == (r,r*(r+1)//2)
        assert model.G_.shape == (r,r*(r+1)*(r+2)//6)
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


# Affine intrusive models (public) ============================================
class TestAffineIntrusiveDiscreteROM:
    """Test _core._affine._intrusive.AffineIntrusiveDiscreteROM."""
    def test_fit(self):
        """Test _core._affine._intrusive.AffineIntrusiveDiscreteROM.fit()."""
        TestAffineIntrusiveMixin()._test_fit(roi.AffineIntrusiveDiscreteROM)

    def test_predict(self):
        """Test _core._affine._intrusive.AffineIntrusiveDiscreteROM.predict()."""
        TestAffineMixin()._test_predict(roi.AffineIntrusiveDiscreteROM)


class TestAffineIntrusiveContinuousROM:
    """Test _core._affine._intrusive.AffineIntrusiveContinuousROM."""
    def test_fit(self):
        """Test _core._affine._intrusive.AffineIntrusiveContinuousROM.fit()."""
        TestAffineIntrusiveMixin()._test_fit(roi.AffineIntrusiveContinuousROM)

    def test_predict(self):
        """Test _core._affine._intrusive.AffineIntrusiveContinuousROM.predict()."""
        TestAffineMixin()._test_predict(roi.AffineIntrusiveContinuousROM)
