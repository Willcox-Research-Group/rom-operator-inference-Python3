# test_intrusive.py
"""Tests for rom_operator_inference._core._intrusive.py."""

import pytest
from scipy import linalg as la

import rom_operator_inference as roi

from . import _get_data, _get_operators


# Mixins (private) ============================================================
class TestIntrusiveMixin:
    """Test _core._intrusive._IntrusiveMixin."""
    def test_check_operators(self):
        """Test _core._intrusive._IntrusiveMixin._check_operators()."""
        model = roi._core._intrusive._IntrusiveMixin()
        model.modelform = "cAHB"
        v = None

        # Try with missing operator keys.
        with pytest.raises(KeyError) as ex:
            model._check_operators({"A":v, "H":v, "B":v})
        assert ex.value.args[0] == "missing operator key 'c'"

        with pytest.raises(KeyError) as ex:
            model._check_operators({"H":v, "B":v})
        assert ex.value.args[0] == "missing operator keys 'c', 'A'"

        # Try with surplus operator keys.
        with pytest.raises(KeyError) as ex:
            model._check_operators({'CC':v, "c":v, "A":v, "H":v, "B":v})
        assert ex.value.args[0] == "invalid operator key 'CC'"

        with pytest.raises(KeyError) as ex:
            model._check_operators({"c":v, "A":v, "H":v, "B":v,
                                    'CC':v, 'LL':v})
        assert ex.value.args[0] == "invalid operator keys 'CC', 'LL'"

        # Correct usage.
        model._check_operators({"c":v, "A":v, "H":v, "B":v})

    def _test_fit(self, ModelClass):
        """Test _core._intrusive._IntrusiveMixin.fit(), the parent method for
        _core._intrusive.IntrusiveDiscreteROM.fit() and
        _core._intrusive.IntrusiveContinuousROM.fit().
        """
        model = ModelClass("cAHB")

        # Get test data.
        n, k, m, r = 30, 50, 10, 5
        X = _get_data(n, k, m)[0]
        Vr = la.svd(X)[0][:,:r]

        # Get test operators.
        c, A, H, Hc, G, Gc, B = _get_operators(n, m)
        B1d = B[:,0]
        operators = {"c":c, "A":A, "H":H, "G":G, "B":B}

        # Try to fit the model with misaligned operators and Vr.
        Abad = A[:,:-2]
        Hbad = H[:,1:]
        Gbad = G[:,:-1]
        cbad = c[::2]
        Bbad = B[1:,:]

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, {"c":cbad, "A":A, "H":H, "B":B})
        assert ex.value.args[0] == "basis Vr and FOM operator c not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, {"c":c, "A":Abad, "H":H, "B":B})
        assert ex.value.args[0] == "basis Vr and FOM operator A not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, {"c":c, "A":A, "H":Hbad, "B":B})
        assert ex.value.args[0] == \
            "basis Vr and FOM operator H not aligned"

        model = ModelClass("cAGB")
        with pytest.raises(ValueError) as ex:
            model.fit(Vr, {"c":c, "A":A, "G":Gbad, "B":B})
        assert ex.value.args[0] == \
            "basis Vr and FOM operator G not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit(Vr, {"c":c, "A":A, "G":G, "B":Bbad})
        assert ex.value.args[0] == "basis Vr and FOM operator B not aligned"

        # Fit the model with each possible modelform.
        for form in ["A", "cA", "H", "cH", "AG", "cAH", "cAHB"]:
            model.modelform = form
            ops = {key:val for key,val in operators.items() if key in form}
            model.fit(Vr, ops)

        model.modelform = "cAHGB"
        model.fit(Vr, {"c":c, "A":A, "H":Hc, "G":Gc, "B":B})

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
        model.modelform = "cAHB"
        model.fit(Vr, {"c":c, "A":A, "H":H, "B":B1d})
        assert model.B.shape == (n,1)
        assert model.B_.shape == (r,1)


# Useable classes (public) ====================================================
class TestIntrusiveDiscreteROM:
    """Test _core._intrusive.IntrusiveDiscreteROM."""
    def test_fit(self):
        """Test _core._intrusive.IntrusiveDiscreteROM.fit()."""
        TestIntrusiveMixin()._test_fit(roi.IntrusiveDiscreteROM)


class TestIntrusiveContinuousROM:
    """Test _core._intrusive.IntrusiveContinuousROM."""
    def test_fit(self):
        """Test _core._intrusive.IntrusiveContinuousROM.fit()."""
        TestIntrusiveMixin()._test_fit(roi.IntrusiveContinuousROM)
