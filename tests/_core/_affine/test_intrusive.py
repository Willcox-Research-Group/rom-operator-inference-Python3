# _core/_affine/test_intrusive.py
"""Tests for rom_operator_inference._core._affine._intrusive."""

import pytest
import numpy as np

import rom_operator_inference as opinf

from .test_base import TestAffineMixin
from .. import _get_operators, MODEL_FORMS, MODEL_KEYS


# Affine intrusive mixin (private) ============================================
class TestAffineIntrusiveMixin:
    """Test _core._affine._intrusive._AffineIntrusiveMixin."""
    class Dummy(opinf._core._affine._intrusive._AffineIntrusiveMixin,
                opinf._core._base._BaseROM):
        def __init__(self, modelform):
            self.modelform = modelform

    def test_process_fit_arguments(self, n=30, r=10):
        """Test _core._intrusive._IntrusiveMixin._process_fit_arguments()."""
        Vr = np.random.random((n,r))

        model = self.Dummy("c")
        operators = {k:None for k in model.modelform}

        # Correct usage.
        model._process_fit_arguments(Vr, operators, operators)
        assert model.n == n
        assert model.r == r
        assert model.Vr is Vr

    def test_project_operators(self, n=7, m=5, r=3):
        """Test _core._intrusive._IntrusiveMixin._project_operators()."""
        # Get test data.
        Vr = np.random.random((n,r))
        shapes = {
                    "c": (n,),
                    "A": (n,n),
                    "H": (n,n**2),
                    "G": (n,n**3),
                    "B": (n,m),
                    "c_": (r,),
                    "A_": (r,r),
                    "H_": (r,r*(r+1)//2),
                    "G_": (r,r*(r+1)*(r+2)//6),
                    "B_": (r,m),
                 }

        # Get test operators.
        c, A, H, G, B = _get_operators(n, m, expanded=True)
        operators = {"c":c, "A":A, "H":H, "G":G, "B":B}
        B1d = B[:,0]

        # TEST SET 1: Ensure affines={} acts like vanilla intrusive.
        # Try to fit the model with operators that are misaligned with Vr.
        cbad = c[::2]
        Abad = A[:,:-2]
        Hbad = H[:,1:]
        Gbad = G[:,:-1]
        Bbad = B[1:,:]

        # Initialize the test model.
        model = self.Dummy(self.Dummy._MODEL_KEYS)
        model.Vr = Vr

        with pytest.raises(ValueError) as ex:
            model._project_operators({},{"c":cbad, "A":A, "H":H, "G":G, "B":B})
        assert ex.value.args[0] == \
            f"c.shape = {cbad.shape}, must be (n,) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._project_operators({},{"c":c, "A":Abad, "H":H, "G":G, "B":B})
        assert ex.value.args[0] == \
            f"A.shape = {Abad.shape}, must be (n,n) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._project_operators({},{"c":c, "A":A, "H":Hbad, "G":G, "B":B})
        assert ex.value.args[0] == \
            f"H.shape = {Hbad.shape}, must be (n,n**2) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._project_operators({},{"c":c, "A":A, "H":H, "G":Gbad, "B":B})
        assert ex.value.args[0] == \
            f"G.shape = {Gbad.shape}, must be (n,n**3) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._project_operators({},{"c":c, "A":A, "H":H, "G":G, "B":Bbad})
        assert ex.value.args[0] == \
            f"B.shape = {Bbad.shape}, must be (n,m) with n = {n}, m = {m}"

        # Test each modelform.
        for form in MODEL_FORMS:
            model = self.Dummy(form)
            model.Vr = Vr
            ops = {key:val for key,val in operators.items() if key in form}
            model._project_operators({}, ops)
            for prefix in MODEL_KEYS:
                attr = prefix+'_'
                assert hasattr(model, prefix)
                assert hasattr(model, attr)
                fom_op = getattr(model, prefix)
                rom_op = getattr(model, attr)
                if prefix in form:
                    assert fom_op is operators[prefix]
                    assert fom_op.shape == shapes[prefix]
                    assert isinstance(rom_op, np.ndarray)
                    assert rom_op.shape == shapes[attr]
                else:
                    assert fom_op is None
                    assert rom_op is None
            if "B" in form:
                assert model.m == m
            else:
                assert model.m == 0

        # Fit the model with 1D inputs (1D array for B)
        model = self.Dummy("cAHB")
        model.Vr = Vr
        model._project_operators({}, {"c":c, "A":A, "H":H, "B":B1d})
        assert model.m == 1
        assert model.B.shape == (n,1)
        assert model.B_.shape == (r,1)

        # TEST SET 2: Nontrivial affines.
        def ident(a):
            return a
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

        model = self.Dummy(self.Dummy._MODEL_KEYS)
        model.Vr = Vr

        with pytest.raises(ValueError) as ex:
            model._project_operators(affines, {"c": [cbad,cbad],
                                               "A": [A, A, A],
                                               "H": [H],
                                               "G": [G],
                                               "B": [B, B]})
        assert ex.value.args[0] == \
            f"c.shape = {cbad.shape}, must be (n,) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._project_operators(affines, {"c": [c, c],
                                               "A": [Abad,Abad,Abad],
                                               "H": [H],
                                               "G": [G],
                                               "B": [B, B]})
        assert ex.value.args[0] == \
            f"A.shape = {Abad.shape}, must be (n,n) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._project_operators(affines, {"c": [c, c],
                                               "A": [A, A, A],
                                               "H": [Hbad],
                                               "G": [G],
                                               "B": [B, B]})
        assert ex.value.args[0] == \
            f"H.shape = {Hbad.shape}, must be (n,n**2) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._project_operators(affines, {"c": [c, c],
                                               "A": [A, A, A],
                                               "H": [H],
                                               "G": [Gbad],
                                               "B": [B, B]})
        assert ex.value.args[0] == \
            f"G.shape = {Gbad.shape}, must be (n,n**3) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._project_operators(affines, {"c": [c, c],
                                               "A": [A, A, A],
                                               "H": [H],
                                               "G": [G],
                                               "B": [Bbad, Bbad]})
        assert ex.value.args[0] == \
            f"B.shape = {Bbad.shape}, must be (n,m) with n = {n}, m = {m}"

        # Test each modelform.
        for form in MODEL_FORMS:
            model = self.Dummy(form)
            model.Vr = Vr
            ops = {key:val for key,val in operators.items() if key in form}
            afs = {key:val for key,val in affines.items() if key in form}
            model._project_operators(afs, ops)
            for prefix in MODEL_KEYS:
                attr = prefix+'_'
                assert hasattr(model, prefix)
                assert hasattr(model, attr)
                fom_op = getattr(model, prefix)
                rom_op = getattr(model, attr)
                if prefix in form:
                    assert isinstance(fom_op,
                                      opinf._core._affine.AffineOperator)
                    assert fom_op.shape == shapes[prefix]
                    assert len(fom_op.matrices) == len(operators[prefix])
                    for fom_op, op in zip(fom_op.matrices, operators[prefix]):
                        assert np.all(fom_op == op)
                    assert isinstance(rom_op,
                                      opinf._core._affine.AffineOperator)
                    assert rom_op.shape == shapes[attr]
                    assert len(rom_op.matrices) == len(operators[prefix])
                else:
                    assert fom_op is None
                    assert rom_op is None
            if "B" in form:
                assert model.m == m
            else:
                assert model.m == 0

        model = self.Dummy("HG")
        model.Vr = Vr
        Hc = np.random.random((n,n*(n + 1)//2))
        Gc = np.random.random((n,n*(n + 1)*(n + 2)//6))
        model._project_operators({"H": [ident], "G": [ident]},
                                 {"H": [Hc], "G": [Gc]})
        assert model.H.shape == (n,n**2)
        assert model.G.shape == (n,n**3)

    def _test_fit(self, ModelClass, n=15, k=500, m=4, r=2):
        """Test _core._affine._intrusive._AffineIntrusiveMixin.fit(),
        parent method of
        _core._affine._intrusive.AffineIntrusiveDiscreteROM.fit() and
        _core._affine._intrusive.AffineIntrusiveContinuousROM.fit().
        """
        # Get test data.
        Vr = np.random.random((n,r))

        # TEST SET 1: Ensure affines={} acts like vanilla intrusive.
        # Get test operators.
        c, A, H, G, B = _get_operators(n, m, expanded=True)
        operators = {"c":c, "A":A, "H":H, "G":G, "B":B}
        B1d = B[:,0]

        # Test each modelform.
        for form in MODEL_FORMS:
            model = ModelClass(form)
            ops = {key:val for key,val in operators.items() if key in form}
            model.fit(Vr, {}, ops)
            if "B" in form:         # Also test with one-dimensional inputs.
                ops["B"] = B1d
                model.fit(Vr, {}, ops)

        # TEST SET 2: Nontrivial affines.
        def ident(a):
            return a
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

        # Test each modelform.
        for form in MODEL_FORMS:
            model = ModelClass(form)
            ops = {key:val for key,val in operators.items() if key in form}
            afs = {key:val for key,val in affines.items() if key in form}
            model.fit(Vr, afs, ops)
            if "B" in form:         # Also test with one-dimensional inputs.
                ops["B"] = [B1d, B1d]
                model.fit(Vr, afs, ops)


# Affine intrusive models (public) ============================================
class TestAffineIntrusiveDiscreteROM:
    """Test _core._affine._intrusive.AffineIntrusiveDiscreteROM."""
    def test_fit(self):
        """Test _core._affine._intrusive.AffineIntrusiveDiscreteROM.fit()."""
        TestAffineIntrusiveMixin()._test_fit(opinf.AffineIntrusiveDiscreteROM)

    def test_predict(self):
        """Test _core._affine._intrusive.AffineIntrusiveDiscreteROM.predict().
        """
        TestAffineMixin()._test_predict(opinf.AffineIntrusiveDiscreteROM)


class TestAffineIntrusiveContinuousROM:
    """Test _core._affine._intrusive.AffineIntrusiveContinuousROM."""
    def test_fit(self):
        """Test _core._affine._intrusive.AffineIntrusiveContinuousROM.fit()."""
        TestAffineIntrusiveMixin()._test_fit(
            opinf.AffineIntrusiveContinuousROM)

    def test_predict(self):
        """Test
        _core._affine._intrusive.AffineIntrusiveContinuousROM.predict().
        """
        TestAffineMixin()._test_predict(opinf.AffineIntrusiveContinuousROM)
