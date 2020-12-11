# _core/test_intrusive.py
"""Tests for rom_operator_inference._core._intrusive.py."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi

from . import MODEL_FORMS, MODEL_KEYS, _get_data, _get_operators


# Mixins (private) ============================================================
class TestIntrusiveMixin:

    class Dummy(roi._core._base._BaseROM,
                roi._core._intrusive._IntrusiveMixin):
        def __init__(self, modelform):
            self.modelform = modelform
        def _construct_f_(*args, **kwargs):
            pass

    """Test _core._intrusive._IntrusiveMixin."""
    def test_check_operators_keys(self):
        """Test _core._intrusive._IntrusiveMixin._check_operators_keys()."""
        model = roi._core._intrusive._IntrusiveMixin()
        model.modelform = "cAHB"
        v = None

        # Try with missing operator keys.
        with pytest.raises(KeyError) as ex:
            model._check_operators_keys({"A":v, "H":v, "B":v})
        assert ex.value.args[0] == "missing operator key 'c'"

        with pytest.raises(KeyError) as ex:
            model._check_operators_keys({"H":v, "B":v})
        assert ex.value.args[0] == "missing operator keys 'c', 'A'"

        # Try with surplus operator keys.
        with pytest.raises(KeyError) as ex:
            model._check_operators_keys({'CC':v, "c":v, "A":v, "H":v, "B":v})
        assert ex.value.args[0] == "invalid operator key 'CC'"

        with pytest.raises(KeyError) as ex:
            model._check_operators_keys({"c":v, "A":v, "H":v, "B":v,
                                         'CC':v, 'LL':v})
        assert ex.value.args[0] == "invalid operator keys 'CC', 'LL'"

        # Correct usage.
        model._check_operators_keys({"c":v, "A":v, "H":v, "B":v})

    def test_process_fit_arguments(self):
        """Test _core._intrusive._IntrusiveMixin._process_fit_arguments()."""
        n, r = 30, 10
        Vr = np.random.random((n,r))

        model = self.Dummy("c")
        operators = {k:None for k in model.modelform}

        # Try without providing a basis.
        with pytest.raises(ValueError) as ex:
            model._process_fit_arguments(None, operators)
        assert ex.value.args[0] == \
            "Vr required for intrusive ROMs (got Vr=None)"

        # Correct usage.
        model._process_fit_arguments(Vr, operators)
        assert model.n == n
        assert model.r == r
        assert model.Vr is Vr

    def test_project_operators(self):
        """Test _core._intrusive._IntrusiveMixin._project_operators()."""
        # Get test data.
        n, m, r = 7, 5, 3
        Vr = np.random.random((n,r))
        shapes = {
                    "c":   (n,),
                    "A":   (n,n),
                    "H":   (n,n**2),
                    "G":   (n,n**3),
                    "B":   (n,m),
                    "c_":  (r,),
                    "A_":  (r,r),
                    "H_": (r,r*(r+1)//2),
                    "G_": (r,r*(r+1)*(r+2)//6),
                    "B_":  (r,m),
                 }

        # Initialize the test model.
        model = self.Dummy("cAHGB")
        model.Vr = Vr
        model.n = n

        # Get test operators.
        c, A, H, G, B = _get_operators(n, m, expanded=True)
        operators = {"c":c, "A":A, "H":H, "G":G, "B":B}
        B1d = B[:,0]

        # Try to fit the model with operators that are misaligned with Vr.
        cbad = c[::2]
        Abad = A[:,:-2]
        Hbad = H[:,1:]
        Gbad = G[:,:-1]
        Bbad = B[1:,:]

        with pytest.raises(ValueError) as ex:
            model._project_operators({"c":cbad, "A":A, "H":H, "G":G, "B":B})
        assert ex.value.args[0] == "basis Vr and FOM operator c not aligned"

        with pytest.raises(ValueError) as ex:
            model._project_operators({"c":c, "A":Abad, "H":H, "G":G, "B":B})
        assert ex.value.args[0] == "basis Vr and FOM operator A not aligned"

        with pytest.raises(ValueError) as ex:
            model._project_operators({"c":c, "A":A, "H":Hbad, "G":G, "B":B})
        assert ex.value.args[0] == "basis Vr and FOM operator H not aligned"

        with pytest.raises(ValueError) as ex:
            model._project_operators({"c":c, "A":A, "H":H, "G":Gbad, "B":B})
        assert ex.value.args[0] == "basis Vr and FOM operator G not aligned"

        with pytest.raises(ValueError) as ex:
            model._project_operators({"c":c, "A":A, "H":H, "G":G, "B":Bbad})
        assert ex.value.args[0] == "basis Vr and FOM operator B not aligned"

        # Test each modelform.
        for form in MODEL_FORMS:
            model.modelform = form
            ops = {key:val for key,val in operators.items() if key in form}
            model._project_operators(ops)
            for prefix in MODEL_KEYS:
                attr = prefix+'_'
                assert hasattr(model, prefix)
                assert hasattr(model, attr)
                value_n = getattr(model, prefix)
                value_r = getattr(model, attr)
                if prefix in form:
                    assert value_n is operators[prefix]
                    assert value_n.shape == shapes[prefix]
                    assert isinstance(value_r, np.ndarray)
                    assert value_r.shape == shapes[attr]
                else:
                    assert value_n is None
                    assert value_r is None
            if "B" in form:
                assert model.m == m
            else:
                assert model.m is None

        # Fit the model with 1D inputs (1D array for B)
        model.modelform = "cAHB"
        model._project_operators({"c":c, "A":A, "H":H, "B":B1d})
        assert model.B.shape == (n,1)
        assert model.B_.shape == (r,1)

    def _test_fit(self, ModelClass):
        """Test _core._intrusive._IntrusiveMixin.fit()."""
        # Get test data.
        n, m, r = 7, 5, 3
        Vr = np.random.random((n,r))

        # Get test operators.
        c, A, H, G, B = _get_operators(n, m, expanded=True)
        operators = {"c":c, "A":A, "H":H, "G":G, "B":B}
        B1d = B[:,0]

        # Test each modelform.
        model = ModelClass("cAHGB")
        for form in MODEL_FORMS:
            model.modelform = form
            ops = {key:val for key,val in operators.items() if key in form}
            model.fit(Vr, ops)
            if "B" in form:         # Also test with one-dimensional inputs.
                ops["B"] = B1d
                model.fit(Vr, ops)


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
