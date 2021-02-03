# _core/test_intrusive.py
"""Tests for rom_operator_inference._core._intrusive.py."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi

from . import MODEL_FORMS, MODEL_KEYS, _get_data, _get_operators


# Mixins (private) ============================================================
class TestIntrusiveMixin:
    """Test _core._intrusive._IntrusiveMixin."""
    class Dummy(roi._core._intrusive._IntrusiveMixin,
                roi._core._base._BaseROM):
        def __init__(self, modelform):
            self.modelform = modelform

    def test_dimension_properties(self, n=20, m=3, r=7):
        """Test the properties _core._base._BaseROM.(n|r|Vr)."""
        model = self.Dummy("cH")
        assert model.n is None
        assert model.m == 0
        assert model.r is None
        assert model.Vr is None

        # Try setting n explicitly.
        with pytest.raises(AttributeError) as ex:
            model.n = n+1
        assert ex.value.args[0] == "can't set attribute (n = Vr.shape[0])"

        # Try setting r explicitly.
        with pytest.raises(AttributeError) as ex:
            model.r = r+1
        assert ex.value.args[0] == "can't set attribute (r = Vr.shape[1])"

        # Correct assignment.
        Vr = np.random.random((n,r))
        model.Vr = Vr
        assert model.n == n
        assert model.m == 0
        assert model.r == r
        assert model.Vr is Vr

        # Correct cleanup.
        del model.Vr
        assert model.Vr is None
        assert model.n is None
        assert model.r is None

        # Try setting Vr to None.
        model = self.Dummy("AB")
        assert model.n is None
        assert model.m is None
        assert model.r is None
        assert model.Vr is None
        with pytest.raises(AttributeError) as ex:
            model.Vr = None
        assert ex.value.args[0] == "Vr=None not allowed for intrusive ROMs"

    def test_operator_properties(self, n=10, m=4, r=2):
        """Test the properties _core._base._BaseROM.(c_|A_|H_|G_|B_)."""
        c,  A,  H,  G,  B  = fom_operators = _get_operators(n, m, True)
        c_, A_, H_, G_, B_ = rom_operators = _get_operators(r, m)

        model = self.Dummy(self.Dummy._MODEL_KEYS)
        model.Vr = np.zeros((n,r))
        model.m = m

        for fom_key, op, op_ in zip("cAHGB", fom_operators, rom_operators):
            rom_key = fom_key + '_'
            assert hasattr(model, fom_key)
            assert hasattr(model, rom_key)
            assert getattr(model, fom_key) is None
            assert getattr(model, rom_key) is None
            setattr(model, fom_key, op)
            setattr(model, rom_key, op_)
            assert getattr(model, fom_key) is op
            assert getattr(model, rom_key) is op_

        model.H = np.zeros((n,n*(n + 1)//2))
        model.G = np.zeros((n,n*(n + 1)*(n + 2)//6))
        model.H_ = np.zeros((r,r**2))
        model.G_ = np.zeros((r,r**3))

    def test_check_fom_operator_shape(self, n=10, m=3, r=4):
        """Test _core._intrusive._IntrusiveMixin._check_fom_operator_shape().
        """
        c, A, H, G, B = operators = _get_operators(n, m, expanded=True)

        # Try correct match but dimension 'r' is missing.
        model = self.Dummy("A")
        with pytest.raises(AttributeError) as ex:
            model._check_fom_operator_shape(A, 'A')
        assert ex.value.args[0] == "no basis 'Vr' (call fit())"

        # Try correct match but dimension 'm' is missing.
        model = self.Dummy("B")
        model.Vr = np.zeros((n,r))
        with pytest.raises(AttributeError) as ex:
            model._check_fom_operator_shape(B, 'B')
        assert ex.value.args[0] == "no input dimension 'm' (call fit())"

        # Try with dimensions set, but improper shapes.
        model = self.Dummy(self.Dummy._MODEL_KEYS)
        model.Vr = np.zeros((n,r))
        model.m = m

        with pytest.raises(ValueError) as ex:
            model._check_fom_operator_shape(c[:-1], 'c')
        assert ex.value.args[0] == \
            f"c.shape = {c[:-1].shape}, must be (n,) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._check_fom_operator_shape(A[:-1,1:], 'A')
        assert ex.value.args[0] == \
            f"A.shape = {A[:-1,1:].shape}, must be (n,n) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._check_fom_operator_shape(H[:-1,:-1], 'H')
        assert ex.value.args[0] == \
            f"H.shape = {H[:-1,:-1].shape}, must be (n,n**2) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._check_fom_operator_shape(G[1:], 'G')
        assert ex.value.args[0] == \
            f"G.shape = {G[1:].shape}, must be (n,n**3) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._check_fom_operator_shape(B[1:-1], 'B')
        assert ex.value.args[0] == \
            f"B.shape = {B[1:-1].shape}, must be (n,m) with n = {n}, m = {m}"

        # Correct usage.
        for key, op in zip("cAHGB", operators):
            model._check_fom_operator_shape(op, key)

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

    def test_process_fit_arguments(self, n=30, r=10):
        """Test _core._intrusive._IntrusiveMixin._process_fit_arguments()."""
        Vr = np.random.random((n,r))

        model = self.Dummy("c")
        operators = {k:None for k in model.modelform}

        # Correct usage.
        model._process_fit_arguments(Vr, operators)
        assert model.n == n
        assert model.r == r
        assert model.Vr is Vr

    def test_project_operators(self, n=7, m=5, r=3):
        """Test _core._intrusive._IntrusiveMixin._project_operators()."""
        # Get test data.
        Vr = np.random.random((n,r))
        shapes = {
                    "c":   (n,),
                    "A":   (n,n),
                    "H":   (n,n**2),
                    "G":   (n,n**3),
                    "B":   (n,m),
                    "c_":  (r,),
                    "A_":  (r,r),
                    "H_":  (r,r*(r+1)//2),
                    "G_":  (r,r*(r+1)*(r+2)//6),
                    "B_":  (r,m),
                 }

        # Initialize the test model.
        model = self.Dummy("cAHGB")
        model.Vr = Vr

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
        assert ex.value.args[0] == \
            f"c.shape = {cbad.shape}, must be (n,) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._project_operators({"c":c, "A":Abad, "H":H, "G":G, "B":B})
        assert ex.value.args[0] == \
            f"A.shape = {Abad.shape}, must be (n,n) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._project_operators({"c":c, "A":A, "H":Hbad, "G":G, "B":B})
        assert ex.value.args[0] == \
            f"H.shape = {Hbad.shape}, must be (n,n**2) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._project_operators({"c":c, "A":A, "H":H, "G":Gbad, "B":B})
        assert ex.value.args[0] == \
            f"G.shape = {Gbad.shape}, must be (n,n**3) with n = {n}"

        with pytest.raises(ValueError) as ex:
            model._project_operators({"c":c, "A":A, "H":H, "G":G, "B":Bbad})
        assert ex.value.args[0] == \
            f"B.shape = {Bbad.shape}, must be (n,m) with n = {n}, m = {m}"

        # Test each modelform.
        for form in MODEL_FORMS:
            model = self.Dummy(form)
            model.Vr = Vr
            ops = {key:val for key,val in operators.items() if key in form}
            model._project_operators(ops)
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
        model._project_operators({"c":c, "A":A, "H":H, "B":B1d})
        assert model.m == 1
        assert model.B.shape == (n,1)
        assert model.B_.shape == (r,1)

    def _test_fit(self, ModelClass, n=7, m=5, r=3):
        """Test _core._intrusive._IntrusiveMixin.fit()."""
        # Get test data.
        Vr = np.random.random((n,r))

        # Get test operators.
        c, A, H, G, B = _get_operators(n, m, expanded=True)
        operators = {"c":c, "A":A, "H":H, "G":G, "B":B}
        B1d = B[:,0]

        # Test each modelform.
        for form in MODEL_FORMS:
            model = ModelClass(form)
            ops = {key:val for key,val in operators.items() if key in form}
            model.fit(Vr, ops)
            if "B" in form:         # Also test with one-dimensional inputs.
                ops["B"] = B1d
                model.fit(Vr, ops)


# Useable classes (public) ====================================================
class TestIntrusiveDiscreteROM:
    """Test _core._intrusive.IntrusiveDiscreteROM."""
    def test_f(self, n=5, m=2):
        """Test _core._intrusive.IntrusiveDiscreteROM.f()."""
        c, A, H, G, B = _get_operators(n, m, expanded=True)
        Vr = np.zeros((n,n//2))

        model = roi._core._intrusive.IntrusiveDiscreteROM("cA")
        model.Vr = Vr
        model.c, model.A = c, A
        x = np.random.random(n)
        y = c + A @ x
        assert np.allclose(model.f(x), y)
        assert np.allclose(model.f(x, -1), y)

        model = roi._core._intrusive.IntrusiveDiscreteROM("HGB")
        model.Vr = Vr
        model.m = m
        model.H, model.G, model.B = H, G, B
        u = np.random.random(m)
        x = np.random.random(n)
        x2 = np.kron(x, x)
        y = H @ x2 + G @ np.kron(x, x2) + B @ u
        assert np.allclose(model.f(x, u), y)

    def test_fit(self):
        """Test _core._intrusive.IntrusiveDiscreteROM.fit()."""
        TestIntrusiveMixin()._test_fit(roi.IntrusiveDiscreteROM)


class TestIntrusiveContinuousROM:
    """Test _core._intrusive.IntrusiveContinuousROM."""
    def test_f(self, n=5, m=2):
        """Test _core._intrusive.IntrusiveContinuousROM.f()."""
        c, A, H, G, B = _get_operators(n, m, expanded=True)
        Vr = np.zeros((n, n//2))

        # Check that the constructed f takes the right number of arguments.
        model = roi._core._intrusive.IntrusiveContinuousROM("cA")
        model.Vr = Vr
        model.c, model.A = c, A
        x = np.random.random(n)
        y = c + A @ x
        assert np.allclose(model.f(0, x), y)
        assert np.allclose(model.f(1, x), y)
        assert np.allclose(model.f(1, x, -1), y)

        model = roi._core._intrusive.IntrusiveContinuousROM("HGB")
        model.Vr = Vr
        model.m = m
        model.H, model.G, model.B = H, G, B
        uu = np.random.random(m)
        u = lambda t: uu + t
        x = np.random.random(n)
        x2 = np.kron(x, x)
        y = H @ x2 + G @ np.kron(x, x2) + B @ uu
        assert np.allclose(model.f(0, x, u), y)
        y = H @ x2 + G @ np.kron(x, x2) + B @ (uu + 1)
        assert np.allclose(model.f(1, x, u), y)

    def test_fit(self):
        """Test _core._intrusive.IntrusiveContinuousROM.fit()."""
        TestIntrusiveMixin()._test_fit(roi.IntrusiveContinuousROM)
