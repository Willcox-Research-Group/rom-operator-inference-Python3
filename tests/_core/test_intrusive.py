# _core/test_intrusive.py
"""Tests for rom_operator_inference._core._intrusive.py."""

import pytest
import numpy as np

import rom_operator_inference as opinf

from . import MODEL_FORMS, MODEL_KEYS, _get_operators


# Mixins (private) ============================================================
class TestIntrusiveMixin:
    """Test _core._intrusive._IntrusiveMixin."""
    class Dummy(opinf._core._intrusive._IntrusiveMixin,
                opinf._core._base._BaseROM):
        def __init__(self, modelform):
            self.modelform = modelform

    def test_dimension_properties(self, n=20, m=3, r=7):
        """Test the properties _core._base._BaseROM.(n|r|Vr)."""
        rom = self.Dummy("cH")
        assert rom.n is None
        assert rom.m == 0
        assert rom.r is None
        assert rom.Vr is None

        # Try setting n explicitly.
        with pytest.raises(AttributeError) as ex:
            rom.n = n+1
        assert ex.value.args[0] == "can't set attribute (n = Vr.shape[0])"

        # Try setting r explicitly.
        with pytest.raises(AttributeError) as ex:
            rom.r = r+1
        assert ex.value.args[0] == "can't set attribute (r = Vr.shape[1])"

        # Correct assignment.
        Vr = np.random.random((n,r))
        rom.Vr = Vr
        assert rom.n == n
        assert rom.m == 0
        assert rom.r == r
        assert rom.Vr is Vr

        # Correct cleanup.
        del rom.Vr
        assert rom.Vr is None
        assert rom.n is None
        assert rom.r is None

        # Try setting Vr to None.
        rom = self.Dummy("AB")
        assert rom.n is None
        assert rom.m is None
        assert rom.r is None
        assert rom.Vr is None
        with pytest.raises(AttributeError) as ex:
            rom.Vr = None
        assert ex.value.args[0] == "Vr=None not allowed for intrusive ROMs"

    def test_operator_properties(self, n=10, m=4, r=2):
        """Test the properties _core._base._BaseROM.(c_|A_|H_|G_|B_)."""
        c, A, H, G, B = fom_operators = _get_operators(n, m, True)
        c_, A_, H_, G_, B_ = rom_operators = _get_operators(r, m)

        rom = self.Dummy(self.Dummy._MODEL_KEYS)
        rom.Vr = np.zeros((n,r))
        rom.m = m

        for fom_key, op, op_ in zip("cAHGB", fom_operators, rom_operators):
            rom_key = fom_key + '_'
            assert hasattr(rom, fom_key)
            assert hasattr(rom, rom_key)
            assert getattr(rom, fom_key) is None
            assert getattr(rom, rom_key) is None
            setattr(rom, fom_key, op)
            setattr(rom, rom_key, op_)
            assert getattr(rom, fom_key) is op
            assert getattr(rom, rom_key) is op_

        rom.H = np.zeros((n,n*(n + 1)//2))
        rom.G = np.zeros((n,n*(n + 1)*(n + 2)//6))
        rom.H_ = np.zeros((r,r**2))
        rom.G_ = np.zeros((r,r**3))

    def test_check_fom_operator_shape(self, n=10, m=3, r=4):
        """Test _core._intrusive._IntrusiveMixin._check_fom_operator_shape().
        """
        c, A, H, G, B = operators = _get_operators(n, m, expanded=True)

        # Try correct match but dimension 'r' is missing.
        rom = self.Dummy("A")
        with pytest.raises(AttributeError) as ex:
            rom._check_fom_operator_shape(A, 'A')
        assert ex.value.args[0] == "no basis 'Vr' (call fit())"

        # Try correct match but dimension 'm' is missing.
        rom = self.Dummy("B")
        rom.Vr = np.zeros((n,r))
        with pytest.raises(AttributeError) as ex:
            rom._check_fom_operator_shape(B, 'B')
        assert ex.value.args[0] == "no input dimension 'm' (call fit())"

        # Try with dimensions set, but improper shapes.
        rom = self.Dummy(self.Dummy._MODEL_KEYS)
        rom.Vr = np.zeros((n,r))
        rom.m = m

        with pytest.raises(ValueError) as ex:
            rom._check_fom_operator_shape(c[:-1], 'c')
        assert ex.value.args[0] == \
            f"c.shape = {c[:-1].shape}, must be (n,) with n = {n}"

        with pytest.raises(ValueError) as ex:
            rom._check_fom_operator_shape(A[:-1,1:], 'A')
        assert ex.value.args[0] == \
            f"A.shape = {A[:-1,1:].shape}, must be (n,n) with n = {n}"

        with pytest.raises(ValueError) as ex:
            rom._check_fom_operator_shape(H[:-1,:-1], 'H')
        assert ex.value.args[0] == \
            f"H.shape = {H[:-1,:-1].shape}, must be (n,n**2) with n = {n}"

        with pytest.raises(ValueError) as ex:
            rom._check_fom_operator_shape(G[1:], 'G')
        assert ex.value.args[0] == \
            f"G.shape = {G[1:].shape}, must be (n,n**3) with n = {n}"

        with pytest.raises(ValueError) as ex:
            rom._check_fom_operator_shape(B[1:-1], 'B')
        assert ex.value.args[0] == \
            f"B.shape = {B[1:-1].shape}, must be (n,m) with n = {n}, m = {m}"

        # Correct usage.
        for key, op in zip("cAHGB", operators):
            rom._check_fom_operator_shape(op, key)

    def test_check_operators_keys(self):
        """Test _core._intrusive._IntrusiveMixin._check_operators_keys()."""
        rom = opinf._core._intrusive._IntrusiveMixin()
        rom.modelform = "cAHB"
        v = None

        # Try with missing operator keys.
        with pytest.raises(KeyError) as ex:
            rom._check_operators_keys({"A":v, "H":v, "B":v})
        assert ex.value.args[0] == "missing operator key 'c'"

        with pytest.raises(KeyError) as ex:
            rom._check_operators_keys({"H":v, "B":v})
        assert ex.value.args[0] == "missing operator keys 'c', 'A'"

        # Try with surplus operator keys.
        with pytest.raises(KeyError) as ex:
            rom._check_operators_keys({'CC':v, "c":v, "A":v, "H":v, "B":v})
        assert ex.value.args[0] == "invalid operator key 'CC'"

        with pytest.raises(KeyError) as ex:
            rom._check_operators_keys({"c":v, "A":v, "H":v, "B":v,
                                       'CC':v, 'LL':v})
        assert ex.value.args[0] == "invalid operator keys 'CC', 'LL'"

        # Correct usage.
        rom._check_operators_keys({"c":v, "A":v, "H":v, "B":v})

    def test_process_fit_arguments(self, n=30, r=10):
        """Test _core._intrusive._IntrusiveMixin._process_fit_arguments()."""
        Vr = np.random.random((n,r))

        rom = self.Dummy("c")
        operators = {k:None for k in rom.modelform}

        # Correct usage.
        rom._process_fit_arguments(Vr, operators)
        assert rom.n == n
        assert rom.r == r
        assert rom.Vr is Vr

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

        # Initialize the test ROM.
        rom = self.Dummy("cAHGB")
        rom.Vr = Vr

        # Get test operators.
        c, A, H, G, B = _get_operators(n, m, expanded=True)
        operators = {"c":c, "A":A, "H":H, "G":G, "B":B}
        B1d = B[:,0]

        # Try to fit the ROM with operators that are misaligned with Vr.
        cbad = c[::2]
        Abad = A[:,:-2]
        Hbad = H[:,1:]
        Gbad = G[:,:-1]
        Bbad = B[1:,:]

        with pytest.raises(ValueError) as ex:
            rom._project_operators({"c":cbad, "A":A, "H":H, "G":G, "B":B})
        assert ex.value.args[0] == \
            f"c.shape = {cbad.shape}, must be (n,) with n = {n}"

        with pytest.raises(ValueError) as ex:
            rom._project_operators({"c":c, "A":Abad, "H":H, "G":G, "B":B})
        assert ex.value.args[0] == \
            f"A.shape = {Abad.shape}, must be (n,n) with n = {n}"

        with pytest.raises(ValueError) as ex:
            rom._project_operators({"c":c, "A":A, "H":Hbad, "G":G, "B":B})
        assert ex.value.args[0] == \
            f"H.shape = {Hbad.shape}, must be (n,n**2) with n = {n}"

        with pytest.raises(ValueError) as ex:
            rom._project_operators({"c":c, "A":A, "H":H, "G":Gbad, "B":B})
        assert ex.value.args[0] == \
            f"G.shape = {Gbad.shape}, must be (n,n**3) with n = {n}"

        with pytest.raises(ValueError) as ex:
            rom._project_operators({"c":c, "A":A, "H":H, "G":G, "B":Bbad})
        assert ex.value.args[0] == \
            f"B.shape = {Bbad.shape}, must be (n,m) with n = {n}, m = {m}"

        # Test each modelform.
        for form in MODEL_FORMS:
            rom = self.Dummy(form)
            rom.Vr = Vr
            ops = {key:val for key,val in operators.items() if key in form}
            rom._project_operators(ops)
            for prefix in MODEL_KEYS:
                attr = prefix+'_'
                assert hasattr(rom, prefix)
                assert hasattr(rom, attr)
                fom_op = getattr(rom, prefix)
                rom_op = getattr(rom, attr)
                if prefix in form:
                    assert fom_op is operators[prefix]
                    assert fom_op.shape == shapes[prefix]
                    assert isinstance(rom_op, np.ndarray)
                    assert rom_op.shape == shapes[attr]
                else:
                    assert fom_op is None
                    assert rom_op is None
            if "B" in form:
                assert rom.m == m
            else:
                assert rom.m == 0

        # Fit the ROM with 1D inputs (1D array for B)
        rom = self.Dummy("cAHB")
        rom.Vr = Vr
        rom._project_operators({"c":c, "A":A, "H":H, "B":B1d})
        assert rom.m == 1
        assert rom.B.shape == (n,1)
        assert rom.B_.shape == (r,1)

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
            rom = ModelClass(form)
            ops = {key:val for key,val in operators.items() if key in form}
            rom.fit(Vr, ops)
            if "B" in form:         # Also test with one-dimensional inputs.
                ops["B"] = B1d
                rom.fit(Vr, ops)


# Useable classes (public) ====================================================
class TestIntrusiveDiscreteROM:
    """Test _core._intrusive.IntrusiveDiscreteROM."""
    def test_f(self, n=5, m=2):
        """Test _core._intrusive.IntrusiveDiscreteROM.f()."""
        c, A, H, G, B = _get_operators(n, m, expanded=True)
        Vr = np.zeros((n,n//2))

        rom = opinf._core._intrusive.IntrusiveDiscreteROM("cA")
        rom.Vr = Vr
        rom.c, rom.A = c, A
        x = np.random.random(n)
        y = c + A @ x
        assert np.allclose(rom.f(x), y)
        assert np.allclose(rom.f(x, -1), y)

        rom = opinf._core._intrusive.IntrusiveDiscreteROM("HGB")
        rom.Vr = Vr
        rom.m = m
        rom.H, rom.G, rom.B = H, G, B
        u = np.random.random(m)
        x = np.random.random(n)
        x2 = np.kron(x, x)
        y = H @ x2 + G @ np.kron(x, x2) + B @ u
        assert np.allclose(rom.f(x, u), y)

    def test_fit(self):
        """Test _core._intrusive.IntrusiveDiscreteROM.fit()."""
        TestIntrusiveMixin()._test_fit(opinf.IntrusiveDiscreteROM)


class TestIntrusiveContinuousROM:
    """Test _core._intrusive.IntrusiveContinuousROM."""
    def test_f(self, n=5, m=2):
        """Test _core._intrusive.IntrusiveContinuousROM.f()."""
        c, A, H, G, B = _get_operators(n, m, expanded=True)
        Vr = np.zeros((n, n//2))

        # Check that the constructed f takes the right number of arguments.
        rom = opinf._core._intrusive.IntrusiveContinuousROM("cA")
        rom.Vr = Vr
        rom.c, rom.A = c, A
        x = np.random.random(n)
        y = c + A @ x
        assert np.allclose(rom.f(0, x), y)
        assert np.allclose(rom.f(1, x), y)
        assert np.allclose(rom.f(1, x, -1), y)

        rom = opinf._core._intrusive.IntrusiveContinuousROM("HGB")
        rom.Vr = Vr
        rom.m = m
        rom.H, rom.G, rom.B = H, G, B
        uu = np.random.random(m)
        def u(t):
            return uu + t
        x = np.random.random(n)
        x2 = np.kron(x, x)
        y = H @ x2 + G @ np.kron(x, x2) + B @ uu
        assert np.allclose(rom.f(0, x, u), y)
        y = H @ x2 + G @ np.kron(x, x2) + B @ (uu + 1)
        assert np.allclose(rom.f(1, x, u), y)

    def test_fit(self):
        """Test _core._intrusive.IntrusiveContinuousROM.fit()."""
        TestIntrusiveMixin()._test_fit(opinf.IntrusiveContinuousROM)
