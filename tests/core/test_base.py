# core/test_base.py
"""Tests for rom_operator_inference.core._base."""

import pytest
import numpy as np

import rom_operator_inference as opinf

from . import MODEL_FORMS, _get_data, _get_operators


class TestBaseROM:
    """Test core._base._BaseROM."""

    class Dummy(opinf.core._base._BaseROM):
        """Instantiable version of _BaseROM."""
        _LHS_LABEL = "dq / dt"
        _STATE_LABEL = "q(t)"
        _INPUT_LABEL = "u(t)"

        def fit(*args, **kwargs):
            pass

        def predict(*args, **kwargs):
            pass

        def save(*args, **kwargs):
            pass

        def load(*args, **kwargs):
            pass

    def test_str(self):
        """Test core._base._BaseROM.__str__() (string representation)."""
        # Continuous ROMs
        rom = self.Dummy("A")
        assert str(rom) == \
            "Reduced-order model structure: dq / dt = Aq(t)"
        rom = self.Dummy("cA")
        assert str(rom) == \
            "Reduced-order model structure: dq / dt = c + Aq(t)"
        rom = self.Dummy("HB")
        assert str(rom) == \
            "Reduced-order model structure: dq / dt = H[q(t) ⊗ q(t)] + Bu(t)"
        rom = self.Dummy("G")
        assert str(rom) == \
            "Reduced-order model structure: dq / dt = G[q(t) ⊗ q(t) ⊗ q(t)]"
        rom = self.Dummy("cH")
        assert str(rom) == \
            "Reduced-order model structure: dq / dt = c + H[q(t) ⊗ q(t)]"

        # Discrete ROMs
        self.Dummy._LHS_LABEL = "q_{j+1}"
        self.Dummy._STATE_LABEL = "q_{j}"
        self.Dummy._INPUT_LABEL = "u_{j}"
        rom = self.Dummy("A")
        assert str(rom) == \
            "Reduced-order model structure: q_{j+1} = Aq_{j}"
        rom = self.Dummy("cB")
        assert str(rom) == \
            "Reduced-order model structure: q_{j+1} = c + Bu_{j}"
        rom = self.Dummy("H")
        assert str(rom) == \
            "Reduced-order model structure: q_{j+1} = H[q_{j} ⊗ q_{j}]"

        # Steady ROMs
        self.Dummy._LHS_LABEL = "g"
        self.Dummy._STATE_LABEL = "q"
        self.Dummy._INPUT_LABEL = "u"

    def test_modelform_properties(self, n=10, r=3, m=5):
        """Test the properties related to core._base._BaseROM.modelform."""
        c_, A_, H_, G_, B_ = _get_operators(r, m)

        # Try with invalid modelform.
        with pytest.raises(ValueError) as ex:
            self.Dummy("bad_form")
        assert ex.value.args[0] == \
            "invalid modelform key 'b'; options are " \
            f"{', '.join(opinf.core._base._BaseROM._MODELFORM_KEYS)}"

        # Check initial attributes exist.
        rom = self.Dummy("BAc")
        assert hasattr(rom, "modelform")
        assert rom.modelform == "cAB"
        assert hasattr(rom, "basis")
        assert hasattr(rom, "n")
        assert hasattr(rom, "m")
        assert hasattr(rom, "r")
        assert hasattr(rom, "c_")
        assert hasattr(rom, "A_")
        assert hasattr(rom, "H_")
        assert hasattr(rom, "G_")
        assert hasattr(rom, "B_")
        assert hasattr(rom, "_projected_operators_")
        assert rom.basis is None
        assert rom.n is None
        assert rom.m is None
        assert rom.r is None
        assert rom.c_ is None
        assert rom.A_ is None
        assert rom.H_ is None
        assert rom.G_ is None
        assert rom.B_ is None
        assert rom._projected_operators_ == ""

        rom = self.Dummy("cGA")
        assert rom.modelform == "cAG"
        assert rom.m == 0
        assert rom.c_ is None
        assert rom.A_ is None
        assert rom.H_ is None
        assert rom.G_ is None
        assert rom.B_ is None

        rom = self.Dummy("BHc")
        assert rom.modelform == "cHB"
        assert rom.c_ is None
        assert rom.A_ is None
        assert rom.H_ is None
        assert rom.G_ is None
        assert rom.B_ is None

    def test_dimension_properties(self, n=20, m=3, r=7):
        """Test the properties core._base._BaseROM.(n|r|basis)."""
        rom = self.Dummy("cH")
        assert rom.n is None
        assert rom.m == 0
        assert rom.r is None
        assert rom.basis is None

        # Case 1: basis != None
        basis = np.random.random((n,r))
        rom.basis = basis
        assert rom.n == n
        assert rom.m == 0
        assert rom.r == r
        assert rom.basis is basis

        # Try setting n with basis already set.
        with pytest.raises(AttributeError) as ex:
            rom.n = n+1
        assert ex.value.args[0] == "can't set attribute (n = basis.shape[0])"

        # Try setting m with no inputs.
        with pytest.raises(AttributeError) as ex:
            rom.m = 1
        assert ex.value.args[0] == "can't set attribute ('B' not in modelform)"

        # Try setting r with basis already set.
        with pytest.raises(AttributeError) as ex:
            rom.r = r+1
        assert ex.value.args[0] == "can't set attribute (r = basis.shape[1])"

        # Case 2: basis = None
        del rom.basis
        assert rom.basis is None
        assert rom.n is None
        rom = self.Dummy("AB")
        assert rom.m is None
        rom.r = r
        rom.m = m
        rom.B_ = np.random.random((r,m))

        # Try setting r with an operator already set.
        with pytest.raises(AttributeError) as ex:
            rom.r = r+1
        assert ex.value.args[0] == "can't set attribute (call fit() to reset)"

        # Try setting m with B_ already set.
        with pytest.raises(AttributeError) as ex:
            rom.m = m+1
        assert ex.value.args[0] == "can't set attribute (m = B_.shape[1])"

        # Case 3: basis has more columns than rows
        rom._clear()
        with pytest.raises(ValueError) as ex:
            rom.basis = np.arange(12).reshape((2, 6))
        assert ex.value.args[0] == "basis must be n x r with n > r"

    def test_operator_properties(self, m=4, r=7):
        """Test the properties core._base._BaseROM.(c_|A_|H_|G_|B_)."""
        c, A, H, G, B = operators = _get_operators(r, m)

        rom = self.Dummy(self.Dummy._MODELFORM_KEYS)
        rom.r = r
        rom.m = m

        for key, op in zip("cAHGB", operators):
            name = key+'_'
            assert hasattr(rom, name)
            assert getattr(rom, name) is None
            setattr(rom, name, op)
            assert getattr(rom, name).entries is op
        rom.H_ = np.random.random((r,r**2))
        rom.G_ = np.random.random((r,r**3))

    def test_set_operators(self, n=60, m=10, r=12):
        """Test core._base._BaseROM._set_operators()."""
        basis = np.random.random((n, r))
        c, A, H, G, B = _get_operators(r, m)

        # Test correct usage.
        rom = self.Dummy("cAH")._set_operators(basis=basis, c_=c, A_=A, H_=H)
        assert isinstance(rom, self.Dummy)
        assert rom.modelform == "cAH"
        assert rom.n == n
        assert rom.r == r
        assert rom.m == 0
        assert rom.basis is basis
        assert rom.c_.entries is c
        assert rom.A_.entries is A
        assert rom.H_.entries is H
        assert rom.B_ is None
        assert rom.G_ is None

        rom = self.Dummy("GB")._set_operators(None, G_=G, B_=B)
        assert isinstance(rom, self.Dummy)
        assert rom.modelform == "GB"
        assert rom.n is None
        assert rom.r == r
        assert rom.m == m
        assert rom.basis is None
        assert rom.c_ is None
        assert rom.A_ is None
        assert rom.H_ is None
        assert rom.G_.entries is G
        assert rom.B_.entries is B

    def test_iter(self, m=2, r=6):
        """Test core._base._BaseROM.__iter__()."""
        rom = self.Dummy("cAH")
        oplist = list(rom)
        assert len(oplist) == 3
        for op in oplist:
            assert op is None

        c, A, H, _, _ = operators = _get_operators(r, m)
        rom._set_operators(None, c_=c, A_=A, H_=H)
        for romop, trueop in zip(rom, operators):
            assert romop.entries is trueop

    def test_eq(self, n=10, r=3):
        """Test core._base._BaseROM.__eq__()."""

        class Dummy2(self.Dummy):
            """Distinct copy of Dummy"""
            pass

        # Different class.
        romL = self.Dummy("A")
        romR = Dummy2("A")
        assert not (romL == romR)
        assert romL != romR

        # Different modelform.
        romR = self.Dummy("cA")
        assert romL != romR

        # No basis or operators.
        romR.modelform = "A"
        assert romL == romR

        # Different bases.
        Vr = np.random.random((n,r))
        romL.basis = Vr
        assert romL != romR
        romL.basis = None
        romR.basis = Vr
        assert romL != romR
        romL.basis = Vr
        romR.basis = 2 * Vr
        assert romL != romR

        # Same basis, no operators.
        romR.basis = Vr
        assert romL == romR

        # Different operators.
        A = np.random.random((r,r))
        romL.A_ = A
        assert romL != romR
        romR.A_ = 2 * A
        assert romL != romR

        # Same everything.
        romR.A_ = A
        assert romL == romR

    # Validation methods ------------------------------------------------------
    def test_check_operator_matches_modelform(self):
        """Test core._base._BaseROM._check_operator_matches_modelform()."""
        # Try key in modelform but operator None.
        rom = self.Dummy(self.Dummy._MODELFORM_KEYS)
        for key in rom._MODELFORM_KEYS:
            with pytest.raises(TypeError) as ex:
                rom._check_operator_matches_modelform(None, key)
            assert ex.value.args[0] == \
                f"'{key}' in modelform requires {key}_ != None"

        # Try key not in modelform but operator not None.
        rom = self.Dummy("")
        for key in rom._MODELFORM_KEYS:
            with pytest.raises(TypeError) as ex:
                rom._check_operator_matches_modelform(10, key)
            assert ex.value.args[0] == \
                f"'{key}' not in modelform requires {key}_ = None"

    def test_check_rom_operator_shape(self, m=4, r=7):
        """Test core._base._BaseROM._check_rom_operator_shape()."""
        c, A, H, G, B = operators = _get_operators(r, m)

        # Try correct match but dimension 'r' is missing.
        rom = self.Dummy("A")
        with pytest.raises(AttributeError) as ex:
            rom._check_rom_operator_shape(A, 'A')
        assert ex.value.args[0] == "no reduced dimension 'r' (call fit())"

        # Try correct match but dimension 'm' is missing.
        rom = self.Dummy("B")
        rom.r = 10
        with pytest.raises(AttributeError) as ex:
            rom._check_rom_operator_shape(B, 'B')
        assert ex.value.args[0] == "no input dimension 'm' (call fit())"

        # Try with dimensions set, but improper shapes.
        rom = self.Dummy(self.Dummy._MODELFORM_KEYS)
        rom.r, rom.m = r, m

        with pytest.raises(ValueError) as ex:
            rom._check_rom_operator_shape(c[:-1], 'c')
        assert ex.value.args[0] == \
            f"c_.shape = {c[:-1].shape}, must be (r,) with r = {r}"

        with pytest.raises(ValueError) as ex:
            rom._check_rom_operator_shape(A[:-1,1:], 'A')
        assert ex.value.args[0] == \
            f"A_.shape = {A[:-1,1:].shape}, must be (r,r) with r = {r}"

        with pytest.raises(ValueError) as ex:
            rom._check_rom_operator_shape(H[:-1,:-1], 'H')
        assert ex.value.args[0] == \
            f"H_.shape = {H[:-1,:-1].shape}, must be (r,r(r+1)/2) with r = {r}"

        with pytest.raises(ValueError) as ex:
            rom._check_rom_operator_shape(G[1:], 'G')
        assert ex.value.args[0] == \
            f"G_.shape = {G[1:].shape}, must be (r,r(r+1)(r+2)/6) with r = {r}"

        with pytest.raises(ValueError) as ex:
            rom._check_rom_operator_shape(B[1:-1], 'B')
        assert ex.value.args[0] == \
            f"B_.shape = {B[1:-1].shape}, must be (r,m) with r = {r}, m = {m}"

        # Correct usage.
        for key, op in zip("cAHGB", operators):
            rom._check_rom_operator_shape(op, key)

    def test_check_inputargs(self):
        """Test _BaseROM._check_inputargs()."""

        # Try with 'B' in modelform but without inputs.
        rom = self.Dummy("cB")
        with pytest.raises(ValueError) as ex:
            rom._check_inputargs(None, 'U')
        assert ex.value.args[0] == \
            "argument 'U' required since 'B' in modelform"

        # Try without 'B' in modelform but with inputs.
        rom = self.Dummy("cA")
        with pytest.raises(ValueError) as ex:
            rom._check_inputargs(1, 'u')
        assert ex.value.args[0] == \
            "argument 'u' invalid since 'B' in modelform"

    def test_is_trained(self, m=4, r=7):
        """Test core._base._BaseROM._check_is_trained()."""
        operators = _get_operators(r, m)
        rom = self.Dummy(self.Dummy._MODELFORM_KEYS)

        # Try without dimensions / operators set.
        with pytest.raises(AttributeError) as ex:
            rom._check_is_trained()
        assert ex.value.args[0] == "model not trained (call fit())"

        # Successful check.
        rom.r, rom.m = r, m
        rom.c_, rom.A_, rom.H_, rom.G_, rom.B_ = operators
        rom._check_is_trained()

    # Projection / reconstruction ---------------------------------------------
    def test_project_operators(self, n=7, m=5, r=3):
        """Test core._base._BaseROM._project_operators()."""
        # Get test data.
        basis = np.random.random((n,r))
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

        # Initialize the ROM and test null input.
        rom = self.Dummy("cAHGB")
        assert rom._project_operators(None) is None
        assert rom._project_operators(dict()) is None

        # Get test operators.
        c, A, H, G, B = _get_operators(n, m, expanded=True)
        operators = {"c":c, "A":A, "H":H, "G":G, "B":B}
        B1d = B[:,0]

        # Try to project without a basis.
        with pytest.raises(ValueError) as ex:
            rom._project_operators({"c": c, "A": A})
        assert ex.value.args[0] == \
            "basis required to project full-order operators"

        # Try to project with invalid operator keys.
        rom = self.Dummy("cAHG")
        rom.basis = basis
        with pytest.raises(KeyError) as ex:
            rom._project_operators({"B": B})
        assert ex.value.args[0] == "invalid operator key 'B'"

        with pytest.raises(KeyError) as ex:
            rom._project_operators({"cc": c, "aaa": A})
        assert ex.value.args[0] == "invalid operator keys 'cc', 'aaa'"

        # Try to fit the ROM with operators that are misaligned with the basis.
        cbad = c[::2]
        Abad = A[:,:-2]
        Hbad = H[:,1:]
        Gbad = G[:,:-1]
        Bbad = B[1:,:]

        rom = self.Dummy("cAHGB")
        rom.basis = basis

        with pytest.raises(ValueError) as ex:
            rom._project_operators({"c":cbad, "A":A, "H":H, "G":G, "B":B})
        assert "matmul: Input operand 1 has a mismatch" in ex.value.args[0]

        with pytest.raises(ValueError) as ex:
            rom._project_operators({"c":c, "A":Abad})
        assert "matmul: Input operand 1 has a mismatch" in ex.value.args[0]

        with pytest.raises(ValueError) as ex:
            rom._project_operators({"H":Hbad, "G":G, "B":B})
        assert "matmul: Input operand 1 has a mismatch" in ex.value.args[0]

        with pytest.raises(ValueError) as ex:
            rom._project_operators({"c":c, "A":A, "H":H, "G":Gbad, "B":B})
        assert "matmul: Input operand 1 has a mismatch" in ex.value.args[0]

        with pytest.raises(ValueError) as ex:
            rom._project_operators({"B":Bbad})
        assert "matmul: Input operand 1 has a mismatch" in ex.value.args[0]

        # Test each modelform.
        for form in MODEL_FORMS:
            rom = self.Dummy(form)
            rom.basis = basis
            ops = {key:val for key,val in operators.items() if key in form}
            rom._project_operators(ops)
            for prefix in self.Dummy._MODELFORM_KEYS:
                attr = prefix+'_'
                assert hasattr(rom, attr)
                rom_op = getattr(rom, attr)
                if prefix in form:
                    assert isinstance(rom_op.entries, np.ndarray)
                    assert rom_op.shape == shapes[attr]
                else:
                    assert rom_op is None
            if "B" in form:
                assert rom.m == m
            else:
                assert rom.m == 0

        # Special case: project input operator with 1D inputs (m = 1).
        rom = self.Dummy("cAHB")
        rom.basis = basis
        rom._project_operators({"c":c, "A":A, "H":H, "B":B1d})
        assert rom.m == 1
        assert rom.B_.shape == (r,1)

        # Special case: c = scalar
        rom = self.Dummy("cAHB")
        rom.basis = basis
        rom._project_operators({"c": 3})
        assert np.allclose(rom.c_.entries, 3*(basis.T @ np.ones(n)))

        # Special case: A = multiple of the identity
        ident = basis.T @ basis
        rom._project_operators({"A": "I"})
        assert np.allclose(rom.A_.entries, ident)
        rom._project_operators({"A": 4})
        assert np.allclose(rom.A_.entries, 4*ident)

        # TODO: Special case: H(q) = q^2

        # TODO: Special case: G(q) = q^3

    def test_project(self, n=60, k=50, r=10):
        """Test core._base._BaseROM.project()."""
        Q, Qdot, _ = _get_data(n, k, 2)
        rom = self.Dummy("c")

        # Try to project without reduced dimension r set.
        with pytest.raises(AttributeError) as ex:
            rom.project(Q, 'things')
        assert ex.value.args[0] == "reduced dimension not set"

        # Try to project with r set but with wrong shape.
        rom.r = r
        with pytest.raises(AttributeError) as ex:
            rom.project(Q, 'arg')
        assert ex.value.args[0] == "basis not set"

        # Try to project with basis set but with wrong shape.
        rom.basis = np.random.random((n, r))
        with pytest.raises(ValueError) as ex:
            rom.project(Q[:-1,:], 'state')
        assert ex.value.args[0] == "state not aligned with basis"

        # Correct usage.
        for S, label in [(Q, 'state'), (Qdot, 'ddts')]:
            S_ = rom.project(S, label)
            assert S_.shape == (r,k)
            assert np.allclose(S_, rom.basis.T @ S)
            S_ = rom.project(S[:r,:], label)
            assert S_.shape == (r,k)
            assert np.all(S_ == S[:r,:])

    def test_reconstruct(self, n=60, k=20, r=8):
        """Test core._base._BaseROM.reconstruct()."""
        Q_, Qdot_, _ = _get_data(r, k, 2)
        rom = self.Dummy("c")

        # Try to reconstruct without basis.
        rom.r = r
        with pytest.raises(AttributeError) as ex:
            rom.reconstruct(Q_, 'arg')
        assert ex.value.args[0] == "basis not set"

        # Try to project with basis set but with wrong shape.
        rom.basis = np.random.random((n, r))
        with pytest.raises(ValueError) as ex:
            rom.reconstruct(Q_[:-1,:], 'state')
        assert ex.value.args[0] == "state not aligned with basis"

        # Correct usage.
        for S_, label in [(Q_, 'state_'), (Qdot_, 'ddts_')]:
            S = rom.reconstruct(S_, label)
            assert S.shape == (n,k)
            assert np.allclose(S, rom.basis @ S_)

    # ROM evaluation ----------------------------------------------------------
    def test_evaluate(self, r=5, m=2):
        """Test core._base._BaseROM.evaluate()."""
        c_, A_, H_, G_, B_ = _get_operators(r, m)

        rom = self.Dummy("cA")
        rom.r = r
        rom.c_, rom.A_ = c_, A_
        q_ = np.random.random(r)
        y_ = c_ + A_ @ q_
        assert np.allclose(rom.evaluate(q_), y_)
        assert np.allclose(rom.evaluate(q_, -1), y_)

        kron2c, kron3c = opinf.utils.kron2c, opinf.utils.kron3c
        rom = self.Dummy("HGB")
        rom.r, rom.m = r, m
        rom.H_, rom.G_, rom.B_ = H_, G_, B_
        u = np.random.random(m)
        q_ = np.random.random(r)
        y_ = H_ @ kron2c(q_) + G_ @ kron3c(q_) + B_ @ u
        assert np.allclose(rom.evaluate(q_, u), y_)

        rom = self.Dummy("AB")
        rom.r, rom.m = r, 1
        B1d_ = B_[:,0]
        rom.A_, rom.B_ = A_, B1d_
        u = np.random.random()
        q_ = np.random.random(r)
        y_ = A_ @ q_ + (B1d_ * u)
        assert np.allclose(rom.evaluate(q_, u), y_)


# class TestBaseParametricROM:
#     """Test core._base._BaseParametricROM."""
#
#     class Dummy:
#         """Instantiable version of _BaseROM."""
#         def fit(*args, **kwargs):
#             pass
#
#         def save(*args, **kwargs):
#             pass
#
#         def load(*args, **kwargs):
#             pass
#
#     class DummyOperator(opinf.core.operators._base._BaseParametricOperator):
#         def __call__(self, parameter):
#             pass
#
#     def test_p(self):
#         raise NotImplementedError
#
#     def test_call(self):
#         raise NotImplementedError
#
#     def test_predict(self):
#         raise NotImplementedError
#
#     def test_evaluate(self):
#         raise NotImplementedError
