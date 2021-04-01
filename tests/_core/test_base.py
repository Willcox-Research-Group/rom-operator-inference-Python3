# _core/test_base.py
"""Tests for rom_operator_inference._core._base.py."""

import os
import h5py
import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as opinf

from . import MODEL_FORMS, _get_data, _get_operators, _trainedmodel


class TestBaseROM:
    """Test _core._base._BaseROM."""

    class Dummy(opinf._core._base._BaseROM):
        """Copy of _BaseROM without the abstract class instantiation error."""
        def __init__(self, modelform):
            self.modelform = modelform

    def test_init(self):
        """Test _core._base._BaseROM.__init__()."""
        with pytest.raises(TypeError) as ex:
            opinf._core._base._BaseROM()
        assert ex.value.args[0] == \
            "__init__() missing 1 required positional argument: 'modelform'"

        with pytest.raises(TypeError) as ex:
            opinf._core._base._BaseROM("cAH", False)
        assert ex.value.args[0] == \
            "__init__() takes 2 positional arguments but 3 were given"

        with pytest.raises(RuntimeError) as ex:
            opinf._core._base._BaseROM("cAH")
        assert ex.value.args[0] == \
            "abstract class instantiation (use _ContinuousROM or _DiscreteROM)"

    def test_modelform_properties(self, n=10, r=3, m=5):
        """Test the properties related to _core._base_._BaseROM.modelform."""
        c_, A_, H_, G_, B_ = _get_operators(r, m)

        # Try with invalid modelform.
        with pytest.raises(ValueError) as ex:
            self.Dummy("bad_form")
        assert ex.value.args[0] == \
            "invalid modelform key 'b'; " \
            f"options are {', '.join(opinf._core._base._BaseROM._MODEL_KEYS)}"

        # Check initial attributes exist.
        rom = self.Dummy("cAB")
        assert hasattr(rom, "modelform")
        assert hasattr(rom, "Vr")
        assert hasattr(rom, "n")
        assert hasattr(rom, "m")
        assert hasattr(rom, "r")
        assert hasattr(rom, "has_constant")
        assert hasattr(rom, "has_linear")
        assert hasattr(rom, "has_quadratic")
        assert hasattr(rom, "has_cubic")
        assert hasattr(rom, "has_inputs")
        assert hasattr(rom, "c_")
        assert hasattr(rom, "A_")
        assert hasattr(rom, "H_")
        assert hasattr(rom, "G_")
        assert hasattr(rom, "B_")
        assert rom.Vr is None
        assert rom.n is None
        assert rom.m is None
        assert rom.r is None
        assert rom.c_ is None
        assert rom.A_ is None
        assert rom.H_ is None
        assert rom.G_ is None
        assert rom.B_ is None

        rom = self.Dummy("cAG")
        assert rom.modelform == "cAG"
        assert rom.m == 0
        assert rom.has_constant is True
        assert rom.has_linear is True
        assert rom.has_quadratic is False
        assert rom.has_cubic is True
        assert rom.has_inputs is False
        assert rom.c_ is None
        assert rom.A_ is None
        assert rom.H_ is None
        assert rom.G_ is None
        assert rom.B_ is None

        rom = self.Dummy("BHc")
        assert rom.modelform == "cHB"
        assert rom.has_constant is True
        assert rom.has_linear is False
        assert rom.has_quadratic is True
        assert rom.has_cubic is False
        assert rom.has_inputs is True
        assert rom.c_ is None
        assert rom.A_ is None
        assert rom.H_ is None
        assert rom.G_ is None
        assert rom.B_ is None

    def test_dimension_properties(self, n=20, m=3, r=7):
        """Test the properties _core._base._BaseROM.(n|r|Vr)."""
        rom = self.Dummy("cH")
        assert rom.n is None
        assert rom.m == 0
        assert rom.r is None
        assert rom.Vr is None

        # Case 1: Vr != None
        Vr = np.random.random((n,r))
        rom.Vr = Vr
        assert rom.n == n
        assert rom.m == 0
        assert rom.r == r
        assert rom.Vr is Vr

        # Try setting n with Vr already set.
        with pytest.raises(AttributeError) as ex:
            rom.n = n+1
        assert ex.value.args[0] == "can't set attribute (n = Vr.shape[0])"

        # Try setting m with no inputs.
        with pytest.raises(AttributeError) as ex:
            rom.m = 1
        assert ex.value.args[0] == "can't set attribute ('B' not in modelform)"

        # Try setting r with Vr already set.
        with pytest.raises(AttributeError) as ex:
            rom.r = r+1
        assert ex.value.args[0] == "can't set attribute (r = Vr.shape[1])"

        # Case 2: Vr = None
        del rom.Vr
        assert rom.Vr is None
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

    def test_operator_properties(self, m=4, r=7):
        """Test the properties _core._base._BaseROM.(c_|A_|H_|G_|B_)."""
        c, A, H, G, B = operators = _get_operators(r, m)

        rom = self.Dummy(self.Dummy._MODEL_KEYS)
        rom.r = r
        rom.m = m

        for key, op in zip("cAHGB", operators):
            name = key+'_'
            assert hasattr(rom, name)
            assert getattr(rom, name) is None
            setattr(rom, name, op)
            assert getattr(rom, name) is op
        rom.H_ = np.random.random((r,r**2))
        rom.G_ = np.random.random((r,r**3))

    def test_check_operator_matches_modelform(self):
        """Test _core._base._BaseROM._check_operator_matches_modelform()."""
        # Try key in modelform but operator None.
        rom = self.Dummy(self.Dummy._MODEL_KEYS)
        for key in rom._MODEL_KEYS:
            with pytest.raises(TypeError) as ex:
                rom._check_operator_matches_modelform(None, key)
            assert ex.value.args[0] == \
                f"'{key}' in modelform requires {key}_ != None"

        # Try key not in modelform but operator not None.
        rom = self.Dummy("")
        for key in rom._MODEL_KEYS:
            with pytest.raises(TypeError) as ex:
                rom._check_operator_matches_modelform(10, key)
            assert ex.value.args[0] == \
                f"'{key}' not in modelform requires {key}_ = None"

    def test_check_rom_operator_shape(self, m=4, r=7):
        """Test _core._base._BaseROM._check_rom_operator_shape()."""
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
        rom = self.Dummy(self.Dummy._MODEL_KEYS)
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

        # Try with has_inputs = True but without inputs.
        rom = self.Dummy("cB")
        with pytest.raises(ValueError) as ex:
            rom._check_inputargs(None, 'U')
        assert ex.value.args[0] == \
            "argument 'U' required since 'B' in modelform"

        # Try with has_inputs = False but with inputs.
        rom = self.Dummy("cA")
        with pytest.raises(ValueError) as ex:
            rom._check_inputargs(1, 'u')
        assert ex.value.args[0] == \
            "argument 'u' invalid since 'B' in modelform"

    def test_is_trained(self, m=4, r=7):
        """Test _core._base._BaseROM._check_is_trained()."""
        operators = _get_operators(r, m)
        rom = self.Dummy(self.Dummy._MODEL_KEYS)

        # Try without dimensions / operators set.
        with pytest.raises(AttributeError) as ex:
            rom._check_is_trained()
        assert ex.value.args[0] == "model not trained (call fit())"

        # Successful check.
        rom.r, rom.m = r, m
        rom.c_, rom.A_, rom.H_, rom.G_, rom.B_ = operators
        rom._check_is_trained()

    def test_set_operators(self, n=60, m=10, r=12):
        """Test _core._base._BaseROM.set_operators()."""
        Vr = np.random.random((n, r))
        c, A, H, G, B = _get_operators(r, m)

        # Test correct usage.
        rom = self.Dummy("cAH").set_operators(Vr=Vr, c_=c, A_=A, H_=H)
        assert isinstance(rom, self.Dummy)
        assert rom.modelform == "cAH"
        assert rom.n == n
        assert rom.r == r
        assert rom.m == 0
        assert rom.Vr is Vr
        assert rom.c_ is c
        assert rom.A_ is A
        assert rom.H_ is H
        assert rom.B_ is None
        assert rom.G_ is None

        rom = self.Dummy("GB").set_operators(None, G_=G, B_=B)
        assert isinstance(rom, self.Dummy)
        assert rom.modelform == "GB"
        assert rom.n is None
        assert rom.r == r
        assert rom.m == m
        assert rom.Vr is None
        assert rom.c_ is None
        assert rom.A_ is None
        assert rom.H_ is None
        assert rom.G_ is G
        assert rom.B_ is B

    def test_project(self, n=60, k=50, r=10):
        """Test _core._base._BaseROM.project()."""
        X, Xdot, _ = _get_data(n, k, 2)
        rom = self.Dummy("c")
        rom.Vr = la.svd(X)[0][:,:r]

        with pytest.raises(ValueError) as ex:
            rom.project(X[:-1,:], 'X')
        assert ex.value.args[0] == "X not aligned with Vr, dimension 0"

        for S, label in [(X, 'X'), (Xdot, 'Xdot')]:
            S_ = rom.project(S, label)
            assert S_.shape == (r,k)
            S_ = rom.project(rom.Vr.T @ S, label)
            assert S_.shape == (r,k)

    def test_fit(self):
        """Test _core._base._BaseROM.fit()."""
        rom = self.Dummy("A")
        with pytest.raises(NotImplementedError) as ex:
            rom.fit()
        assert ex.value.args[0] == "fit() implemented by child classes"

        with pytest.raises(NotImplementedError) as ex:
            rom.fit(1, 2, 3, 4, 5, 6, 7, a=8)
        assert ex.value.args[0] == "fit() implemented by child classes"

    def test_predict(self):
        """Test _core._base._BaseROM.fit()."""
        rom = self.Dummy("A")
        with pytest.raises(NotImplementedError) as ex:
            rom.predict()
        assert ex.value.args[0] == "predict() implemented by child classes"

        with pytest.raises(NotImplementedError) as ex:
            rom.predict(1, 2, 3, 4, 5, 6, 7, a=8)
        assert ex.value.args[0] == "predict() implemented by child classes"


class TestDiscreteROM:
    """Test _core._base._DiscreteROM."""
    def test_f_(self, r=5, m=2):
        """Test _core._base.DiscreteROM.f_()."""
        c_, A_, H_, G_, B_ = _get_operators(r, m)

        rom = opinf._core._base._DiscreteROM("cA")
        rom.r = r
        rom.c_, rom.A_ = c_, A_
        x_ = np.random.random(r)
        y_ = c_ + A_ @ x_
        assert np.allclose(rom.f_(x_), y_)
        assert np.allclose(rom.f_(x_, -1), y_)

        kron2c, kron3c = opinf.utils.kron2c, opinf.utils.kron3c
        rom = opinf._core._base._DiscreteROM("HGB")
        rom.r, rom.m = r, m
        rom.H_, rom.G_, rom.B_ = H_, G_, B_
        u = np.random.random(m)
        x_ = np.random.random(r)
        y_ = H_ @ kron2c(x_) + G_ @ kron3c(x_) + B_ @ u
        assert np.allclose(rom.f_(x_, u), y_)

    def test_predict(self):
        """Test _core._base._DiscreteROM.predict()."""
        rom = opinf._core._base._DiscreteROM('')

        # Get test data.
        n, k, m, r = 60, 50, 20, 10
        X = _get_data(n, k, m)[0]
        Vr = la.svd(X)[0][:,:r]

        niters = 5
        x0 = X[:,0]
        U = np.ones((m, niters-1))

        # Try to predict with invalid initial condition.
        x0_ = Vr.T @ x0
        rom = _trainedmodel(False, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(x0_[:-1], niters, U)
        assert ex.value.args[0] == "x0 not aligned with Vr, dimension 0"

        # Try to predict with bad niters argument.
        with pytest.raises(ValueError) as ex:
            rom.predict(x0, -18, U)
        assert ex.value.args[0] == \
            "argument 'niters' must be a nonnegative integer"

        # Try to predict with badly-shaped discrete inputs.
        rom = _trainedmodel(False, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(x0, niters, np.random.random((m-1, niters-1)))
        assert ex.value.args[0] == \
            f"invalid input shape ({(m-1,niters-1)} != {(m,niters-1)}"

        rom = _trainedmodel(False, "cAHB", Vr, m=1)
        with pytest.raises(ValueError) as ex:
            rom.predict(x0, niters, np.random.random((2, niters-1)))
        assert ex.value.args[0] == \
            f"invalid input shape ({(2,niters-1)} != {(1,niters-1)}"

        # Try to predict with continuous inputs.
        rom = _trainedmodel(False, "cAHB", Vr, m)
        with pytest.raises(TypeError) as ex:
            rom.predict(x0, niters, lambda t: np.ones(m-1))
        assert ex.value.args[0] == "input U must be an array, not a callable"

        for form in MODEL_FORMS:
            if "B" not in form:             # No control inputs.
                rom = _trainedmodel(False, form, Vr, None)
                out = rom.predict(x0, niters)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,niters)
            else:                           # Has Control inputs.
                # Predict with 2D inputs.
                rom = _trainedmodel(False, form, Vr, m)
                out = rom.predict(x0, niters, U)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,niters)

                # Predict with 1D inputs.
                rom = _trainedmodel(False, form, Vr, 1)
                out = rom.predict(x0, niters, np.ones(niters))
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,niters)

        # Predict with no basis gives result in low-dimensional space.
        rom = _trainedmodel(False, "cA", Vr, None)
        rom.Vr = None
        out = rom.predict(Vr.T @ x0, niters)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r,niters)


class TestContinuousROM:
    """Test _core._base._ContinuousROM."""
    def test_f_(self, r=5, m=2):
        """Test _core._base.ContinuousROM.f_()."""
        c_, A_, H_, G_, B_ = _get_operators(r, m)

        # Check that the constructed f takes the right number of arguments.
        rom = opinf._core._base._ContinuousROM("cA")
        rom.r = r
        rom.c_, rom.A_ = c_, A_
        x_ = np.random.random(r)
        y_ = c_ + A_ @ x_
        assert np.allclose(rom.f_(0, x_), y_)
        assert np.allclose(rom.f_(1, x_), y_)
        assert np.allclose(rom.f_(1, x_, -1), y_)

        kron2c, kron3c = opinf.utils.kron2c, opinf.utils.kron3c
        rom = opinf._core._base._ContinuousROM("HGB")
        rom.r, rom.m = r, m
        rom.H_, rom.G_, rom.B_ = H_, G_, B_
        uu = np.random.random(m)
        def u(t):
            return uu + t
        y_ = H_ @ kron2c(x_) + G_ @ kron3c(x_) + B_ @ uu
        assert np.allclose(rom.f_(0, x_, u), y_)
        y_ = H_ @ kron2c(x_) + G_ @ kron3c(x_) + B_ @ (uu+1)
        assert np.allclose(rom.f_(1, x_, u), y_)

    def test_predict(self):
        """Test _core._base._ContinuousROM.predict()."""
        # Get test data.
        n, k, m, r = 60, 50, 20, 10
        X = _get_data(n, k, m)[0]
        Vr = la.svd(X)[0][:,:r]

        nt = 5
        x0 = X[:,0]
        t = np.linspace(0, .01*nt, nt)
        def u(t):
            return np.ones(m)
        Upred = np.ones((m, nt))

        # Try to predict with invalid initial condition.
        x0_ = Vr.T @ x0
        rom = _trainedmodel(True, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(x0_[1:], t, u)
        assert ex.value.args[0] == "x0 not aligned with Vr, dimension 0"

        # Try to predict with bad time array.
        with pytest.raises(ValueError) as ex:
            rom.predict(x0, np.vstack((t,t)), u)
        assert ex.value.args[0] == "time 't' must be one-dimensional"

        # Predict without inputs.
        for form in MODEL_FORMS:
            if "B" not in form:
                rom = _trainedmodel(True, form, Vr, None)
                out = rom.predict(x0, t)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,t.size)

        # Predict with no basis gives result in low-dimensional space.
        rom = _trainedmodel(True, "cA", Vr, None)
        rom.Vr = None
        out = rom.predict(Vr.T @ x0, t)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r,t.size)

        # Try to predict with badly-shaped discrete inputs.
        rom = _trainedmodel(True, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(x0, t, np.random.random((m-1, nt)))
        assert ex.value.args[0] == \
            f"invalid input shape ({(m-1,nt)} != {(m,nt)}"

        rom = _trainedmodel(True, "cAHB", Vr, m=1)
        with pytest.raises(ValueError) as ex:
            rom.predict(x0, t, np.random.random((2, nt)))
        assert ex.value.args[0] == \
            f"invalid input shape ({(2,nt)} != {(1,nt)}"

        # Try to predict with badly-shaped continuous inputs.
        rom = _trainedmodel(True, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(x0, t, lambda t: np.ones(m-1))
        assert ex.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(m,)}"
        with pytest.raises(ValueError) as ex:
            rom.predict(x0, t, lambda t: 1)
        assert ex.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(m,)}"

        rom = _trainedmodel(True, "cAHB", Vr, m=1)
        with pytest.raises(ValueError) as ex:
            rom.predict(x0, t, u)
        assert ex.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(1,)}" \
            " or scalar"

        # Try to predict with continuous inputs with bad return type
        rom = _trainedmodel(True, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(x0, t, lambda t: set([5]))
        assert ex.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(m,)}"

        for form in MODEL_FORMS:
            if "B" in form:
                # Predict with 2D inputs.
                rom = _trainedmodel(True, form, Vr, m)
                # continuous input.
                out = rom.predict(x0, t, u)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,nt)
                # discrete input.
                out = rom.predict(x0, t, Upred)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,nt)

                # Predict with 1D inputs.
                rom = _trainedmodel(True, form, Vr, 1)
                # continuous input.
                out = rom.predict(x0, t, lambda t: 1)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,nt)
                out = rom.predict(x0, t, lambda t: np.array([1]))
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,nt)
                # discrete input.
                out = rom.predict(x0, t, np.ones_like(t))
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,nt)


class TestNonparametricMixin:
    """Test _core._base._NonparametricMixin."""

    class Dummy(opinf._core._base._NonparametricMixin,
                opinf._core._base._BaseROM):
        def __init__(self, modelform):
            self.modelform = modelform

    def test_O_(self, r=9, m=4):
        """Test _core._base._NonparametricMixin.O_."""
        c, A, H, G, B = _get_operators(r, m)
        for form in MODEL_FORMS:
            rom = self.Dummy(form)
            rom.set_operators(None,
                              c_=c if 'c' in form else None,
                              A_=A if 'A' in form else None,
                              H_=H if 'H' in form else None,
                              G_=G if 'G' in form else None,
                              B_=B if 'B' in form else None)
            O_ = rom.O_
            d = opinf.lstsq.lstsq_size(form, r, m if 'B' in form else 0)
            assert O_.shape == (r,d)

            # Spot check.
            if form == "cB":
                assert np.all(O_ == np.hstack((c[:,np.newaxis], B)))
            elif form == "AB":
                assert np.all(O_ == np.hstack((A, B)))
            elif form == "HG":
                assert np.all(O_ == np.hstack((H, G)))

    def test_str(self):
        """Test _core._base._NonparametricMixin.__str__()
        (string representation).
        """
        # Continuous ROMs
        rom = opinf.InferredContinuousROM("A")
        assert str(rom) == \
            "Reduced-order model structure: dx / dt = Ax(t)"
        rom = opinf.InferredContinuousROM("cA")
        assert str(rom) == \
            "Reduced-order model structure: dx / dt = c + Ax(t)"
        rom = opinf.InferredContinuousROM("HB")
        assert str(rom) == \
            "Reduced-order model structure: dx / dt = H(x(t) ⊗ x(t)) + Bu(t)"
        rom = opinf.InferredContinuousROM("G")
        assert str(rom) == \
            "Reduced-order model structure: dx / dt = G(x(t) ⊗ x(t) ⊗ x(t))"
        rom = opinf.InferredContinuousROM("cH")
        assert str(rom) == \
            "Reduced-order model structure: dx / dt = c + H(x(t) ⊗ x(t))"

        # Discrete ROMs
        rom = opinf.IntrusiveDiscreteROM("A")
        assert str(rom) == \
            "Reduced-order model structure: x_{j+1} = Ax_{j}"
        rom = opinf.IntrusiveDiscreteROM("cB")
        assert str(rom) == \
            "Reduced-order model structure: x_{j+1} = c + Bu_{j}"
        rom = opinf.IntrusiveDiscreteROM("H")
        assert str(rom) == \
            "Reduced-order model structure: x_{j+1} = H(x_{j} ⊗ x_{j})"

    def test_save_model(self):
        """Test _core._base._NonparametricMixin.save_model()."""
        # Clean up after old tests.
        target = "savemodeltest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Get a test model.
        n, m, r = 15, 2, 5
        Vr = np.random.random((n,r))
        rom = _trainedmodel("inferred", "cAHGB", Vr, m)

        def _checkfile(filename, mdl, hasbasis):
            assert os.path.isfile(filename)
            with h5py.File(filename, 'r') as data:
                # Check metadata.
                assert "meta" in data
                assert len(data["meta"]) == 0
                assert data["meta"].attrs["modelclass"] == \
                    mdl.__class__.__name__
                assert data["meta"].attrs["modelform"] == mdl.modelform

                # Check basis
                if hasbasis:
                    assert "Vr" in data
                    assert np.allclose(data["Vr"], Vr)

                # Check operators
                assert "operators" in data
                if "c" in mdl.modelform:
                    assert np.allclose(data["operators/c_"], mdl.c_)
                else:
                    assert "c_" not in data["operators"]
                if "A" in mdl.modelform:
                    assert np.allclose(data["operators/A_"], mdl.A_)
                else:
                    assert "A_" not in data["operators"]
                if "H" in mdl.modelform:
                    assert np.allclose(data["operators/H_"], mdl.H_)
                else:
                    assert "H_" not in data["operators"]
                if "G" in mdl.modelform:
                    assert np.allclose(data["operators/G_"], mdl.G_)
                else:
                    assert "G_" not in data["operators"]
                if "B" in mdl.modelform:
                    assert np.allclose(data["operators/B_"], mdl.B_)
                else:
                    assert "B_" not in data["operators"]

        rom.save_model(target[:-3], save_basis=False)
        _checkfile(target, rom, False)

        with pytest.raises(FileExistsError) as ex:
            rom.save_model(target, overwrite=False)
        assert ex.value.args[0] == target

        rom.save_model(target, save_basis=True, overwrite=True)
        _checkfile(target, rom, True)

        rom = _trainedmodel("inferred", "c", Vr, 0)
        rom.save_model(target, overwrite=True)
        _checkfile(target, rom, True)

        rom = _trainedmodel("inferred", "AB", Vr, m)
        rom.Vr = None
        rom.save_model(target, save_basis=True, overwrite=True)
        _checkfile(target, rom, False)

        # Check that save_model() and load_model() are inverses.
        rom.Vr = Vr
        rom.save_model(target, save_basis=True, overwrite=True)
        rom2 = opinf.load_model(target)
        for attr in ["n", "m", "r", "modelform", "__class__"]:
            assert getattr(rom, attr) == getattr(rom2, attr)
        for attr in ["A_", "B_", "Vr"]:
            assert np.allclose(getattr(rom, attr), getattr(rom2, attr))
        for attr in ["c_", "H_", "G_"]:
            assert getattr(rom, attr) is getattr(rom2, attr) is None

        # Check Vr = None functionality.
        rom.Vr = None
        rom.save_model(target, overwrite=True)
        rom2 = opinf.load_model(target)
        for attr in ["m", "r", "modelform", "__class__"]:
            assert getattr(rom, attr) == getattr(rom2, attr)
        for attr in ["A_", "B_",]:
            assert np.allclose(getattr(rom, attr), getattr(rom2, attr))
        for attr in ["n", "c_", "H_", "G_", "Vr"]:
            assert getattr(rom, attr) is getattr(rom2, attr) is None

        os.remove(target)


class TestParametricMixin:
    """Test _core._base._ParametricMixin."""

    class DummyDiscrete(opinf._core._base._ParametricMixin,
                        opinf._core._base._DiscreteROM):
        pass

    class DummyContinuous(opinf._core._base._ParametricMixin,
                          opinf._core._base._ContinuousROM):
        pass

    def test_call(self, r=10, m=3):
        """Test _core._base._ParametricMixin.__call__()."""
        # Define dummy operators to use.
        c1, A1, H1, G1, B1 = _get_operators(r, m)
        c2, A2, H2, G2, B2 = _get_operators(r, m)
        def c(*args, **kwargs):
            return c1
        def A(*args, **kwargs):
            return A1
        def H(*args, **kwargs):
            return H1
        def G(*args, **kwargs):
            return G1
        def B(*args, **kwargs):
            return B1
        c.shape = (r,)
        A.shape = (r,r)
        H.shape = (r,r*(r + 1)//2)
        G.shape = (r,r*(r + 1)*(r + 2)//6)
        B.shape = (r,m)

        rom = self.DummyDiscrete("cAH")
        rom.r = r
        rom.c_, rom.A_, rom.H_ = c2, A, H
        newrom = rom(1)
        assert isinstance(newrom,
                          opinf._core._base._DiscreteParametricEvaluationROM)
        assert newrom.c_ is c2
        assert newrom.A_ is A1
        assert newrom.H_ is H1

        rom.c_, rom.A_, rom.H_ = c, A2, H2
        newrom = rom(2)
        assert isinstance(newrom,
                          opinf._core._base._DiscreteParametricEvaluationROM)
        assert newrom.c_ is c1
        assert newrom.A_ is A2
        assert newrom.H_ is H2

        rom = self.DummyContinuous("GB")
        rom.r, rom.m = r, m
        rom.G_, rom.B_ = G2, B
        newrom = rom(3)
        assert isinstance(newrom,
                          opinf._core._base._ContinuousParametricEvaluationROM)
        assert newrom.G_ is G2
        assert newrom.B_ is B1

        rom.G_, rom.B_ = G, B2
        newrom = rom(3)
        assert isinstance(newrom,
                          opinf._core._base._ContinuousParametricEvaluationROM)
        assert newrom.G_ is G1
        assert newrom.B_ is B2

        badrom = opinf._core._base._ParametricMixin()
        with pytest.raises(RuntimeError) as ex:
            badrom(10)
        assert len(ex.value.args) == 0

    def test_str(self, r=10, m=3):
        """Test _core._base._ParametricMixin.__str__()."""

        # Define dummy operators to use.
        def c(*args, **kwargs):
            pass
        def A(*args, **kwargs):
            pass
        def H(*args, **kwargs):
            pass
        def G(*args, **kwargs):
            pass
        def B(*args, **kwargs):
            pass
        c.shape = (r,)
        A.shape = (r,r)
        H.shape = (r,r*(r + 1)//2)
        G.shape = (r,r*(r + 1)*(r + 2)//6)
        B.shape = (r,m)

        # Continuous ROMs
        rom = self.DummyContinuous("A")
        assert str(rom) == \
            "Reduced-order model structure: dx / dt = Ax(t)"

        rom.r = r
        rom.A_ = A
        assert str(rom) == \
            "Reduced-order model structure: dx / dt = A(µ)x(t)"

        rom = self.DummyContinuous("cA")
        rom.r = r
        rom.c_, rom.A_ = c, A
        assert str(rom) == \
            "Reduced-order model structure: dx / dt = c(µ) + A(µ)x(t)"

        rom = self.DummyContinuous("HB")
        rom.r, rom.m = r, m
        assert str(rom) == \
            "Reduced-order model structure: dx / dt = H(x(t) ⊗ x(t)) + Bu(t)"
        rom = self.DummyContinuous("G")
        rom.r = r
        rom.G_ = G
        assert str(rom) == \
            "Reduced-order model structure: dx / dt = G(µ)(x(t) ⊗ x(t) ⊗ x(t))"

        # Discrete ROMs
        rom = self.DummyDiscrete("cH")
        assert str(rom) == \
            "Reduced-order model structure: x_{j+1} = c + H(x_{j} ⊗ x_{j})"
        rom.r = r
        rom.c_ = c
        assert str(rom) == \
            "Reduced-order model structure: x_{j+1} = c(µ) + H(x_{j} ⊗ x_{j})"
