# _core/test_base.py
"""Tests for rom_operator_inference._core._base.py."""

import os
import h5py
import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi

from . import (MODEL_KEYS, MODEL_FORMS,
               _get_data, _get_operators, _trainedmodel)


class TestBaseROM:
    """Test _core._base._BaseROM."""

    class Dummy(roi._core._base._BaseROM):
        """Copy of _BaseROM without the abstract class instantiation error."""
        def __init__(self, modelform):
            self.modelform = modelform

    def test_init(self):
        """Test _core._base._BaseROM.__init__()."""
        with pytest.raises(TypeError) as ex:
            roi._core._base._BaseROM()
        assert ex.value.args[0] == \
            "__init__() missing 1 required positional argument: 'modelform'"

        with pytest.raises(TypeError) as ex:
            roi._core._base._BaseROM("cAH", False)
        assert ex.value.args[0] == \
            "__init__() takes 2 positional arguments but 3 were given"

        with pytest.raises(RuntimeError) as ex:
            roi._core._base._BaseROM("cAH")
        assert ex.value.args[0] == \
            "abstract class instantiation (use _ContinuousROM or _DiscreteROM)"

    def test_modelform_properties(self, n=10, r=3, m=5):
        """Test the properties related to _core._base_._BaseROM.modelform."""
        c_, A_, H_, G_, B_ = _get_operators(r, m)
        Vr = np.random.random((n,r))

        # Try with invalid modelform.
        with pytest.raises(ValueError) as ex:
            self.Dummy("bad_form")
        assert ex.value.args[0] == \
            "invalid modelform key 'b'; " \
            f"options are {', '.join(roi._core._base._BaseROM._MODEL_KEYS)}"

        # Check initial attributes exist.
        model = self.Dummy("cAB")
        assert hasattr(model, "modelform")
        assert hasattr(model, "Vr")
        assert hasattr(model, "n")
        assert hasattr(model, "m")
        assert hasattr(model, "r")
        assert hasattr(model, "has_constant")
        assert hasattr(model, "has_linear")
        assert hasattr(model, "has_quadratic")
        assert hasattr(model, "has_cubic")
        assert hasattr(model, "has_inputs")
        assert hasattr(model, "c_")
        assert hasattr(model, "A_")
        assert hasattr(model, "H_")
        assert hasattr(model, "G_")
        assert hasattr(model, "B_")
        assert model.Vr is None
        assert model.n is None
        assert model.m is None
        assert model.r is None
        assert model.c_ is None
        assert model.A_ is None
        assert model.H_ is None
        assert model.G_ is None
        assert model.B_ is None

        model = self.Dummy("cAG")
        assert model.modelform == "cAG"
        assert model.m == 0
        assert model.has_constant is True
        assert model.has_linear is True
        assert model.has_quadratic is False
        assert model.has_cubic is True
        assert model.has_inputs is False
        assert model.c_ is None
        assert model.A_ is None
        assert model.H_ is None
        assert model.G_ is None
        assert model.B_ is None

        model = self.Dummy("BHc")
        assert model.modelform == "cHB"
        assert model.has_constant is True
        assert model.has_linear is False
        assert model.has_quadratic is True
        assert model.has_cubic is False
        assert model.has_inputs is True
        assert model.c_ is None
        assert model.A_ is None
        assert model.H_ is None
        assert model.G_ is None
        assert model.B_ is None

    def test_dimension_properties(self, n=20, m=3, r=7):
        """Test the properties _core._base._BaseROM.(n|r|Vr)."""
        model = self.Dummy("cH")
        assert model.n is None
        assert model.m == 0
        assert model.r is None
        assert model.Vr is None

        # Case 1: Vr != None
        Vr = np.random.random((n,r))
        model.Vr = Vr
        assert model.n == n
        assert model.m == 0
        assert model.r == r
        assert model.Vr is Vr

        # Try setting n with Vr already set.
        with pytest.raises(AttributeError) as ex:
            model.n = n+1
        assert ex.value.args[0] == "can't set attribute (n = Vr.shape[0])"

        # Try setting m with no inputs.
        with pytest.raises(AttributeError) as ex:
            model.m = 1
        assert ex.value.args[0] == "can't set attribute ('B' not in modelform)"

        # Try setting r with Vr already set.
        with pytest.raises(AttributeError) as ex:
            model.r = r+1
        assert ex.value.args[0] == "can't set attribute (r = Vr.shape[1])"

        # Case 2: Vr = None
        del model.Vr
        assert model.Vr is None
        assert model.n is None
        model = self.Dummy("AB")
        assert model.m is None
        model.r = r
        model.m = m
        model.B_ = np.random.random((r,m))

        # Try setting r with an operator already set.
        with pytest.raises(AttributeError) as ex:
            model.r = r+1
        assert ex.value.args[0] == "can't set attribute (call fit() to reset)"

        # Try setting m with B_ already set.
        with pytest.raises(AttributeError) as ex:
            model.m = m+1
        assert ex.value.args[0] == "can't set attribute (m = B_.shape[1])"

    def test_operator_properties(self, m=4, r=7):
        """Test the properties _core._base._BaseROM.(c_|A_|H_|G_|B_)."""
        c, A, H, G, B = operators = _get_operators(r, m)

        model = self.Dummy(self.Dummy._MODEL_KEYS)
        model.r = r
        model.m = m

        for key, op in zip("cAHGB", operators):
            name = key+'_'
            assert hasattr(model, name)
            assert getattr(model, name) is None
            setattr(model, name, op)
            assert getattr(model, name) is op
        model.H_ = np.random.random((r,r**2))
        model.G_ = np.random.random((r,r**3))

    def test_check_operator_matches_modelform(self):
        """Test _core._base._BaseROM._check_operator_matches_modelform()."""
        # Try key in modelform but operator None.
        model = self.Dummy(self.Dummy._MODEL_KEYS)
        for key in model._MODEL_KEYS:
            with pytest.raises(TypeError) as ex:
                model._check_operator_matches_modelform(None, key)
            assert ex.value.args[0] == \
                f"'{key}' in modelform requires {key}_ != None"

        # Try key not in modelform but operator not None.
        model = self.Dummy("")
        for key in model._MODEL_KEYS:
            with pytest.raises(TypeError) as ex:
                model._check_operator_matches_modelform(10, key)
            assert ex.value.args[0] == \
                f"'{key}' not in modelform requires {key}_ = None"

    def test_check_rom_operator_shape(self, m=4, r=7):
        """Test _core._base._BaseROM._check_rom_operator_shape()."""
        c, A, H, G, B = operators = _get_operators(r, m)

        # Try correct match but dimension 'r' is missing.
        model = self.Dummy("A")
        with pytest.raises(AttributeError) as ex:
            model._check_rom_operator_shape(A, 'A')
        assert ex.value.args[0] == "no reduced dimension 'r' (call fit())"

        # Try correct match but dimension 'm' is missing.
        model = self.Dummy("B")
        model.r = 10
        with pytest.raises(AttributeError) as ex:
            model._check_rom_operator_shape(B, 'B')
        assert ex.value.args[0] == "no input dimension 'm' (call fit())"

        # Try with dimensions set, but improper shapes.
        model = self.Dummy(self.Dummy._MODEL_KEYS)
        model.r, model.m = r, m

        with pytest.raises(ValueError) as ex:
            model._check_rom_operator_shape(c[:-1], 'c')
        assert ex.value.args[0] == \
            f"c_.shape = {c[:-1].shape}, must be (r,) with r = {r}"

        with pytest.raises(ValueError) as ex:
            model._check_rom_operator_shape(A[:-1,1:], 'A')
        assert ex.value.args[0] == \
            f"A_.shape = {A[:-1,1:].shape}, must be (r,r) with r = {r}"

        with pytest.raises(ValueError) as ex:
            model._check_rom_operator_shape(H[:-1,:-1], 'H')
        assert ex.value.args[0] == \
            f"H_.shape = {H[:-1,:-1].shape}, must be (r,r(r+1)/2) with r = {r}"

        with pytest.raises(ValueError) as ex:
            model._check_rom_operator_shape(G[1:], 'G')
        assert ex.value.args[0] == \
            f"G_.shape = {G[1:].shape}, must be (r,r(r+1)(r+2)/6) with r = {r}"

        with pytest.raises(ValueError) as ex:
            model._check_rom_operator_shape(B[1:-1], 'B')
        assert ex.value.args[0] == \
            f"B_.shape = {B[1:-1].shape}, must be (r,m) with r = {r}, m = {m}"

        # Correct usage.
        for key, op in zip("cAHGB", operators):
            model._check_rom_operator_shape(op, key)

    def test_check_inputargs(self):
        """Test _BaseROM._check_inputargs()."""

        # Try with has_inputs = True but without inputs.
        model = self.Dummy("cB")
        with pytest.raises(ValueError) as ex:
            model._check_inputargs(None, 'U')
        assert ex.value.args[0] == \
            "argument 'U' required since 'B' in modelform"

        # Try with has_inputs = False but with inputs.
        model = self.Dummy("cA")
        with pytest.raises(ValueError) as ex:
            model._check_inputargs(1, 'u')
        assert ex.value.args[0] == \
            "argument 'u' invalid since 'B' in modelform"

    def test_is_trained(self, m=4, r=7):
        """Test _core._base._BaseROM._check_is_trained()."""
        operators = _get_operators(r, m)
        model = self.Dummy(self.Dummy._MODEL_KEYS)

        # Try without dimensions / operators set.
        with pytest.raises(AttributeError) as ex:
            model._check_is_trained()
        assert ex.value.args[0] == "model not trained (call fit())"

        # Successful check.
        model.r, model.m = r, m
        model.c_, model.A_, model.H_, model.G_, model.B_ = operators
        model._check_is_trained()

    def test_set_operators(self, n=60, m=10, r=12):
        """Test _core._base._BaseROM.set_operators()."""
        Vr = np.random.random((n, r))
        c, A, H, G, B = _get_operators(r, m)

        # Test correct usage.
        model = self.Dummy("cAH").set_operators(Vr=Vr, c_=c, A_=A, H_=H)
        assert isinstance(model, self.Dummy)
        assert model.modelform == "cAH"
        assert model.n == n
        assert model.r == r
        assert model.m == 0
        assert model.Vr is Vr
        assert model.c_ is c
        assert model.A_ is A
        assert model.H_ is H
        assert model.B_ is None
        assert model.G_ is None

        model = self.Dummy("GB").set_operators(None, G_=G, B_=B)
        assert isinstance(model, self.Dummy)
        assert model.modelform == "GB"
        assert model.n is None
        assert model.r == r
        assert model.m == m
        assert model.Vr is None
        assert model.c_ is None
        assert model.A_ is None
        assert model.H_ is None
        assert model.G_ is G
        assert model.B_ is B

    def test_project(self, n=60, k=50, r=10):
        """Test _core._base._BaseROM.project()."""
        X, Xdot, _ = _get_data(n, k, 2)
        model = self.Dummy("c")
        model.Vr = la.svd(X)[0][:,:r]

        with pytest.raises(ValueError) as ex:
            model.project(X[:-1,:], 'X')
        assert ex.value.args[0] == "X not aligned with Vr, dimension 0"

        for S, label in [(X, 'X'), (Xdot, 'Xdot')]:
            S_ = model.project(S, label)
            assert S_.shape == (r,k)
            S_ = model.project(model.Vr.T @ S, label)
            assert S_.shape == (r,k)

    def test_fit(self):
        """Test _core._base._BaseROM.fit()."""
        model = self.Dummy("A")
        with pytest.raises(NotImplementedError) as ex:
            model.fit()
        assert ex.value.args[0] == "fit() implemented by child classes"

        with pytest.raises(NotImplementedError) as ex:
            model.fit(1, 2, 3, 4, 5, 6, 7, a=8)
        assert ex.value.args[0] == "fit() implemented by child classes"

    def test_predict(self):
        """Test _core._base._BaseROM.fit()."""
        model = self.Dummy("A")
        with pytest.raises(NotImplementedError) as ex:
            model.predict()
        assert ex.value.args[0] == "predict() implemented by child classes"

        with pytest.raises(NotImplementedError) as ex:
            model.predict(1, 2, 3, 4, 5, 6, 7, a=8)
        assert ex.value.args[0] == "predict() implemented by child classes"


class TestDiscreteROM:
    """Test _core._base._DiscreteROM."""
    def test_f_(self, r=5, m=2):
        """Test _core._base.DiscreteROM.f_()."""
        c_, A_, H_, G_, B_ = _get_operators(r, m)

        model = roi._core._base._DiscreteROM("cA")
        model.r = r
        model.c_, model.A_ = c_, A_
        x_ = np.random.random(r)
        y_ = c_ + A_ @ x_
        assert np.allclose(model.f_(x_), y_)
        assert np.allclose(model.f_(x_, -1), y_)

        model = roi._core._base._DiscreteROM("HGB")
        model.r, model.m = r, m
        model.H_, model.G_, model.B_ = H_, G_, B_
        u = np.random.random(m)
        x_ = np.random.random(r)
        y_ = H_ @ roi.utils.kron2c(x_) + G_ @ roi.utils.kron3c(x_) + B_ @ u
        assert np.allclose(model.f_(x_, u), y_)

    def test_predict(self):
        """Test _core._base._DiscreteROM.predict()."""
        model = roi._core._base._DiscreteROM('')

        # Get test data.
        n, k, m, r = 60, 50, 20, 10
        X = _get_data(n, k, m)[0]
        Vr = la.svd(X)[0][:,:r]

        niters = 5
        x0 = X[:,0]
        U = np.ones((m, niters-1))

        # Try to predict with invalid initial condition.
        x0_ = Vr.T @ x0
        model = _trainedmodel(False, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            model.predict(x0_[:-1], niters, U)
        assert ex.value.args[0] == "x0 not aligned with Vr, dimension 0"

        # Try to predict with bad niters argument.
        with pytest.raises(ValueError) as ex:
            model.predict(x0, -18, U)
        assert ex.value.args[0] == \
            "argument 'niters' must be a nonnegative integer"

        # Try to predict with badly-shaped discrete inputs.
        model = _trainedmodel(False, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            model.predict(x0, niters, np.random.random((m-1, niters-1)))
        assert ex.value.args[0] == \
            f"invalid input shape ({(m-1,niters-1)} != {(m,niters-1)}"

        model = _trainedmodel(False, "cAHB", Vr, m=1)
        with pytest.raises(ValueError) as ex:
            model.predict(x0, niters, np.random.random((2, niters-1)))
        assert ex.value.args[0] == \
            f"invalid input shape ({(2,niters-1)} != {(1,niters-1)}"

        # Try to predict with continuous inputs.
        model = _trainedmodel(False, "cAHB", Vr, m)
        with pytest.raises(TypeError) as ex:
            model.predict(x0, niters, lambda t: np.ones(m-1))
        assert ex.value.args[0] == "input U must be an array, not a callable"

        for form in MODEL_FORMS:
            if "B" not in form:             # No control inputs.
                model = _trainedmodel(False, form, Vr, None)
                out = model.predict(x0, niters)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,niters)
            else:                           # Has Control inputs.
                # Predict with 2D inputs.
                model = _trainedmodel(False, form, Vr, m)
                out = model.predict(x0, niters, U)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,niters)

                # Predict with 1D inputs.
                model = _trainedmodel(False, form, Vr, 1)
                out = model.predict(x0, niters, np.ones(niters))
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,niters)

        # Predict with no basis gives result in low-dimensional space.
        model = _trainedmodel(False, "cA", Vr, None)
        model.Vr = None
        out = model.predict(Vr.T @ x0, niters)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r,niters)


class TestContinuousROM:
    """Test _core._base._ContinuousROM."""
    def test_f_(self, r=5, m=2):
        """Test _core._base.ContinuousROM.f_()."""
        c_, A_, H_, G_, B_ = _get_operators(r, m)

        # Check that the constructed f takes the right number of arguments.
        model = roi._core._base._ContinuousROM("cA")
        model.r = r
        model.c_, model.A_ = c_, A_
        x_ = np.random.random(r)
        y_ = c_ + A_ @ x_
        assert np.allclose(model.f_(0, x_), y_)
        assert np.allclose(model.f_(1, x_), y_)
        assert np.allclose(model.f_(1, x_, -1), y_)

        model = roi._core._base._ContinuousROM("HGB")
        model.r, model.m = r, m
        model.H_, model.G_, model.B_ = H_, G_, B_
        uu = np.random.random(m)
        u = lambda t: uu + t
        y_ = H_ @ roi.utils.kron2c(x_) + G_ @ roi.utils.kron3c(x_) + B_ @ uu
        assert np.allclose(model.f_(0, x_, u), y_)
        y_ = H_ @ roi.utils.kron2c(x_) + G_ @ roi.utils.kron3c(x_) + B_ @(uu+1)
        assert np.allclose(model.f_(1, x_, u), y_)

    def test_predict(self):
        """Test _core._base._ContinuousROM.predict()."""
        model = roi._core._base._ContinuousROM('')

        # Get test data.
        n, k, m, r = 60, 50, 20, 10
        X = _get_data(n, k, m)[0]
        Vr = la.svd(X)[0][:,:r]

        nt = 5
        x0 = X[:,0]
        t = np.linspace(0, .01*nt, nt)
        u = lambda t: np.ones(m)
        Upred = np.ones((m, nt))

        # Try to predict with invalid initial condition.
        x0_ = Vr.T @ x0
        model = _trainedmodel(True, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            model.predict(x0_[1:], t, u)
        assert ex.value.args[0] == "x0 not aligned with Vr, dimension 0"

        # Try to predict with bad time array.
        with pytest.raises(ValueError) as ex:
            model.predict(x0, np.vstack((t,t)), u)
        assert ex.value.args[0] == "time 't' must be one-dimensional"

        # Predict without inputs.
        for form in MODEL_FORMS:
            if "B" not in form:
                model = _trainedmodel(True, form, Vr, None)
                out = model.predict(x0, t)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,t.size)

        # Predict with no basis gives result in low-dimensional space.
        model = _trainedmodel(True, "cA", Vr, None)
        model.Vr = None
        out = model.predict(Vr.T @ x0, t)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r,t.size)

        # Try to predict with badly-shaped discrete inputs.
        model = _trainedmodel(True, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, np.random.random((m-1, nt)))
        assert ex.value.args[0] == \
            f"invalid input shape ({(m-1,nt)} != {(m,nt)}"

        model = _trainedmodel(True, "cAHB", Vr, m=1)
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, np.random.random((2, nt)))
        assert ex.value.args[0] == \
            f"invalid input shape ({(2,nt)} != {(1,nt)}"

        # Try to predict with badly-shaped continuous inputs.
        model = _trainedmodel(True, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, lambda t: np.ones(m-1))
        assert ex.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(m,)}"
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, lambda t: 1)
        assert ex.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(m,)}"

        model = _trainedmodel(True, "cAHB", Vr, m=1)
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, u)
        assert ex.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(1,)}" \
            " or scalar"

        # Try to predict with continuous inputs with bad return type
        model = _trainedmodel(True, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, lambda t: set([5]))
        assert ex.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(m,)}"

        for form in MODEL_FORMS:
            if "B" in form:
                # Predict with 2D inputs.
                model = _trainedmodel(True, form, Vr, m)
                # continuous input.
                out = model.predict(x0, t, u)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,nt)
                # discrete input.
                out = model.predict(x0, t, Upred)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,nt)

                # Predict with 1D inputs.
                model = _trainedmodel(True, form, Vr, 1)
                # continuous input.
                out = model.predict(x0, t, lambda t: 1)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,nt)
                out = model.predict(x0, t, lambda t: np.array([1]))
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,nt)
                # discrete input.
                out = model.predict(x0, t, np.ones_like(t))
                assert isinstance(out, np.ndarray)
                assert out.shape == (n,nt)


class TestNonparametricMixin:
    """Test _core._base._NonparametricMixin."""

    class Dummy(roi._core._base._NonparametricMixin,
                roi._core._base._BaseROM):
        def __init__(self, modelform):
            self.modelform = modelform

    def test_O_(self, r=9, m=4):
        """Test _core._base._NonparametricMixin.O_."""
        c, A, H, G, B = _get_operators(r, m)
        for form in MODEL_FORMS:
            model = self.Dummy(form)
            model.set_operators(None,
                                c_=c if 'c' in form else None,
                                A_=A if 'A' in form else None,
                                H_=H if 'H' in form else None,
                                G_=G if 'G' in form else None,
                                B_=B if 'B' in form else None)
            O_ = model.O_
            d = roi.lstsq.lstsq_size(form, r, m if 'B' in form else 0)
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
        model = roi.InferredContinuousROM("A")
        assert str(model) == \
            "Reduced-order model structure: dx / dt = Ax(t)"
        model = roi.InferredContinuousROM("cA")
        assert str(model) == \
            "Reduced-order model structure: dx / dt = c + Ax(t)"
        model = roi.InferredContinuousROM("HB")
        assert str(model) == \
            "Reduced-order model structure: dx / dt = H(x(t) ⊗ x(t)) + Bu(t)"
        model = roi.InferredContinuousROM("G")
        assert str(model) == \
            "Reduced-order model structure: dx / dt = G(x(t) ⊗ x(t) ⊗ x(t))"
        model = roi.InferredContinuousROM("cH")
        assert str(model) == \
            "Reduced-order model structure: dx / dt = c + H(x(t) ⊗ x(t))"

        # Discrete ROMs
        model = roi.IntrusiveDiscreteROM("A")
        assert str(model) == \
            "Reduced-order model structure: x_{j+1} = Ax_{j}"
        model = roi.IntrusiveDiscreteROM("cB")
        assert str(model) == \
            "Reduced-order model structure: x_{j+1} = c + Bu_{j}"
        model = roi.IntrusiveDiscreteROM("H")
        assert str(model) == \
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
        model = _trainedmodel("inferred", "cAHGB", Vr, m)

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

        model.save_model(target[:-3], save_basis=False)
        _checkfile(target, model, False)

        with pytest.raises(FileExistsError) as ex:
            model.save_model(target, overwrite=False)
        assert ex.value.args[0] == target

        model.save_model(target, save_basis=True, overwrite=True)
        _checkfile(target, model, True)

        model = _trainedmodel("inferred", "c", Vr, 0)
        model.save_model(target, overwrite=True)
        _checkfile(target, model, True)

        model = _trainedmodel("inferred", "AB", Vr, m)
        model.Vr = None
        model.save_model(target, save_basis=True, overwrite=True)
        _checkfile(target, model, False)

        # Check that save_model() and load_model() are inverses.
        model.Vr = Vr
        model.save_model(target, save_basis=True, overwrite=True)
        model2 = roi.load_model(target)
        for attr in ["n", "m", "r", "modelform", "__class__"]:
            assert getattr(model, attr) == getattr(model2, attr)
        for attr in ["A_", "B_", "Vr"]:
            assert np.allclose(getattr(model, attr), getattr(model2, attr))
        for attr in ["c_", "H_", "G_"]:
            assert getattr(model, attr) is getattr(model2, attr) is None

        # Check Vr = None functionality.
        model.Vr = None
        model.save_model(target, overwrite=True)
        model2 = roi.load_model(target)
        for attr in ["m", "r", "modelform", "__class__"]:
            assert getattr(model, attr) == getattr(model2, attr)
        for attr in ["A_", "B_",]:
            assert np.allclose(getattr(model, attr), getattr(model2, attr))
        for attr in ["n", "c_", "H_", "G_", "Vr"]:
            assert getattr(model, attr) is getattr(model2, attr) is None

        os.remove(target)


class TestParametricMixin:
    """Test _core._base._ParametricMixin."""

    class DummyDiscrete(roi._core._base._ParametricMixin,
                        roi._core._base._DiscreteROM):
        pass

    class DummyContinuous(roi._core._base._ParametricMixin,
                          roi._core._base._ContinuousROM):
        pass

    def test_call(self, r=10, m=3):
        """Test _core._base._ParametricMixin.__call__()."""
        # Define dummy operators to use.
        c1, A1, H1, G1, B1 = _get_operators(r, m)
        c2, A2, H2, G2, B2 = _get_operators(r, m)
        def c(*args, **kwargs): return c1
        def A(*args, **kwargs): return A1
        def H(*args, **kwargs): return H1
        def G(*args, **kwargs): return G1
        def B(*args, **kwargs): return B1
        c.shape = (r,)
        A.shape = (r,r)
        H.shape = (r,r*(r + 1)//2)
        G.shape = (r,r*(r + 1)*(r + 2)//6)
        B.shape = (r,m)

        model = self.DummyDiscrete("cAH")
        model.r = r
        model.c_, model.A_, model.H_ = c2, A, H
        newmodel = model(1)
        assert isinstance(newmodel,
                          roi._core._base._DiscreteParametricEvaluationROM)
        assert newmodel.c_ is c2
        assert newmodel.A_ is A1
        assert newmodel.H_ is H1

        model.c_, model.A_, model.H_ = c, A2, H2
        newmodel = model(2)
        assert isinstance(newmodel,
                          roi._core._base._DiscreteParametricEvaluationROM)
        assert newmodel.c_ is c1
        assert newmodel.A_ is A2
        assert newmodel.H_ is H2

        model = self.DummyContinuous("GB")
        model.r, model.m = r, m
        model.G_, model.B_ = G2, B
        newmodel = model(3)
        assert isinstance(newmodel,
                          roi._core._base._ContinuousParametricEvaluationROM)
        assert newmodel.G_ is G2
        assert newmodel.B_ is B1

        model.G_, model.B_ = G, B2
        newmodel = model(3)
        assert isinstance(newmodel,
                          roi._core._base._ContinuousParametricEvaluationROM)
        assert newmodel.G_ is G1
        assert newmodel.B_ is B2

        badmodel = roi._core._base._ParametricMixin()
        with pytest.raises(RuntimeError) as ex:
            badmodel(10)
        assert len(ex.value.args) == 0

    def test_str(self, r=10, m=3):
        """Test _core._base._ParametricMixin.__str__()."""

        # Define dummy operators to use.
        def c(*args, **kwargs): pass
        def A(*args, **kwargs): pass
        def H(*args, **kwargs): pass
        def G(*args, **kwargs): pass
        def B(*args, **kwargs): pass
        c.shape = (r,)
        A.shape = (r,r)
        H.shape = (r,r*(r + 1)//2)
        G.shape = (r,r*(r + 1)*(r + 2)//6)
        B.shape = (r,m)

        # Continuous ROMs
        model = self.DummyContinuous("A")
        assert str(model) == \
            "Reduced-order model structure: dx / dt = Ax(t)"

        model.r = r
        model.A_ = A
        assert str(model) == \
            "Reduced-order model structure: dx / dt = A(µ)x(t)"

        model = self.DummyContinuous("cA")
        model.r = r
        model.c_, model.A_ = c, A
        assert str(model) == \
            "Reduced-order model structure: dx / dt = c(µ) + A(µ)x(t)"

        model = self.DummyContinuous("HB")
        model.r, model.m = r, m
        assert str(model) == \
            "Reduced-order model structure: dx / dt = H(x(t) ⊗ x(t)) + Bu(t)"
        model = self.DummyContinuous("G")
        model.r = r
        model.G_ = G
        assert str(model) == \
            "Reduced-order model structure: dx / dt = G(µ)(x(t) ⊗ x(t) ⊗ x(t))"

        # Discrete ROMs
        model = self.DummyDiscrete("cH")
        assert str(model) == \
            "Reduced-order model structure: x_{j+1} = c + H(x_{j} ⊗ x_{j})"
        model.r = r
        model.c_ = c
        assert str(model) == \
            "Reduced-order model structure: x_{j+1} = c(µ) + H(x_{j} ⊗ x_{j})"
