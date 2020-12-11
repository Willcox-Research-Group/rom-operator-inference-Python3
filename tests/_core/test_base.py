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
    def test_init(self):
        """Test _BaseROM.__init__()."""
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

        model = roi._core._base._ContinuousROM("cA")
        assert hasattr(model, "modelform")
        assert hasattr(model, "has_inputs")
        assert model.modelform == "cA"
        assert model.has_constant is True
        assert model.has_linear is True
        assert model.has_quadratic is False
        assert model.has_inputs is False

        model.modelform = "BHc"
        assert model.modelform == "cHB"
        assert model.has_constant is True
        assert model.has_linear is False
        assert model.has_quadratic is True
        assert model.has_inputs is True

    def test_check_modelform(self, n=10, r=3, m=5):
        """Test _BaseROM._check_modelform()."""
        c_, A_, H_, G_, B_ = _get_operators(r, m)
        Vr = np.random.random((n,r))

        # Try with invalid modelform.
        model = roi._core._base._ContinuousROM("bad_form")
        with pytest.raises(ValueError) as ex:
            model._check_modelform(trained=False)
        assert ex.value.args[0] == \
            "invalid modelform key 'b'; " \
            f"options are {', '.join(model._MODEL_KEYS)}"

        # Try with completely untrained model.
        model.modelform = "cAH"
        with pytest.raises(AttributeError) as ex:
            model._check_modelform(trained=True)
        assert ex.value.args[0] == \
            "null reduced dimension 'r'; call fit() to train model"

        # Try with reduced dimension set but not input dimension.
        model.modelform = "cAHB"
        model.r = r
        with pytest.raises(AttributeError) as ex:
            model._check_modelform(trained=True)
        assert ex.value.args[0] == \
            "null input dimension 'm'; call fit() to train model"

        # Try with missing operators.
        model.modelform = "cAHB"
        model.r, model.m = r, m
        model.A_, model.H_, model.G_, model.B_ = A_, H_, None, B_
        with pytest.raises(AttributeError) as ex:
            model._check_modelform(trained=True)
        assert ex.value.args[0] == \
            "attribute 'c_' missing; call fit() to train model"
        model.c_ = c_

        # Try with null required operator.
        model.A_ = None
        with pytest.raises(AttributeError) as ex:
            model._check_modelform(trained=True)
        assert ex.value.args[0] == \
            "attribute 'A_' is None; call fit() to train model"
        model.A_ = A_

        # Try with operators that shouldn't be present.
        model.G_ = G_
        with pytest.raises(AttributeError) as ex:
            model._check_modelform(trained=True)
        assert ex.value.args[0] == \
            "attribute 'G_' should be None; call fit() to train model"
        model.G_ = None

        # Try with a badly shaped operator.
        model.A_ = np.random.random((r,r-1))
        with pytest.raises(ValueError) as ex:
            model._check_modelform(trained=True)
        assert ex.value.args[0] == \
            f"'A_.shape' must be {(r,r)} for r = {r} (got {(r,r-1)})"

    def test_set_operators(self):
        """Test _core._base._BaseROM._set_operators()."""
        n, m, r = 60, 20, 30
        Vr = np.random.random((n, r))
        c, A, H, G, B = _get_operators(r, m)

        # Test correct usage.
        model = roi._core._base._ContinuousROM("cAH")._set_operators(
                                                    Vr=Vr, c_=c, A_=A, H_=H)
        assert isinstance(model, roi._core._base._ContinuousROM)
        assert model.modelform == "cAH"
        assert model.n == n
        assert model.r == r
        assert model.m is None
        assert np.allclose(model.Vr, Vr)
        assert np.allclose(model.c_, c)
        assert np.allclose(model.A_, A)
        assert np.allclose(model.H_, H)
        assert model.B_ is None
        assert model.G_ is None

        model = roi._core._base._DiscreteROM("GB")._set_operators(None,
                                                                  G_=G, B_=B)
        assert isinstance(model, roi._core._base._DiscreteROM)
        assert model.modelform == "GB"
        assert model.n is None
        assert model.r == r
        assert model.m == m
        assert model.Vr is None
        assert model.c_ is None
        assert model.A_ is None
        assert model.H_ is None
        assert np.allclose(model.G_, G)
        assert np.allclose(model.B_, B)

    def test_check_inputargs(self):
        """Test _BaseROM._check_inputargs()."""

        # Try with has_inputs = True but without inputs.
        model = roi._core._base._DiscreteROM("cB")
        with pytest.raises(ValueError) as ex:
            model._check_inputargs(None, 'U')
        assert ex.value.args[0] == \
            "argument 'U' required since 'B' in modelform"

        # Try with has_inputs = False but with inputs.
        model.modelform = "cA"
        with pytest.raises(ValueError) as ex:
            model._check_inputargs(1, 'u')
        assert ex.value.args[0] == \
            "argument 'u' invalid since 'B' in modelform"

    def test_project(self):
        """Test _core._base._BaseROM.project()."""
        n, k, m, r = 60, 50, 20, 10
        X, Xdot, U = _get_data(n, k, m)
        model = roi._core._base._ContinuousROM("c")
        model.n, model.r, model.m = n, r, m
        model.Vr = la.svd(X)[0][:,:r]

        with pytest.raises(ValueError) as ex:
            model.project(X[:-1,:], 'X')
        assert ex.value.args[0] == "X not aligned with Vr, dimension 0"

        for S, label in [(X, 'X'), (Xdot, 'Xdot')]:
            S_ = model.project(S, label)
            assert S_.shape == (r,k)
            S_ = model.project(model.Vr.T @ S, label)
            assert S_.shape == (r,k)


class TestDiscreteROM:
    """Test _core._base._DiscreteROM."""
    def test_construct_f_(self):
        """Test _core._base.DiscreteROM._construct_f_()."""
        model = roi._core._base._DiscreteROM('')

        # Check that the constructed f takes the right number of arguments.
        model.modelform = "cA"
        model.c_, model.A_ = np.random.random(5), np.random.random((5,5))
        model.H_, model.G_, model.B_ = None, None, None
        model.r = 5
        model._construct_f_()
        with pytest.raises(TypeError) as ex:
            model.f_(1, 2)
        assert ex.value.args[0] == \
            "<lambda>() takes 1 positional argument but 2 were given"

        model.modelform = "HGB"
        model.H_ = np.random.random((2,3))
        model.G_ = np.random.random((2,4))
        model.B_ = np.random.random((2,5))
        model.r, model.m = 2, 5
        model.c_, model.A_ = None, None
        model._construct_f_()
        with pytest.raises(TypeError) as ex:
            model.f_(1)
        assert ex.value.args[0] == \
            "<lambda>() missing 1 required positional argument: 'u'"

    def test_fit(self):
        """Test _core._base._DiscreteROM.fit()."""
        model = roi._core._base._DiscreteROM("A")
        with pytest.raises(NotImplementedError) as ex:
            model.fit()
        assert ex.value.args[0] == \
            "fit() must be implemented by child classes"

        with pytest.raises(NotImplementedError) as ex:
            model.fit(1, 2, 3, 4, 5, 6, 7, a=8)
        assert ex.value.args[0] == \
            "fit() must be implemented by child classes"

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
        model.Vr, model.n = None, None
        out = model.predict(Vr.T @ x0, niters)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r,niters)


class TestContinuousROM:
    """Test _core._base._ContinuousROM."""
    def test_construct_f_(self):
        """Test incorrect usage of _core._base.ContinuousROM._construct_f_()."""
        model = roi._core._base._ContinuousROM('')

        # Check that the constructed f takes the right number of arguments.
        model.modelform = "cA"
        model.c_, model.A_ = np.random.random(5), np.random.random((5,5))
        model.H_, model.G_, model.B_ = None, None, None
        model.r = 5
        model._construct_f_()
        with pytest.raises(TypeError) as ex:
            model.f_(1)
        assert ex.value.args[0] == \
            "<lambda>() missing 1 required positional argument: 'x_'"

    def test_fit(self):
        """Test _core._base._ContinuousROM.fit()."""
        model = roi._core._base._ContinuousROM("A")
        with pytest.raises(NotImplementedError) as ex:
            model.fit()
        assert ex.value.args[0] == \
            "fit() must be implemented by child classes"

        with pytest.raises(NotImplementedError) as ex:
            model.fit(1, 2, 3, 4, 5, 6, 7, a=8)
        assert ex.value.args[0] == \
            "fit() must be implemented by child classes"

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
        model.Vr, model.n = None, None
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

    class Dummy(roi._core._base._BaseROM, roi._core._base._NonparametricMixin):
        def __init__(self, modelform):
            self.modelform = modelform

    def test_operator_matrix_(self):
        """Test _core._base._NonparametricMixin.operator_matrix_."""
        m, r =  4, 9
        c, A, H, G, B = _get_operators(r, m)
        model = self.Dummy("")
        model.m, model.r = m, r
        model.c_, model.A_, model.H_, model.G_, model.B_ = c, A, H, G, B

        for form in MODEL_FORMS:
            model.modelform = form
            model.m = m if 'B' in form else 0
            O = model.operator_matrix_
            d = roi.lstsq.lstsq_size(form, r, model.m)
            assert O.shape == (r,d)

            # Spot check.
            if form == "cB":
                assert np.all(O == np.hstack((c[:,np.newaxis], B)))
            elif form == "AB":
                assert np.all(O == np.hstack((A, B)))
            elif form == "HG":
                assert np.all(O == np.hstack((H, G)))

    def test_str(self):
        """Test _core._base._NonparametricMixin.__str__()
        (string representation).
        """
        # Continuous ROMs
        model = roi.InferredContinuousROM("A")
        assert str(model) == \
            "Reduced-order model structure: dx / dt = Ax(t)"
        model.modelform = "cA"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = c + Ax(t)"
        model.modelform = "HB"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = H(x(t) ⊗ x(t)) + Bu(t)"
        model.modelform = "G"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = G(x(t) ⊗ x(t) ⊗ x(t))"
        model.modelform = "cH"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = c + H(x(t) ⊗ x(t))"

        # Discrete ROMs
        model = roi.IntrusiveDiscreteROM("A")
        assert str(model) == \
            "Reduced-order model structure: x_{j+1} = Ax_{j}"
        model.modelform = "cB"
        assert str(model) == \
            "Reduced-order model structure: x_{j+1} = c + Bu_{j}"
        model.modelform = "H"
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
        model.Vr, model.n = None, None
        model.save_model(target, overwrite=True)
        model2 = roi.load_model(target)
        for attr in ["m", "r", "modelform", "__class__"]:
            assert getattr(model, attr) == getattr(model2, attr)
        for attr in ["A_", "B_",]:
            assert np.allclose(getattr(model, attr), getattr(model2, attr))
        for attr in ["n", "c_", "H_", "G_", "Vr"]:
            assert getattr(model, attr) is getattr(model2, attr) is None

        # Try to save a bad model.
        A_ = model.A_
        del model.A_
        with pytest.raises(AttributeError) as ex:
            model.save_model(target, overwrite=True)

        os.remove(target)


class TestParametricMixin:
    """Test _core._base._ParametricMixin."""
    def test_str(self):
        """Test _core._base._ParametricMixin.__str__()
        (string representation).
        """
        # Continuous ROMs
        model = roi.InterpolatedInferredContinuousROM("A")
        assert str(model) == \
            "Reduced-order model structure: dx / dt = Ax(t)"
        model.c_ = lambda t: t
        model.A_ = lambda t: t
        model.modelform = "cA"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = c(µ) + A(µ)x(t)"
        model.H_ = None
        model.G_ = lambda t: t
        model.B_ = None
        model.modelform = "HB"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = H(x(t) ⊗ x(t)) + Bu(t)"
        model.modelform = "G"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = G(µ)(x(t) ⊗ x(t) ⊗ x(t))"

        # Discrete ROMs
        model = roi.AffineIntrusiveDiscreteROM("cH")
        assert str(model) == \
            "Reduced-order model structure: x_{j+1} = c + H(x_{j} ⊗ x_{j})"
        model.c_ = lambda t: t
        model.H_ = None
        assert str(model) == \
            "Reduced-order model structure: x_{j+1} = c(µ) + H(x_{j} ⊗ x_{j})"
