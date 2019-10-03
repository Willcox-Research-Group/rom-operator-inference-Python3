# test_core.py
"""Tests for rom_operator_inference._core.py."""

import pytest
import warnings
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi


def _get_data(n=200, k=50, m=20):
    X = np.random.random((n,k))
    Xdot = np.zeros((n,k))
    U = np.ones((m,k))

    return X, Xdot, U

def _get_operators(n=200, m=20):
    A = np.eye(n)
    H = np.zeros((n,n**2))
    F = np.zeros((n,n*(n+1)//2))
    c = np.random.random(n)
    B = np.random.random((n,m))
    return A, H, F, c, B

_contmodel = roi._core._BaseContinuousModel._trained_model_from_operators
# _discmodel = roi._core._BaseDiscreteModel._trained_model_from_operators

class TestBaseModel:
    """Test _core._BaseModel."""
    def test_init(self):
        """Test _BaseModel.__init__()."""
        with pytest.raises(TypeError) as ex:
            roi._core._BaseModel()
        assert ex.value.args[0] == \
            "__init__() missing 1 required positional argument: 'modelform'"

        with pytest.raises(TypeError) as ex:
            roi._core._BaseModel("LQc", False, None)
        assert ex.value.args[0] == \
            "__init__() takes from 2 to 3 positional arguments " \
            "but 4 were given"

        model = roi._core._BaseModel("LQc")
        assert hasattr(model, "modelform")
        assert hasattr(model, "has_inputs")

    def test_check_modelform(self):
        """Test _BaseModel._check_modelform()."""
        model = roi._core._BaseModel("bad_form", False)
        with pytest.raises(ValueError) as ex:
            model._check_modelform()
        assert ex.value.args[0] == \
            "invalid modelform 'bad_form'; " \
            f"options are {model._VALID_MODEL_FORMS}"

    def test_check_hasinputs(self):
        """Test _BaseModel._check_hasinputs()."""
        model = roi._core._BaseModel("some model form", None)

        # Try with has_inputs = True but without inputs.
        model.has_inputs = True
        with pytest.raises(ValueError) as ex:
            model._check_hasinputs(None, 'U')
        assert ex.value.args[0] == \
            "argument 'U' required since has_inputs=True"

        # Try with has_inputs = False but with inputs.
        model.has_inputs = False
        with pytest.raises(ValueError) as ex:
            model._check_hasinputs(1, 'u')
        assert ex.value.args[0] == \
            "argument 'u' invalid since has_inputs=False"


class TestBaseContinuousModel:
    """Test _core._BaseContinuousModel."""
    def test_construct_f_(self):
        """Test incorrect usage of BaseContinuousModel._construct_f_()."""
        model = roi._core._BaseContinuousModel(None, False)

        model.has_inputs = False
        with pytest.raises(RuntimeError) as ex:
            model._construct_f_(lambda t: 1)
        assert ex.value.args[0] == "improper use of _construct_f_()!"

        model.has_inputs = True
        with pytest.raises(RuntimeError) as ex:
            model._construct_f_()
        assert ex.value.args[0] == "improper use of _construct_f_()!"

    def test_trained_model_from_operators(self):
        """Test BaseContinuousModel._trained_model_from_operators."""
        # Try with no operators given.
        with pytest.raises(AttributeError) as ex:
            roi._core._BaseContinuousModel._trained_model_from_operators("LQc",
                                                                         True)
        assert ex.value.args[0] == \
            "attributes 'Vr', 'n', 'r', 'm', 'A_', 'F_', 'c_', 'B_' required"

        # Try with everything given but one attribute.
        with pytest.raises(AttributeError) as ex:
            roi._core._BaseContinuousModel._trained_model_from_operators(
                            "LQc", True, n=1, r=1, m=1, A_=1, F_=1, c_=1, B_=1)
        assert ex.value.args[0] == "attribute 'Vr' required"

        # Correct usage.
        A, H, F, c, B = _get_operators(n=200, m=20)
        roi._core._BaseContinuousModel._trained_model_from_operators(
            "LQc", False, n=200, m=20, r=10, A_=A, F_=F, c_=c, B_=B, Vr=None)

    def test_str(self):
        """Test BaseContinuousModel.__str__() (string representation)."""
        model = roi._core._BaseContinuousModel(None, False)

        model.modelform = "L"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = Ax(t)"
        model.modelform = "Lc"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = Ax(t) + c"
        model.modelform = "Q"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = H(x ⊗ x)(t)"
        model.modelform = "Qc"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = H(x ⊗ x)(t) + c"
        model.modelform = "LQ"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = Ax(t) + H(x ⊗ x)(t)"
        model.modelform = "LQc"
        assert str(model) == \
            "Reduced-order model structure: dx / dt = Ax(t) + H(x ⊗ x)(t) + c"

    def test_fit(self):
        """Test _core._BaseContinuousModel.fit()."""
        model = roi._core._BaseContinuousModel("L", False)
        with pytest.raises(NotImplementedError) as ex:
            model.fit()
        assert ex.value.args[0] == \
            "fit() must be implemented by child classes"

        with pytest.raises(NotImplementedError) as ex:
            model.fit(1, 2, 3, 4, 5, 6, 7, a=8)
        assert ex.value.args[0] == \
            "fit() must be implemented by child classes"

    def test_predict(self):
        """Test _core._BaseContinuousModel.predict()."""
        model = roi._core._BaseContinuousModel(None, None)

        # Get test data.
        n, k, m, r = 200, 100, 20, 10
        X, Xdot, U = _get_data(n, k, m)
        Vr = la.svd(X)[0][:,:r]
        U1d = np.ones(k)

        # Get test (reduced) operators.
        A, H, F, c, B = _get_operators(r, m)
        B1d = B[:,0]

        nt = 5
        x0 = X[:,0]
        t = np.linspace(0, .01*nt, nt)
        u = lambda t: np.ones(m)
        Upred = np.ones((m, nt))

        # Try to predict before fitting.
        model.modelform = "LQc"
        model.has_inputs = True
        with pytest.raises(AttributeError) as ex:
            model.predict(x0, t, u)
        assert ex.value.args[0] == "model not trained (call fit() first)"
        model = _contmodel("LQc", True,
                           n=n, m=m, r=r, Vr=Vr, A_=A, F_=F, c_=c, B_=B)

        # Try to predict with invalid initial condition.
        x0_ = Vr.T @ x0
        with pytest.raises(ValueError) as ex:
            model.predict(x0_, t, u)
        assert ex.value.args[0] == f"invalid initial state size ({r} != {n})"

        # Try to predict with weird time array.
        with pytest.raises(ValueError) as ex:
            model.predict(x0, np.vstack((t,t)), u)
        assert ex.value.args[0] == "time 't' must be one-dimensional"

        # Try to predict without inputs when required and vice versa
        model.has_inputs = True
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t)
        assert ex.value.args[0] == \
            "argument 'u' required since has_inputs=True"

        model.has_inputs = False
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, u)
        assert ex.value.args[0] == \
            "argument 'u' invalid since has_inputs=False"

        # Change has_inputs between fit() and predict().
        model = _contmodel("LQc", True,
                           n=n, m=m, r=r, Vr=Vr, A_=A, F_=F, c_=c, B_=B)
        model.has_inputs = False
        with pytest.raises(AttributeError) as ex:
            model.predict(x0, t)
        assert ex.value.args[0] == \
            "`has_inputs` attribute altered between fit()" \
            " and predict(); call fit() again to retrain"

        model = _contmodel("LQc", False,
                           n=n, m=None, r=r, Vr=Vr, A_=A, F_=F, c_=c, B_=None)
        model.has_inputs = True
        with pytest.raises(AttributeError) as ex:
            model.predict(x0, t, u)
        assert ex.value.args[0] == \
            "`has_inputs` attribute altered between fit()" \
            " and predict(); call fit() again to retrain"

        # Predict without inputs.
        model.has_inputs = False
        for form in model._VALID_MODEL_FORMS:
            model.modelform = form
            model._construct_f_()
            model.predict(x0, t)

        # Try to predict with badly-shaped discrete inputs.
        model = _contmodel("LQc", True,
                           n=n, m=m, r=r, Vr=Vr, A_=A, F_=F, c_=c, B_=B)
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, np.random.random((m-1, nt)))
        assert ex.value.args[0] == \
            f"invalid input shape ({(m-1,nt)} != {(m,nt)}"

        model = _contmodel("LQc", True,
                           n=n, m=1, r=r, Vr=Vr, A_=A, F_=F, c_=c, B_=B1d)
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, np.random.random((2, nt)))
        assert ex.value.args[0] == \
            f"invalid input shape ({(2,nt)} != {(1,nt)}"

        # Try to predict with badly-shaped continuous inputs.
        model = _contmodel("LQc", True,
                           n=n, m=m, r=r, Vr=Vr, A_=A, F_=F, c_=c, B_=B)
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, lambda t: np.ones(m-1))
        assert ex.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(m,)}"
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, lambda t: 1)
        assert ex.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(m,)}"

        model = _contmodel("LQc", True,
                           n=n, m=1, r=r, Vr=Vr, A_=A, F_=F, c_=c, B_=B1d)
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, u)
        assert ex.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(1,)}" \
            " or scalar"

        # Try to predict with continuous inputs with bad return type
        model = _contmodel("LQc", True,
                           n=n, m=m, r=r, Vr=Vr, A_=A, F_=F, c_=c, B_=B)
        with pytest.raises(ValueError) as ex:
            model.predict(x0, t, lambda t: set([5]))
        assert ex.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(m,)}"

        # Predict with 2D inputs.
        model = _contmodel("LQc", True,
                           n=n, m=m, r=r, Vr=Vr, A_=A, F_=F, c_=c, B_=B)
        for form in ["L", "Lc", "Q", "Qc", "LQ", "LQc"]:
            model.modelform = form
            # continuous input.
            out = model.predict(x0, t, u)
            assert isinstance(out, np.ndarray)
            assert out.shape == (n,nt)
            # discrete input.
            out = model.predict(x0, t, Upred)
            assert isinstance(out, np.ndarray)
            assert out.shape == (n,nt)

        # Predict with 1D inputs.
        model = _contmodel("LQc", True,
                           n=n, m=1, r=r, Vr=Vr, A_=A, F_=F, c_=c,
                           B_=B1d.reshape((-1,1)))
        for form in ["L", "Lc", "Q", "Qc", "LQ", "LQc"]:
            model.modelform = form
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


class TestIntrusiveContinuousModel:
    """Test _core.IntrusiveContinuousModel."""

    def test_fit(self):
        model = roi.IntrusiveContinuousModel("LQc", has_inputs=False)

        # Get test data.
        n, k, m, r = 200, 100, 20, 10
        X, Xdot, U = _get_data(n, k, m)
        Vr = la.svd(X)[0][:,:r]

        # Get test operators.
        A, H, F, c, B = _get_operators(n, m)
        B1d = B[:,0]
        operators = [A, H, c, B]

        # Try to fit the model with misaligned operators and Vr.
        Abad = A[:,:-2]
        Hbad = H[:,1:]
        cbad = c[::2]
        Bbad = B[1:,:]
        model.modelform = "LQc"
        model.has_inputs = True

        with pytest.raises(ValueError) as ex:
            model.fit([A, H, B], Vr)
        assert ex.value.args[0] == "expected 4 operators, got 3"

        with pytest.raises(ValueError) as ex:
            model.fit([Abad, H, c, B], Vr)
        assert ex.value.args[0] == "basis Vr and FOM operator A not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit([A, Hbad, c, B], Vr)
        assert ex.value.args[0] == \
            "basis Vr and FOM operator H (F) not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit([A, H, cbad, B], Vr)
        assert ex.value.args[0] == "basis Vr and FOM operator c not aligned"

        with pytest.raises(ValueError) as ex:
            model.fit([A, H, c, Bbad], Vr)
        assert ex.value.args[0] == "basis Vr and FOM operator B not aligned"

        # Fit the model with each possible modelform.
        model.has_inputs = False
        model.modelform = "L"
        model.fit([A], Vr)
        model.modelform = "Lc"
        model.fit([A, c], Vr)
        model.modelform = "Q"
        model.fit([H], Vr)
        model.fit([roi.utils.H2F(H)], Vr)
        model.modelform = "Qc"
        model.fit([H, c], Vr)
        model.modelform = "LQ"
        model.fit([A, H], Vr)
        model.modelform = "LQc"
        model.fit([A, H, c], Vr)
        model.modelform = "LQc"
        model.has_inputs = True
        model.fit([A, H, c, B], Vr)

        # Test fit output sizes.
        assert model.n == n
        assert model.r == r
        assert model.m == m
        assert model.A.shape == (n,n)
        assert model.F.shape == (n,n*(n+1)//2)
        assert model.H.shape == (n,n**2)
        assert model.c.shape == (n,)
        assert model.B.shape == (n,m)
        assert model.A_.shape == (r,r)
        assert model.F_.shape == (r,r*(r+1)//2)
        assert model.H_.shape == (r,r**2)
        assert model.c_.shape == (r,)
        assert model.B_.shape == (r,m)

        # Fit the model with 1D inputs (1D array for B)
        model.modelform = "LQc"
        model.has_inputs = True
        model.fit([A, H, c, B1d], Vr)
        assert model.B.shape == (n,1)
        assert model.B_.shape == (r,1)


class TestInferredContinuousModel:
    """Test _core.InferredContinuousModel."""
    def test_fit(self):
        model = roi.InferredContinuousModel("LQc", has_inputs=False)

        # Get test data.
        n, k, m, r = 200, 100, 20, 10
        X, Xdot, U = _get_data(n, k, m)
        Vr = la.svd(X)[0][:,:r]

        # Try to fit the model with misaligned X and Xdot.
        with pytest.raises(ValueError) as ex:
            model.fit(X, Xdot[:,1:-1], Vr)
        assert ex.value.args[0] == \
            f"X and Xdot different shapes ({(n,k)} != {(n,k-2)})"

        # Try to fit the model with misaligned X and Vr.
        with pytest.raises(ValueError) as ex:
            model.fit(X, Xdot, Vr[1:-1,:])
        assert ex.value.args[0] == \
            f"X and Vr not aligned, first dimension {n} != {n-2}"

        # Fit the model with each possible modelform.
        model.has_inputs = False
        for form in model._VALID_MODEL_FORMS:
            model.modelform = form
            model.fit(X, Xdot, Vr)

        # Test fit output sizes.
        model.modelform = "LQc"
        model.has_inputs = True
        model.fit(X, Xdot, Vr, U=U)
        assert model.n == n
        assert model.r == r
        assert model.m == m
        assert model.A_.shape == (r,r)
        assert model.F_.shape == (r,r*(r+1)//2)
        assert model.H_.shape == (r,r**2)
        assert model.c_.shape == (r,)
        assert model.B_.shape == (r,m)
        assert hasattr(model, "residual_")

        # Try again with one-dimensional inputs.
        m = 1
        U = np.ones(k)
        model.fit(X, Xdot, Vr, U=U)
        n, r, m = model.n, model.r, model.m
        assert model.n == n
        assert model.r == r
        assert model.m == 1
        assert model.A_.shape == (r,r)
        assert model.F_.shape == (r,r*(r+1)//2)
        assert model.H_.shape == (r,r**2)
        assert model.c_.shape == (r,)
        assert model.B_.shape == (r,1)
        assert hasattr(model, "residual_")

class TestInterpolatedInferredContinuousModel:
    """Test _core.InterpolatedInferredContinuousModel."""
    def test_fit(self):
        """Test _core.InterpolatedInferredContinuousModel.fit()."""
        model = roi.InterpolatedInferredContinuousModel("LQc", False)

        # Get data for fitting.
        n, m, k, r = 50, 10, 20, 5
        X1, Xdot1, U1 = _get_data(n, k, m)
        X2, Xdot2, U2 = X1+1, Xdot1.copy(), U1+1
        Xs = [X1, X2]
        Xdots = [Xdot1, Xdot2]
        Us = [U1, U2]
        ps = [1, 2]
        Vr = la.svd(np.hstack(Xs))[0][:,:r]

        # Try with non-scalar parameters.
        with pytest.raises(ValueError) as ex:
            model.fit([np.array([1,1]), np.array([2,2])], Xs, Xdots, Vr)
        assert ex.value.args[0] == "only scalar parameter values are supported"

        # Try with bad number of Xs.
        with pytest.raises(ValueError) as ex:
            model.fit(ps, [X1, X2, X2+1], Xdots, Vr)
        assert ex.value.args[0] == \
            "num parameter samples != num state snapshot sets (2 != 3)"

        # Try with bad number of Xdots.
        with pytest.raises(ValueError) as ex:
            model.fit(ps, Xs, Xdots + [Xdot1], Vr)
        assert ex.value.args[0] == \
            "num parameter samples != num velocity snapshot sets (2 != 3)"

        # Try with varying input sizes.
        model.has_inputs = True
        with pytest.raises(ValueError) as ex:
            model.fit(ps, Xs, Xdots, Vr, [U1, U2[:-1]])
        assert ex.value.args[0] == \
            "dimension 'm' inconsistent across training sets"

        # Fit correctly with no inputs.
        model.has_inputs = False
        model.fit(ps, Xs, Xdots, Vr)
        for attr in ["models_", "dataconds_", "residuals_", "fs_"]:
            assert hasattr(model, attr)
            assert len(getattr(model, attr)) == len(model.models_)

        # Fit correctly with inputs.
        model.has_inputs = True
        model.fit(ps, Xs, Xdots, Vr, Us)

    def test_predict(self):
        """Test _core.InterpolatedInferredContinuousModel.predict()."""
        model = roi.InterpolatedInferredContinuousModel("LQc", False)

        # Get data for fitting.
        n, m, k, r = 50, 10, 20, 5
        X1, Xdot1, U1 = _get_data(n, k, m)
        X2, Xdot2, U2 = X1+1, Xdot1.copy(), U1+1
        Xs = [X1, X2]
        Xdots = [Xdot1, Xdot2]
        Us = [U1, U2]
        ps = [1, 2]
        Vr = la.svd(np.hstack(Xs))[0][:,:r]

        # Parameters for predicting.
        x0 = np.random.random(n)
        nt = 5
        t = np.linspace(0, .01*nt, nt)
        u = lambda t: np.ones(10)

        # Fit / predict with no inputs.
        model.fit(ps, Xs, Xdots, Vr)
        model.predict(1, x0, t)
        model.predict(1.5, x0, t)

        # Fit / predict with inputs.
        model.has_inputs = True
        model.fit(ps, Xs, Xdots, Vr, Us)
        model.predict(1, x0, t, u)
        model.predict(1.5, x0, t, u)
