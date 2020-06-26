# _core.py
"""Classes for reduction of dynamical systems."""

import os
import h5py
import warnings
import itertools
import numpy as np
from scipy import linalg as la
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp, IntegrationWarning

from .utils import (lstsq_reg,
                    expand_Hc as Hc2H, compress_H as H2Hc,
                    expand_Gc as Gc2G, compress_G as G2Gc,
                    kron2c, kron3c)


# Helper functions and classes (public) =======================================
def select_model(time, rom_strategy, parametric=False):
    """Select the appropriate ROM model class for the situation.

    Parameters
    ----------
    time : str {"discrete", "continuous"}
        The type of full-order model to be reduced. Options:
        * "discrete": solve a discrete dynamical system,
          x_{j+1} = f(x_{j}, u_{j}), x_{0} = x0.
        * "continuous": solve an ordinary differential equation,
          dx / dt = f(t, x(t), u(t)), x(0) = x0.

    rom_strategy : str {"inferred", "intrusive"}
        Whether to use Operator Inference or intrusive projection to compute
        the operators of the reduced model. Options:
        * "inferred": use Operator Inference, i.e., solve a least-squares
          problem based on snapshot data.
        * "intrusive": use intrusive projection, i.e., project known full-order
          operators to the reduced space.

    parametric : str {"affine", "interpolated"} or False
        Whether or not the model depends on an external parameter, and how to
        handle the parametric dependence. Options:
        * False (default): the problem is nonparametric.
        * "affine": one or more operators in the problem depends affinely on
          the parameter, i.e., A(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * A_{i}.
          Only valid for rom_strategy="intrusive".
        * "interpolated": construct individual models for each sample parameter
          and interpolate them for general parameter inputs. Only valid for
          rom_strategy="inferred", and only when the parameter is a scalar.

    Returns
    -------
    ModelClass : type
        One of the ROM classes derived from _BaseROM:
        * InferredDiscreteROM
        * InferredContinuousROM
        * IntrusiveDiscreteROM
        * IntrusiveContinuousROM
        * AffineIntrusiveDiscreteROM
        * AffineIntrusiveContinuousROM
        * InterpolatedInferredDiscreteROM
        * InterpolatedInferredContinuousROM
    """
    # Validate parameters.
    time_options = {"discrete", "continuous"}
    rom_strategy_options = {"inferred", "intrusive"}
    parametric_options = {False, "affine", "interpolated"}

    if time not in time_options:
        raise ValueError(f"input `time` must be one of {time_options}")
    if rom_strategy not in rom_strategy_options:
        raise ValueError(
                f"input `rom_strategy` must be one of {rom_strategy_options}")
    if parametric not in parametric_options:
        raise ValueError(
                f"input `parametric` must be one of {parametric_options}")

    t, r, p = time, rom_strategy, parametric

    if t == "discrete" and r == "inferred" and not p:
        return InferredDiscreteROM
    elif t == "continuous" and r == "inferred" and not p:
        return InferredContinuousROM
    elif t == "discrete" and r == "intrusive" and not p:
        return IntrusiveDiscreteROM
    elif t == "continuous" and r == "intrusive" and not p:
        return IntrusiveContinuousROM
    elif t == "discrete" and r == "intrusive" and p == "affine":
        return AffineIntrusiveDiscreteROM
    elif t == "continuous" and r == "intrusive" and p == "affine":
        return AffineIntrusiveContinuousROM
    elif t == "discrete" and r == "inferred" and p == "interpolated":
        return InterpolatedInferredDiscreteROM
    elif t == "continuous" and r == "inferred" and p == "interpolated":
        return InterpolatedInferredContinuousROM
    else:
        raise NotImplementedError("model type invalid or not implemented")


def trained_model_from_operators(ModelClass, modelform, Vr,
                                 c_=None, A_=None,
                                 H_=None, Hc_=None,
                                 G_=None, Gc_=None, B_=None):
    """Construct a prediction-capable ROM object from the operators of
    the reduced model.

    Parameters
    ----------
    ModelClass : type
        One of the ROM classes (e.g., IntrusiveContinuousROM).

    modelform : str
        The structure of the model, a substring of "cAHB".

    Vr : (n,r) ndarray or None
        The basis for the linear reduced space (e.g., POD basis matrix).
        If None, then r is inferred from one of the reduced operators.

    c_ : (r,) ndarray or None
        Reduced constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Reduced linear state matrix, or None if 'c' is not in `modelform`.

    H_ : (r,r**2) ndarray or None
        Reduced quadratic state matrix (full size), or None if 'H' is not in
        `modelform`.

    Hc_ : (r,r(r+1)/2) ndarray or None
        Reduced quadratic state matrix (compact), or None if 'H' is not in
        `modelform`. Only used if `H_` is also None.

    G_ : (r,r**3) ndarray or None
        Reduced cubic state matrix (full size), or None if 'G' is not in
        `modelform`.

    Gc_ : (r,r(r+1)(r+2)/6) ndarray or None
        Reduced cubic state matrix (compact), or None if 'G' is not in
        `modelform`. Only used if `G_` is also None.

    B_ : (r,m) ndarray or None
        Reduced input matrix, or None if 'B' is not in `modelform`.

    Returns
    -------
    model : ModelClass object
        A new model, ready for predict() calls.
    """
    # Check that the ModelClass is valid.
    if not issubclass(ModelClass, _BaseROM):
        raise TypeError("ModelClass must be derived from _BaseROM")

    # Construct the new model object.
    model = ModelClass(modelform)
    model._check_modelform(trained=False)

    # Insert the reduced operators.
    model.m = None if B_ is None else 1 if B_.ndim == 1 else B_.shape[1]
    model.c_, model.A_, model.B_ = c_, A_, B_
    model.Hc_ = H2Hc(H_) if H_ else Hc_
    model.Gc_ = G2Gc(G_) if G_ else Gc_

    # Insert the basis (if given) and determine dimensions.
    model.Vr = Vr
    if Vr is not None:
        model.n, model.r = Vr.shape
    else:
        # Determine the dimension r from the existing operators.
        model.n = None
        for op in [model.c_, model.A_, model.Hc_, model.Gc_, model.B_]:
            if op is not None:
                model.r = op.shape[0]
                break

    # Construct the complete reduced model operator from the arguments.
    model._construct_f_()

    return model


def load_model(loadfile):
    """Load a serialized model from an HDF5 file, created previously from
    a ROM object's save_model() method.

    Parameters
    ----------
    loadfile : str
        The file to load from, which should end in '.h5'.

    Returns
    -------
    model : ROM class
        The trained reduced-order model.
    """
    if not os.path.isfile(loadfile):
        raise FileNotFoundError(loadfile)

    with h5py.File(loadfile, 'r') as data:
        if "meta" not in data:
            raise ValueError("invalid save format (meta/ not found)")
        if "operators" not in data:
            raise ValueError("invalid save format (operators/ not found)")

        # Load metadata.
        modelclass = data["meta"].attrs["modelclass"]
        try:
            ModelClass = eval(modelclass)
        except NameError as ex:
            raise ValueError(f"invalid modelclass '{modelclass}' (meta.attrs)")
        # is_parametric = issubclass(ModelClass, _ParametricMixin)
        modelform = data["meta"].attrs["modelform"]

        # Load basis if present.
        Vr = data["Vr"][:] if "Vr" in data else None

        # Load operators.
        operators = {}
        if 'c' in modelform:
            operators["c_"] = data["operators/c_"][:]
        if 'A' in modelform:
            operators["A_"] = data["operators/A_"][:]
        if 'H' in modelform:
            operators["Hc_"] = data["operators/Hc_"][:]
        if 'G' in modelform:
            operators["Gc_"] = data["operators/Gc_"][:]
        if 'B' in modelform:
            operators["B_"] = data["operators/B_"][:]

        # Load any other saved attributes.
        if "other" in data:
            attrs = {key: dset[0] if dset.shape == (1,) else dset[:]
                                   for key, dset in data["other"].items()}
        else:
            attrs = {}

        # TODO: loading (and saving) for Parametric operators.

    # Load the model.
    model = trained_model_from_operators(ModelClass,modelform,Vr, **operators)

    # Attach extra attributes.
    for key, val in attrs.items():
        setattr(model, key, val)

    return model


class AffineOperator:
    """Class for representing a linear operator with affine structure, i.e.,

        A(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * A_{i}.

    The matrix A(µ) is constructed by calling the object once the coefficient
    functions and component matrices are set.

    Attributes
    ----------
    nterms : int
        The number of terms in the sum defining the linear operator.

    coefficient_functions : list of `nterms` callables
        The coefficient scalar-valued functions that define the operator.
        Each must take the same sized input and return a scalar.

    matrices : list of `nterms` ndarrays of the same shape
        The component matrices defining the linear operator.
    """
    def __init__(self, coeffs, matrices=None):
        """Save the coefficient functions and component matrices (optional).

        Parameters
        ----------
        coeffs : list of `nterms` callables
            The coefficient scalar-valued functions that define the operator.
            Each must take the same sized input and return a scalar.

        matrices : list of `nterms` ndarrays of the same shape
            The component matrices defining the linear operator.
            Can also be assigned later by setting the `matrices` attribute.
        """
        self.coefficient_functions = coeffs
        self._nterms = len(coeffs)
        if matrices:
            self.matrices = matrices
        else:
            self._ready = False

    @property
    def nterms(self):
        """The number of component matrices."""
        return self._nterms

    @property
    def matrices(self):
        """The component matrices."""
        return self._matrices

    @matrices.setter
    def matrices(self, ms):
        """Set the component matrices, checking that the shapes are equal."""
        if len(ms) != self.nterms:
            _noun = "matrix" if self.nterms == 1 else "matrices"
            raise ValueError(f"expected {self.nterms} {_noun}, got {len(ms)}")

        # Check that each matrix in the list has the same shape.
        shape = ms[0].shape
        for m in ms:
            if m.shape != shape:
                raise ValueError("affine operator matrix shapes do not match "
                                 f"({m.shape} != {shape})")

        # Store matrix list and shape, and mark as ready (for __call__()).
        self._matrices = ms
        self.shape = shape
        self._ready = True

    def validate_coeffs(self, µ):
        """Check that each coefficient function 1) is a callable function,
        2) takes in the right sized inputs, and 3) returns scalar values.

        Parameters
        ----------
        µ : float or (p,) ndarray
            A test input for the coefficient functions.
        """
        for θ in self.coefficient_functions:
            if not callable(θ):
                raise ValueError("coefficients of affine operator must be "
                                 "callable functions")
            elif not np.isscalar(θ(µ)):
                raise ValueError("coefficient functions of affine operator "
                                 "must return a scalar")

    def __call__(self, µ):
        """Evaluate the affine operator at the given parameter."""
        if not self._ready:
            raise RuntimeError("component matrices not initialized!")
        return np.sum([θi(µ)*Ai for θi,Ai in zip(self.coefficient_functions,
                                                 self.matrices)], axis=0)

    def __eq__(self, other):
        """Test whether the component matrices of two AffineOperator objects
        are numerically equal. The coefficient functions are *NOT* compared.
        """
        if not isinstance(other, AffineOperator):
            return False
        if self.nterms != other.nterms:
            return False
        if not (self._ready and other._ready):
            return False
        return all([np.allclose(self.matrices[l], other.matrices[l])
                                            for l in range(self.nterms)])


# Base classes (private) ======================================================
class _BaseROM:
    """Base class for all rom_operator_inference reduced model classes."""
    _MODEL_KEYS = "cAHGB"       # Constant, Linear, Quadratic, Cubic, Input.

    def __init__(self, modelform):
        if not isinstance(self, (_ContinuousROM, _DiscreteROM)):
            raise RuntimeError("abstract class instantiation "
                               "(use _ContinuousROM or _DiscreteROM)")
        self.modelform = modelform

    @property
    def modelform(self):
        return self._form

    @modelform.setter
    def modelform(self, form):
        self._form = ''.join(sorted(form,
                                    key=lambda k: self._MODEL_KEYS.find(k)))

    @property
    def has_constant(self):
        return "c" in self.modelform

    @property
    def has_linear(self):
        return "A" in self.modelform

    @property
    def has_quadratic(self):
        return "H" in self.modelform

    @property
    def has_cubic(self):
        return "G" in self.modelform

    @property
    def has_inputs(self):
        return "B" in self.modelform

    # @property
    # def has_outputs(self):
    #     return "C" in self._form

    def _check_modelform(self, trained=False):
        """Ensure that self.modelform is valid."""
        for key in self.modelform:
            if key not in self._MODEL_KEYS:
                raise ValueError(f"invalid modelform key '{key}'; options "
                                 "are " + ', '.join(self._MODEL_KEYS))

        if trained:
            # Make sure that the required attributes exist and aren't None,
            # and that nonrequired attributes exist but are None.
            for key, s in zip("cAHGB", ["c_", "A_", "Hc_", "Gc_", "B_"]):
                if not hasattr(self, s):
                    raise AttributeError(f"attribute '{s}' missing;"
                                         " call fit() to train model")
                attr = getattr(self, s)
                if key in self.modelform and attr is None:
                    raise AttributeError(f"attribute '{s}' is None;"
                                         " call fit() to train model")
                elif key not in self.modelform and attr is not None:
                    raise AttributeError(f"attribute '{s}' should be None;"
                                         " call fit() to train model")

    def _check_inputargs(self, u, argname):
        """Check that self.has_inputs agrees with input arguments."""
        if self.has_inputs and u is None:
            raise ValueError(f"argument '{argname}' required"
                             " since 'B' in modelform")

        if not self.has_inputs and u is not None:
            raise ValueError(f"argument '{argname}' invalid"
                             " since 'B' in modelform")

    def project(self, S, label="input"):
        """Check the dimensions of S and project it if needed."""
        if S.shape[0] not in {self.r, self.n}:
            raise ValueError(f"{label} not aligned with Vr, dimension 0")
        return self.Vr.T @ S if S.shape[0] == self.n else S

    @property
    def operator_norm_(self):
        """Calculate the squared Frobenius norm of the ROM operators."""
        self._check_modelform(trained=True)
        total = 0
        if self.has_constant:
            total += np.sum(self.c_**2)
        if self.has_linear:
            total += np.sum(self.A_**2)
        if self.has_quadratic:
            total += np.sum(self.Hc_**2)
        if self.has_cubic:
            total += np.sum(self.Gc_**2)
        if self.has_inputs:
            total += np.sum(self.B_**2)
        return total


class _DiscreteROM(_BaseROM):
    """Base class for models that solve the discrete ROM problem,

        x_{j+1} = f(x_{j}, u_{j}),         x_{0} = x0.

    The problem may also be parametric, i.e., x and f may depend on an
    independent parameter µ.
    """
    def _construct_f_(self):
        """Define the attribute self.f_ based on the computed operators."""
        self._check_modelform(trained=True)

        # No control inputs, so f = f(x).
        if self.modelform == "c":
            f_ = lambda x_: self.c_
        elif self.modelform == "A":
            f_ = lambda x_: self.A_@x_
        elif self.modelform == "H":
            f_ = lambda x_: self.Hc_@kron2c(x_)
        elif self.modelform == "G":
            f_ = lambda x_: self.Gc_@kron3c(x_)
        elif self.modelform == "cA":
            f_ = lambda x_: self.c_ + self.A_@x_
        elif self.modelform == "cH":
            f_ = lambda x_: self.c_ + self.Hc_@kron2c(x_)
        elif self.modelform == "cG":
            f_ = lambda x_: self.c_ + self.Gc_@kron3c(x_)
        elif self.modelform == "AH":
            f_ = lambda x_: self.A_@x_ + self.Hc_@kron2c(x_)
        elif self.modelform == "AG":
            f_ = lambda x_: self.A_@x_ + self.Gc_@kron3c(x_)
        elif self.modelform == "HG":
            f_ = lambda x_: self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_)
        elif self.modelform == "cAH":
            f_ = lambda x_: self.c_ + self.A_@x_ + self.Hc_@kron2c(x_)
        elif self.modelform == "cAG":
            f_ = lambda x_: self.c_ + self.A_@x_ + self.Gc_@kron3c(x_)
        elif self.modelform == "cHG":
            f_ = lambda x_: self.c_ + self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_)
        elif self.modelform == "AHG":
            f_ = lambda x_: self.A_@x_ + self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_)
        elif self.modelform == "cAHG":
            f_ = lambda x_: self.c_ + self.A_@x_ + self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_)

        # Has control inputs, so f = f(x, u).
        elif self.modelform == "B":
            f_ = lambda x_,u: self.B_@u
        elif self.modelform == "cB":
            f_ = lambda x_,u: self.c_ + self.B_@u
        elif self.modelform == "AB":
            f_ = lambda x_,u: self.A_@x_ + self.B_@u
        elif self.modelform == "HB":
            f_ = lambda x_,u: self.Hc_@kron2c(x_) + self.B_@u
        elif self.modelform == "GB":
            f_ = lambda x_,u: self.Gc_@kron3c(x_) + self.B_@u
        elif self.modelform == "cAB":
            f_ = lambda x_,u: self.c_ + self.A_@x_ + self.B_@u
        elif self.modelform == "cHB":
            f_ = lambda x_,u: self.c_ + self.Hc_@kron2c(x_) + self.B_@u
        elif self.modelform == "cGB":
            f_ = lambda x_,u: self.c_ + self.Gc_@kron3c(x_) + self.B_@u
        elif self.modelform == "AHB":
            f_ = lambda x_,u: self.A_@x_ + self.Hc_@kron2c(x_) + self.B_@u
        elif self.modelform == "AGB":
            f_ = lambda x_,u: self.A_@x_ + self.Gc_@kron3c(x_) + self.B_@u
        elif self.modelform == "HGB":
            f_ = lambda x_,u: self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_) + self.B_@u
        elif self.modelform == "cAHB":
            f_ = lambda x_,u: self.c_ + self.A_@x_ + self.Hc_@kron2c(x_) + self.B_@u
        elif self.modelform == "cAGB":
            f_ = lambda x_,u: self.c_ + self.A_@x_ + self.Gc_@kron3c(x_) + self.B_@u
        elif self.modelform == "cHGB":
            f_ = lambda x_,u: self.c_ + self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_) + self.B_@u
        elif self.modelform == "AHGB":
            f_ = lambda x_,u: self.A_@x_ + self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_) + self.B_@u
        elif self.modelform == "cAHGB":
            f_ = lambda x_,u: self.c_ + self.A_@x_ + self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_) + self.B_@u

        self.f_ = f_

    def fit(self, *args, **kwargs):             # pragma: no cover
        raise NotImplementedError("fit() must be implemented by child classes")

    def predict(self, x0, niters, U=None):
        """Step forward the learned ROM `niters` steps.

        Parameters
        ----------
        x0 : (n,) or (r,) ndarray
            The initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).

        niters : int
            The number of times to step the system forward.

        U : (m,niters-1) ndarray
            The inputs for the next niters-1 time steps.

        Returns
        -------
        X_ROM : (n,niters) or (r,niters) ndarray
            The approximate solution to the system, including the given
            initial condition. If the basis Vr is None, return solutions in the
            reduced r-dimensional subspace (r,niters). Otherwise, map solutions
            to the full n-dimensional space with Vr (n,niters).
        """
        # Verify modelform.
        self._check_modelform(trained=True)
        self._check_inputargs(U, 'U')

        # Project initial conditions (if needed).
        x0_ = self.project(x0, 'x0')

        # Verify iteration argument.
        if not isinstance(niters, int) or niters < 0:
            raise ValueError("argument 'niters' must be a nonnegative integer")

        # Create the solution array and fill in the initial condition.
        X_ = np.empty((self.r,niters))
        X_[:,0] = x0_.copy()

        # Run the iteration.
        if self.has_inputs:
            if callable(U):
                raise TypeError("input U must be an array, not a callable")
            # Validate shape of input, reshaping if input is 1d.
            U = np.atleast_2d(U)
            if U.ndim != 2 or U.shape[0] != self.m or U.shape[1] < niters - 1:
                raise ValueError("invalid input shape "
                                 f"({U.shape} != {(self.m,niters-1)}")
            for j in range(niters-1):
                X_[:,j+1] = self.f_(X_[:,j], U[:,j])    # f(xj,uj)
        else:
            for j in range(niters-1):
                X_[:,j+1] = self.f_(X_[:,j])            # f(xj)

        # Reconstruct the approximation to the full-order model if possible.
        return self.Vr @ X_ if self.Vr is not None else X_


class _ContinuousROM(_BaseROM):
    """Base class for models that solve the continuous (ODE) ROM problem,

        dx / dt = f(t, x(t), u(t)),         x(0) = x0.

    The problem may also be parametric, i.e., x and f may depend on an
    independent parameter µ.
    """
    def _construct_f_(self):
        """Define the attribute self.f_ based on the computed operators."""
        self._check_modelform(trained=True)

        # self._jac = None
        # No control inputs.
        if self.modelform == "c":
            f_ = lambda t,x_: self.c_
            # self._jac = np.zeros((self.r, self.r))
        elif self.modelform == "A":
            f_ = lambda t,x_: self.A_@x_
            # self._jac = self.A_
        elif self.modelform == "H":
            f_ = lambda t,x_: self.Hc_@kron2c(x_)
        elif self.modelform == "G":
            f_ = lambda t,x_: self.Gc_@kron3c(x_)
        elif self.modelform == "cA":
            f_ = lambda t,x_: self.c_ + self.A_@x_
            # self._jac = self.A_
        elif self.modelform == "cH":
            f_ = lambda t,x_: self.c_ + self.Hc_@kron2c(x_)
        elif self.modelform == "cG":
            f_ = lambda t,x_: self.c_ + self.Gc_@kron3c(x_)
        elif self.modelform == "AH":
            f_ = lambda t,x_: self.A_@x_ + self.Hc_@kron2c(x_)
        elif self.modelform == "AG":
            f_ = lambda t,x_: self.A_@x_ + self.Gc_@kron3c(x_)
        elif self.modelform == "HG":
            f_ = lambda t,x_: self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_)
        elif self.modelform == "cAH":
            f_ = lambda t,x_: self.c_ + self.A_@x_ + self.Hc_@kron2c(x_)
        elif self.modelform == "cAG":
            f_ = lambda t,x_: self.c_ + self.A_@x_ + self.Gc_@kron3c(x_)
        elif self.modelform == "cHG":
            f_ = lambda t,x_: self.c_ + self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_)
        elif self.modelform == "AHG":
            f_ = lambda t,x_: self.A_@x_ + self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_)
        elif self.modelform == "cAHG":
            f_ = lambda t,x_: self.c_ + self.A_@x_ + self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_)

        # Has control inputs.
        elif self.modelform == "B":
            f_ = lambda t,x_,u: self.B_@u(t)
            # self._jac = np.zeros((self.r, self.r))
        elif self.modelform == "cB":
            f_ = lambda t,x_,u: self.c_ + self.B_@u(t)
            # self._jac = np.zeros((self.r, self.r))
        elif self.modelform == "AB":
            f_ = lambda t,x_,u: self.A_@x_ + self.B_@u(t)
            # self._jac = self.A_
        elif self.modelform == "HB":
            f_ = lambda t,x_,u: self.Hc_@kron2c(x_) + self.B_@u(t)
        elif self.modelform == "GB":
            f_ = lambda t,x_,u: self.Gc_@kron3c(x_) + self.B_@u(t)
        elif self.modelform == "cAB":
            f_ = lambda t,x_,u: self.c_ + self.A_@x_ + self.B_@u(t)
            # self._jac = self.A_
        elif self.modelform == "cHB":
            f_ = lambda t,x_,u: self.c_ + self.Hc_@kron2c(x_) + self.B_@u(t)
        elif self.modelform == "cGB":
            f_ = lambda t,x_,u: self.c_ + self.Gc_@kron3c(x_) + self.B_@u(t)
        elif self.modelform == "AHB":
            f_ = lambda t,x_,u: self.A_@x_ + self.Hc_@kron2c(x_) + self.B_@u(t)
        elif self.modelform == "AGB":
            f_ = lambda t,x_,u: self.A_@x_ + self.Gc_@kron3c(x_) + self.B_@u(t)
        elif self.modelform == "HGB":
            f_ = lambda t,x_,u: self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_) + self.B_@u(t)
        elif self.modelform == "cAHB":
            f_ = lambda t,x_,u: self.c_ + self.A_@x_ + self.Hc_@kron2c(x_) + self.B_@u(t)
        elif self.modelform == "cAGB":
            f_ = lambda t,x_,u: self.c_ + self.A_@x_ + self.Gc_@kron3c(x_) + self.B_@u(t)
        elif self.modelform == "cHGB":
            f_ = lambda t,x_,u: self.c_ + self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_) + self.B_@u(t)
        elif self.modelform == "AHGB":
            f_ = lambda t,x_,u: self.A_@x_ + self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_) + self.B_@u(t)
        elif self.modelform == "cAHGB":
            f_ = lambda t,x_,u: self.c_ + self.A_@x_ + self.Hc_@kron2c(x_) + self.Gc_@kron3c(x_) + self.B_@u(t)

        self.f_ = f_

    def fit(self, *args, **kwargs):             # pragma: no cover
        raise NotImplementedError("fit() must be implemented by child classes")

    def predict(self, x0, t, u=None, **options):
        """Simulate the learned ROM with scipy.integrate.solve_ivp().

        Parameters
        ----------
        x0 : (n,) or (r,) ndarray
            The initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).

        t : (nt,) ndarray
            The time domain over which to integrate the reduced-order system.

        u : callable or (m,nt) ndarray
            The input as a function of time (preferred) or the input at the
            times `t`. If given as an array, u(t) is approximated by a cubic
            spline interpolating the known data points.

        options
            Arguments for solver.integrate.solve_ivp(), such as the following:
            method : str
                The ODE solver for the reduced-order system.
                * 'RK45' (default): Explicit Runge-Kutta method of order 5(4).
                * 'RK23': Explicit Runge-Kutta method of order 3(2).
                * 'Radau': Implicit Runge-Kutta method of the Radau IIA family
                    of order 5.
                * 'BDF': Implicit multi-step variable-order (1 to 5) method
                    based on a backward differentiation formula for the
                    derivative.
                * 'LSODA': Adams/BDF method with automatic stiffness detection
                    and switching. This wraps the Fortran solver from ODEPACK.
            max_step : float
                The maximimum allowed integration step size.
            See https://docs.scipy.org/doc/scipy/reference/integrate.html.

        Returns
        -------
        X_ROM : (n,nt) or (r,nt) ndarray
            The approximate solution to the system over the time domain `t`.
            If the basis Vr is None, return solutions in the reduced
            r-dimensional subspace (r,nt). Otherwise, map the solutions to the
            full n-dimensional space with Vr (n,nt).
        """
        # Verify modelform.
        self._check_modelform(trained=True)
        self._check_inputargs(u, 'u')

        # Project initial conditions (if needed).
        x0_ = self.project(x0, 'x0')

        # Verify time domain.
        if t.ndim != 1:
            raise ValueError("time 't' must be one-dimensional")
        nt = t.shape[0]

        # Interpret control input argument `u`.
        if self.has_inputs:
            if callable(u):         # If u is a function, check output shape.
                out = u(t[0])
                if np.isscalar(out):
                    if self.m == 1:     # u : R -> R, wrap output as array.
                        _u = u
                        u = lambda s: np.array([_u(s)])
                    else:               # u : R -> R, but m != 1.
                        raise ValueError("input function u() must return"
                                         f" ndarray of shape (m,)={(self.m,)}")
                elif not isinstance(out, np.ndarray):
                    raise ValueError("input function u() must return"
                                     f" ndarray of shape (m,)={(self.m,)}")
                elif out.shape != (self.m,):
                    message = "input function u() must return" \
                              f" ndarray of shape (m,)={(self.m,)}"
                    if self.m == 1:
                        raise ValueError(message + " or scalar")
                    raise ValueError(message)
            else:                   # u is an (m,nt) array.
                U = np.atleast_2d(u)
                if U.shape != (self.m,nt):
                    raise ValueError("invalid input shape "
                                     f"({U.shape} != {(self.m,nt)}")
                u = CubicSpline(t, U, axis=1)

        # Integrate the reduced-order model.
        fun = (lambda t,x_: self.f_(t, x_, u)) if self.has_inputs else self.f_
        self.sol_ = solve_ivp(fun,              # Integrate f_(t, x_, u)
                              [t[0], t[-1]],    # over this time interval
                              x0_,              # with this initial condition
                              t_eval=t,         # evaluated at these points
                              # jac=self._jac,    # with this Jacobian
                              **options)        # with these solver options.

        # Raise warnings if the integration failed.
        if not self.sol_.success:               # pragma: no cover
            warnings.warn(self.sol_.message, IntegrationWarning)

        # Reconstruct the approximation to the full-order model.
        return self.Vr @ self.sol_.y if self.Vr is not None else self.sol_.y


# Basic mixins (private) ======================================================
class _InferredMixin:
    """Mixin class for reduced model classes that use Operator Inference."""

    @staticmethod
    def _check_training_data_shapes(datasets):
        """Ensure that each data set has the same number of columns."""
        k = datasets[0].shape[1]
        for data in datasets:
            if data.shape[1] != k:
                raise ValueError("data sets not aligned, dimension 1")

    def fit(self, Vr, X, rhs, U=None, P=0):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, X and rhs are assumed to already be projected (r,k).

        X : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        rhs : (n,k) or (r,k) ndarray
            Column-wise next-iteration (discrete model) or velocity
            (continuous model) training data. Each column is a snapshot, and
            either full order (n rows) or projected to reduced order (r rows).

        U : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see utils.lstsq_reg(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        # Check modelform and inputs.
        self._check_modelform()
        self._check_inputargs(U, 'U')

        # Store dimensions and check that number of samples is consistent.
        if Vr is not None:
            self.n, self.r = Vr.shape   # Full dimension, reduced dimension.
        else:
            self.n = None
            self.r = X.shape[0]
        _tocheck = [X, rhs]
        if self.has_inputs:             # Input dimension.
            if U.ndim == 1:
                U = U.reshape((1,-1))
                self.m = 1
            else:
                self.m = U.shape[0]
            _tocheck.append(U)
        else:
            self.m = None
        self._check_training_data_shapes(_tocheck)
        k = X.shape[1]

        # Project states and rhs to the reduced subspace (if not done already).
        self.Vr = Vr
        X_ = self.project(X, 'X')
        rhs_ = self.project(rhs, 'rhs')

        # Construct the "Data matrix" D = [X^T, (X ⊗ X)^T, U^T, 1].
        D_blocks = []
        if self.has_constant:
            D_blocks.append(np.ones((k,1)))

        if self.has_linear:
            D_blocks.append(X_.T)

        if self.has_quadratic:
            X2_ = kron2c(X_)
            D_blocks.append(X2_.T)
            _r2 = X2_.shape[0]  # = r(r+1)/2, size of compact quadratic Kron.

        if self.has_cubic:
            X3_ = kron3c(X_)
            D_blocks.append(X3_.T)
            _r3 = X3_.shape[0]  # = r(r+1)(r+2)/6, size of compact cubic Kron.

        if self.has_inputs:
            D_blocks.append(U.T)
            m = U.shape[0]
            self.m = m

        D = np.hstack(D_blocks)
        R = rhs_.T

        # Solve for the reduced-order model operators via least squares.
        Otrp, res, _, sval = lstsq_reg(D, R, P)

        # Record info about the least squares solution.
        # Condition number of the raw data matrix.
        self.datacond_ = np.linalg.cond(D)
        # Condition number of regularized data matrix.
        self.dataregcond_ = abs(sval[0]/sval[-1]) if sval[-1] > 0 else np.inf
        # Squared Frobenius data misfit (without regularization).
        self.misfit_ = np.sum(((D @ Otrp) - R)**2)
        # Squared Frobenius residual of the regularized least squares problem.
        self.residual_ = np.sum(res) if res.size > 0 else self.misfit_

        # Extract the reduced operators from Otrp.
        i = 0
        if self.has_constant:
            self.c_ = Otrp[i:i+1][0]        # Note that c_ is one-dimensional.
            i += 1
        else:
            self.c_ = None

        if self.has_linear:
            self.A_ = Otrp[i:i+self.r].T
            i += self.r
        else:
            self.A_ = None

        if self.has_quadratic:
            self.Hc_ = Otrp[i:i+_r2].T
            i += _r2
        else:
            self.Hc_ = None

        if self.has_cubic:
            self.Gc_ = Otrp[i:i+_r3].T
            i += _r3
        else:
            self.Gc_ = None

        if self.has_inputs:
            self.B_ = Otrp[i:i+self.m].T
            i += self.m
        else:
            self.B_ = None

        self._construct_f_()
        return self


class _IntrusiveMixin:
    """Mixin class for reduced model classes that use intrusive projection."""
    def _check_operators(self, operators):
        """Check the keys of the `operators` argument."""
        # Check for missing operator keys.
        missing = [repr(key) for key in self.modelform if key not in operators]
        if missing:
            _noun = "key" + ('' if len(missing) == 1 else 's')
            raise KeyError(f"missing operator {_noun} {', '.join(missing)}")

        # Check for unnecessary operator keys.
        surplus = [repr(key) for key in operators if key not in self.modelform]
        if surplus:
            _noun = "key" + ('' if len(surplus) == 1 else 's')
            raise KeyError(f"invalid operator {_noun} {', '.join(surplus)}")

    def fit(self, Vr, operators):
        """Compute the reduced model operators via intrusive projection.

        Parameters
        ----------
        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).
            This cannot be set to None, as it is required for projection.

        operators: dict(str -> ndarray)
            The operators that define the full-order model f.
            Keys must match the modelform:
            * 'c': constant term c.
            * 'A': linear state matrix A.
            * 'H': quadratic state matrix H (either full H or compact Hc).
            * 'G': cubic state matrix H (either full G or compact Gc).
            * 'B': input matrix B.

        Returns
        -------
        self
        """
        # Verify modelform.
        self._check_modelform()
        self._check_operators(operators)

        # Store dimensions.
        self.Vr = Vr
        self.n, self.r = Vr.shape           # Dim of system, num basis vectors.

        # Project FOM operators.
        if self.has_constant:               # Constant term.
            self.c = operators['c']
            if self.c.shape != (self.n,):
                raise ValueError("basis Vr and FOM operator c not aligned")
            self.c_ = self.Vr.T @ self.c
        else:
            self.c, self.c_ = None, None

        if self.has_linear:                 # Linear state matrix.
            self.A = operators['A']
            if self.A.shape != (self.n,self.n):
                raise ValueError("basis Vr and FOM operator A not aligned")
            self.A_ = self.Vr.T @ self.A @ self.Vr
        else:
            self.A, self.A_ = None, None

        if self.has_quadratic:              # Quadratic state matrix.
            H_or_Hc = operators['H']
            _n2 = self.n * (self.n + 1) // 2
            if H_or_Hc.shape == (self.n,self.n**2):         # It's H.
                self.H = H_or_Hc
                self.Hc = H2Hc(self.H)
            elif H_or_Hc.shape == (self.n,_n2):             # It's Hc.
                self.Hc = H_or_Hc
                self.H = Hc2H(self.Hc)
            else:
                raise ValueError("basis Vr and FOM operator H not aligned")
            H_ = self.Vr.T @ self.H @ np.kron(self.Vr, self.Vr)
            self.Hc_ = H2Hc(H_)
        else:
            self.Hc, self.H, self.Hc_ = None, None, None

        if self.has_cubic:
            G_or_Gc = operators['G']
            _n3 = self.n * (self.n + 1) * (self.n + 2) // 6
            if G_or_Gc.shape == (self.n,self.n**3):         # It's G.
                self.G = G_or_Gc
                self.Gc = G2Gc(self.G)
            elif G_or_Gc.shape == (self.n,_n3):             # It's Gc.
                self.Gc = G_or_Gc
                self.G = Gc2G(self.Gc)
            else:
                raise ValueError("basis Vr and FOM operator G not aligned")
            G_ = self.Vr.T @ self.G @ np.kron(self.Vr,np.kron(self.Vr,self.Vr))
            self.Gc_ = G2Gc(G_)
        else:
            self.Gc, self.G, self.Gc_ = None, None, None

        if self.has_inputs:                 # Linear input matrix.
            self.B = operators['B']
            if self.B.shape[0] != self.n:
                raise ValueError("basis Vr and FOM operator B not aligned")
            if self.B.ndim == 2:
                self.m = self.B.shape[1]
            else:                                   # One-dimensional input
                self.B = self.B.reshape((-1,1))
                self.m = 1
            self.B_ = self.Vr.T @ self.B
        else:
            self.B, self.B_, self.m = None, None, None

        self._construct_f_()
        return self


class _NonparametricMixin:
    """Mixin class for non-parametric reduced model classes."""
    @property
    def H_(self):
        """Matricized quadratic tensor; operates on full Kronecker product."""
        return None if self.Hc_ is None else Hc2H(self.Hc_)

    @property
    def G_(self):
        """Matricized cubic tensor; operates on full cubic Kronecker product.
        """
        return None if self.Gc_ is None else Gc2G(self.Gc_)

    def __str__(self):
        """String representation: the structure of the model."""
        discrete = isinstance(self, _DiscreteROM)
        x = "x_{j}" if discrete else "x(t)"
        u = "u_{j}" if discrete else "u(t)"
        lhs = "x_{j+1}" if discrete else "dx / dt"
        out = []
        if self.has_constant:
            out.append("c")
        if self.has_linear:
            out.append(f"A{x}")
        if self.has_quadratic:
            out.append(f"H({x} ⊗ {x})")
        if self.has_cubic:
            out.append(f"G({x} ⊗ {x} ⊗ {x})")
        if self.has_inputs:
            out.append(f"B{u}")
        return f"Reduced-order model structure: {lhs} = " + " + ".join(out)

    def save_model(self, savefile, save_basis=True, overwrite=False):
        """Serialize the learned model, saving it in HDF5 format.
        The model can then be loaded with rom_operator_inference.load_model().

        Parameters
        ----------
        savefile : str
            The file to save to. If it does not end with '.h5', this extension
            will be tacked on to the end.

        savebasis : bool
            If True, save the basis Vr as well as the reduced operators.
            If False, only save reduced operators.

        overwrite : bool
            If True and the specified file already exists, overwrite the file.
            If False and the specified file already exists, raise an error.
        """
        # Ensure that the model is trained (or there is nothing to save).
        self._check_modelform(trained=True)

        # Ensure the file is saved in HDF5 format.
        if not savefile.endswith(".h5"):
            savefile += ".h5"

        # Prevent overwriting and existing file on accident.
        if os.path.isfile(savefile) and not overwrite:
            raise FileExistsError(savefile)

        with h5py.File(savefile, 'w') as f:
            # Store metadata: ROM class and model form.
            meta = f.create_dataset("meta", shape=(0,))
            meta.attrs["modelclass"] = self.__class__.__name__
            meta.attrs["modelform"] = self.modelform

            # Store basis (optionally) if it exists.
            if (self.Vr is not None) and save_basis:
                f.create_dataset("Vr", data=self.Vr)

            # Store reduced operators.
            if self.has_constant:
                f.create_dataset("operators/c_", data=self.c_)
            if self.has_linear:
                f.create_dataset("operators/A_", data=self.A_)
            if self.has_quadratic:
                f.create_dataset("operators/Hc_", data=self.Hc_)
            if self.has_cubic:
                f.create_dataset("operators/Gc_", data=self.Gc_)
            if self.has_inputs:
                f.create_dataset("operators/B_", data=self.B_)

            # Store additional useful attributes.
            for attr in ["datacond_", "dataregcond_", "residual_", "misfit_"]:
                if hasattr(self, attr):
                    val = getattr(self, attr)
                    if np.isscalar(val):
                        val = [val]
                    f.create_dataset(f"other/{attr}", data=val)


class _ParametricMixin:
    """Mixin class for parametric reduced model classes."""
    def __call__(self, µ):
        """Construct the reduced model corresponding to the parameter µ."""
        c_  = self.c_(µ)  if callable(self.c_)  else self.c_
        A_  = self.A_(µ)  if callable(self.A_)  else self.A_
        Hc_ = self.Hc_(µ) if callable(self.Hc_) else self.Hc_
        Gc_ = self.Gc_(µ) if callable(self.Gc_) else self.Gc_
        B_  = self.B_(µ)  if callable(self.B_)  else self.B_
        cl = _DiscreteROM if isinstance(self, _DiscreteROM) else _ContinuousROM
        return trained_model_from_operators(ModelClass=cl,
                                            modelform=self.modelform,
                                            Vr=self.Vr,
                                            A_=A_, Hc_=Hc_, Gc_=Gc_, c_=c_,
                                            B_=B_)

    def __str__(self):
        """String representation: the structure of the model."""
        if not hasattr(self, "c_"):             # Untrained -> Nonparametric
            return _NonparametricMixin.__str__(self)
        discrete = isinstance(self, _DiscreteROM)

        x = "x_{j}" if discrete else "x(t)"
        u = "u_{j}" if discrete else "u(t)"
        lhs = "x_{j+1}" if discrete else "dx / dt"
        out = []
        if self.has_constant:
            out.append("c(µ)" if callable(self.c_)  else "c")
        if self.has_linear:
            A = "A(µ)" if callable(self.A_)  else "A"
            out.append(A + f"{x}")
        if self.has_quadratic:
            H = "H(µ)" if callable(self.Hc_) else "H"
            out.append(H + f"({x} ⊗ {x})")
        if self.has_cubic:
            G = "G(µ)" if callable(self.Gc_) else "G"
            out.append(G + f"({x} ⊗ {x} ⊗ {x})")
        if self.has_inputs:
            B = "B(µ)" if callable(self.B_)  else "B"
            out.append(B + f"{u}")
        return f"Reduced-order model structure: {lhs} = "+" + ".join(out)


# Specialized mixins (private) ================================================
class _InterpolatedMixin(_InferredMixin, _ParametricMixin):
    """Mixin class for interpolatory parametric reduced model classes."""
    @property
    def cs_(self):
        """The constant terms for each submodel."""
        return [m.c_ for m in self.models_] if self.has_constant else None

    @property
    def As_(self):
        """The linear state matrices for each submodel."""
        return [m.A_ for m in self.models_] if self.has_linear else None

    @property
    def Hs_(self):
        """The full quadratic state matrices for each submodel."""
        return [m.H_ for m in self.models_] if self.has_quadratic else None

    @property
    def Hcs_(self):
        """The compact quadratic state matrices for each submodel."""
        return [m.Hc_ for m in self.models_] if self.has_quadratic else None

    @property
    def Gs_(self):
        """The full cubic state matrices for each submodel."""
        return [m.G_ for m in self.models_] if self.has_cubic else None

    @property
    def Gcs_(self):
        """The compact cubic state matrices for each submodel."""
        return [m.Gc_ for m in self.models_] if self.has_cubic else None

    @property
    def Bs_(self):
        """The linear input matrices for each submodel."""
        return [m.B_ for m in self.models_] if self.has_inputs else None

    @property
    def fs_(self):
        """The reduced-order operators for each submodel."""
        return [m.f_ for m in self.models_]

    @property
    def dataconds_(self):
        """The condition numbers of the raw data matrices for each submodel."""
        return np.array([m.datacond_ for m in self.models_])

    @property
    def dataregconds_(self):
        """The condition numbers of the regularized data matrices for each
        submodel.
        """
        return np.array([m.dataregcond_ for m in self.models_])

    @property
    def residuals_(self):
        """The regularized least-squares residuals for each submodel."""
        return np.array([m.residual_ for m in self.models_])

    @property
    def misfits_(self):
        """The (nonregularized) least-squares data misfits for each
        submodel.
        """
        return np.array([m.misfit_ for m in self.models_])

    def __len__(self):
        """The number of trained models."""
        return len(self.models_) if hasattr(self, "models_") else 0

    def fit(self, ModelClass, Vr, µs, Xs, Xdots, Us=None, P=0):
        """Solve for the reduced model operators via ordinary least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        ModelClass: class
            ROM class, either _ContinuousROM or _DiscreteROM, to use for the
            newly constructed model.

        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, Xs and rhss are assumed to already be projected (r,k).

        µs : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        Xs : list of s (n,k) or (r,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

        Xdots : list of s (n,k) or (r,k) ndarrays or None
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].
            Igored if the model is discrete (according to `ModelClass`).

        Us : list of s (m,k) or (k,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see utils.lstsq_reg(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        # Check modelform and inputs.
        self._check_modelform(trained=False)
        self._check_inputargs(Us, 'Us')
        is_continuous = issubclass(ModelClass, _ContinuousROM)

        # Check that parameters are one-dimensional.
        if not np.isscalar(µs[0]):
            raise ValueError("only scalar parameter values are supported")

        # Check that the number of params matches the number of snapshot sets.
        s = len(µs)
        if len(Xs) != s:
            raise ValueError("num parameter samples != num state snapshot "
                             f"sets ({s} != {len(Xs)})")
        if is_continuous and len(Xdots) != s:
            raise ValueError("num parameter samples != num velocity snapshot "
                             f"sets ({s} != {len(Xdots)})")
        elif not is_continuous:
            Xdots = [None] * s

        # Check and store dimensions.
        if Vr is not None:
            self.n, self.r = Vr.shape
        else:
            self.n = None
            self.r = Xs[0].shape[0]
        self.m = None

        # Check that the arrays in each list have the same number of columns.
        _tocheck = [Xs]
        if is_continuous:
            _tocheck.append(Xdots)
        if self.has_inputs:
            self.m = Us[0].shape[0] if Us[0].ndim == 2 else 1
            # Check that the input dimension is the same in each data set.
            for U in Us:
                m = U.shape[0] if U.ndim == 2 else 1
                if m != self.m:
                    raise ValueError("control inputs not aligned")
        else:
            Us = [None]*s
        for dataset in _tocheck:
            self._check_training_data_shapes(dataset)

        # TODO: figure out how to handle P (scalar, array, list(arrays)).

        # Train one model per parameter sample.
        self.Vr = Vr
        self.models_ = []
        for µ, X, Xdot, U in zip(µs, Xs, Xdots, Us):
            model = ModelClass(self.modelform)
            if is_continuous:
                model.fit(Vr, X, Xdot, U, P)
            else:
                model.fit(Vr, X, U, P)
            model.parameter = µ
            self.models_.append(model)

        # Construct interpolators.
        self.c_ = CubicSpline(µs, self.cs_)  if self.has_constant  else None
        self.A_ = CubicSpline(µs, self.As_)  if self.has_linear    else None
        self.Hc_= CubicSpline(µs, self.Hcs_) if self.has_quadratic else None
        self.H_ = CubicSpline(µs, self.Hs_)  if self.has_quadratic else None
        self.Gc_= CubicSpline(µs, self.Gcs_) if self.has_cubic     else None
        self.G_ = CubicSpline(µs, self.Gs_)  if self.has_cubic     else None
        self.B_ = CubicSpline(µs, self.Bs_)  if self.has_inputs    else None

        return self


class _AffineMixin(_ParametricMixin):
    """Mixin class for affinely parametric reduced model classes."""

    def _check_affines(self, affines, µ=None):
        """Check the keys of the affines argument."""
        # Check for unnecessary affine keys.
        surplus = [repr(key) for key in affines if key not in self.modelform]
        if surplus:
            _noun = "key" + ('' if len(surplus) == 1 else 's')
            raise KeyError(f"invalid affine {_noun} {', '.join(surplus)}")

        if µ is not None:
            for a in affines.values():
                AffineOperator(a).validate_coeffs(µ)


class _AffineIntrusiveMixin(_IntrusiveMixin, _AffineMixin):
    """Mixin class for affinely parametric intrusive reduced model classes."""
    def fit(self, Vr, affines, operators):
        """Solve for the reduced model operators via intrusive projection.

        Parameters
        ----------
        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).
            This cannot be set to None, as it is required for projection.

        affines : dict(str -> list(callables))
            Functions that define the structures of the affine operators.
            Keys must match the modelform:
            * 'c': Constant term c(µ).
            * 'A': Linear state matrix A(µ).
            * 'H': Quadratic state matrix H(µ).
            * 'G': Cubic state matrix G(µ).
            * 'B': linear Input matrix B(µ).
            For example, if the constant term has the affine structure
            c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'c' -> [θ1, θ2, θ3].

        operators: dict(str -> ndarray or list(ndarrays))
            The operators that define the full-order model f(t,x;µ).
            Keys must match the modelform:
            * 'c': Constant term c(µ).
            * 'A': Linear state matrix A(µ).
            * 'H': Quadratic state matrix H(µ).
            * 'G': Cubic state matrix G(µ).
            * 'B': Input matrix B(µ).
            Terms with affine structure should be given as a list of the
            component matrices. For example, if the linear state matrix has
            the form A(µ) = θ1(µ)A1 + θ2(µ)A2, then 'A' -> [A1, A2].

        Returns
        -------
        self
        """
        # Verify modelform, affines, and operators.
        self._check_modelform(trained=False)
        self._check_affines(affines, None)
        self._check_operators(operators)

        # Store dimensions.
        self.Vr = Vr
        self.n, self.r = self.Vr.shape      # Dim of system, num basis vectors.

        # Project FOM operators.
        if self.has_constant:               # Constant term.
            if 'c' in affines:
                self.c = AffineOperator(affines['c'], operators['c'])
                if self.c.shape != (self.n,):
                    raise ValueError("basis Vr and FOM operator c not aligned")
                self.c_ = AffineOperator(affines['c'],
                                          [self.Vr.T @ c
                                           for c in self.c.matrices])
            else:
                self.c = operators['c']
                if self.c.shape != (self.n,):
                    raise ValueError("basis Vr and FOM operator c not aligned")
                self.c_ = self.Vr.T @ self.c
        else:
            self.c, self.c_ = None, None

        if self.has_linear:                 # Linear state matrix.
            if 'A' in affines:
                self.A = AffineOperator(affines['A'], operators['A'])
                if self.A.shape != (self.n,self.n):
                    raise ValueError("basis Vr and FOM operator A not aligned")
                self.A_ = AffineOperator(affines['A'],
                                          [self.Vr.T @ A @ self.Vr
                                           for A in self.A.matrices])
            else:
                self.A = operators['A']
                if self.A.shape != (self.n,self.n):
                    raise ValueError("basis Vr and FOM operator A not aligned")
                self.A_ = self.Vr.T @ self.A @ self.Vr
        else:
            self.A, self.A_ = None, None

        if self.has_quadratic:              # Quadratic state matrix.
            _n2 = self.n * (self.n + 1) // 2
            if 'H' in affines:
                H_or_Hc = AffineOperator(affines['H'], operators['H'])
                if H_or_Hc.shape == (self.n,self.n**2):     # It's H.
                    self.H = H_or_Hc
                    self.Hc = AffineOperator(affines['H'],
                                             [H2Hc(H)
                                              for H in H_or_Hc.matrices])
                elif H_or_Hc.shape == (self.n,_n2):         # It's Hc.
                    self.Hc = H_or_Hc
                    self.H = AffineOperator(affines['H'],
                                             [Hc2H(Hc)
                                              for Hc in H_or_Hc.matrices])
                else:
                    raise ValueError("basis Vr and FOM operator H not aligned")
                Vr2 = np.kron(self.Vr, self.Vr)
                self.H_ = AffineOperator(affines['H'],
                                          [self.Vr.T @ H @ Vr2
                                           for H in self.H.matrices])
                self.Hc_ = AffineOperator(affines['H'],
                                          [H2Hc(H_)
                                           for H_ in self.H_.matrices])
            else:
                H_or_Hc = operators['H']
                if H_or_Hc.shape == (self.n,self.n**2):     # It's H.
                    self.H = H_or_Hc
                    self.Hc = H2Hc(self.H)
                elif H_or_Hc.shape == (self.n,_n2):         # It's Hc.
                    self.Hc = H_or_Hc
                    self.H = Hc2H(self.Hc)
                else:
                    raise ValueError("basis Vr and FOM operator H not aligned")
                self.H_ = self.Vr.T @ self.H @ np.kron(self.Vr, self.Vr)
                self.Hc_ = H2Hc(self.H_)
        else:
            self.Hc, self.H, self.Hc_ = None, None, None

        if self.has_cubic:                  # Cubic state matrix.
            _n3 = self.n * (self.n + 1) * (self.n + 2) // 6
            if 'G' in affines:
                G_or_Gc = AffineOperator(affines['G'], operators['G'])
                if G_or_Gc.shape == (self.n,self.n**3):     # It's G.
                    self.G = G_or_Gc
                    self.Gc = AffineOperator(affines['G'],
                                             [G2Gc(G)
                                              for G in G_or_Gc.matrices])
                elif G_or_Gc.shape == (self.n,_n3):         # It's Gc.
                    self.Gc = G_or_Gc
                    self.G = AffineOperator(affines['G'],
                                             [Gc2G(Gc)
                                              for Gc in G_or_Gc.matrices])
                else:
                    raise ValueError("basis Vr and FOM operator G not aligned")
                Vr3 = np.kron(np.kron(self.Vr, self.Vr), self.Vr)
                self.G_ = AffineOperator(affines['G'],
                                          [self.Vr.T @ G @ Vr3
                                           for G in self.G.matrices])
                self.Gc_ = AffineOperator(affines['G'],
                                          [G2Gc(G_)
                                           for G_ in self.G_.matrices])
            else:
                G_or_Gc = operators['G']
                if G_or_Gc.shape == (self.n,self.n**3):     # It's G.
                    self.G = G_or_Gc
                    self.Gc = G2Gc(self.G)
                elif G_or_Gc.shape == (self.n,_n3):         # It's Gc.
                    self.Gc = G_or_Gc
                    self.G = Gc2G(self.Gc)
                else:
                    raise ValueError("basis Vr and FOM operator G not aligned")
                Vr3 = np.kron(np.kron(self.Vr, self.Vr), self.Vr)
                self.G_ = self.Vr.T @ self.G @ Vr3
                self.Gc_ = G2Gc(self.G_)
        else:
            self.Gc, self.G, self.Gc_ = None, None, None

        if self.has_inputs:                 # Linear input matrix.
            if 'B' in affines:
                self.B = AffineOperator(affines['B'], operators['B'])
                if self.B.shape[0] != self.n:
                    raise ValueError("basis Vr and FOM operator B not aligned")
                if len(self.B.shape) == 2:
                    self.m = self.B.shape[1]
                else:                                   # One-dimensional input
                    self.B = AffineOperator(affines['B'],
                                             [B.reshape((-1,1))
                                              for B in self.B.matrices])
                    self.m = 1
                self.B_ = AffineOperator(affines['B'],
                                          [self.Vr.T @ B
                                           for B in self.B.matrices])
            else:
                self.B = operators['B']
                if self.B.shape[0] != self.n:
                    raise ValueError("basis Vr and FOM operator B not aligned")
                if self.B.ndim == 2:
                    self.m = self.B.shape[1]
                else:                                   # One-dimensional input
                    self.B = self.B.reshape((-1,1))
                    self.m = 1
                self.B_ = self.Vr.T @ self.B
        else:
            self.B, self.B_, self.m = None, None, None

        return self


# Useable classes (public) ====================================================
# Nonparametric Operator Inference models -------------------------------------
class InferredDiscreteROM(_InferredMixin, _NonparametricMixin, _DiscreteROM):
    """Reduced order model for a discrete dynamical system of
    the form

        x_{j+1} = f(x_{j}, u_{j}),              x_{0} = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving an ordinary
    least-squares problem.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
        'G' : Cubic state term G(x⊗x⊗x).
        'B' : Input term Bu.
        For example, modelform=="AB" means f(x,u) = Ax + Bu.

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax.

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x).

    has_cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x).

    has_inputs : bool
        Whether or not there is a linear input term Bu.

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    datacond_ : float
        Condition number of the raw data matrix for the least-squares problem.

    dataregcond_ : float
        Condition number of the regularized data matrix for the least-squares
        problem. Same as datacond_ if there is no regularization.

    residual_ : float
        The squared Frobenius-norm residual of the regularized least-squares
        problem for computing the reduced-order model operators.

    misfit_ : float
        The squared Frobenius-norm data misfit of the (nonregularized)
        least-squares problem for computing the reduced-order model operators.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)/2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used
        directly in solving the ROM.

    Gc_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : (r,r**3) ndarray or None
        Learned ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used
        directly in solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : callable((r,) ndarray, (m,) ndarray) -> (r,)
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        The signature is f_(x_) if 'B' is not in `modelform` (no inputs) and
        f_(x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and inputs if appropriate) to reduced state. Calculated in fit().
    """
    def fit(self, Vr, X, U=None, P=0):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, X is assumed to already be projected (r,k).

        X : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        U : (m,k-1) or (k-1,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see utils.lstsq_reg(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        return _InferredMixin.fit(self, Vr,
                                  X[:,:-1], X[:,1:],    # x_j's and x_{j+1}'s.
                                  U[...,:X.shape[1]-1] if U is not None else U,
                                  P)


class InferredContinuousROM(_InferredMixin, _NonparametricMixin,
                            _ContinuousROM):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t, x(t), u(t)),             x(0) = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving an ordinary
    least-squares problem.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax(t).
        'H' : Quadratic state term H(x⊗x)(t).
        'G' : Cubic state term G(x⊗x⊗x)(t).
        'B' : Input term Bu(t).
        For example, modelform=="AB" means f(t,x(t),u(t)) = Ax(t) + Bu(t).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax(t).

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x)(t).

    has_cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x)(t).

    has_inputs : bool
        Whether or not there is a linear input term Bu(t).

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    datacond_ : float
        Condition number of the raw data matrix for the least-squares problem.

    dataregcond_ : float
        Condition number of the regularized data matrix for the least-squares
        problem. Same as datacond_ if there is no regularization.

    residual_ : float
        The squared Frobenius-norm residual of the regularized least-squares
        problem for computing the reduced-order model operators.

    misfit_ : float
        The squared Frobenius-norm data misfit of the (nonregularized)
        least-squares problem for computing the reduced-order model operators.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)/2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used
        directly in solving the ROM.

    Gc_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : (r,r**3) ndarray or None
        Learned ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used
        directly in solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : callable(float, (r,) ndarray, func?) -> (r,) ndarray
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        The signature is f_(t, x_) if 'B' is not in `modelform` (no inputs) and
        f_(t, x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and possibly an input function) to reduced state. Calculated in fit().

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, Vr, X, Xdot, U=None, P=0):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, X and Xdot are assumed to already be projected (r,k).

        X : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        Xdot : (n,k) or (r,k) ndarray
            Column-wise velocity training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        U : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see utils.lstsq_reg(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        return _InferredMixin.fit(self, Vr, X, Xdot, U, P)


# Nonparametric intrusive models ----------------------------------------------
class IntrusiveDiscreteROM(_IntrusiveMixin, _NonparametricMixin, _DiscreteROM):
    """Reduced order model for a discrete dynamical system of the form

        x_{j+1} = f(x_{j}, u_{j}),              x_{0} = x0.

    The user must specify the model form of the full-order model (FOM)
    operator f and the associated operators; the operators for the reduced
    model (ROM) are computed explicitly by projecting the full-order operators.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
        'G' : Cubic state term G(x⊗x⊗x).
        'B' : Input term Bu.
        For example, modelform=="AB" means f(x,u) = Ax + Bu.

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax.

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x).

    has_cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x).

    has_inputs : bool
        Whether or not there is a linear input term Bu.

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    datacond_ : float
        Condition number of the raw data matrix for the least-squares problem.

    dataregcond_ : float
        Condition number of the regularized data matrix for the least-squares
        problem. Same as datacond_ if there is no regularization.

    residual_ : float
        The squared Frobenius-norm residual of the regularized least-squares
        problem for computing the reduced-order model operators.

    misfit_ : float
        The squared Frobenius-norm data misfit of the (nonregularized)
        least-squares problem for computing the reduced-order model operators.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)/2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used
        directly in solving the ROM.

    Gc_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : (r,r**3) ndarray or None
        Learned ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used
        directly in solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : callable((r,) ndarray, (m,) ndarray) -> (r,)
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        The signature is f_(x_) if 'B' is not in `modelform` (no inputs) and
        f_(x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and inputs if appropriate) to reduced state. Calculated in fit().
    """
    pass


class IntrusiveContinuousROM(_IntrusiveMixin, _NonparametricMixin,
                             _ContinuousROM):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t, x(t), u(t)),             x(0) = x0.

    The user must specify the model form of the full-order model (FOM)
    operator f and the associated operators; the operators for the reduced
    model (ROM) are computed explicitly by projecting the full-order operators.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax(t).
        'H' : Quadratic state term H(x⊗x)(t).
        'B' : Input term Bu(t).
        For example, modelform=="AB" means f(t,x(t),u(t)) = Ax(t) + Bu(t).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax(t).

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x)(t).

    has_cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x)(t).

    has_inputs : bool
        Whether or not there is a linear input term Bu(t).

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the projected reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    c : (n,) ndarray or None
        FOM constant term, or None if 'c' is not in `modelform`.

    A : (n,n) ndarray or None
        FOM linear state matrix, or None if 'A' is not in `modelform`.

    Hc : (n,n(n+1)/2) ndarray or None
        FOM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`.

    H : (n,n**2) ndarray or None
        FOM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`.

    Gc : (n,n(n+1)(n+2)/6) ndarray or None
        FOM cubic state matrix (compact), or None if 'G' is not in `modelform`.

    G : (n,n**3) ndarray or None
        FOM cubic state matrix (full), or None if 'G' is not in `modelform`.

    B : (n,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)/2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used in
        solving the ROM.

    Gc_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : (r,r**3) ndarray or None
        Learned ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used in
        solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : callable(float, (r,) ndarray, func?) -> (r,) ndarray
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        The signature is f_(t, x_) if 'B' is not in `modelform` (no inputs) and
        f_(t, x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and possibly an input function) to reduced state. Calculated in fit().

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    pass


# Interpolated Operator Inference models --------------------------------------
class InterpolatedInferredDiscreteROM(_InterpolatedMixin, _DiscreteROM):
    """Reduced order model for a high-dimensional discrete dynamical system,
    parametrized by a scalar µ, of the form

        x_{j+1}(µ) = f(x_{j}(µ), u_{j}; µ),     x_{0}(µ) = x0(µ).

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving several
    ordinary least-squares problems, then interpolating those models with
    respect to the scalar parameter µ.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
        'G' : Cubic state term G(x⊗x⊗x).
        'B' : Input term Bu.
        For example, modelform=="AB" means f(x,u) = Ax + Bu.

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax.

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x).

    has_cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x).

    has_inputs : bool
        Whether or not there is a linear input term Bu.

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    s : int
        The number of training parameter samples, hence also the number of
        reduced models computed via inference and used in the interpolation.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    dataconds_ : (s,) ndarray
        Condition numbers of the raw data matrices for each least-squares
        problem.

    dataregconds_ : (s,) ndarray
        Condition numbers of the regularized data matrices for each
        least-squares problem.

    residuals_ : (s,) ndarray
        The squared Frobenius-norm residuals of the regularized least-squares
        problems for computing each set of reduced-order model operators.

    misfits_ : (s,) ndarray
        The squared Frobenius-norm data misfits of the (nonregularized)
        least-squares problems for computing each set of reduced-order model
        operators.

    cs_ : list of s (r,) ndarrays or None
        Learned ROM constant terms, or None if 'c' is not in `modelform`.

    As_ : list of s (r,r) ndarrays or None
        Learned ROM linear state matrices, or None if 'A' not in `modelform`.

    Hcs_ : list of s (r,r(r+1)/2) ndarrays or None
        Learned ROM quadratic state matrices (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    Hs_ : list of s (r,r**2) ndarrays or None
        Learned ROM quadratic state matrices (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hcs_ if desired; not used in
        solving the ROM.

    Gcs_ : list of s (r,r(r+1)(r+2)/6) ndarrays or None
        Learned ROM cubic state matrices (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    Gs_ : list of s (r,r**3) ndarrays or None
        Learned ROM cubic state matrices (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gcs_ if desired; not used in
        solving the ROM.

    Bs_ : list of s (r,m) ndarrays or None
        Learned ROM input matrices, or None if 'B' not in `modelform`.

    fs_ : list of func(float, (r,) ndarray) -> (r,) ndarray
        The complete ROM operators for each parameter sample, defined by
        cs_, As_, and/or Hcs_.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, Vr, µs, Xs, Us=None, P=0):
        """Solve for the reduced model operators via ordinary least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, Xs are assumed to already be projected (r,k).

        µs : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        Xs : list of s (n,k) or (r,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

        Us : list of s (m,k-1) or (k-1,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see utils.lstsq_reg(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        return _InterpolatedMixin.fit(self, InferredDiscreteROM,
                                      Vr, µs, Xs, None, Us, P)

    def predict(self, µ, x0, niters, U=None):
        """Construct a ROM for the parameter µ by interolating the entries of
        the learned models, then step forward this new ROM `niters` steps.

        Parameters
        ----------
        µ : float
            The parameter of interest for the prediction.

        x0 : (n,) or (r,) ndarray
            The initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).

        niters : int
            The number of times to step the system forward.

        U : (m,niters-1) ndarray
            The inputs for the next niters-1 time steps.

        Returns
        -------
        X_ROM : (n,niters) ndarray
            The approximate solutions to the full-order system, including the
            given initial condition.
        """
        # Check modelform and inputs.
        self._check_modelform(trained=True)
        self._check_inputargs(U, 'U')

        model = self(µ)     # See __call__().
        return model.predict(x0, niters, U)


class InterpolatedInferredContinuousROM(_InterpolatedMixin, _ContinuousROM):
    """Reduced order model for a system of high-dimensional ODEs, parametrized
    by a scalar µ, of the form

         dx / dt = f(t, x(t;µ), u(t); µ),       x(0;µ) = x0(µ).

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving several
    ordinary least-squares problems, then interpolating those models with
    respect to the scalar parameter µ.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c(µ)
        'A' : Linear state term A(µ)x(t).
        'H' : Quadratic state term H(µ)(x⊗x)(t).
        'H' : Cubic state term G(µ)(x⊗x⊗x)(t).
        'B' : Input term B(µ)u(t).
        For example, modelform=="cA" means f(t, x(t); µ) = c(µ) + A(µ)x(t;µ).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c(µ).

    has_linear : bool
        Whether or not there is a linear state term A(µ)x(t).

    has_quadratic : bool
        Whether or not there is a quadratic state term H(µ)(x⊗x)(t).

    has_cubic : bool
        Whether or not there is a cubic state term G(µ)(x⊗x⊗x)(t).

    has_inputs : bool
        Whether or not there is a linear input term B(µ)u(t).

    n : int
        The dimension of the original model.

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    s : int
        The number of training parameter samples, hence also the number of
        reduced models computed via inference and used in the interpolation.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    dataconds_ : (s,) ndarray
        Condition numbers of the raw data matrices for each least-squares
        problem.

    dataregconds_ : (s,) ndarray
        Condition numbers of the regularized data matrices for each
        least-squares problem.

    residuals_ : (s,) ndarray
        The squared Frobenius-norm residuals of the regularized least-squares
        problems for computing each set of reduced-order model operators.

    misfits_ : (s,) ndarray
        The squared Frobenius-norm data misfits of the (nonregularized)
        least-squares problems for computing each set of reduced-order model
        operators.

    cs_ : list of s (r,) ndarrays or None
        Learned ROM constant terms, or None if 'c' is not in `modelform`.

    As_ : list of s (r,r) ndarrays or None
        Learned ROM linear state matrices, or None if 'A' not in `modelform`.

    Hcs_ : list of s (r,r(r+1)/2) ndarrays or None
        Learned ROM quadratic state matrices (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    Hs_ : list of s (r,r**2) ndarrays or None
        Learned ROM quadratic state matrices (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hcs_ if desired; not used in
        solving the ROM.

    Gcs_ : list of s (r,r(r+1)(r+2)/6) ndarrays or None
        Learned ROM cubic state matrices (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    Gs_ : list of s (r,r**3) ndarrays or None
        Learned ROM cubic state matrices (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gcs_ if desired; not used in
        solving the ROM.

    Bs_ : list of s (r,m) ndarrays or None
        Learned ROM input matrices, or None if 'B' not in `modelform`.

    fs_ : list of func(float, (r,) ndarray) -> (r,) ndarray
        The complete ROM operators for each parameter sample, defined by
        cs_, As_, and/or Hcs_.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, Vr, µs, Xs, Xdots, Us=None, P=0):
        """Solve for the reduced model operators via ordinary least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, Xs and Xdots are assumed to already be projected (r,k).

        µs : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        Xs : list of s (n,k) or (r,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

        Xdots : list of s (n,k) or (r,k) ndarrays
            Column-wise velocity training data (each column is a snapshot),
            either full order (n rows) ro projected to reduced order (r rows).
            The ith array Xdots[i] corresponds to the ith parameter, µs[i].

        Us : list of s (m,k) or (k,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see utils.lstsq_reg(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        return _InterpolatedMixin.fit(self, InferredContinuousROM,
                                      Vr, µs, Xs, Xdots, Us, P)

    def predict(self, µ, x0, t, u=None, **options):
        """Construct a ROM for the parameter µ by interolating the entries of
        the learned models, then simulate this interpolated ROM with
        scipy.integrate.solve_ivp().

        Parameters
        ----------
        µ : float
            The parameter of interest for the prediction.

        x0 : (n,) or (r,) ndarray
            The initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).

        t : (nt,) ndarray
            The time domain over which to integrate the reduced-order system.

        u : callable or (m,nt) ndarray
            The input as a function of time (preferred) or the input at the
            times `t`. If given as an array, u(t) is approximated by a cubic
            spline interpolating the known data points.

        options
            Arguments for solver.integrate.solve_ivp(), such as the following:
            method : str
                The ODE solver for the reduced-order system.
                * 'RK45' (default): Explicit Runge-Kutta method of order 5(4).
                * 'RK23': Explicit Runge-Kutta method of order 3(2).
                * 'Radau': Implicit Runge-Kutta method of the Radau IIA family
                    of order 5.
                * 'BDF': Implicit multi-step variable-order (1 to 5) method
                    based on a backward differentiation formula for the
                    derivative.
                * 'LSODA': Adams/BDF method with automatic stiffness detection
                    and switching. This wraps the Fortran solver from ODEPACK.
            max_step : float
                The maximimum allowed integration step size.
            See https://docs.scipy.org/doc/scipy/reference/integrate.html.

        Returns
        -------
        X_ROM : (n,nt) ndarray
            The approximate solution to the full-order system over `t`.
        """
        # Check modelform and inputs.
        self._check_modelform(trained=True)
        self._check_inputargs(u, 'u')

        model = self(µ)     # See __call__().
        out = model.predict(x0, t, u, **options)
        self.sol_ = model.sol_
        return out


# Affine intrusive models -----------------------------------------------------
class AffineIntrusiveDiscreteROM(_AffineIntrusiveMixin, _DiscreteROM):
    """Reduced order model for a high-dimensional, parametrized discrete
    dynamical system of the form

        x_{j+1}(µ) = f(x_{j}(µ), u_{j}; µ),     x_{0}(µ) = x0(µ),

    where one or more of the operators that compose f have an affine
    dependence on the parameter, e.g., A(µ) = θ1(µ)A1 + θ2(µ)A2 + θ3(µ)A3.
    The user must specify the model form of the full-order model (FOM)
    operator f and the associated operators; the operators for the reduced
    model (ROM) are explicitly computed by projecting the full-order operators.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
        'G' : Cubic state term G(x⊗x⊗x).
        'B' : Input term Bu.
        For example, modelform=="AB" means f(x,u) = Ax + Bu.

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax.

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x).

    cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x).

    has_inputs : bool
        Whether or not there is a linear input term Bu.

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    c : callable(µ) -> (n,) ndarray; (n,) ndarray; or None
        FOM constant term, or None if 'c' is not in `modelform`.

    A : callable(µ) -> (n,n) ndarray; (n,n) ndarray; or None
        FOM linear state matrix, or None if 'A' is not in `modelform`.

    Hc : callable(µ) -> (n,n(n+1)/2) ndarray; (n,n(n+1)/2) ndarray; or None
        FOM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`.

    H : callable(µ) -> (n,n**2) ndarray; (n,n**2) ndarray; or None
        FOM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`.

    Gc : callable(µ) -> (n,n(n+1)(n+2)/6) ndarray; (n,n(n+1)(n+2)/6) ndarray;
          or None
        FOM cubic state matrix (compact), or None if 'G' is not
        in `modelform`.

    G : callable(µ) -> (n,n**3) ndarray; (n,n**3) ndarray; or None
        FOM cubic state matrix (full size), or None if 'G' is not
        in `modelform`.

    B : callable(µ) -> (n,m) ndarray; (n,m) ndarray; or None
        FOM input matrix, or None if 'B' is not in `modelform`.

    c_ : callable(µ) -> (r,) ndarray; (r,) ndarray; or None
        Computed ROM constant term, or None if 'c' is not in `modelform`.

    A_ : callable(µ) -> (r,r) ndarray; (r,r) ndarray; or None
        Computed ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : callable(µ) -> (r,r(r+1)/2) ndarray; (r,r(r+1)/2) ndarray; or None
        Computed ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : callable(µ) -> (r,r**2) ndarray; (r,r**2) ndarray; or None
        Computed ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used in
        solving the ROM.

    Gc_ : callable(µ) -> (r,r(r+1)(r+2)/6) ndarray; (r,r(r+1)(r+2)/6) ndarray;
          or None
        Computed ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : callable(µ) -> (r,r**3) ndarray; (r,r**3) ndarray; or None
        Computed ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used in
        solving the ROM.

    B_ : callable(µ) -> (r,m) ndarray; (r,m) ndarray; or None
        Computed ROM input matrix, or None if 'B' is not in `modelform`.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def predict(self, µ, x0, niters, U=None):
        """Construct a ROM for the parameter µ by exploiting the affine
        structure of the ROM operators, then step the resulting ROM forward
        `niters` steps.

        Parameters
        ----------
        µ : (p,) ndarray
            The parameter of interest for the prediction.

        x0 : (n,) or (r,) ndarray
            The initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).

        niters : int
            The number of times to step the system forward.

        U : (m,niters-1) ndarray
            The inputs for the next niters-1 time steps.

        Returns
        -------
        X_ROM : (n,niters) ndarray
            The approximate solutions to the full-order system, including the
            given initial condition.
        """
        # Check modelform and inputs.
        self._check_modelform(trained=True)
        self._check_inputargs(U, 'U')

        # TODO: Make sure the parameter µ has the correct dimension.
        # Use the affine structure of the operators to construct a new model.
        model = self(µ)
        return model.predict(x0, niters, U)


class AffineIntrusiveContinuousROM(_AffineIntrusiveMixin, _ContinuousROM):
    """Reduced order model for a high-dimensional, parametrized system of ODEs
    of the form

        dx / dt = f(t, x(t), u(t); µ),          x(0;µ) = x0(µ),

    where one or more of the operators that compose f have an affine
    dependence on the parameter, e.g., A(µ) = θ1(µ)A1 + θ2(µ)A2 + θ3(µ)A3.
    The user must specify the model form of the full-order model (FOM)
    operator f and the associated operators; the operators for the reduced
    model (ROM) are explicitly computed by projecting the full-order operators.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        * 'c' : Constant term c(µ).
        * 'A' : Linear state term A(µ)x(t).
        * 'H' : Quadratic state term H(µ)(x⊗x)(t).
        * 'G' : Cubic state term G(µ)(x⊗x⊗x)(t).
        * 'B' : Linear input term B(µ)u(t).
        For example, modelform=="cA" means f(t, x(t); µ) = c(µ) + A(µ)x(t;µ).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c(µ).

    has_linear : bool
        Whether or not there is a linear term A(µ)x(t).

    has_quadratic : bool
        Whether or not there is a quadratic term H(µ)(x⊗x)(t).

    has_cubic : bool
        Whether or not there is a cubic term G(µ)(x⊗x⊗x)(t).

    has_inputs : bool
        Whether or not there is an input term B(µ)u(t).

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the projected reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    c : callable(µ) -> (n,) ndarray; (n,) ndarray; or None
        FOM constant term, or None if 'c' is not in `modelform`.

    A : callable(µ) -> (n,n) ndarray; (n,n) ndarray; or None
        FOM linear state matrix, or None if 'A' is not in `modelform`.

    Hc : callable(µ) -> (n,n(n+1)/2) ndarray; (n,n(n+1)/2) ndarray; or None
        FOM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`.

    H : callable(µ) -> (n,n**2) ndarray; (n,n**2) ndarray; or None
        FOM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`.

    Gc : callable(µ) -> (n,n(n+1)(n+2)/6) ndarray; (n,n(n+1)(n+2)/6) ndarray;
          or None
        FOM cubic state matrix (compact), or None if 'G' is not in `modelform`.

    G : callable(µ) -> (n,n**3) ndarray; (n,n**3) ndarray; or None
        FOM cubic state matrix (full), or None if 'G' is not in `modelform`.

    B : callable(µ) -> (n,m) ndarray; (n,m) ndarray; or None
        FOM input matrix, or None if 'B' is not in `modelform`.

    c_ : callable(µ) -> (r,) ndarray; (r,) ndarray; or None
        Computed ROM constant term, or None if 'c' is not in `modelform`.

    A_ : callable(µ) -> (r,r) ndarray; (r,r) ndarray; or None
        Computed ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : callable(µ) -> (r,r(r+1)/2) ndarray; (r,r(r+1)/2) ndarray; or None
        Computed ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : callable(µ) -> (r,r**2) ndarray; (r,r**2) ndarray; or None
        Computed ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used in
        solving the ROM.

    Gc_ : callable(µ) -> (r,r(r+1)(r+2)/6) ndarray; (r,r(r+1)(r+2)/6) ndarray;
          or None
        Computed ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : callable(µ) -> (r,r**3) ndarray; (r,r**3) ndarray; or None
        Computed ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used in
        solving the ROM.

    B_ : callable(µ) -> (r,m) ndarray; (r,m) ndarray; or None
        Computed ROM input matrix, or None if 'B' is not in `modelform`.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def predict(self, µ, x0, t, u=None, **options):
        """Construct a ROM for the parameter µ by exploiting the affine
        structure of the ROM operators, then simulate the resulting ROM with
        scipy.integrate.solve_ivp().

        Parameters
        ----------
        µ : (p,) ndarray
            The parameter of interest for the prediction.

        x0 : (n,) or (r,) ndarray
            The initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).

        t : (nt,) ndarray
            The time domain over which to integrate the reduced-order system.

        u : callable or (m,nt) ndarray
            The input as a function of time (preferred) or the input at the
            times `t`. If given as an array, u(t) is approximated by a cubic
            spline interpolating the known data points.

        options
            Arguments for solver.integrate.solve_ivp(), such as the following:
            method : str
                The ODE solver for the reduced-order system.
                * 'RK45' (default): Explicit Runge-Kutta method of order 5(4).
                * 'RK23': Explicit Runge-Kutta method of order 3(2).
                * 'Radau': Implicit Runge-Kutta method of the Radau IIA family
                    of order 5.
                * 'BDF': Implicit multi-step variable-order (1 to 5) method
                    based on a backward differentiation formula for the
                    derivative.
                * 'LSODA': Adams/BDF method with automatic stiffness detection
                    and switching. This wraps the Fortran solver from ODEPACK.
            max_step : float
                The maximimum allowed integration step size.
            See https://docs.scipy.org/doc/scipy/reference/integrate.html.

        Returns
        -------
        X_ROM : (n,nt) ndarray
            The approximate solution to the full-order system over `t`.
        """
        # Check modelform and inputs.
        self._check_modelform(trained=True)
        self._check_inputargs(u, 'u')

        # TODO: Make sure the parameter µ has the correct dimension.
        # Use the affine structure of the operators to construct a new model.
        model = self(µ)
        out = model.predict(x0, t, u, **options)
        self.sol_ = model.sol_
        return out


# Future additions ------------------------------------------------------------
# TODO: Account for state / input interactions (N).
# TODO: save_model() for parametric forms.
# TODO: jacobians for each model form in the continuous case.
# TODO: self.p = parameter size for parametric classes (+ shape checking)
