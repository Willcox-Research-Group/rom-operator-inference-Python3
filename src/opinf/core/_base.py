# core/_base.py
"""Abstract base classes for reduced-order models."""

__all__ = []

import abc
import numpy as np

from .. import pre
from . import operators

_isparametricop = operators.is_parametric_operator


class _BaseROM(abc.ABC):
    """Base class for all opinf reduced model classes."""
    _MODELFORM_KEYS = "cAHGB"   # Constant, Linear, Quadratic, Cubic, Input.
    _LHS_ARGNAME = "lhs"
    _LHS_LABEL = None
    _STATE_LABEL = None
    _INPUT_LABEL = None

    def __init__(self, modelform):
        """Set the model form (ROM structure)."""
        self.modelform = modelform

    def __str__(self):
        """String representation: structure of the model, dimensions, etc."""
        # Build model structure.
        lhs, q, u = self._LHS_LABEL, self._STATE_LABEL, self._INPUT_LABEL
        out, terms = [], []
        if 'c' in self.modelform:
            terms.append("c")
        if 'A' in self.modelform:
            terms.append(f"A{q}")
        if 'H' in self.modelform:
            terms.append(f"H[{q} ⊗ {q}]")
        if 'G' in self.modelform:
            terms.append(f"G[{q} ⊗ {q} ⊗ {q}]")
        if 'B' in self.modelform:
            terms.append(f"B{u}")
        structure = " + ".join(terms)
        out.append(f"Reduced-order model structure: {lhs} = {structure}")

        # Report dimensions.
        if self.n:
            out.append(f"Full-order dimension    n = {self.n:d}")
        if self.m:
            out.append(f"Input/control dimension m = {self.m:d}")
        if self.r:
            out.append(f"Reduced-order dimension r = {self.r:d}")
            # TODO: out.append(f"Total degrees of freedom = {}")
        return '\n'.join(out)

    def __repr__(self):
        """Unique ID + string representation."""
        uniqueID = f"<{self.__class__.__name__} object at {hex(id(self))}>"
        return f"{uniqueID}\n{str(self)}"

    def _clear(self):
        """Set private attributes as None, erasing any previously stored basis,
        dimensions, or ROM operators.
        """
        self.__m = None if 'B' in self.modelform else 0
        self.__r = None
        self.__basis = None
        self.__c_ = None
        self.__A_ = None
        self.__H_ = None
        self.__G_ = None
        self.__B_ = None
        self._projected_operators_ = ""

    # Properties: modelform ---------------------------------------------------
    @property
    def modelform(self):
        """Structure of the reduced-order model."""
        return self.__form

    @modelform.setter
    def modelform(self, form):
        """Set the modelform, which - if successful - resets the entire ROM."""
        form = ''.join(sorted(form,
                              key=lambda k: self._MODELFORM_KEYS.find(k)))
        for key in form:
            if key not in self._MODELFORM_KEYS:
                raise ValueError(f"invalid modelform key '{key}'; options "
                                 "are " + ', '.join(self._MODELFORM_KEYS))
        self.__form = form
        self._clear()

    # Properties: dimensions --------------------------------------------------
    @property
    def n(self):
        """Dimension of the full-order model."""
        return self.basis.shape[0] if self.basis is not None else None

    @n.setter
    def n(self, n):
        """Setting this dimension is not allowed."""
        raise AttributeError("can't set attribute (n = basis.shape[0])")

    @property
    def m(self):
        """Dimension of the input term, if present."""
        return self.__m

    @m.setter
    def m(self, m):
        """Set input dimension; only allowed if 'B' in modelform
        and the operator B_ is None.
        """
        if 'B' not in self.modelform and m != 0:
            raise AttributeError("can't set attribute ('B' not in modelform)")
        elif self.B_ is not None:
            raise AttributeError("can't set attribute (m = B_.shape[1])")
        self.__m = m

    @property
    def r(self):
        """Dimension of the reduced-order model."""
        return self.__r

    @r.setter
    def r(self, r):
        """Set ROM dimension; only allowed if the basis is None."""
        if self.basis is not None:
            raise AttributeError("can't set attribute (r = basis.shape[1])")
        if any(op is not None for op in self):
            raise AttributeError("can't set attribute (call fit() to reset)")
        self.__r = r

    @property
    def _r2(self):
        """Dimension of the compact quadratic Kronecker product."""
        r = self.r
        return r * (r + 1) // 2

    @property
    def _r3(self):
        """Dimension of the compact cubic Kronecker product."""
        r = self.r
        return r * (r + 1) * (r + 2) // 6

    # Properties: basis -------------------------------------------------------
    @property
    def basis(self):
        """Basis for the linear reduced space (e.g., POD ), of shape (n, r)."""
        return self.__basis

    @basis.setter
    def basis(self, basis):
        """Set the basis, thereby fixing the dimensions n and r."""
        if basis is not None:
            if not isinstance(basis, pre.basis._base._BaseBasis):
                basis = pre.LinearBasis().fit(basis)
            if basis.shape[0] < basis.shape[1]:
                raise ValueError("basis must be n x r with n > r")
            self.__r = basis.shape[1]
        self.__basis = basis

    @basis.deleter
    def basis(self):
        self.__basis = None

    # Properties: reduced-order operators -------------------------------------
    @property
    def c_(self):
        """ROM constant operator, of shape (r,)."""
        return self.__c_

    @c_.setter
    def c_(self, c_):
        self._check_operator_matches_modelform(c_, 'c')
        if c_ is not None:
            if not operators.is_operator(c_):
                c_ = operators.ConstantOperator(c_)
            self._check_rom_operator_shape(c_, 'c')
        self.__c_ = c_

    @property
    def A_(self):
        """ROM linear state operator, of shape (r, r)."""
        return self.__A_

    @A_.setter
    def A_(self, A_):
        # TODO: what happens if model.A_ = something but model.r is None?
        self._check_operator_matches_modelform(A_, 'A')
        if A_ is not None:
            if not operators.is_operator(A_):
                A_ = operators.LinearOperator(A_)
            self._check_rom_operator_shape(A_, 'A')
        self.__A_ = A_

    @property
    def H_(self):
        """ROM quadratic state opeator, of shape (r, r(r+1)/2)."""
        return self.__H_

    @H_.setter
    def H_(self, H_):
        self._check_operator_matches_modelform(H_, 'H')
        if H_ is not None:
            if not operators.is_operator(H_):
                H_ = operators.QuadraticOperator(H_)
            self._check_rom_operator_shape(H_, 'H')
        self.__H_ = H_

    @property
    def G_(self):
        """ROM cubic state operator, of shape (r, r(r+1)(r+2)/6)."""
        return self.__G_

    @G_.setter
    def G_(self, G_):
        self._check_operator_matches_modelform(G_, 'G')
        if G_ is not None:
            if not operators.is_operator(G_):
                G_ = operators.CubicOperator(G_)
            self._check_rom_operator_shape(G_, 'G')
        self.__G_ = G_

    @property
    def B_(self):
        """ROM input operator, of shape (r, m)."""
        return self.__B_

    @B_.setter
    def B_(self, B_):
        self._check_operator_matches_modelform(B_, 'B')
        if B_ is not None:
            if not operators.is_operator(B_):
                B_ = operators.LinearOperator(B_)
            self._check_rom_operator_shape(B_, 'B')
        self.__B_ = B_

    def _set_operators(self, basis,
                       c_=None, A_=None, H_=None, G_=None, B_=None):
        """Set the ROM operators and corresponding dimensions.

        Parameters
        ----------
        basis : (n, r) ndarray or None
            Basis for the linear reduced space (e.g., POD basis matrix).
            If None, then r is inferred from one of the reduced operators.
        c_ : (r,) ndarray or None
            Reduced-order constant term.
        A_ : (r, r) ndarray or None
            Reduced-order linear state matrix.
        H_ : (r, r(r+1)/2) ndarray or None
            Reduced-order (compact) quadratic state matrix.
        G_ : (r, r(r+1)(r+2)/6) ndarray or None
            Reduced-order (compact) cubic state matrix.
        B_ : (r, m) ndarray or None
            Reduced-order input matrix.

        Returns
        -------
        self
        """
        self._clear()
        operators = [c_, A_, H_, G_, B_]

        # Save the low-dimensional basis. Sets self.n and self.r if given.
        self.basis = basis

        # Set the input dimension 'm'.
        if 'B' in self.modelform:
            if B_ is not None:
                self.m = 1 if len(B_.shape) == 1 else B_.shape[1]
        else:
            self.m = 0

        # Determine the ROM dimension 'r' if no basis was given.
        if basis is None:
            self.r = None
            for op in operators:
                if op is not None:
                    self.r = op.shape[0]
                    break

        # Insert the operators. Raises exceptions if shapes are bad, etc.
        self.c_, self.A_, self.H_, self.G_, self.B_, = c_, A_, H_, G_, B_

        return self

    def __iter__(self):
        for key in self.modelform:
            yield getattr(self, f"{key}_")

    def __eq__(self, other):
        """Two ROMs are equal if they are of the same type and have the same
        bases and reduced-order operators.
        """
        if self.__class__ != other.__class__:
            return False
        if self.modelform != other.modelform:
            return False
        if self.basis is None:
            if other.basis is not None:
                return False
        else:
            if other.basis is None:
                return False
            if self.basis != other.basis:
                return False
        for opL, opR in zip(self, other):
            if not (opL is opR is None) and opL != opR:
                return False
        return True

    # Validation methods ------------------------------------------------------
    def _check_operator_matches_modelform(self, operator, key):
        """Raise a TypeError if the given operator is incompatible with the
        modelform.

        Parameters
        ----------
        operator : ndarray or None
            Operator (ndarray, etc.) data to be attached as an attribute.
        key : str
            A single character from 'cAHGB', indicating which operator to set.
        """
        if (key in self.modelform) and (operator is None):
            raise TypeError(f"'{key}' in modelform requires {key}_ != None")
        if (key not in self.modelform) and (operator is not None):
            raise TypeError(f"'{key}' not in modelform requires {key}_ = None")

    def _check_rom_operator_shape(self, operator, key):
        """Ensure that the given operator has the correct shape."""
        # First, check that the required dimensions exist.
        if self.r is None:
            raise AttributeError("no reduced dimension 'r' (call fit())")
        if key == 'B' and (self.m is None):
            raise AttributeError("no input dimension 'm' (call fit())")
        r, m = self.r, self.m

        # Check operator shape.
        if key == "c" and operator.shape != (r,):
            raise ValueError(f"c_.shape = {operator.shape}, "
                             f"must be (r,) with r = {r}")
        elif key == "A" and operator.shape != (r, r):
            raise ValueError(f"A_.shape = {operator.shape}, "
                             f"must be (r, r) with r = {r}")
        elif key == "H" and operator.shape != (r, r*(r + 1)//2):
            raise ValueError(f"H_.shape = {operator.shape}, must be "
                             f"(r, r(r+1)/2) with r = {r}")
        elif key == "G" and operator.shape != (r, r*(r + 1)*(r + 2)//6):
            raise ValueError(f"G_.shape = {operator.shape}, must be "
                             f"(r, r(r+1)(r+2)/6) with r = {r}")
        elif key == "B" and operator.shape != (r, m):
            raise ValueError(f"B_.shape = {operator.shape}, must be "
                             f"(r, m) with r = {r}, m = {m}")

    def _check_inputargs(self, u, argname):
        """Check that the modelform agrees with input arguments."""
        if 'B' in self.modelform and u is None:
            raise ValueError(f"argument '{argname}' required"
                             " since 'B' in modelform")

        if 'B' not in self.modelform and u is not None:
            raise ValueError(f"argument '{argname}' invalid"
                             " since 'B' in modelform")

    def _check_is_trained(self):
        """Ensure that the model is trained and ready for prediction."""
        try:
            for key in self.modelform:
                op = getattr(self, key+'_')
                self._check_operator_matches_modelform(op, key)
                self._check_rom_operator_shape(op, key)
        except Exception as e:
            raise AttributeError("model not trained (call fit())") from e

    # Dimensionality reduction ------------------------------------------------
    def _project_operators(self, known_operators):
        """Project known full-order operators to the reduced-order space.

        Parameters
        ----------
        known_operators : dict(str -> ndarray)
            Dictionary of known full-order or reduced-order operators.
            Corresponding reduced-order operators are computed directly
            through projection; remaining operators are inferred.
            Keys must match the modelform, values are ndarrays:
            * 'c': (n,) constant term c.
            * 'A': (n, n) linear state matrix A.
            * 'H': (n, n**2) quadratic state matrix H.
            * 'G': (n, n**3) cubic state matrix G.
            * 'B': (n, m) input matrix B.
        """
        # Do nothing if there are no operators to project.
        if known_operators is None or len(known_operators) == 0:
            return

        # If there is no basis, we must have only reduced-order operators.
        if self.basis is None:
            # Require r so we can tell between full and reduced order.
            if self.r is None:
                raise ValueError("dimension r required to use known operators")
            elif not (all(op.shape[0] == self.r
                      for op in known_operators.values())):
                raise ValueError("basis required "
                                 "to project full-order operators")

        # Validate the keys of the operator dictionary.
        surplus = [repr(key)
                   for key in known_operators.keys()
                   if key not in self.modelform]
        if surplus:
            _noun = "key" + ('' if len(surplus) == 1 else 's')
            raise KeyError(f"invalid operator {_noun} {', '.join(surplus)}")

        # Project full-order operators.
        Vr = self.basis.entries
        if ('H' in self.modelform) or ('G' in self.modelform):
            Vr2 = np.kron(Vr, Vr)

        if 'c' in known_operators:          # Constant term.
            c = known_operators['c']        # c = multiple of vector of ones.
            if np.isscalar(c):
                c = c * Vr.sum(axis=0)
            if c.shape[0] != self.r:
                c = Vr.T @ c
            self.c_ = c

        if 'A' in known_operators:          # Linear state matrix.
            A = known_operators['A']
            if isinstance(A, str) and A.lower() in ("i", "id", "identity"):
                A = 1
            if np.isscalar(A):              # A = multiple of identity.
                A = A * np.eye(self.r)
            if A.shape[0] != self.r:
                A = Vr.T @ A @ Vr
            self.A_ = A

        if 'H' in known_operators:          # Quadratic state matrix.
            H = known_operators['H']
            # TODO: fast projection.
            # TODO: special case for q^2.
            if H.shape[0] != self.r:
                H = Vr.T @ H @ Vr2
            self.H_ = H

        if 'G' in known_operators:          # Cubic state matrix.
            G = known_operators['G']
            # TODO: fast projection?
            # TODO: special case for q^3.
            if G.shape[0] != self.r:
                G = Vr.T @ G @ np.kron(Vr, Vr2)
            self.G_ = G

        if 'B' in known_operators:          # Linear input matrix.
            B = known_operators['B']
            if B.ndim == 1:
                B = B.reshape((-1, 1))
            self.m = B.shape[1]
            if B.shape[0] != self.r:
                B = Vr.T @ B
            self.B_ = B

        # Save keys of known operators.
        self._projected_operators_ = ''.join(known_operators.keys())

    def encode(self, state, label="argument"):
        """Map high-dimensional states to low-dimensional latent coordinates.

        Parameters
        ----------
        state : (n, ...) or (r, ...) ndarray
            High- or low-dimensional state vector or a collection of these.
            If state.shape[0] == r (already low-dimensional), do nothing.
        label : str
            Name for state (used only in error reporting).

        Returns
        -------
        state_ : (r, ...) ndarray
            Low-dimensional encoding of `state`.
        """
        if self.r is None:
            raise AttributeError("reduced dimension not set")
        if state.shape[0] not in (self.r, self.n):
            if self.basis is None:
                raise AttributeError("basis not set")
            raise ValueError(f"{label} not aligned with basis")
        return self.basis.encode(state) if state.shape[0] == self.n else state

    def decode(self, state_, label="argument"):
        """Map low-dimensional latent coordinates to high-dimensional states.

        Parameters
        ----------
        state_ : (r, ...) ndarray
            Low-dimensional latent coordinate vector or a collection of these.
        label : str
            Name for state_ (used only in error reporting).

        Returns
        -------
        state : (n, ...) ndarray
            High-dimensional decoding of `state_`.
        """
        if self.basis is None:
            raise AttributeError("basis not set")
        if state_.shape[0] != self.r:
            raise ValueError(f"{label} not aligned with basis")
        return self.basis.decode(state_)

    # def project(self, state):
    #     """Project high-dimensional states to the subspace that can be
    #     represented by the basis by encoding the state in low-dimensional
    #     latent coordinates, then decoding those coordinates:
    #     project(Q) = decode(encode(Q)).
    #
    #     Parameters
    #     ----------
    #     state : (n, ...) ndarray
    #         High-dimensional state vector or a collection of these.
    #
    #     Returns
    #     -------
    #     state_projected : (n, ...) ndarray
    #         High-dimensional state vector, or a collection of k such vectors
    #         organized as the columns of a matrix, projected to the range of
    #         the basis.
    #     """
    #     return self.decode(self.encode(state))

    # ROM evaluation ----------------------------------------------------------
    def evaluate(self, state_, input_=None):
        """Evaluate the right-hand side of the model, i.e., the f() of

        * g = f(q, u)                   (steady state)
        * q_{j+1} = f(q_{j}, u_{j})     (discrete time)
        * dq / dt = f(q(t), u(t))       (continuous time)

        Parameters
        ----------
        state_ : (r,) ndarray
            Low-dimensional state vector.
        input_ : (m,) ndarray or None
            Input vector corresponding to the state.

        Returns
        -------
        f(state_, input_): (r,) ndarray
            Evaluation of the right-hand side of the model.
        """
        out = np.zeros(self.r, dtype=float)
        if 'c' in self.modelform:
            out += self.c_()
        if 'A' in self.modelform:
            out += self.A_(state_)
        if 'H' in self.modelform:
            out += self.H_(state_)
        if 'G' in self.modelform:
            out += self.G_(state_)
        if 'B' in self.modelform:
            out += self.B_(input_)
        return out

    # TODO: jacobian(self, state_)

    # Abstract public methods (must be implemented by child classes) ----------
    @abc.abstractmethod
    def fit(*args, **kwargs):
        """Train the reduced-order model with the specified data."""
        raise NotImplementedError                       # pragma: no cover

    @abc.abstractmethod
    def predict(*args, **kwargs):
        """Solve the reduced-order model under specified conditions."""
        raise NotImplementedError                       # pragma: no cover

    # Model persistence (not required but suggested) --------------------------
    def save(*args, **kwargs):
        """Save the reduced-order structure / operators in HDF5 format."""
        raise NotImplementedError("use pickle/joblib")

    def load(*args, **kwargs):
        """Load a previously saved reduced-order model from an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")


class _BaseParametricROM(_BaseROM):
    """Base class for all parametric reduced-order model classes."""
    # Must be specified by child classes.
    _ModelClass = NotImplemented

    # ModelClass properties ---------------------------------------------------
    @property
    def _LHS_ARGNAME(self):                             # pragma: no cover
        return self._ModelClass._LHS_ARGNAME

    @property
    def _LHS_LABEL(self):                               # pragma: no cover
        return self._ModelClass._LHS_LABEL

    @property
    def _STATE_LABEL(self):                             # pragma: no cover
        return self._ModelClass._STATE_LABEL

    @property
    def _INPUT_LABEL(self):                             # pragma: no cover
        return self._ModelClass._INPUT_LABEL

    @property
    def ModelClass(self):
        """Class of nonparametric ROM to represent this parametric ROM
        at a particular parameter, a subclass of core._base._BaseROM:
        >>> type(MyParametricROM(init_args).fit(fit_args)(parameter_value)).
        """
        return self._ModelClass

    # Constructor -------------------------------------------------------------
    def __init__(self, modelform):
        """Set the modelform.

        Parameters
        ----------
        modelform : str
            See _BaseROM.modelform.
        """
        _BaseROM.__init__(self, modelform)

        # Valiate the ModelClass.
        if not issubclass(self.ModelClass, _BaseROM):
            raise RuntimeError("invalid ModelClass "
                               f"'{self.ModelClass.__name__}'")

    def _clear(self):
        """Set private attributes as None, erasing any previously stored basis,
        dimensions, or ROM operators.
        """
        _BaseROM._clear(self)
        self.__p = None

    # Properties: dimensions --------------------------------------------------
    @property
    def p(self):
        """Dimension of the parameter space."""
        return self.__p

    def _set_parameter_dimension(self, parameters):
        """Extract and save the dimension of the parameter space."""
        shape = np.shape(parameters)
        if len(shape) == 1:
            self.__p = 1
        elif len(shape) == 2:
            self.__p = shape[1]
        else:
            raise ValueError("parameter values must be scalars or 1D arrays")

    # Parametric evaluation ---------------------------------------------------
    def __call__(self, parameter):
        """Construct a non-parametric ROM at the given parameter value."""
        # Evaluate the parametric operators at the parameter value.
        c_ = self.c_(parameter) if _isparametricop(self.c_) else self.c_
        A_ = self.A_(parameter) if _isparametricop(self.A_) else self.A_
        H_ = self.H_(parameter) if _isparametricop(self.H_) else self.H_
        G_ = self.G_(parameter) if _isparametricop(self.G_) else self.G_
        B_ = self.B_(parameter) if _isparametricop(self.B_) else self.B_

        # Construct a nonparametric ROM with the evaluated operators.
        rom = self.ModelClass(self.modelform)
        return rom._set_operators(basis=self.basis,
                                  c_=c_, A_=A_, H_=H_, G_=G_, B_=B_)

    def evaluate(self, parameter, *args, **kwargs):
        """Evaluate the right-hand side of the model at the given parameter."""
        return self(parameter).evaluate(*args, **kwargs)

    def predict(self, parameter, *args, **kwargs):
        """Solve the reduced-order model at the given parameter."""
        return self(parameter).predict(*args, **kwargs)
