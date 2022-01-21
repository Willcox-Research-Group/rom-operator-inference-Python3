"""Operator classes for the individual components of polynomial models.

That is, for models with the form
    dq / dt = c + Aq(t) + H[q(t) ⊗ q(t)] + G[q(t) ⊗ q(t) ⊗ q(t)],
these classes represent the operators c (constant), A (linear), H (quadratic),
and G (cubic).

Classes
-------
* _BaseOperator
* _AffineMixin(_ParametricMixin)
"""

__all__ = [
            "ConstantOperator",
            "LinearOperator",
            "QuadraticOperator",
            "CubicOperator",
          ]

import abc
import numpy as np

from ..utils import kron2c_indices, kron3c_indices, compress_H, compress_G


# Base class ==================================================================
class _BaseOperator(abc.ABC):
    """Base class for operators that are part of reduced-order models.
    Call the instantiated object to evaluate the operator on an input.

    Attributes
    ----------
    entries : ndarray
        Actual NumPy array representing the operator.
    shape : tuple
        Shape of the operator entries array.
    """
    @abc.abstractmethod
    def __init__(self, entries):
        """Set operator entries."""
        self.__entries = entries

    @staticmethod
    def _validate_entries(entries):
        """Ensure argument is a NumPy array and screen for NaN, Inf entries."""
        if not isinstance(entries, np.ndarray):
            raise TypeError("operator entries must be NumPy array")
        if np.any(np.isnan(entries)):
            raise ValueError("operator entries must not be NaN")
        elif np.any(np.isinf(entries)):
            raise ValueError("operator entries must not be Inf")

    @property
    def entries(self):
        """Discrete representation of the operator."""
        return self.__entries

    @property
    def shape(self):
        """Shape of the operator."""
        return self.entries.shape

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __eq__(self, other):
        """Test whether two Operator objects are numerically equal."""
        if not isinstance(other, self.__class__):
            return False
        return np.all(self.entries == other.entries)


# Non-parametric operators ====================================================
class ConstantOperator(_BaseOperator):
    """Constant terms."""
    def __init__(self, entries):
        self._validate_entries(entries)

        # Flatten operator if needed or report dimension error.
        if entries.ndim > 1:
            if entries.ndim == 2 and 1 in entries.shape:
                entries = np.ravel(entries)
            else:
                raise ValueError("constant operator must be one-dimensional")

        _BaseOperator.__init__(self, entries)

    def __call__(self, *args):
        return self.entries


class LinearOperator(_BaseOperator):
    """Linear state or input operator.

    Example
    -------
    >>> import numpy as np
    >>> A = opinf._core.LinearOperator(np.random.random((10, 10)))
    >>> q = np.random.random(10)
    >>> A(q)                        # Evaluate Aq.
    """
    def __init__(self, entries, square=False):
        """Check dimensions and set operator entries."""
        self._validate_entries(entries)

        if entries.ndim == 1:
            entries = entries.reshape((-1, 1))
        if square and (entries.shape[0] != entries.shape[1]):
            raise ValueError("expected square array for linear operator")
        if entries.ndim != 2:
            raise ValueError("linear operator must be two-dimensional")

        _BaseOperator.__init__(self, entries)

    def __call__(self, q):
        return self.entries @ np.atleast_1d(q)


class QuadraticOperator(_BaseOperator):
    """Operator for quadratic interactions of a state/input with itself
    (compact Kronecker).

    Example
    -------
    >>> import numpy as np
    >>> H = opinf._core.QuadraticOperator(np.random.random((10, 100)))
    >>> q = np.random.random(10)
    >>> H(q)                        # Evaluate H[q ⊗ q].
    """
    def __init__(self, entries):
        """Check dimensions and set operator entries."""
        self._validate_entries(entries)

        # TODO: allow reshaping from three-dimensional tensor?
        if entries.ndim != 2:
            raise ValueError("quadratic operator must be two-dimensional")
        r1, r2 = entries.shape
        # TODO: relax this requirement?
        # If so, need to try to compress if r2 is not a perfect square.
        if r2 == r1**2:
            entries = compress_H(entries)
        elif r2 != r1 * (r1 + 1) // 2:
            raise ValueError("invalid dimensions for quadratic operator")
        self._mask = kron2c_indices(r1)

        _BaseOperator.__init__(self, entries)

    def __call__(self, q):
        return self.entries @ np.prod(np.atleast_1d(q)[self._mask], axis=1)


class CrossQuadraticOperator(QuadraticOperator):
    """Quadratic terms of different states / inputs (full Kronecker)."""
    def __init__(self, entries):
        self._validate_entries(entries)

        _BaseOperator.__init__(self, entries)

    def __call__(self, q1, q2):
        return self.entries @ np.kron(q1, q2)
        # TODO: what about two-dimensional inputs?
        # la.khatri_rao() will do this, but that *requires* that the
        # inputs q1 and q2 are both two-dimensional.
        # TODO: and what about scalar inputs? (special case of r=1 or m=1).


class CubicOperator(_BaseOperator):
    """Cubic terms of a state/input with itself (compact Kronecker).

    Example
    -------
    >>> import numpy as np
    >>> G = opinf._core.CubicOperator(np.random.random((10, 1000)))
    >>> q = np.random.random(10)
    >>> G(q)                        # Evaluate G[q ⊗ q ⊗ q].
    """
    def __init__(self, entries):
        self._validate_entries(entries)

        # TODO: allow reshaping from three-dimensional tensor?
        if entries.ndim != 2:
            raise ValueError("cubic operator must be two-dimensional")
        r1, r2 = entries.shape
        # TODO: relax this requirement?
        # If so, need to try to compress if r2 is not a perfect square.
        if r2 == r1**3:
            entries = compress_G(entries)
        elif r2 != r1 * (r1 + 1) * (r1 + 2) // 6:
            raise ValueError("invalid dimensions for cubic operator")
        self._mask = kron3c_indices(r1)

        _BaseOperator.__init__(self, entries)

    def __call__(self, q):
        return self.entries @ np.prod(np.atleast_1d(q)[self._mask], axis=1)


# Affine parametric operators =================================================
class _BaseAffineOperator(abc.ABC):
    """Base class for representing operators with affine structure, i.e.,

        A(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * A_{i}.

    The matrix A(µ) is constructed by calling the object once the coefficient
    functions and component matrices are set.

    Attributes
    ----------
    coefficient_functions : list of `nterms` callables
        Coefficient scalar-valued functions in the affine expansion.
        Each must take the same sized input and return a scalar.
    matrices : list of `nterms` ndarrays, all of the same shape
        Component matrices in each term of the affine expansion.
    """
    @abc.abstractmethod
    def __init__(self, OperatorClass, coeffs, matrices, **kwargs):
        """Save the coefficient functions and component matrices.

        Parameters
        ----------
        OperatorClass : class
            Class of operator to construct, a subclass of _core._BaseOperator.
        coeffs : list of `nterms` callables
            Coefficient scalar-valued functions in the affine expansion.
            Each must take the same sized input and return a scalar.
        matrices : list of `nterms` ndarrays, all of the same shape
            Component matrices in each term of the affine expansion.
        """
        if not issubclass(OperatorClass, _BaseOperator):
            raise TypeError(f"invalid operatortype '{OperatorClass.__name__}'")
        self.__opclass = OperatorClass

        if any(not callable(θ) for θ in coeffs):
            raise TypeError("coefficients of affine operator must be callable")
        self.__θs = coeffs

        # Check that the right number of terms are included.
        # if (n_coeffs := len(coeffs) != (n_matrices := len(matrices)):
        n_coeffs, n_matrices = len(coeffs), len(matrices)
        if n_coeffs != n_matrices:
            raise ValueError(f"{n_coeffs} = len(coeffs) "
                             f"!= len(matrices) = {n_matrices}")

        # Check that each matrix in the list has the same shape.
        shape = matrices[0].shape
        if any(A.shape != shape for A in matrices):
            raise ValueError("affine component matrix shapes do not match")

        self.__matrices = matrices
        self.__kwargs = kwargs

    @property
    def coefficient_functions(self):
        """Coefficient scalar-valued functions in the affine expansion."""
        return self.__θs

    @property
    def matrices(self):
        """Component matrices in each term of the affine expansion."""
        return self.__matrices

    # @property
    # def shape(self):
    #     """Shape: the shape of the component matrices."""
    #     return self.matrices[0].shape

    @staticmethod
    def validate_coeffs(θs, µ):
        """Check that each coefficient function 1) is a callable function,
        2) takes in the right sized inputs, and 3) returns scalar values.

        Parameters
        ----------
        µ : float or (p,) ndarray
            A test input for the coefficient functions.
        """
        for θ in θs:
            if not callable(θ):
                raise TypeError("coefficient functions of affine operator "
                                "must be callable")
            elif not np.isscalar(θ(µ)):
                raise ValueError("coefficient functions of affine operator "
                                 "must return a scalar")

    def __call__(self, µ):
        """Evaluate the affine operator at the given parameter."""
        entries = np.sum([θi(µ)*Ai for θi,Ai in zip(self.coefficient_functions,
                                                    self.matrices)],
                         axis=0)
        return self.__opclass(entries, **self.__kwargs)

    def __len__(self):
        """Length: number of terms in the affine expansion."""
        return len(self.coefficient_functions)

    def __eq__(self, other):
        """Test whether the component matrices of two AffineOperator objects
        are numerically equal. Coefficient functions are *NOT* compared.
        """
        if not isinstance(other, self.__class__):
            return False
        if self.__opclass is not other.__opclass:
            return False
        if len(self) != len(other):
            return False
        return all(np.all(left == right)
                   for left, right in zip(self.matrices, other.matrices))


class AffineConstantOperator(_BaseAffineOperator):
    """Constant operator with affine structure, i.e.,

        c(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * c_{i}.

    The vector c(µ) is constructed by calling the object once the coefficient
    functions and component matrices are set.

    Attributes
    ----------
    coefficient_functions : list of `nterms` callables
        Coefficient scalar-valued functions in the affine expansion.
        Each must take the same sized input and return a scalar.
    vectors : list of `nterms` one-dimensional ndarrays
        Component vectors in each term of the affine expansion.
    """
    def __init__(self, coeffs, vectors):
        """Save the coefficient functions and component vectors.

        Parameters
        ----------
        coeffs : list of `nterms` callables
            Coefficient scalar-valued functions in the affine expansion.
            Each must take the same sized input and return a scalar.
        vectors : list of `nterms` one-dimensional ndarrays
            Component vectors in each term of the affine expansion.
        """
        _BaseAffineOperator.__init__(self, ConstantOperator, coeffs, vectors)


class AffineLinearOperator(_BaseAffineOperator):
    """Linear operator with affine structure, i.e.,

        A(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * A_{i}.

    The matrix A(µ) is constructed by calling the object once the coefficient
    functions and component matrices are set.

    Attributes
    ----------
    coefficient_functions : list of `nterms` callables
        Coefficient scalar-valued functions in the affine expansion.
        Each must take the same sized input and return a scalar.
    matrices : list of `nterms` ndarrays, all of the same shape
        Component matrices in each term of the affine expansion.
    """
    def __init__(self, coeffs, matrices):
        """Save the coefficient functions and component matrices.

        Parameters
        ----------
        coeffs : list of `nterms` callables
            Coefficient scalar-valued functions in the affine expansion.
            Each must take the same sized input and return a scalar.
        matrices : list of `nterms` ndarrays, all of the same shape
            Component matrices in each term of the affine expansion.
        """
        _BaseAffineOperator.__init__(self, LinearOperator, coeffs, matrices)


class AffineQuadraticOperator(_BaseAffineOperator):
    """Quadratic operator with affine structure, i.e.,

        H(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * H_{i}.

    The matrix H(µ) is constructed by calling the object once the coefficient
    functions and component matrices are set.

    Attributes
    ----------
    coefficient_functions : list of `nterms` callables
        Coefficient scalar-valued functions in the affine expansion.
        Each must take the same sized input and return a scalar.
    matrices : list of `nterms` ndarrays, all of the same shape
        Component matrices in each term of the affine expansion.
    """
    def __init__(self, coeffs, matrices):
        """Save the coefficient functions and component matrices.

        Parameters
        ----------
        coeffs : list of `nterms` callables
            Coefficient scalar-valued functions in the affine expansion.
            Each must take the same sized input and return a scalar.
        matrices : list of `nterms` ndarrays, all of the same shape
            Component matrices in each term of the affine expansion.
        """
        _BaseAffineOperator.__init__(self, QuadraticOperator, coeffs, matrices)


class AffineCubicOperator(_BaseAffineOperator):
    """Cubic operator with affine structure, i.e.,

        G(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * G_{i}.

    The matrix G(µ) is constructed by calling the object once the coefficient
    functions and component matrices are set.

    Attributes
    ----------
    coefficient_functions : list of `nterms` callables
        Coefficient scalar-valued functions in the affine expansion.
        Each must take the same sized input and return a scalar.
    matrices : list of `nterms` ndarrays, all of the same shape
        Component matrices in each term of the affine expansion.
    """
    def __init__(self, coeffs, matrices):
        """Save the coefficient functions and component matrices.

        Parameters
        ----------
        coeffs : list of `nterms` callables
            Coefficient scalar-valued functions in the affine expansion.
            Each must take the same sized input and return a scalar.
        matrices : list of `nterms` ndarrays, all of the same shape
            Component matrices in each term of the affine expansion.
        """
        _BaseAffineOperator.__init__(self, CubicOperator, coeffs, matrices)


# Interpolating parametric operators ==========================================
# TODO
