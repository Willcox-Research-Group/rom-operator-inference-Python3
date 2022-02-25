# core/operators/_affine.py

__all__ = [
    "AffineConstantOperator",
    "AffineLinearOperator",
    "AffineQuadraticOperator",
    # AffineCrossQuadraticOperator",
    "AffineCubicOperator",
]

import abc
import numpy as np

from ._nonparametric import (_BaseNonparametricOperator,
                             ConstantOperator, LinearOperator,
                             QuadraticOperator, CubicOperator)


# TODO: symbol (for printing)
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
            Class of operator to construct, a subclass of
            core.operators._BaseNonparametricOperator.
        coeffs : list of `nterms` callables
            Coefficient scalar-valued functions in the affine expansion.
            Each must take the same sized input and return a scalar.
        matrices : list of `nterms` ndarrays, all of the same shape
            Component matrices in each term of the affine expansion.
        """
        if not issubclass(OperatorClass, _BaseNonparametricOperator):
            raise TypeError(f"invalid operatortype '{OperatorClass.__name__}'")
        self.__opclass = OperatorClass

        if any(not callable(theta) for theta in coeffs):
            raise TypeError("coefficients of affine operator must be callable")
        self.__thetas = coeffs

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
        return self.__thetas

    @property
    def matrices(self):
        """Component matrices in each term of the affine expansion."""
        return self.__matrices

    # @property
    # def shape(self):
    #     """Shape: the shape of the component matrices."""
    #     return self.matrices[0].shape

    @staticmethod
    def validate_coeffs(thetas, mu):
        """Check that each coefficient function 1) is a callable function,
        2) takes in the right sized inputs, and 3) returns scalar values.

        Parameters
        ----------
        mu : float or (p,) ndarray
            A test input for the coefficient functions.
        """
        for theta in thetas:
            if not callable(theta):
                raise TypeError("coefficient functions of affine operator "
                                "must be callable")
            elif not np.isscalar(theta(mu)):
                raise ValueError("coefficient functions of affine operator "
                                 "must return a scalar")

    def __call__(self, mu):
        """Evaluate the affine operator at the given parameter."""
        entries = np.sum([thetai(mu)*Ai for thetai, Ai in zip(
                          self.coefficient_functions, self.matrices)],
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
