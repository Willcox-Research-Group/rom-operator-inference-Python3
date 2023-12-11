# models/monolithic/parametric/_base.py
"""Base class for monolithic parametric dynamical systems models."""

__all__ = []

# import abc
import numpy as np

from .._base import _MonolithicModel
from .... import errors
from .... import operators_new as _operators


class _ParametricMonolithicModel(_MonolithicModel):
    """Base class for parametric monolithic models.

    Parent class: :class:`_MonolithicModel`

    Child classes: (**TODO**)

    * ``ParametricSteadyModel``
    * ``ParametricDiscreteModel``
    * ``ParametricContinuousModel``
    """

    _ModelClass = NotImplemented  # Must be specified by child classes.

    # ModelClass properties ---------------------------------------------------
    @property
    def _LHS_ARGNAME(self):  # pragma: no cover
        """Name of LHS argument in fit(), e.g., "ddts"."""
        return self._ModelClass._LHS_ARGNAME

    @property
    def _LHS_LABEL(self):  # pragma: no cover
        """String representation of LHS, e.g., "dq / dt"."""
        return self._ModelClass._LHS_LABEL

    @property
    def _STATE_LABEL(self):  # pragma: no cover
        """String representation of state, e.g., "q(t)"."""
        return self._ModelClass._STATE_LABEL

    @property
    def _INPUT_LABEL(self):  # pragma: no cover
        """String representation of input, e.g., "u(t)"."""
        return self._ModelClass._INPUT_LABEL

    @property
    def ModelClass(self):
        """Nonparametric model class that represents this parametric model
        when evaluated at a particular parameter value, a subclass of
        :class:`_MonolithicModel`.

        Examples
        --------
        >>> model = MyParametricModel(init_args).fit(fit_args)
        >>> model_evaluated = model.evaluate(parameter_value).
        >>> type(model_evaluated) is MyParametricModel.ModelClass
        True
        """
        return self._ModelClass

    # Constructor -------------------------------------------------------------
    def __init__(self, modelform):
        """Define the model structure.

        Parameters
        ----------
        operators : list of :mod:`opinf.operators` objects
            Operators comprising the terms of the model.
        """
        # Validate the ModelClass.
        if not issubclass(self.ModelClass, _MonolithicModel):
            raise RuntimeError(
                "invalid ModelClass " f"'{self.ModelClass.__name__}'"
            )

        _MonolithicModel.__init__(self, modelform)

    def _clear(self):
        """Set private attributes as None, erasing any previously stored basis,
        dimensions, or ROM operators.
        """
        _MonolithicModel._clear(self)
        self.__p = None

    # Properties: operators ---------------------------------------------------
    _operator_abbreviations = dict()

    def _isvalidoperator(self, op):
        """Only interpolated operators are allowed CURRENTLY."""
        return type(op) in [
            _operators.InterpolatedConstantOperator,
            _operators.InterpolatedLinearOperator,
            _operators.InterpolatedQuadraticOperator,
            _operators.InterpolatedCubicOperator,
            _operators.InterpolatedInputOperator,
            _operators.InterpolatedStateInputOperator,
        ]

    def _get_operator_of_type(self, OpClass):
        """Return the first operator in the model corresponding to the
        operator class ``OpClass``.
        """

    # Properties: dimensions --------------------------------------------------
    @staticmethod
    def _check_parameter_dimension_consistency(ops):
        """Ensure all operators have the same parameter dimension."""
        ps = {
            op.parameter_dimension
            for op in ops
            if op.parameter_dimension is not None
        }
        if len(ps) > 1:
            raise errors.DimensionalityError(
                "operators not aligned "
                "(parameter dimension must be the same for all operators)"
            )
        return ps.pop() if len(ps) == 1 else None

    @property
    def state_dimension(self):
        """Dimension :math:`r` of the state."""
        return self.__r

    @state_dimension.setter
    def state_dimension(self, r):
        """Set the state dimension.
        Not allowed if any existing operators have ``state_dimension != r``.
        """
        if self.__operators is not None:
            for op in self.operators:
                if op.entries is not None and op.state_dimension != r:
                    raise AttributeError(
                        "can't set attribute "
                        f"(existing operators have r = {self.__r})"
                    )
        self.__r = r

    @property
    def parameter_dimension(self):
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
    # def evaluate(self, parameter):
    #     """Construct a non-parametric ROM at the given parameter value."""
    #     # Evaluate the parametric operators at the parameter value.
    #     c_ = self.c_(parameter) if _isparametricop(self.c_) else self.c_
    #     A_ = self.A_(parameter) if _isparametricop(self.A_) else self.A_
    #     H_ = self.H_(parameter) if _isparametricop(self.H_) else self.H_
    #     G_ = self.G_(parameter) if _isparametricop(self.G_) else self.G_
    #     B_ = self.B_(parameter) if _isparametricop(self.B_) else self.B_

    #     # Construct a nonparametric ROM with the evaluated operators.
    #     rom = self.ModelClass(self.modelform)
    #     return rom._set_operators(basis=self.basis,
    #                               c_=c_, A_=A_, H_=H_, G_=G_, B_=B_)

    def rhs(self, parameter, state, input_=None):
        """Evaluate the right-hand side of the model at the given parameter."""
        return self(parameter).rhs(state, input_)  # pragma: no cover

    def jacobian(self, parameter, state, input_):
        """Evaluate the Jacobian of the model at the given parameter."""
        return self(parameter).jacobian(state, input_)  # pragma: no cover

    def predict(self, parameter, *args, **kwargs):
        """Solve the reduced-order model at the given parameter."""
        return self(parameter).predict(*args, **kwargs)
