# models/mono/_nonparametric.py
"""Nonparametric monolithic dynamical systems models."""

__all__ = [
    "SteadyModel",
    "DiscreteModel",
    "ContinuousModel",
]

import abc
import warnings
import numpy as np
import scipy.integrate as spintegrate
import scipy.interpolate as spinterpolate

from ._base import _Model
from ... import errors, utils
from ... import operators as _operators


# Base class ==================================================================
class _NonparametricModel(_Model):
    """Base class for nonparametric monolithic models.

    Parent class: :class:`opinf.models.mono._base._Model`

    Child classes:

    * :class:`opinf.models.DiscreteModel`
    * :class:`opinf.models.ContinuousModel`
    """

    _LHS_ARGNAME = "lhs"  # Name of LHS argument in fit(), e.g., "ddts".
    _LHS_LABEL = None  # String representation of LHS, e.g., "dq / dt".
    _STATE_LABEL = None  # String representation of state, e.g., "q(t)".
    _INPUT_LABEL = None  # String representation of input, e.g., "u(t)".

    # Properties: operators ---------------------------------------------------
    _operator_abbreviations = {
        "c": _operators.ConstantOperator,
        "A": _operators.LinearOperator,
        "H": _operators.QuadraticOperator,
        "G": _operators.CubicOperator,
        "B": _operators.InputOperator,
        "N": _operators.StateInputOperator,
    }

    @staticmethod
    def _isvalidoperator(op):
        """Return True if and only if ``op`` is a valid operator object
        for this class of model.
        """
        return _operators.is_nonparametric(op)

    @staticmethod
    def _check_operator_types_unique(ops):
        """Raise a ValueError if any two operators represent the same kind
        of operation (e.g., two constant operators).
        """
        if len({type(op) for op in ops}) != len(ops):
            raise ValueError("duplicate type in list of operators to infer")

    def _get_operator_of_type(self, OpClass):
        """Return the first operator of type ``OpClass``."""
        for op in self.operators:
            if isinstance(op, OpClass):
                return op

    # String representation ---------------------------------------------------
    def __str__(self):
        """String representation: structure of the model, dimensions, etc."""
        # Build model structure.
        out, terms = [], []
        for op in self.operators:
            terms.append(op._str(self._STATE_LABEL, self._INPUT_LABEL))
        structure = " + ".join(terms)
        out.append(f"Model structure: {self._LHS_LABEL} = {structure}")

        # Report dimensions.
        if self.state_dimension:
            out.append(f"State dimension r = {self.state_dimension:d}")
        if self.input_dimension:
            out.append(f"Input dimension m = {self.input_dimension:d}")

        return "\n".join(out)

    def __repr__(self):
        """Unique ID + string representation."""
        return utils.str2repr(self)

    # Properties: operator inference ------------------------------------------
    @property
    def operator_matrix(self):
        r""":math:`r \times d(r, m)` operator matrix, e.g.,
        :math:`\Ohat = [~\chat~~\Ahat~~\Hhat~~\Bhat~]`.

        This matrix **does not** includes the entries of any operators whose
        entries are known *a priori*.
        """
        self._check_is_trained()
        return np.column_stack(
            [
                self.operators[i].entries
                for i in self._indices_of_operators_to_infer
            ]
        )

    # Fitting -----------------------------------------------------------------
    def _process_fit_arguments(self, states, lhs, inputs):
        """Prepare training data for Operator Inference by extracting
        dimensions, validating data sizes, and modifying the left-hand side
        data if there are any known operators.
        """
        # Clear non-intrusive operator data.
        self._clear()

        def _check_valid_dimension0(dataset, label):
            """Dimension 0 must be r (state dimensions)."""
            if (dim := dataset.shape[0]) != self.state_dimension:
                raise errors.DimensionalityError(
                    f"{label}.shape[0] = {dim} != r = {self.state_dimension}"
                )

        # Process states, extract model dimension if needed.
        states = np.atleast_2d(states)
        if self.state_dimension is None:
            self.state_dimension = states.shape[0]
        _check_valid_dimension0(states, "states")

        def _check_valid_dimension1(dataset, label):
            """Dimension 1 must be k (number of snapshots)."""
            if (dim := dataset.shape[1]) != (k := states.shape[1]):
                raise errors.DimensionalityError(
                    f"{label}.shape[-1] = {dim} != {k} = states.shape[-1]"
                )

        # Process LHS.
        lhs = np.atleast_2d(lhs)
        _check_valid_dimension0(lhs, self._LHS_ARGNAME)
        _check_valid_dimension1(lhs, self._LHS_ARGNAME)

        # Process inputs, extract input dimension if needed.
        self._check_inputargs(inputs, "inputs")
        if self._has_inputs:
            inputs = np.atleast_2d(inputs)
            if not self.input_dimension:
                self.input_dimension = inputs.shape[0]
            if inputs.shape[0] != self.input_dimension:
                raise errors.DimensionalityError(
                    f"inputs.shape[0] = {inputs.shape[0]} "
                    f"!= {self.input_dimension} = m"
                )
            _check_valid_dimension1(inputs, "inputs")

        # Subtract known operator evaluations from the LHS.
        for i in self._indices_of_known_operators:
            lhs = lhs - self.operators[i].apply(states, inputs)

        return states, lhs, inputs

    def _assemble_data_matrix(self, states, inputs):
        r"""Construct the Operator Inference data matrix :math:`\D`
        from state snapshots and/or input data.

        For example, if the model has the structure
        .. math::
            \ddt\qhat(t)
            = \chat + \Ahat\qhat(t)
            + \Hhat[\qhat(t)\otimes\qhat(t)] + \Bhat\u(t),

        then the data matrix is
        :math:`\D = [~
        \mathbf{1}~~
        \widehat{\Q}\trp~~
        (\widehat{\Q}\odot\widehat{\Q})\trp~~
        \U\trp~]`,

        where :math:`\widehat{\Q}` is ``states``
        and :math:`\U` is `inputs`.

        Parameters
        ----------
        states : (r, k) ndarray
            Snapshot training data.
        inputs : (m, k) ndarray or None
            Inputs corresponding to the snapshots.

        Returns
        -------
        D : (k, d(r, m)) ndarray
            Operator Inference data matrix.
        """
        return np.hstack(
            [
                self.operators[i].datablock(states, inputs).T
                for i in self._indices_of_operators_to_infer
            ]
        )

    def _extract_operators(self, Ohat):
        """Extract and save the inferred operators from the solution to the
        Operator Inference regression problem.

        Parameters
        ----------
        Ohat : (r, d(r, m)) ndarray
            Matrix of operator entries, concatenated horizontally.
        """
        index = 0
        for i in self._indices_of_operators_to_infer:
            endex = index + self.operators[i].operator_dimension(
                self.state_dimension, self.input_dimension
            )
            self.operators[i].set_entries(Ohat[:, index:endex])
            index = endex

    def _fit_solver(self, states, lhs, inputs=None):
        """Construct a solver object mapping the regularizer to solutions
        of the Operator Inference least-squares problem.
        """
        # Set up non-intrusive learning.
        states_, lhs_, inputs_ = self._process_fit_arguments(
            states, lhs, inputs
        )
        D = self._assemble_data_matrix(states_, inputs_)
        self.solver.fit(D, lhs_)

    def refit(self):
        """Solve the Operator Inference regression using the data from the
        last :meth:`fit()` call, then extract the inferred operators.

        This method is useful for calibrating the model operators with
        different ``solver`` hyperparameters without reprocessing any training
        data. For example, if ``solver`` is an :class:`opinf.lstsq.L2Solver`,
        changing its ``regularizer`` attribute and calling this method solves
        the regression with the new regression value without re-factorizing the
        data matrix.
        """
        # Fully intrusive case (nothing to learn).
        if self._fully_intrusive:
            warnings.warn(
                "all operators initialized explicitly, nothing to learn",
                errors.OpInfWarning,
            )
            return self

        # Execute non-intrusive learning.
        self._extract_operators(self.solver.solve())

    def fit(self, states, lhs, inputs=None):
        r"""Learn the model operators from data.

        The operators are inferred by solving the regression problem

        .. math::
           \min_{\Ohat}\sum_{j=0}^{k-1}\left\|
           \fhat(\qhat_j, \u_j) - \zhat_j
           \right\|_2^2
           = \min_{\Ohat}\left\|\D\Ohat\trp - \dot{\Qhat}\trp\right\|_F^2

        where
        :math:`\zhat = \fhat(\qhat, \u)` is the model and

        * :math:`\qhat_j\in\RR^r` is a measurement of the state,
        * :math:`\u_j\in\RR^m` is a measurement of the input, and
        * :math:`\zhat_j\in\RR^r` is a measurement of the left-hand side
          of the model.

        The *operator matrix* :math:`\Ohat\in\RR^{r\times d(r,m)}` is such that
        :math:`\fhat(\q,\u) = \Ohat\d(\qhat,\u)` for some data vector
        :math:`\d(\qhat,\u)\in\RR^{d(r,m)}`; the *data matrix*
        :math:`\D\in\RR^{k\times d(r,m)}` is given by
        :math:`[~\d(\qhat_0,\u_0)~~\cdots~~\d(\qhat_{k-1},\u_{k-1})~]\trp`.
        Finally,
        :math:`\Zhat = [~\zhat_0~~\cdots~~\zhat_{k-1}~]\in\RR^{r\times k}`.
        See the :mod:`opinf.operators` module for more explanation.

        The strategy for solving the regression, as well as any additional
        regularization or constraints, are specified by the ``solver`` set
        in the constructor.

        Parameters
        ----------
        states : (r, k) ndarray
            Snapshot training data. Each column is a single snapshot.
        lhs : (r, k) ndarray
            Left-hand side training data. Each column ``lhs[:, j]``
            corresponds to the snapshot ``states[:, j]``.
            The interpretation of this argument depends on the setting:
            forcing data for steady-state problems, next iteration for
            discrete-time problems, and time derivatives of the state for
            continuous-time problems.
        inputs : (m, k) or (k,) ndarray or None
            Input training data. Each column ``inputs[:, j]`` corresponds
            to the snapshot ``states[:, j]``.
            May be a one-dimensional array if ``m=1`` (scalar input).

        Returns
        -------
        self
        """
        # Fully intrusive case (nothing to learn).
        if self._fully_intrusive:
            warnings.warn(
                "all operators initialized explicitly, nothing to learn",
                errors.OpInfWarning,
            )
            return self

        self._fit_solver(states, lhs, inputs)
        self.refit()
        return self

    # Model evaluation --------------------------------------------------------
    def rhs(self, state, input_=None):
        r"""Evaluate the right-hand side of the model by applying each operator
        and summing the results.

        This is the function :math:`\Ophat(\qhat, \u)`
        where the model can be written as one of the following:

        * :math:`\ddt\qhat(t) = \Ophat(\qhat(t), \u(t))` (continuous time)
        * :math:`\qhat_{j+1} = \Ophat(\qhat_j, \u_j)` (discrete time)
        * :math:`\widehat{\mathbf{g}} = \Ophat(\qhat, \u)` (steady state)

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or None
            Input vector corresponding to the state.

        Returns
        -------
        out : (r,) ndarray
            Evaluation of the right-hand side of the model.
        """
        out = np.zeros_like(state)
        for op in self.operators:
            out += op.apply(state, input_)
        return out

    def jacobian(self, state, input_=None):
        r"""Construct and sum the state Jacobian of each model operator.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function :math:`\ddqhat\Ophat(\qhat, \u)`
        where the model can be written as one of the following:

        * :math:`\ddt\qhat(t) = \Ophat(\qhat(t), \u(t))` (continuous time)
        * :math:`\qhat_{j+1} = \Ophat(\qhat_{j}, \u_{j})` (discrete time)
        * :math:`\widehat{\mathbf{g}} = \Ophat(\qhat, \u)` (steady state)

        Parameters
        ----------
        state : (r,) ndarray
            State vector :math:`\qhat`.
        input_ : (m,) ndarray or None
            Input vector :math:`\u`.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian of the right-hand side of the model.
        """
        r = self.state_dimension
        out = np.zeros_like(state, shape=(r, r))
        for op in self.operators:
            out += op.jacobian(state, input_)
        return out

    @abc.abstractmethod
    def predict(*args, **kwargs):  # pragma: no cover
        """Solve the model under specified conditions."""
        raise NotImplementedError

    # Model persistence -------------------------------------------------------
    def save(self, savefile, overwrite=False):
        """Serialize the model, saving it in HDF5 format.
        The model can be recovered with the :meth:`load()` class method.

        Parameters
        ----------
        savefile : str
            File to save to, with extension ``.h5`` (HDF5).
        overwrite : bool
            If ``True`` and the specified ``savefile`` already exists,
            overwrite the file. If ``False`` (default) and the specified
            ``savefile`` already exists, raise an error.
        """
        with utils.hdf5_savehandle(savefile, overwrite=overwrite) as hf:
            # Metadata.
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["num_operators"] = len(self.operators)
            meta.attrs["r"] = (
                int(self.state_dimension) if self.state_dimension else 0
            )
            meta.attrs["m"] = (
                int(self.input_dimension) if self.input_dimension else 0
            )

            # Store operator data.
            hf.create_dataset(
                "indices_infer", data=self._indices_of_operators_to_infer
            )
            hf.create_dataset(
                "indices_known", data=self._indices_of_known_operators
            )
            for i, op in enumerate(self.operators):
                op.save(hf.create_group(f"operator_{i}"))
            if self.solver is not None:
                self.solver.save(hf.create_group("solver"))

    @classmethod
    def load(cls, loadfile: str):
        """Load a serialized model from an HDF5 file, created previously from
        the :meth:`save()` method.

        Parameters
        ----------
        loadfile : str
            Path to the file where the model was stored via :meth:`save()`.

        Returns
        -------
        model : _NonparametricModel
            Loaded model.
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            # Load metadata.
            num_operators = int(hf["meta"].attrs["num_operators"])
            indices_infer = [int(i) for i in hf["indices_infer"][:]]
            indices_known = [int(i) for i in hf["indices_known"][:]]

            # Load operators.
            ops = []
            for i in range(num_operators):
                gp = hf[f"operator_{i}"]
                OpClassName = gp["meta"].attrs["class"]
                ops.append(getattr(_operators, OpClassName).load(gp))

            # Construct the model.
            model = cls(ops)
            model._indices_of_operators_to_infer = indices_infer
            model._indices_of_known_operators = indices_known
            if r := int(hf["meta"].attrs["r"]):
                model.state_dimension = r
            if (m := int(hf["meta"].attrs["m"])) and model._has_inputs:
                model.input_dimension = m

        return model


# Public classes ==============================================================
class SteadyModel(_NonparametricModel):  # pragma: no cover
    r"""Nonparametric steady state model :math:`\zhat = \fhat(\qhat)`.

    Here,

    * :math:`\qhat(t)\in\RR^{r}` is the model state, and
    * :math:`\u(t)\in\RR^{m}` is the (optional) input.

    The structure of :math:`\fhat` is specified through the
    ``operators`` argument.

    Parameters
    ----------
    operators : list of :mod:`opinf.operators` objects
        Operators comprising the terms of the model.
    solver : :mod:`opinf.lstsq` object or float > 0 or None
        Solver for the least-squares regression. Defaults:

        * ``None``: :class:`opinf.lstsq.PlainSolver`.
          SVD-based solve without regularization.
        * float > 0: :class:`opinf.lstsq.L2Solver`.
          SVD-based solve with scalar Tikhonov regularization.
    """

    _LHS_ARGNAME = "forcing"
    _LHS_LABEL = "z"
    _STATE_LABEL = "q"
    _INPUT_LABEL = None
    # TODO: disallow input terms?

    def fit(self, states, forcing=None):
        r"""Learn the model operators from data.

        The operators are inferred by solving the regression problem

        .. math::
           \min_{\Ohat}\sum_{j=0}^{k-1}\left\|
           \fhat(\qhat_j) - \zhat_{j}
           \right\|_2^2
           = \min_{\Ohat}\left\|
           \D\Ohat\trp - [~\zhat_0~~\cdots~~\zhat_{k-1}~]\trp
           \right\|_F^2

        where :math:`\zhat = \fhat(\qhat)` is the model and

        * :math:`\qhat_j\in\RR^r` is a measurement of the state,
        * :math:`\zhat_j\in\RR^r` is a measurement of the forcing term
          corresponding to :math:`\qhat_j`,
        * :math:`\Ohat\in\RR^{r\times d(r,m)}` is the *operator matrix* such
          that :math:`\fhat(\q,\u) = \Ohat\d(\qhat,\u)` for some data vector
          :math:`\d(\qhat,\u)\in\RR^{d(r,m)}`, and
        * :math:`\D\in\RR^{k\times d(r,m)}` is the *data matrix* given by
          :math:`[~\d(\qhat_0,\u_0)~~\cdots~~\d(\qhat_{k-1},\u_{k-1})~]\trp`.

        See the :mod:`opinf.operators` module for further explanation.

        The strategy for solving the regression, as well as any additional
        regularization or constraints, are specified by the ``solver``.

        Parameters
        ----------
        states : (r, k) ndarray
            Snapshot training data. Each column is a single snapshot.
        forcing : (r, k) ndarray or None
            Forcing training data. Each column ``forcing[:, j]``
            corresponds to the snapshot ``states[:, j]``.
            If ``None``, set ``forcing = 0``.

        Returns
        -------
        self
        """
        return _NonparametricModel.fit(self, states, forcing, inputs=None)

    def rhs(self, state):
        r"""Evaluate the right-hand side of the model by applying each operator
        and summing the results.

        This is the function :math:`\fhat(\qhat)` where the model is given by
        :math:`\zhat = \fhat(\qhat)`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector :math:`\qhat`.

        Returns
        -------
        g: (r,) ndarray
            Evaluation of the right-hand-side of the model.
        """
        return _NonparametricModel.rhs(self, state, None)

    def jacobian(self, state):
        r"""Sum the state Jacobian of each model operator.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function :math:`\ddqhat\fhat(\qhat)`
        where the model is given by :math:`\zhat = \fhat(\qhat)`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector :math:`\qhat`.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian of the right-hand side of the model.
        """
        return _NonparametricModel.jacobian(self, state, input_=None)

    def predict(self, forcing, guess=None):
        """Solve the model with the given forcing and initial guess."""
        raise NotImplementedError("TODO")


class DiscreteModel(_NonparametricModel):
    r"""Nonparametric discrete dynamical system model
    :math:`\qhat_{j+1} = \fhat(\qhat_{j}, \u_{j})`.

    Here,

    * :math:`\qhat_j\in\RR^{r}` is the :math:`j`-th iteration
      of the model state, and
    * :math:`\u_j\in\RR^{m}` is the (optional) corresponding input.

    The structure of :math:`\fhat` is specified through the
    ``operators`` attribute.

    Parameters
    ----------
    operators : list of :mod:`opinf.operators` objects
        Operators comprising the terms of the model.
    solver : :mod:`opinf.lstsq` object or float > 0 or None
        Solver for the least-squares regression. Defaults:

        * ``None``: :class:`opinf.lstsq.PlainSolver`.
          SVD-based solve without regularization.
        * float > 0: :class:`opinf.lstsq.L2Solver`.
          SVD-based solve with scalar Tikhonov regularization.
    """

    _LHS_ARGNAME = "nextstates"
    _LHS_LABEL = r"q_{j+1}"
    _STATE_LABEL = r"q_{j}"
    _INPUT_LABEL = r"u_{j}"

    @staticmethod
    def stack_trajectories(statelist, inputlist=None):
        """Translate a collection of state trajectories and (optionally) inputs
        to arrays that are appropriate arguments for :meth:`fit()`.

        Parameters
        ----------
        statelist : list of s (r, k_i) ndarrays
            Collection of state trajectories.
        inputlist : list of s (m, k_i) ndarrays
            Collection of inputs corresponding to the state trajectories.

        Returns
        -------
        states : (r, sum_i(k_i)) ndarray
            Snapshot matrix with data from all but the final snapshot of each
            trajectory in ``statelist``.
        nextstates : (r, sum_i(k_i)) ndarray
            Snapshot matrix with data from all but the first snapshot of each
            trajectory in ``statelist``.
        inputs : (r, sum_i(k_i)) ndarray
            Input matrix with data from all but the last input for each
            trajectory. Only returned if ``inputlist`` is provided.
        """
        states = np.hstack([Q[:, :-1] for Q in statelist])
        nextstates = np.hstack([Q[:, 1:] for Q in statelist])
        if inputlist is not None:
            inputs = np.hstack(
                [
                    U[..., : (S.shape[1] - 1)]
                    for S, U in zip(statelist, inputlist)
                ]
            )
            return states, nextstates, inputs
        return states, nextstates

    def fit(self, states, nextstates=None, inputs=None):
        r"""Learn the model operators from data.

        The operators are inferred by solving the regression problem

        .. math::
           \min_{\Ohat}\sum_{j=0}^{k-1}\left\|
           \fhat(\qhat_j, \u_j) - \zhat_{j}
           \right\|_2^2
           = \min_{\Ohat}\left\|
           \D\Ohat\trp - [~\zhat_0~~\cdots~~\zhat_{k-1}~]\trp
           \right\|_F^2

        where
        :math:`\zhat_j = \fhat(\qhat_j, \u_j)` is the model and

        * :math:`\qhat_j\in\RR^r` is a measurement of the state,
        * :math:`\u_j\in\RR^m` is a measurement of the input, and
        * :math:`\zhat_j\in\RR^r` is a measurement of the next state iteration
          after :math:`\qhat_j`,
        * :math:`\Ohat\in\RR^{r\times d(r,m)}` is the *operator matrix* such
          that :math:`\fhat(\q,\u) = \Ohat\d(\qhat,\u)` for some data vector
          :math:`\d(\qhat,\u)\in\RR^{d(r,m)}`, and
        * :math:`\D\in\RR^{k\times d(r,m)}` is the *data matrix* given by
          :math:`[~\d(\qhat_0,\u_0)~~\cdots~~\d(\qhat_{k-1},\u_{k-1})~]\trp`,

        See the :mod:`opinf.operators` module for further explanation.

        The strategy for solving the regression, as well as any additional
        regularization or constraints, are specified by the ``solver``.

        Parameters
        ----------
        states : (r, k) ndarray
            Snapshot training data. Each column is a single snapshot.
        nextstates : (r, k) ndarray or None
            Next iteration training data. Each column ``nextstates[:, j]``
            is the iteration following ``states[:, j]``.
            If ``None``, use ``nextstates[:, j] = states[:, j+1]``.
        inputs : (m, k) or (k,) ndarray or None
            Input training data. Each column ``inputs[:, j]`` corresponds
            to the snapshot ``states[:, j]``.
            May be a one-dimensional array if ``m=1`` (scalar input).

        Returns
        -------
        self
        """
        if nextstates is None:
            nextstates = states[:, 1:]
            states = states[:, :-1]
        if inputs is not None:
            inputs = inputs[..., : states.shape[1]]
        return _NonparametricModel.fit(self, states, nextstates, inputs=inputs)

    def rhs(self, state, input_=None):
        r"""Evaluate the right-hand side of the model by applying each operator
        and summing the results.

        This is the function :math:`\fhat(\qhat, \u)`
        where the model is given by
        :math:`\qhat_{j+1} = \fhat(\qhat_{j}, \u_{j})`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector :math:`\qhat`.
        input_ : (m,) ndarray or None
            Input vector :math:`\u`.

        Returns
        -------
        nextstate : (r,) ndarray
            Evaluation of the right-hand side of the model.
        """
        return _NonparametricModel.rhs(self, state, input_)

    def jacobian(self, state, input_=None):
        r"""Sum the state Jacobian of each model operator.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function :math:`\ddqhat\fhat(\qhat, \u)`
        where the model is given by
        :math:`\qhat_{j+1} = \fhat(\qhat_{j}, \u_{j})`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector :math:`\qhat`.
        input_ : (m,) ndarray or None
            Input vector :math:`\u`.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian of the right-hand side of the model.
        """
        return _NonparametricModel.jacobian(self, state, input_)

    def predict(self, state0, niters, inputs=None):
        """Step forward the discrete dynamical system
        ``niters`` steps. Essentially, this amounts to the following.

        .. code-block:: python

           >>> states[:, 0] = state0
           >>> states[:, 1] = model.rhs(states[:, 0], inputs[:, 0])
           >>> states[:, 2] = model.rhs(states[:, 1], inputs[:, 1])
           ...                                     # Repeat `niters` times.

        Parameters
        ----------
        state0 : (r,) ndarray
            Initial state.
        niters : int
            Number of times to step the system forward.
        inputs : (m, niters-1) ndarray or None
            Inputs for the next ``niters - 1`` time steps.

        Returns
        -------
        states : (r, niters) ndarray
            Solution to the system, including the initial condition ``state0``.
        """
        self._check_is_trained()

        # Check initial condition dimension and process inputs.
        if (_shape := np.shape(state0)) != (self.state_dimension,):
            raise errors.DimensionalityError(
                "initial condition not aligned with model "
                f"(state0.shape = {_shape} != "
                f"({self.state_dimension},) = (r,))"
            )
        self._check_inputargs(inputs, "inputs")

        # Verify iteration argument.
        if not isinstance(niters, int) or niters < 1:
            raise ValueError("argument 'niters' must be a positive integer")

        # Create the solution array and fill in the initial condition.
        states = np.empty((self.state_dimension, niters))
        states[:, 0] = state0.copy()

        # Run the iteration.
        if self._has_inputs:
            if callable(inputs):
                raise TypeError("inputs must be NumPy array, not callable")

            # Validate shape of input, reshaping if input is 1d.
            U = np.atleast_2d(inputs)
            if (
                U.ndim != 2
                or U.shape[0] != self.input_dimension
                or U.shape[1] < niters - 1
            ):
                raise ValueError(
                    f"inputs.shape = ({U.shape} "
                    f"!= {(self.input_dimension, niters-1)} = (m, niters-1)"
                )
            for j in range(niters - 1):
                states[:, j + 1] = self.rhs(states[:, j], U[:, j])
        else:
            for j in range(niters - 1):
                states[:, j + 1] = self.rhs(states[:, j])

        # Return state results.
        return states


class ContinuousModel(_NonparametricModel):
    r"""Nonparametric system of ordinary differential equations
    :math:`\ddt\qhat(t) = \fhat(\qhat(t), \u(t))`.

    Here,

    * :math:`\qhat(t)\in\RR^{r}` is the model state, and
    * :math:`\u(t)\in\RR^{m}` is the (optional) input.

    The structure of :math:`\fhat` is specified through the
    ``operators`` argument.

    Parameters
    ----------
    operators : list of :mod:`opinf.operators` objects
        Operators comprising the terms of the model.
    solver : :mod:`opinf.lstsq` object or float > 0 or None
        Solver for the least-squares regression. Defaults:

        * ``None``: :class:`opinf.lstsq.PlainSolver`.
          SVD-based solve without regularization.
        * float > 0: :class:`opinf.lstsq.L2Solver`.
          SVD-based solve with scalar Tikhonov regularization.
    """

    _LHS_ARGNAME = "ddts"
    _LHS_LABEL = "dq / dt"
    _STATE_LABEL = "q(t)"
    _INPUT_LABEL = "u(t)"

    def fit(self, states, ddts, inputs=None):
        r"""Learn the model operators from data.

        The operators are inferred by solving the regression problem

        .. math::
           \min_{\Ohat}\sum_{j=0}^{k-1}\left\|
           \fhat(\qhat_j, \u_j) - \dot{\qhat}_j
           \right\|_2^2
           = \min_{\Ohat}\left\|
           \D\Ohat\trp - [~\dot{\qhat}_0~~\cdots~~\dot{\qhat}_{k-1}~]\trp
           \right\|_F^2

        where
        :math:`\ddt\qhat(t) = \fhat(\qhat(t), \u(t))` is the model and

        * :math:`\qhat_j\in\RR^r` is the state at some time :math:`t_j`,
        * :math:`\u_j\in\RR^m` is the input at time :math:`t_j`,
        * :math:`\dot{\qhat}_j\in\RR^r` is the time derivative of the state
          at time :math:`t_j`, i.e.,
          :math:`\dot{\qhat}_j = \ddt\qhat(t)\big|_{t=t_j}`,
        * :math:`\Ohat\in\RR^{r\times d(r,m)}` is the *operator matrix* such
          that :math:`\fhat(\q,\u) = \Ohat\d(\qhat,\u)` for some data vector
          :math:`\d(\qhat,\u)\in\RR^{d(r,m)}`, and
        * :math:`\D\in\RR^{k\times d(r,m)}` is the *data matrix* given by
          :math:`[~\d(\qhat_0,\u_0)~~\cdots~~\d(\qhat_{k-1},\u_{k-1})~]\trp`.

        See the :mod:`opinf.operators` module for further explanation.

        The strategy for solving the regression, as well as any additional
        regularization or constraints, are specified by the ``solver``.

        Parameters
        ----------
        states : (r, k) ndarray
            Snapshot training data. Each column is a single snapshot.
        ddts : (r, k) ndarray
            Snapshot time derivative data. Each column
            ``ddts[:, j]`` corresponds to the snapshot ``states[:, j]``.
        inputs : (m, k) or (k,) ndarray or None
            Input training data. Each column ``inputs[:, j]`` corresponds
            to the snapshot ``states[:, j]``.
            May be a one-dimensional array if ``m=1`` (scalar input).

        Returns
        -------
        self
        """
        return _NonparametricModel.fit(self, states, ddts, inputs=inputs)

    def rhs(self, t, state, input_func=None):
        r"""Evaluate the right-hand side of the model by applying each operator
        and summing the results.

        This is the right-hand side of the model, i.e., the function
        :math:`\fhat(\qhat(t), \u(t))` where the model is given by
        :math:`\ddt \qhat(t) = \fhat(\qhat(t), \u(t))`.

        Parameters
        ----------
        t : float
            Time :math:`t`, a scalar.
        state : (r,) ndarray
            State vector :math:`\qhat(t)` corresponding to time ``t``.
        input_func : callable(float) -> (m,), or None
            Input function that maps time ``t`` to the input vector
            :math:`\u(t)`.

        Returns
        -------
        dqdt : (r,) ndarray
            Evaluation of the right-hand side of the model.
        """
        input_ = None if not self._has_inputs else input_func(t)
        return _NonparametricModel.rhs(self, state, input_)

    def jacobian(self, t, state, input_func=None):
        r"""Sum the state Jacobian of each model operator.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function :math:`\ddqhat\fhat(\qhat(t), \u(t))`
        where the model is given by
        :math:`\ddt\qhat(t) = \fhat(\qhat(t), \u(t))`.

        Parameters
        ----------
        t : float
            Time :math:`t`, a scalar.
        state : (r,) ndarray
            State vector :math:`\qhat(t)` corresponding to time ``t``.
        input_func : callable(float) -> (m,), or None
            Input function that maps time ``t`` to the input vector
            :math:`\u(t)`.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian of the right-hand side of the model.
        """
        input_ = None if not self._has_inputs else input_func(t)
        return _NonparametricModel.jacobian(self, state, input_)

    def predict(self, state0, t, input_func=None, **options):
        """Solve the system of ordinary differential equations.
        This method wraps :func:`scipy.integrate.solve_ivp()`.

        Parameters
        ----------
        state0 : (r,) ndarray
            Initial state vector.
        t : (nt,) ndarray
            Time domain over which to integrate the model.
        input_func : callable or (m, nt) ndarray
            Input as a function of time (preferred) or the input values at the
            times ``t``. If given as an array, cubic spline interpolation on
            the known data points is used as needed.
        options
            Arguments for :func:`scipy.integrate.solve_ivp()`.
            Common options:

            * **method : str** ODE solver for the model.

              * ``'RK45'`` (default): Explicit Runge--Kutta method
                of order 5(4).
              * ``'RK23'``: Explicit Runge--Kutta method
                of order 3(2).
              * ``'Radau'``: Implicit Runge--Kutta method
                of the Radau IIA family of order 5.
              * ``'BDF'``: Implicit multi-step variable-order (1 to 5) method
                based on a backward differentiation formula for the derivative.
              * ``'LSODA'``: Adams/BDF method with automatic stiffness
                detection and switching.

            * **max_step : float** Maximimum allowed integration step size.

        Returns
        -------
        states : (r, nt) ndarray
            Computed solution to the system over the time domain ``t``.
            A more detailed report on the integration results is stored as
            the ``predict_result_`` attribute.
        """
        self._check_is_trained()

        # Check initial condition dimension and process inputs.
        if (_shape := np.shape(state0)) != (self.state_dimension,):
            raise errors.DimensionalityError(
                "initial condition not aligned with model "
                f"(state0.shape = {_shape} != "
                f"({self.state_dimension},) = (r,))"
            )
        self._check_inputargs(input_func, "input_func")

        # Verify time domain.
        if t.ndim != 1:
            raise ValueError("time 't' must be one-dimensional")
        nt = t.shape[0]

        # Interpret control input argument.
        if self._has_inputs:
            if not callable(input_func):
                # input_func must be (m, nt) ndarray. Interpolate -> callable.
                U = np.atleast_2d(input_func)
                if U.shape != (self.input_dimension, nt):
                    raise ValueError(
                        f"input_func.shape = {U.shape} "
                        f"!= {(self.input_dimension, nt)} = (m, len(t))"
                    )
                input_func = spinterpolate.CubicSpline(t, U, axis=1)

            # Check dimension of input_func() outputs.
            _tmp = input_func(t[0])
            if self.input_dimension == 1 and np.isscalar(_tmp):
                original_input_func = input_func

                def input_func(t):
                    """Wrap outputs of input_func() as an array."""
                    return np.array([original_input_func(t)])

                _tmp = input_func(t[0])

            if not isinstance(_tmp, np.ndarray) or _tmp.shape != (
                self.input_dimension,
            ):
                raise errors.DimensionalityError(
                    "input_func() must return ndarray"
                    f" of shape (m,) = ({self.input_dimension},)"
                )

        if "method" in options and options["method"] in (
            # These methods require the Jacobian.
            "BDF",
            "Radau",
            "LSODA",
        ):
            options["jac"] = self.jacobian

        # Integrate the model.
        out = spintegrate.solve_ivp(
            self.rhs,  # Integrate this function
            [t[0], t[-1]],  # over this time interval
            state0,  # from this initial condition
            args=(input_func,),  # with this input function
            t_eval=t,  # evaluated at these points
            **options,  # using these solver options.
        )

        # Warn if the integration failed.
        if not out.success:  # pragma: no cover
            warnings.warn(out.message, spintegrate.IntegrationWarning)

        # Return state results.
        self.predict_result_ = out
        return out.y


# "Frozen" classes for parametric evaluation ==================================
class _FrozenMixin:
    """Mixin for evaluations of parametric models (disables fit())."""

    def _clear(self):
        raise NotImplementedError(
            "_clear() is disabled for this class, "
            "call fit() on the parametric model object"
        )

    @property
    def solver(self):
        return None

    @solver.setter
    def solver(self, solver):
        pass

    def fit(*args, **kwargs):
        raise NotImplementedError(
            "fit() is disabled for this class, "
            "call fit() on the parametric model object"
        )


class _FrozenSteadyModel(_FrozenMixin, SteadyModel):
    """Nonparametric steady-state model that is the evaluation of
    a parametric model.
    """

    pass  # pragma: no cover


class _FrozenDiscreteModel(_FrozenMixin, DiscreteModel):
    """Nonparametric discrete-time model that is the evaluation of
    a parametric model.
    """

    pass  # pragma: no cover


class _FrozenContinuousModel(_FrozenMixin, ContinuousModel):
    """Nonparametric continuous-time model that is the evaluation of
    a parametric model.
    """

    pass  # pragma: no cover
