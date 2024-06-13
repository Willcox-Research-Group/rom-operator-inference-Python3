# pre/_multi.py
"""Transformer for states with multiple variables."""

__all__ = [
    "TransformerMulti",
]

import warnings
import numpy as np

from .. import errors, utils
from ._base import TransformerTemplate


requires_trained = utils.requires2(
    "state_dimension",
    "transformer not trained, call fit() or fit_transform()",
)


class TransformerMulti:
    r"""Join transformers together for states with multiple variables.

    This class is for states that can be written (after discretization) as

    .. math::
       \q = \left[\begin{array}{c}
       \q_{0} \\ \q_{1} \\ \vdots \\ \q_{n_q - 1}
       \end{array}\right]
       \in \RR^{n},

    where each :math:`\q_{i} \in \RR^{n_i}` represents a single discretized
    state variable of dimension :math:`n_i \in \NN`. The dimension of the
    joint state :math:`\q` is :math:`n = \sum_{i=0}^{n_q - 1} n_i`, i.e.,
    ``state_dimension = sum(variable_sizes)``.
    Individual transformers are calibrated for each state variable.

    Parameters
    ----------
    transformers : list
        Initialized (but not necessarily trained) transformer objects,
        one for each state variable.
    variable_sizes : list or None
        Dimensions for each state variable, :math:`n_0,\ldots,n_{n_q-1}`.
        If ``None`` (default), set :math:`n_i` to
        ``transformers[i].state_dimension``; if any of these are ``None``,
        assume all state variables have the same dimension, i.e.,
        :math:`n_0 = n_1 = \cdots = n_x \in \NN` with :math:`n_x` to be
        determined in :meth:`fit`. In this case, :math:`n = n_q n_x`.
    """

    def __init__(self, transformers, variable_sizes=None):
        """Initialize the transformers."""
        self.transformers = transformers

        if variable_sizes is not None:
            if len(variable_sizes) != len(transformers):
                raise ValueError("len(variable_sizes) != len(transformers)")
            for tf, ni in zip(transformers, variable_sizes):
                TransformerTemplate.state_dimension.fset(tf, ni)

    # Properties --------------------------------------------------------------
    @property
    def transformers(self) -> tuple:
        """Transformers for each state variable."""
        return self.__transformers

    @transformers.setter
    def transformers(self, tfs):
        """Set the transformers."""
        if (num_variables := len(tfs)) == 0:
            raise ValueError("at least one transformer required")
        elif num_variables == 1:
            warnings.warn("only one variable detected", errors.OpInfWarning)

        # Check inheritance and set default variable names.
        for i, tf in enumerate(tfs):
            if not isinstance(tf, TransformerTemplate):
                warnings.warn(
                    f"transformers[{i}] does not inherit from "
                    "TransformerTemplate, unexpected behavior may occur",
                    errors.OpInfWarning,
                )
            if tf.name is None:
                tf.name = f"variable {i}"

        # Store transformers and set slice dimensions.
        self.__nq = num_variables
        self.__transformers = tuple(tfs)

    @property
    def num_variables(self) -> int:
        r"""Number of state variables :math:`n_q \in \NN`."""
        return self.__nq

    @property
    def variable_names(self) -> tuple:
        """Names for each state variable."""
        return tuple(tf.name for tf in self.transformers)

    @property
    def variable_sizes(self) -> tuple:
        r"""Dimensions of each state variable,
        :math:`n_0,\ldots,n_{n_q-1}\in\NN`.
        """
        return tuple(tf.state_dimension for tf in self.transformers)

    @property
    def state_dimension(self) -> int:
        r"""Total dimension :math:`n = \sum_{i=0}^{n_q-1} n_i \in \NN` of the
        joint state.
        """
        if None in (sizes := self.variable_sizes):
            return None
        return sum(sizes)

    # Magic methods -----------------------------------------------------------
    def __len__(self) -> int:
        """Length = number of state variables."""
        return self.num_variables

    def __getitem__(self, key):
        """Get the transformer for variable i."""
        if key in self.variable_names:
            key = self.variable_names.index(key)
        return self.transformers[key]

    def __eq__(self, other) -> bool:
        """Test two TransformerMulti objects for equality."""
        if not (
            isinstance(other, self.__class__)
            or isinstance(self, other.__class__)
        ):
            return False
        if self.num_variables != other.num_variables:
            return False
        return all(
            t1 == t2 for t1, t2 in zip(self.transformers, other.transformers)
        )

    def __str__(self) -> str:
        """String representation: str() of each transformer."""
        out = [f"{self.num_variables}-variable {self.__class__.__name__}"]
        namelength = max(len(name) for name in self.variable_names)
        for tf in self.transformers:
            out.append(f"* {{:>{namelength}}} | {tf}".format(tf.name))
        return "\n".join(out)

    def __repr__(self) -> str:
        """Unique ID + string representation."""
        return utils.str2repr(self)

    # Access convenience methods ----------------------------------------------
    @requires_trained
    def _check_shape(self, states):
        """Verify the shape of the snapshot set Q."""
        if (nQ := len(states)) != (n := self.state_dimension):
            raise errors.DimensionalityError(
                f"len(states) = {nQ:d} != {n:d} = state_dimension"
            )

    def _slices(self, varindex=None):
        """Get slices for one or all state variable(s).

        Parameters
        ----------
        varindex : int
            Index of the variable to get a slice for.
            If ``None`` (default), get slices for all state variables.

        Returns
        -------
        slice or tuple
            Slice for the state variable at index ``varindex``
            or a tuple of slices for all state variables.
        """
        dimsum = np.cumsum((0,) + self.variable_sizes)
        if varindex is not None:
            return slice(dimsum[varindex], dimsum[varindex + 1])

        return tuple(
            [
                slice(dimsum[i], dimsum[i + 1])
                for i in range(self.num_variables)
            ]
        )

    def get_var(self, var, states):
        """Extract a single variable from the joint state.

        Parameters
        ----------
        var : int or str
            Index or name of the variable to extract.
        states : (n, ...) ndarray
            Joint state vector or snapshot matrix.

        Returns
        -------
        state_variable : (n_i, ...) ndarray
            One state variable, extracted from ``states``.
        """
        self._check_shape(states)
        if var in (names := self.variable_names):
            var = names.index(var)
        return states[self._slices(var)]

    def split(self, states):
        """Split the joint state into the individual state variables.

        Parameters
        ----------
        states : (n, ...) ndarray
            Joint state vector or snapshot matrix.

        Returns
        -------
        state_variable : list of nq (nx, ...) ndarrays
            Individual state variables, extracted from ``states``.
        """
        self._check_shape(states)
        return [states[s] for s in self._slices()]

    # Main routines -----------------------------------------------------------
    def fit(self, states):
        """Learn (but do not apply) the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of `k` `n`-dimensional snapshots.

        Returns
        -------
        self
        """
        self.fit_transform(states, inplace=False)
        return self

    def fit_transform(self, states, inplace=False):
        """Learn and apply the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of `k` `n`-dimensional snapshots.
            The first ``variable_sizes[0]`` entries correspond to the first
            state variable, the next ``variable_sizes[1]`` entries correspond
            to the second state variable, and so on.
            If ``variable_sizes`` are not yet prescribed, assume that each
            state variable has the same dimension.
        inplace : bool
            If ``True``, overwrite ``states`` data during transformation.
            If ``False``, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of `k` transformed `n`-dimensional snapshots.
        """
        if self.state_dimension is None:
            # Assume all state dimensions are equal.
            nx, remainder = divmod(states.shape[0], self.num_variables)
            if remainder != 0:
                raise errors.DimensionalityError(
                    "len(states) must be evenly divisible by "
                    f"the number of variables n_q = {self.num_variables}"
                )
            for tf in self.transformers:
                TransformerTemplate.state_dimension.fset(tf, nx)

        new_states = np.concatenate(
            [
                transformer.fit_transform(state_variable, inplace=inplace)
                for transformer, state_variable in zip(
                    self.transformers, self.split(states)
                )
            ],
            axis=0,
        )
        return states if inplace else new_states

    def transform(self, states, inplace=False):
        """Apply the learned transformation.

        Parameters
        ----------
        states : (n, ...) ndarray
            Matrix of `n`-dimensional snapshots, or a single snapshot.
        inplace : bool
            If ``True``, overwrite ``states`` during transformation.
            If ``False``, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, ...) ndarray
            Matrix of `n`-dimensional transformed snapshots, or a single
            transformed snapshot.
        """
        states_transformed = np.concatenate(
            [
                tf.transform(Q, inplace=inplace)
                for tf, Q in zip(self.transformers, self.split(states))
            ]
        )
        return states if inplace else states_transformed

    def transform_ddts(self, ddts, inplace: bool = False):
        r"""Apply the learned transformation to snapshot time derivatives.

        If the transformation is denoted by :math:`\mathcal{T}(q)`,
        this function implements :math:`\mathcal{T}'` such that
        :math:`\mathcal{T}'(\ddt q) = \ddt \mathcal{T}(q)`.

        Parameters
        ----------
        ddts : (n, ...) ndarray
            Matrix of `n`-dimensional snapshot time derivatives, or a
            single snapshot time derivative.
        inplace : bool
            If True, overwrite ``ddts`` during the transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        ddts_transformed : (n, ...) ndarray
            Transformed snapshot time derivatives.
        """
        ddts_transformed = [
            tf.transform_ddts(dQdt, inplace=inplace)
            for tf, dQdt in zip(self.transformers, self.split(ddts))
        ]
        if any(dt is NotImplemented for dt in ddts_transformed):
            return NotImplemented
        return ddts if inplace else np.concatenate(ddts_transformed)

    @requires_trained
    def inverse_transform(self, states_transformed, inplace=False, locs=None):
        """Apply the inverse of the learned transformation.

        Parameters
        ----------
        states_transformed : (n, ...) or (num_variables*p, ...) ndarray
            Matrix of `n`-dimensional transformed snapshots, or a single
            transformed snapshot.
        inplace : bool
            If ``True``, overwrite ``states_transformed`` during the inverse
            transformation. If ``False``, create a copy of the data to
            untransform.
        locs : slice or (p,) ndarray of integers or None
            If given, assume ``states_transformed`` contains the `p` entries
            of each transformed state variable at the indices specified by
            ``locs``. This option requires each state variable to have the
            same dimension.

        Returns
        -------
        states_untransformed: (n, ...) or (num_variables*p, ...) ndarray
            Matrix of `n`-dimensional untransformed snapshots, or the
            :math:`n_q p` entries of such at the indices specified by ``locs``.
        """
        if locs is not None:
            if len(set(self.variable_sizes)) > 1:
                raise ValueError(
                    "'locs != None' requires that "
                    "all transformers have the same state_dimension"
                )
            variables_transformed = np.split(
                states_transformed,
                self.num_variables,
                axis=0,
            )
        else:
            variables_transformed = self.split(states_transformed)

        states = np.concatenate(
            [
                tf.inverse_transform(Q, inplace=inplace, locs=locs)
                for tf, Q in zip(self.transformers, variables_transformed)
            ]
        )
        return states_transformed if inplace else states

    # Model persistence -------------------------------------------------------
    def save(self, savefile, overwrite=False):
        """Save the transformer to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the transformer to.
        overwrite : bool
            If ``True``, overwrite the file if it already exists. If ``False``
            (default), raise a ``FileExistsError`` if the file already exists.
        """
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            hf.create_dataset("num_variables", data=[self.num_variables])
            for i, tf in enumerate(self.transformers):
                tf.save(hf.create_group(f"variable{i}"))

    @classmethod
    def load(cls, loadfile, TransformerClasses):
        """Load a previously saved transformer from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            File where the transformer was stored via :meth:`save()`.
        TransformerClasses : Iterable[type]
            Classes of the transformers for each state variable.

        Returns
        -------
        TransformerMulti
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            num_variables = int(hf["num_variables"][0])

            if isinstance(TransformerClasses, type):
                TransformerClasses = [TransformerClasses] * num_variables
            if (nclasses := len(TransformerClasses)) != num_variables:
                raise ValueError(
                    f"file contains {num_variables:d} transformers "
                    f"but {nclasses:d} classes provided"
                )

            transformers = [
                TransformerClasses[i].load(hf[f"variable{i}"])
                for i in range(num_variables)
            ]

            return cls(transformers)

    # Verification ------------------------------------------------------------
    def verify(self, tol: float = 1e-4):
        r"""Verify that :meth:`transform()` and :meth:`inverse_transform()`
        are consistent and that :meth:`transform_ddts()`, if implemented in
        each transformer, is consistent with :meth:`transform()`.

        * The :meth:`transform()` / :meth:`inverse_transform()` consistency
          check verifies that
          ``inverse_transform(transform(states)) == states``.
        * The :meth:`transform_ddts()` consistency check uses
          :meth:`opinf.ddt.ddt()` to estimate the time derivatives of the
          states and the transformed states, then verfies that the relative
          difference between
          ``transform_ddts(opinf.ddt.ddt(states, t))`` and
          ``opinf.ddt.ddt(transform(states), t)`` is less than ``tol``.
          If this check fails, consider using a finer time mesh.

        Parameters
        ----------
        tol : float > 0
            Tolerance for the finite difference check of
            :meth:`transform_ddts()`.
            Only used if :meth:`transform_ddts()` is implemented.
        """
        for tf, name in zip(self.transformers, self.variable_names):
            print(f"{name}:", end="\t")
            tf.verify(tol)
        return TransformerTemplate.verify(self, tol)

    def _verify_locs(self, states, states_transformed):
        """Verify :meth:`inverse_transform()` with ``locs != None``."""
        if len(sizes := set(self.variable_sizes)) != 1:
            return  # Cannot use locs unless all variable sizes equal.
        nx = sizes.pop()
        locs = np.sort(np.random.choice(nx, size=(nx // 3), replace=False))

        states_at_locs = np.concatenate([Q[locs] for Q in self.split(states)])
        states_transformed_at_locs = np.concatenate(
            [Qt[locs] for Qt in self.split(states_transformed)]
        )
        states_recovered_at_locs = self.inverse_transform(
            states_transformed_at_locs,
            locs=locs,
        )

        if states_recovered_at_locs.shape != states_at_locs.shape:
            raise errors.VerificationError(  # pragma: no cover
                "inverse_transform(states_transformed_at_locs, locs).shape "
                "!= states_at_locs.shape"
            )
        if not np.allclose(states_recovered_at_locs, states_at_locs):
            raise errors.VerificationError(  # pragma: no cover
                "transform() and inverse_transform() are not inverses "
                "(locs != None)"
            )
