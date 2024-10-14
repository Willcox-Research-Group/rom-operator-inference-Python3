# pre/_multi.py
"""Transformer for states with multiple variables."""

__all__ = [
    "TransformerPipeline",
    "NullTransformer",
    "TransformerMulti",
]

import warnings
import numpy as np

from .. import errors, utils
from ._base import TransformerTemplate, requires_trained


# Horizontal joining ==========================================================
class TransformerPipeline(TransformerTemplate):
    r"""Chain multiple transformers.

    Given :math:`\tau\in\NN` transformers
    :math:`\mathcal{T}_1,\ldots,\mathcal{T}_{\tau}`, this class defines the
    compositional transformer
    :math:`\mathcal{T} = \mathcal{T}_{\tau}\circ\cdots\circ\mathcal{T}_1`.

    Parameters
    ----------
    transformers : tuple of instantiated Transformer objects
        Transformers to be chained together;
        `transformers[0]` is applied first, then `transformers[1]`, and so on.
    name : str or None
        Label for the state variable that this transformer acts on.

    Notes
    -----
    This class connects multiple transformers "horizontally"; see
    :class:`TransformerMulti` to connect multiple transformers "vertically",
    i.e., to assign different transformations for different parts of the state.
    """

    def __init__(self, transformers, name=None):
        """Set the transformers."""
        super().__init__(name=name)

        if isinstance(transformers, list):
            transformers = tuple(transformers)
        else:
            raise TypeError("'transformers' should be a list or tuple")

        statedims = set()
        for i, tf in enumerate(transformers):
            if not isinstance(tf, (TransformerTemplate, TransformerMulti)):
                warnings.warn(
                    f"transformers[{i}] does not inherit from "
                    "TransformerTemplate, unexpected behavior may occur",
                    errors.OpInfWarning,
                )
            statedims.add(tf.state_dimension)
        statedims.discard(None)
        if len(statedims) > 1:
            raise ValueError("transformers have inconsistent state_dimension")

        if len(transformers) == 1:
            warnings.warn(
                "only one transformer provided to TransformerPipeline",
                errors.OpInfWarning,
            )
        self.__transformers = tuple(transformers)

    # Properties --------------------------------------------------------------
    @property
    def transformers(self) -> tuple:
        """Transformers being chained together;
        `transformers[0]` is applied first, then `transformers[1]`, and so on.
        """
        return self.__transformers

    @property
    def num_transformers(self) -> int:
        """Number of transformers chained together."""
        return len(self.transformers)

    @property
    def state_dimension(self) -> int:
        r"""Dimension :math:`n` of the state."""
        for tf in self.transformers:
            if (r := tf.state_dimension) is not None:
                return r
        return None

    def __str__(self):
        lines = super().__str__().split("\n  ")
        lines.append(f"num_transformers: {self.num_transformers}")
        lines.append("transformers")
        for tf in self.transformers:
            tfstr = str(tf).split("\n  ")
            tfstr[0] = f"  {tfstr[0]}"
            lines.append("\n      ".join(tfstr))
        return "\n  ".join(lines)

    def __len__(self):
        return self.num_transformers

    # Main routines -----------------------------------------------------------
    def _chain(self, method, states, inplace=False, **kwargs):
        """Apply the specified method for each transformer (forward)."""
        func = getattr(self.transformers[0], method)
        Y = func(states, inplace=inplace, **kwargs)
        for tf in self.transformers[1:]:
            func = getattr(tf, method)
            Y = func(Y, inplace=True, **kwargs)
        return Y

    def fit_transform(self, states, inplace=False):
        return self._chain("fit_transform", states, inplace=inplace)

    def transform(self, states, inplace=False):
        return self._chain("transform", states, inplace=inplace)

    def inverse_transform(self, states_transformed, inplace=False, locs=None):
        Y = self.transformers[-1].inverse_transform(
            states_transformed,
            inplace=inplace,
            locs=locs,
        )
        for tf in reversed(self.transformers[:-1]):
            Y = tf.inverse_transform(Y, inplace=True, locs=locs)
        return Y

    def transform_ddts(self, ddts, inplace=False):
        return self._chain("transform_ddts", ddts, inplace=inplace)

    # Model persistence -------------------------------------------------------
    def save(self, savefile: str, overwrite: bool = False):
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["name"] = str(self.name)
            meta.attrs["num_transformers"] = self.num_transformers
            for i, tf in enumerate(self.transformers):
                tf.save(hf.create_group(f"transformer_{i:0>2d}"))

    @classmethod
    def load(cls, loadfile: str, TransformerClasses):
        """Load a previously saved transformer from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            File where the transformer was stored via :meth:`save()`.
        TransformerClasses : Iterable[type]
            Classes of the transformers for each state variable.

        Returns
        -------
        TransformerPipeline
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            meta = hf["meta"]
            name = str(meta.attrs["name"])
            num_transformers = int(meta.attrs["num_transformers"])
            if (nclasses := len(TransformerClasses)) != num_transformers:
                raise ValueError(
                    f"file contains {num_transformers:d} transformers "
                    f"but {nclasses:d} classes provided"
                )

            transformers = [
                TransformerClasses[i].load(hf[f"transformer_{i:0>2d}"])
                for i in range(num_transformers)
            ]

            return cls(transformers, name=name)


# Vertical joining ============================================================
class NullTransformer(TransformerTemplate):
    r"""Identity transformation :math:`\q\mapsto\q`.

    This transformer can be used in conjunction with :class:`TransformerMulti`
    if separate transformations are desired for individual state variables but
    one of those state variable is to remain unchanged.

    Parameters
    ----------
    name : str or None
        Label for the state variable that this transformer "acts" on.
    """

    # Main routines -----------------------------------------------------------
    def fit(self, states):
        """Set the state dimension.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of `k` `n`-dimensional snapshots.

        Returns
        -------
        self
        """
        self.state_dimension = states.shape[0]
        return self

    def fit_transform(self, states, inplace: bool = True):
        """Do nothing but set the state dimension; this transformation does not
        affect the states.

        Parameters
        ----------
        states : (n, ...) ndarray
            Matrix of `n`-dimensional snapshots, or a single snapshot.
        inplace : bool
            If ``True`` (default), return ``states``.
            If ``False``, return a copy of ``states``.

        Returns
        -------
        states: (n, ...) ndarray
            State snapshots, or a copy of them if ``inplace=False``.
        """
        self.fit(states)
        return states if inplace else states.copy()

    def transform(self, states, inplace: bool = False):
        """Do nothing; this transformation does not affect the states.

        Parameters
        ----------
        states : (n, ...) ndarray
            Matrix of `n`-dimensional snapshots, or a single snapshot.
        inplace : bool
            If ``True`` (default), return ``states``.
            If ``False``, return a copy of ``states``.

        Returns
        -------
        states: (n, ...) ndarray
            State snapshots, or a copy of them if ``inplace=False``.

        Raises
        ------
        ValueError
            If the ``states`` do not align with the :attr:`state_dimension`.
        """
        self._check_shape(states)
        return states if inplace else states.copy()

    def transform_ddts(self, ddts, inplace: bool = True):
        r"""Do nothing; this transformation does not affect derivatives.

        Parameters
        ----------
        ddts : (n, ...) ndarray
            Matrix of `n`-dimensional snapshot time derivatives, or a
            single snapshot time derivative.
        inplace : bool
            If ``True`` (default), return ``ddts``.
            If ``False``, return a create a copy of ``ddts``.

        Returns
        -------
        ddts : (n, ...) ndarray
            Snapshot time derivatives, or a copy of them if ``inplace=False``.

        Raises
        ------
        ValueError
            If the ``states`` do not align with the :attr:`state_dimension`.
        """
        self._check_shape(ddts)
        return ddts if inplace else ddts.copy()

    def inverse_transform(self, states_transformed, inplace=False, locs=None):
        """Do nothing; this transformation does not affect the states.

        Parameters
        ----------
        states_transformed : (n, ...) or (p, ...)  ndarray
            Matrix of `n`-dimensional transformed snapshots, or a single
            transformed snapshot.
        inplace : bool
            If ``True`` (default), return ``states_transformed``.
            If ``False``, return a create a copy of ``states_transformed``.
        locs : slice or (p,) ndarray of integers or None
            If given, assume ``states_transformed`` contains the transformed
            snapshots at only the `p` indices described by ``locs``.

        Returns
        -------
        states_transformed: (n, ...) or (p, ...) ndarray
            Transformed states, or a copy of them if ``inplace=False``.

        Raises
        ------
        ValueError
            If the ``states_transformed`` do not align with the ``locs`` (when
            provided) or the :attr:`state_dimension` (when ``locs`` is not
            provided).
        """
        if locs is not None:
            locs = self._check_locs(locs, states_transformed)
        else:
            self._check_shape(states_transformed)
        return states_transformed if inplace else states_transformed.copy()

    # Model persistence -------------------------------------------------------
    def save(self, savefile, overwrite=False):
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["name"] = str(self.name)
            if (n := self.state_dimension) is not None:
                meta.attrs["state_dimension"] = n

    @classmethod
    def load(cls, loadfile):
        with utils.hdf5_loadhandle(loadfile) as hf:
            meta = hf["meta"]
            name = meta.attrs["name"]
            ntf = cls(name=(None if name == "None" else name))
            if "state_dimension" in meta.attrs:
                ntf.state_dimension = int(meta.attrs["state_dimension"])
            return ntf


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
        one for each state variable. Entries of this list can be ``None``,
        in which case a :class:`NullTransformer` is used.
    variable_sizes : list or None
        Dimensions for each state variable, :math:`n_0,\ldots,n_{n_q-1}`.
        If ``None`` (default), set :math:`n_i` to
        ``transformers[i].state_dimension``; if any of these are ``None``,
        assume all state variables have the same dimension, i.e.,
        :math:`n_0 = n_1 = \cdots = n_x \in \NN` with :math:`n_x` to be
        determined in :meth:`fit`. In this case, :math:`n = n_q n_x`.

    Notes
    -----
    This class connects multiple transformers "vertically" by assigning
    different transformations for different parts of the state; see
    :class:`TransformerPipeline` to connect multiple transformers
    "horizontally", i.e., to perform one transformation after another on the
    entire state.
    """

    def __init__(self, transformers, variable_sizes=None):
        """Initialize the transformers."""
        if (num_variables := len(transformers)) == 0:
            raise ValueError("at least one transformer required")
        elif num_variables == 1:
            warnings.warn("only one variable detected", errors.OpInfWarning)

        # Check inheritance and set default variable names.
        tfs = [NullTransformer() if tf is None else tf for tf in transformers]
        for i, tf in enumerate(tfs):
            if not isinstance(tf, TransformerTemplate):
                warnings.warn(
                    f"transformers[{i}] does not inherit from "
                    "TransformerTemplate, unexpected behavior may occur",
                    errors.OpInfWarning,
                )
            if tf.name is None:
                tf.name = f"variable {i}"

        # Check variable sizes.
        if variable_sizes is not None:
            if len(variable_sizes) != num_variables:
                raise ValueError("len(variable_sizes) != len(transformers)")
            for tf, ni in zip(tfs, variable_sizes):
                if tf.state_dimension is None:
                    tf.state_dimension = ni
                elif (tf_n := tf.state_dimension) != ni:
                    raise ValueError(
                        f"transformers[{i}].state_dimension = {tf_n} "
                        f"!= {ni} = variable_sizes[{i}]"
                    )

        # Store transformers and set slice dimensions.
        self.__nq = num_variables
        self.__transformers = tuple(tfs)

    # Properties --------------------------------------------------------------
    @property
    def transformers(self) -> tuple:
        """Transformers for each state variable."""
        return self.__transformers

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
        lines = [self.__class__.__name__]
        if (n := self.state_dimension) is not None:
            lines.append(f"state_dimension: {n}")
        lines.append(f"num_variables:   {self.num_variables}")
        lines.append("transformers")
        for tf in self.transformers:
            tfstr = str(tf).split("\n  ")
            tfstr[0] = f"  {tfstr[0]}"
            lines.append("\n      ".join(tfstr))
        return "\n  ".join(lines)

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
    def save(self, savefile: str, overwrite: bool = False):
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            hf.create_dataset("num_variables", data=[self.num_variables])
            for i, tf in enumerate(self.transformers):
                tf.save(hf.create_group(f"variable{i}"))

    @classmethod
    def load(cls, loadfile: str, TransformerClasses):
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
