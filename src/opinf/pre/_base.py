# pre/transform/_base.py
"""Template class for transformers."""

__all__ = [
    "TransformerTemplate",
    "TransformerMulti",
]

import abc
import numbers
import numpy as np
import scipy.linalg as la

from .. import errors, ddt, utils


# Base class ==================================================================
class TransformerTemplate(abc.ABC):
    """Template class for transformers.

    Classes that inherit from this template must implement the methods
    :meth:`fit_transform()`, :meth:`transform()`, and
    :meth:`inverse_transform()`. The optional :meth:`transform_ddts()` method
    is used by the ROM class when snapshot time derivative data are available
    in the native state variables.

    See :class:`SnapshotTransformer` for an example.

    The default implementation of :meth:`fit()` simply calls
    :meth:`fit_transform()`.
    """

    # Main routines -----------------------------------------------------------
    def fit(self, states):
        """Learn (but do not apply) the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.

        Returns
        -------
        self
        """
        self.fit_transform(states, inplace=False)
        return self

    @abc.abstractmethod
    def transform(self, states, inplace=False):
        """Apply the learned transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        inplace : bool
            If True, overwrite the input data during the transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        states_transformed : (n, k) ndarray
            Matrix of k transformed snapshots of dimension n.
        """
        raise NotImplementedError  # pragma: no cover

    def transform_ddts(self, ddts, inplace=False):
        r"""Apply the learned transformation to snapshot time derivatives.

        If the transformation is denoted by :math:`\mathcal{T}(q)`,
        this function implements :math:`\mathcal{T}'` such that
        :math:`\mathcal{T}'(\ddt q) = \ddt \mathcal{T}(q)`.

        Parameters
        ----------
        ddts : (n, k) ndarray
            Matrix of k snapshot time derivatives.
        inplace : bool
            If True, overwrite the input data during the transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        ddts_transformed : (n, k) ndarray
            Matrix of k transformed snapshot time derivatives.
        """
        return NotImplemented  # pragma: no cover

    @abc.abstractmethod
    def fit_transform(self, states, inplace=False):
        """Learn and apply the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        inplace : bool
            If True, overwrite the input data during the transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of k transformed snapshots of dimension n.
        """
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def inverse_transform(self, states_transformed, inplace=False, locs=None):
        """Apply the inverse of the learned transformation.

        Parameters
        ----------
        states_transformed : (n, ...) or (p, ...)  ndarray
            Matrix of `n`-dimensional transformed snapshots, or a single
            transformed snapshot.
        inplace : bool
            If ``True``, overwrite ``states_transformed`` during the inverse
            transformation. If ``False``, create a copy of the data to
            untransform.
        locs : slice or (p,) ndarray of integers or None
            If given, assume ``states_transformed`` contains the transformed
            snapshots at only the `p` indices described by ``locs``.

        Returns
        -------
        states: (n, ...) or (p, ...) ndarray
            Matrix of `n`-dimensional untransformed snapshots, or the `p`
            entries of such at the indices specified by ``locs``.
        """
        raise NotImplementedError  # pragma: no cover

    # Model persistence -------------------------------------------------------
    def save(self, savefile, overwrite=False):
        """Save the transformer to an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")  # pragma: no cover

    @classmethod
    def load(cls, loadfile):
        """Load a transformer from an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")  # pragma: no cover

    # Verification ------------------------------------------------------------
    def verify(self, states, t=None, tol: float = 1e-4):
        r"""Verify that :meth:`transform()` and :meth:`inverse_transform()`
        are consistent and that :meth:`transform_ddts()`, if implemented,
        is consistent with :meth:`transform()`.

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
        states : (n, k)
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        t : (k,) ndarray or None
            Time domain corresponding to the states.
            Only required if :meth:`transform_ddts()` is implemented.
        tol : float > 0
            Tolerance for the finite difference check of
            :meth:`transform_ddts()`.
            Only used if :meth:`transform_ddts()` is implemented.
        """
        if not np.ndim(states) == 2:
            raise ValueError(
                "two-dimensional states required for verification"
            )

        # Verify transform().
        states_transformed = self.transform(states, inplace=False)
        if states_transformed.shape != states.shape:
            raise errors.VerificationError(
                "transform(states).shape != states.shape"
            )
        if states_transformed is states:
            raise errors.VerificationError(
                "transform(states, inplace=False) is states"
            )
        states_copy = states.copy()
        states_transformed = self.transform(states_copy, inplace=True)
        if states_transformed is not states_copy:
            raise errors.VerificationError(
                "transform(states, inplace=True) is not states"
            )

        # Verify inverse_transform().
        states_recovered = self.inverse_transform(
            states_transformed,
            inplace=False,
        )
        if states_recovered.shape != states.shape:
            raise errors.VerificationError(
                "inverse_transform(transform(states)).shape != states.shape"
            )
        if states_recovered is states_transformed:
            raise errors.VerificationError(
                "inverse_transform(states_transformed, inplace=False) "
                "is states_transformed"
            )
        states_transformed_copy = states_transformed.copy()
        states_recovered = self.inverse_transform(
            states_transformed_copy,
            inplace=True,
        )
        if states_recovered is not states_transformed_copy:
            raise errors.VerificationError(
                "inverse_transform(states_transformed, inplace=True) "
                "is not states_transformed"
            )
        if not np.allclose(states_recovered, states):
            raise errors.VerificationError(
                "transform() and inverse_transform() are not inverses"
            )

        # Check locs argument of inverse_transform().
        if isinstance(self, _MultivarMixin):
            self._verify_locs(states, states_transformed)
        else:
            n = states.shape[0]
            locs = np.sort(np.random.choice(n, size=(n // 3), replace=False))
            states_transformed_at_locs = states_transformed[locs]
            states_recovered_at_locs = self.inverse_transform(
                states_transformed_at_locs,
                locs=locs,
            )
            states_at_locs = states[locs]
            if states_recovered_at_locs.shape != states_at_locs.shape:
                raise errors.VerificationError(
                    "inverse_transform(transform(states)[locs], locs).shape "
                    "!= states[locs].shape"
                )
            if not np.allclose(states_recovered_at_locs, states_at_locs):
                raise errors.VerificationError(
                    "transform() and inverse_transform() are not inverses "
                    "(locs != None)"
                )
        print("transform() and inverse_transform() are consistent")

        # Finite difference check for transform_ddts().
        if self.transform_ddts(states) is NotImplemented:
            return
        if t is None:
            raise ValueError(
                "time domain 't' required for finite difference check"
            )
        ddts = ddt.ddt(states, t)
        ddts_transformed = self.transform_ddts(ddts, inplace=False)
        if ddts_transformed is ddts:
            raise errors.VerificationError(
                "transform_ddts(ddts, inplace=False) is ddts"
            )
        ddts_est = ddt.ddt(states_transformed, t)
        if (
            diff := la.norm(ddts_transformed - ddts_est) / la.norm(ddts_est)
        ) > tol:
            raise errors.VerificationError(
                "transform_ddts() failed finite difference check,\n\t"
                "|| transform_ddts(d/dt[states]) - d/dt[transform(states)] || "
                f" / || d/dt[transform(states)] || = {diff} > {tol = }"
            )
        ddts_transformed = self.transform_ddts(ddts, inplace=True)
        if ddts_transformed is not ddts:
            raise errors.VerificationError(
                "transform_ddts(ddts, inplace=True) is not ddts"
            )
        print("transform() and transform_ddts() are consistent")


# Mixins ======================================================================
class _UnivarMixin:
    """Mixin for transformers and bases with a single state variable."""

    def __init__(self, name: str = None):
        """Initialize attributes."""
        self.__n = None
        self.__name = name

    @property
    def full_state_dimension(self):
        r"""Dimension :math:`n` of the state snapshots."""
        return self.__n

    @full_state_dimension.setter
    def full_state_dimension(self, n):
        """Set the state dimension."""
        self.__n = int(n)

    @property
    def name(self):
        """Label for the state variable."""
        return self.__name

    @name.setter
    def name(self, label):
        """Set the state variable name."""
        self.__name = label


class _MultivarMixin:
    r"""Mixin for transfomers and bases with multiple state variable.

    This class is for states that can be written (after discretization) as

    .. math::
       \q = \left[\begin{array}{c}
       \q_{0} \\ \q_{1} \\ \vdots \\ \q_{n_q - 1}
       \end{array}\right]
       \in \RR^{n},

    where each :math:`\q_{i} \in \NN^{n_x}` represents a single discretized
    state variable. The full state dimension is :math:`n = n_q n_x`, i.e.,
    ``full_state_dimension = num_variables * variable_size``.

    Parameters
    ----------
    num_variables : int
        Number of state variables :math:`n_q \in \NN`, i.e., the number of
        individual transformations to learn.
    variable_names : list(str) or None
        Name for each state variable.
        Defaults to ``("variable 0", "variable 1", ...)``.
    """

    def __init__(self, num_variables: int, variable_names=None):
        """Initialize variable information."""
        if (
            not isinstance(num_variables, numbers.Number)
            or num_variables // 1 != num_variables
            or num_variables < 1
        ):
            raise TypeError("'num_variables' must be a positive integer")

        self.__nq = int(num_variables)
        self.full_state_dimension = None
        self.variable_names = variable_names

    # Properties --------------------------------------------------------------
    @property
    def num_variables(self):
        r"""Number of state variables :math:`n_q \in \NN`."""
        return self.__nq

    @property
    def variable_names(self):
        """Name for each state variable."""
        return self.__variable_names

    @variable_names.setter
    def variable_names(self, names):
        """Set the variable_names."""
        if names is None:
            names = [f"variable {i}" for i in range(self.num_variables)]
        if len(names) != self.num_variables:
            raise ValueError(
                f"variable_names must have length {self.num_variables}"
            )
        self.__variable_names = tuple(names)

    @property
    def full_state_dimension(self):
        """Total dimension :math:`n = n_q n_x` of all state variables."""
        return self.__n

    @full_state_dimension.setter
    def full_state_dimension(self, n):
        """Set the total and individual variable dimensions."""
        if n is None:
            self.__n = None
            self.__nx = None
            self.__slices = None
            return

        variable_size, remainder = divmod(n, self.num_variables)
        if remainder != 0:
            raise ValueError(
                "'full_state_dimension' must be evenly divisible "
                "by 'num_variables'"
            )
        self.__n = int(n)
        self.__nx = variable_size
        self.__slices = [
            slice(i * variable_size, (i + 1) * variable_size)
            for i in range(self.num_variables)
        ]

    @property
    def variable_size(self):
        r"""Size :math:`n_x \in \NN` of each state variable (mesh size)."""
        return self.__nx

    # Convenience methods -----------------------------------------------------
    def __len__(self):
        """Length = number of state variables."""
        return self.__nq

    @utils.requires("full_state_dimension")
    def _check_shape(self, Q):
        """Verify the shape of the snapshot set Q."""
        if (nQ := Q.shape[0]) != self.full_state_dimension:
            raise errors.DimensionalityError(
                f"states.shape[0] = {nQ:d} "
                f"!= {self.num_variables:d} * {self.variable_size:d} "
                "= num_variables * variable_size = full_state_dimension"
            )

    def get_var(self, var, states):
        """Extract a single variable from the full state.

        Parameters
        ----------
        var : int or str
            Index or name of the variable to extract.
        states : (n, ...) ndarray
            Full state vector or snapshot matrix.

        Returns
        -------
        state_variable : (nx, ...) ndarray
            One state variable, extracted from ``states``.
        """
        self._check_shape(states)
        if var in self.variable_names:
            var = self.variable_names.index(var)
        return states[self.__slices[var]]

    def split(self, states):
        """Split the full state into the individual state variables.

        Parameters
        ----------
        states : (n, ...) ndarray
            Full state vector or snapshot matrix.

        Returns
        -------
        state_variable : list of nq (nx, ...) ndarrays
            Individual state variables, extracted from ``states``.
        """
        self._check_shape(states)
        return np.split(states, self.num_variables, axis=0)

    # Verification ------------------------------------------------------------
    def _verify_locs(self, states, states_transformed):
        """Verify :meth:`inverse_transform()` with ``locs != None``."""
        nx = self.variable_size
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
            raise errors.VerificationError(
                "inverse_transform(states_transformed_at_locs, locs).shape "
                "!= states_at_locs.shape"
            )
        if not np.allclose(states_recovered_at_locs, states_at_locs):
            raise errors.VerificationError(
                "transform() and inverse_transform() are not inverses "
                "(locs != None)"
            )


# Collection of transformers ==================================================
class TransformerMulti(TransformerTemplate, _MultivarMixin):
    r"""Transformer for states with multiple variables.

    This class is for states that can be written (after discretization) as

    .. math::
       \q = \left[\begin{array}{c}
       \q_{0} \\ \q_{1} \\ \vdots \\ \q_{n_q - 1}
       \end{array}\right]
       \in \RR^{n},

    where each :math:`\q_{i} \in \NN^{n_x}` represents a single discretized
    state variable. The full state dimension is :math:`n = n_q n_x`, i.e.,
    ``full_state_dimension = num_variables * variable_size``. Individual
    transformers are calibrated for each state variable.

    Parameters
    ----------
    transformers : list
        Initialized (but not necessarily trained) transformer objects,
        one for each state variable.
    variable_names : tuple(str) or None
        Name for each state variable.
        Defaults to ``("variable 0", "variable 1", ...)``.
    """

    def __init__(
        self,
        transformers,
        variable_names=None,
    ):
        """Set transformation hyperparameters and initialize transformers."""
        _MultivarMixin.__init__(self, len(transformers), variable_names)
        self.transformers = transformers

    # Properties: calibrated quantities ---------------------------------------
    @property
    def transformers(self):
        """Transformers for each state variable."""
        return self.__transformers

    @transformers.setter
    def transformers(self, tfs):
        """Set the transformers"""
        if len(tfs) != self.num_variables:
            raise ValueError("len(transformers) != num_variables")
        self.__transformers = tuple(tfs)

    # Magic methods -----------------------------------------------------------
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
        for name, tf in zip(self.variable_names, self.transformers):
            out.append(f"* {{:>{namelength}}} | {tf}".format(name))
        return "\n".join(out)

    def __repr__(self) -> str:
        """Unique ID + string representation."""
        return utils.str2repr(self)

    # Main routines -----------------------------------------------------------
    def _apply(self, method, states, inplace, locs=None):
        """Apply a method of each transformer to the corresponding chunk of
        ``states``.
        """
        if self.full_state_dimension is None:
            raise AttributeError(
                "transformer not trained (call fit() or fit_transform())"
            )
        options = dict(inplace=inplace)
        if locs is not None:
            options["locs"] = locs
        else:
            self._check_shape(states)

        variables = np.split(states, self.num_variables, axis=0)
        newstates = [
            getattr(transformer, method)(var, **options)
            for transformer, var in zip(self.transformers, variables)
        ]
        if any(Q is NotImplemented for Q in newstates):
            return NotImplemented

        if inplace and locs is None:
            return states
        return np.concatenate(newstates, axis=0)

    def fit_transform(self, states, inplace=False):
        """Learn and apply the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of `k` `n`-dimensional snapshots.
            The first ``variable_size`` entries correspond to the first state
            variable, the next ``variable_size`` entries correspond to the
            second state variable, and so on.
        inplace : bool
            If ``True``, overwrite ``states`` data during transformation.
            If ``False``, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of `k` transformed `n`-dimensional snapshots.
        """
        old_dimension = self.full_state_dimension
        try:
            self.full_state_dimension = states.shape[0]
            return self._apply("fit_transform", states, inplace)
        except Exception:
            self.full_state_dimension = old_dimension
            raise

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
        return self._apply("transform", states, inplace)

    def transform_ddts(self, ddts, inplace: bool = False):
        r"""Apply the learned transformation to snapshot time derivatives.

        Denoting the transformation by
        :math:`\mathcal{T}(\q) = \alpha(\q - \bar{\q}) + \beta`,
        this is the function :math:`\mathcal{T}'(\z) = \alpha\z`.
        Hence, :math:`\mathcal{T}'(\ddt q) = \ddt \mathcal{T}(q)`.

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
        return self._apply("transform_ddts", ddts, inplace)

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
            ``locs``.

        Returns
        -------
        states: (n, ...) or (num_variables*p, ...) ndarray
            Matrix of `n`-dimensional untransformed snapshots, or the
            :math:`n_q p` entries of such at the indices specified by ``locs``.
        """
        return self._apply(
            "inverse_transform",
            states_transformed,
            inplace,
            locs,
        )

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
            # Metadata
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["num_variables"] = self.num_variables
            meta.attrs["variable_names"] = self.variable_names
            if (n := self.full_state_dimension) is not None:
                meta.attrs["full_state_dimension"] = n

            # Save individual transformers.
            for i, tf in enumerate(self.transformers):
                tf.save(hf.create_group(f"variable{i}"))

    @classmethod
    def load(cls, loadfile, TransformerClasses):
        """Load a previously saved transformer from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            File where the transformer was stored via :meth:`save()`.

        Returns
        -------
        TransformerMulti
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            # Load metadata.
            num_variables = int(hf["meta"].attrs["num_variables"])
            names = hf["meta"].attrs["variable_names"].tolist()

            # Load individual transformers.
            transformers = [
                TransformerClasses[i].load(hf[f"variable{i}"])
                for i in range(num_variables)
            ]

            # Initialize object and (if available) set state dimension.
            obj = cls(transformers, variable_names=names)
            if "full_state_dimension" in hf["meta"].attrs:
                obj.full_state_dimension = int(
                    hf["meta"].attrs["full_state_dimension"]
                )

            return obj
