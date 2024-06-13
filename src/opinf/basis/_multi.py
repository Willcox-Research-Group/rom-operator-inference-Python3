# basis/_multi.py
"""Basis for states with multiple variables."""

__all__ = [
    "BasisMulti",
]

import warnings
import numpy as np

from .. import errors, utils
from ._base import BasisTemplate


class BasisMulti:
    r"""Join bases together for states with multiple variables.

    This class is for states that can be written (after discretization) as

    .. math::
       \q = \left[\begin{array}{c}
       \q_0 \\ \q_1 \\ \vdots \\ \q_{n_q - 1}
       \end{array}\right]
       \in \RR^n,

    where each :math:`\q_i \in \RR^{n_i}` represents a single discretized
    state variable, to be compressed (individually) to :math:`r_i` degrees of
    freedom. The compressed state is therefore

    .. math::
       \qhat = \left[\begin{array}{c}
       \qhat_0 \\ \qhat_1 \\ \vdots \\ \qhat_{n_q - 1}
       \end{array}\right]
       \in \RR^r,

    where each :math:`\qhat_i \in \RR^{r_i}` is the compressed version of the
    state variable :math:`\q_i \in \RR^{n_i}`.
    The full state dimension is :math:`n = \sum_{i=0}^{n_q - 1} n_i`, i.e.,
    ``full_state_dimension = sum(full_variable_sizes)``; the reduced state
    dimension is :math:`r = \sum_{i=0}^{n_q - 1}r_i`, i.e.,
    ``reduced_state_dimension = sum(reduced_variable_sizes)``.

    Parameters
    ----------
    bases : list
        Initialized (but not necessarily trained) basis objects,
        one for each state variable.
    full_variable_sizes : list or None
        Dimensions for each state variable, :math:`n_0,\ldots,n_{n_q-1}`.
        If ``None`` (default), set :math:`n_i` to
        ``bases[i].full_state_dimension``; if any of these are ``None``,
        assume all state variables have the same dimension, i.e.,
        :math:`n_0 = n_1 = \cdots = n_x \in \NN` with :math:`n_x` to be
        determined in :meth:`fit`. In this case, :math:`n = n_q n_x`.
    """

    def __init__(self, bases, full_variable_sizes=None):
        """Initialize the bases."""
        self.bases = bases

        if full_variable_sizes is not None:
            if len(full_variable_sizes) != len(bases):
                raise ValueError("len(full_variable_sizes) != len(bases)")
            for basis, ni in zip(bases, full_variable_sizes):
                BasisTemplate.full_state_dimension.fset(basis, ni)

    # Properties --------------------------------------------------------------
    @property
    def bases(self) -> tuple:
        """Bases for each state variable."""
        return self.__bases

    @bases.setter
    def bases(self, bs):
        """Set the bases."""
        if (num_variables := len(bs)) == 0:
            raise ValueError("at least one basis required")
        elif num_variables == 1:
            warnings.warn("only one variable detected", errors.OpInfWarning)

        # Check inheritance and set default variable names.
        for i, basis in enumerate(bs):
            if not isinstance(basis, BasisTemplate):
                warnings.warn(
                    f"bases[{i}] does not inherit from "
                    "BasisTemplate, unexpected behavior may occur",
                    errors.OpInfWarning,
                )
            if basis.name is None:
                basis.name = f"variable {i}"

        # Store bases and set slice dimensions..
        self.__nq = num_variables
        self.__bases = tuple(bs)

    @property
    def num_variables(self) -> int:
        r"""Number of state variables :math:`n_q \in \NN`."""
        return self.__nq

    @property
    def variable_names(self) -> tuple:
        """Names for each state variable."""
        return tuple(basis.name for basis in self.bases)

    @property
    def full_variable_sizes(self) -> tuple:
        r"""Dimensions of each state variable,
        :math:`n_0, \ldots, n_{n_q - 1} \in \NN`.
        """
        return tuple(basis.full_state_dimension for basis in self.bases)

    @property
    def full_state_dimension(self) -> int:
        r"""Total dimension :math:`n = \sum_{i=0}^{n_q-1} n_i \in \NN` of the
        joint full state.
        """
        if None in (ns := self.full_variable_sizes):
            return None
        return sum(ns)

    @property
    def reduced_variable_sizes(self) -> tuple:
        r"""Dimensions of each compressed state variable,
        :math:`r_0, \ldots, r_{n_q - 1} \in \NN`.
        """
        return tuple(basis.reduced_state_dimension for basis in self.bases)

    @property
    def reduced_state_dimension(self):
        r"""Total dimension :math:`r = \sum_{i=0}^{n_q - 1}r_i \in \NN`
        of the joint reduced state.
        """
        if None in (rs := self.reduced_variable_sizes):
            return None
        return sum(rs)

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions :math:`(n, r)` of the basis."""
        return (self.full_state_dimension, self.reduced_state_dimension)

    # Magic methods -----------------------------------------------------------
    def __len__(self) -> int:
        """Length = number of state variables."""
        return self.num_variables

    def __getitem__(self, key):
        """Get the basis for variable i."""
        if key in self.variable_names:
            key = self.variable_names.index(key)
        return self.bases[key]

    def __eq__(self, other) -> bool:
        """Test two BasisMulti objects for equality."""
        if not (
            isinstance(other, self.__class__)
            or isinstance(self, other.__class__)
        ):
            return False
        if self.num_variables != other.num_variables:
            return False
        return all(b1 == b2 for b1, b2 in zip(self.bases, other.bases))

    def __str__(self) -> str:
        """String representation: str() of each basis."""
        out = [f"{self.num_variables}-variable {self.__class__.__name__}"]
        out += [str(basis) for basis in self.bases]
        return "\n| ".join("\n\n".join(out).split("\n"))

    def __repr__(self) -> str:
        """Unique ID + string representation."""
        return utils.str2repr(self)

    # Access convenience methods ----------------------------------------------
    def _check_shape_full(self, states):
        """Verify the shape of the full states."""
        if (n := self.full_state_dimension) is None:
            raise AttributeError("full_state_dimension not set, call fit()")
        if (n2 := states.shape[0]) != n:
            raise errors.DimensionalityError(
                f"states.shape[0] = {n2:d} != {n:d} = full_state_dimension"
            )

    def _check_shape_reduced(self, states_compressed):
        """Verify the shape of the reduced states."""
        if (r := self.reduced_state_dimension) is None:
            raise AttributeError("reduced_state_dimension not set, call fit()")
        if (r2 := states_compressed.shape[0]) != r:
            raise errors.DimensionalityError(
                f"states_compressed.shape[0] = {r2:d} "
                f"!= {r:d} = reduced_state_dimension"
            )

    def _check_is_reduced(self, states) -> bool:
        """Verify the shape of the full or reduced vector / matrix Q.

        Parameters
        ----------
        states : (n, ...) or (r, ...) ndarray
            Joint full or reduced state vector / snapshot matrix.

        Returns
        -------
        is_reduced : bool
            ``False`` if ``states`` has `n` rows;
            ``True`` if ``states`` has `r` rows.

        Raises
        ------
        errors.DimensionalityError
            If states is neither `n`- nor `r`-dimensional.
        """
        if (n := self.full_state_dimension) is not None:
            if states.shape[0] == n:
                return False
        if (r := self.reduced_state_dimension) is not None:
            if states.shape[0] == r:
                return True

        if n is r is None:
            raise AttributeError("dimension attributes not set")
        raise errors.DimensionalityError(
            f"states.shape[0] must be "
            f"full_state_dimension (n = {n:d}) or "
            f"reduced_state_dimension (r = {r:d})"
        )

    def _slices(self, varindex: int = None, reduced: bool = False):
        """Get slices for one or all full or reduced state variable(s).

        Parameters
        ----------
        varindex : int
            Index of the variable to get a slice for.
            If ``None`` (default), get slices for all state variables.
        reduced : bool
            If ``True``, slice the reduced state variables.
            If ``False`` (default), slice the full state variables.

        Returns
        -------
        slice or tuple
            Slice for the full state variable at index ``varindex``
            or a tuple of slices for all state variables.
        """
        ds = (
            self.reduced_variable_sizes
            if reduced
            else self.full_variable_sizes
        )
        dimsum = np.cumsum((0,) + ds)

        if varindex is not None:
            return slice(dimsum[varindex], dimsum[varindex + 1])

        return [
            slice(dimsum[i], dimsum[i + 1]) for i in range(self.num_variables)
        ]

    def get_var(self, var, states):
        """Extract a single variable from the joint full or reduced state.

        Parameters
        ----------
        var : int or str
            Index or name of the variable to extract.
        states : (n, ...) or (r, ...) ndarray
            Joint full or reduced state vector / snapshot matrix.

        Returns
        -------
        state_variable : (n_i, ...) or (r_i, ...) ndarray
            One full or reduced state variable, extracted from ``states``.
        """
        if var in self.variable_names:
            var = self.variable_names.index(var)
        if self._check_is_reduced(states):
            return states[self._slices(var, reduced=True)]
        return states[self._slices(var, reduced=False)]

    def split(self, states):
        """Split a full or reduced state vector into the individual variables.

        Parameters
        ----------
        states : (r, ...) ndarray
            Joint full or reduced state vector / snapshot matrix to split.

        Returns
        -------
        arrs : list ``num_variables`` ndarrays
            Individual full or reduced state variables.
        """
        if self._check_is_reduced(states):
            return [states[s] for s in self._slices(reduced=True)]
        return [states[s] for s in self._slices(reduced=False)]

    # Main routines -----------------------------------------------------------
    def fit(self, states):
        """Construct the joint basis by calling ``fit()`` on the basis for
        each variable.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of `k` `n`-dimensional snapshots.
            The first ``full_variable_sizes[0]`` entries correspond to the
            first state variable, the next ``full_variable_sizes[1]`` entries
            correspond to the second state variable, and so on.
            If ``full_variable_sizes`` are not yet prescribed, assume that each
            state variable has the same dimension.

        Returns
        -------
        self
        """
        if self.full_state_dimension is None:
            # Assume all state dimensions are equal.
            nx, remainder = divmod(states.shape[0], self.num_variables)
            if remainder != 0:
                raise errors.DimensionalityError(
                    "len(states) must be evenly divisible by "
                    f"the number of variables n_q = {self.num_variables}"
                )
            for basis in self.bases:
                BasisTemplate.full_state_dimension.fset(basis, nx)

        self._check_shape_full(states)
        for basis, Q in zip(self.bases, self.split(states)):
            basis.fit(Q)

        return self

    def compress(self, states):
        """Map high-dimensional states to low-dimensional latent coordinates.

        Parameters
        ----------
        states : (n, ...) ndarray
            Matrix of `n`-dimensional state vectors, or a single state vector.
            The first ``full_variable_sizes[0]`` entries correspond to the
            first state variable, the next ``full_variable_sizes[1]`` entries
            correspond to the second state variable, and so on.

        Returns
        -------
        states_compressed : (r, ...) ndarray
            Matrix of `r`-dimensional latent coordinate vectors, or a single
            coordinate vector. The first ``reduced_variable_sizes[0]``
            entries correspond to the first reduced state variable, the next
            ``reduced_variable_sizes[1]`` entries correspond to the second
            reduced state variable, and so on.
        """
        self._check_shape_full(states)

        return np.concatenate(
            [
                basis.compress(Q)
                for basis, Q in zip(self.bases, self.split(states))
            ],
            axis=0,
        )

    def decompress(self, states_compressed, locs=None):
        r"""Map low-dimensional latent coordinates to high-dimensional states.

        Parameters
        ----------
        states_compressed : (r, ...) ndarray
            Matrix of `r`-dimensional latent coordinate vectors, or a single
            coordinate vector. The first ``reduced_variable_sizes[0]``
            entries correspond to the first reduced state variable, the next
            ``reduced_variable_sizes[1]`` entries correspond to the second
            reduced state variable, and so on.
        locs : slice or (p,) ndarray of integers or None
            If given, decompress each state variable at *only* the
            `p` specified locations (indices) described by ``locs``.
            This option requires each full state variable to have the
            same dimension.

        Returns
        -------
        states_decompressed : (n, ...) or (num_variables*p, ...) ndarray
            Matrix of `n`-dimensional decompressed state vectors, or the
            :math:`n_q \cdot p` entries of such at the entries specified
            by ``locs``.
        """
        self._check_shape_reduced(states_compressed)
        if locs is not None and len(set(self.full_variable_sizes)) > 1:
            raise ValueError(
                "'locs != None' requires that "
                "all bases have the same full_state_dimension"
            )

        return np.concatenate(
            [
                basis.decompress(Q_, locs=locs)
                for basis, Q_ in zip(self.bases, self.split(states_compressed))
            ]
        )

    def project(self, state):
        """Project a high-dimensional state vector to the subset of the
        high-dimensional space that can be represented by the basis.

        This is done by

        1. expressing the state in low-dimensional latent coordinates, then
        2. reconstructing the high-dimensional state corresponding to those
           coordinates.

        In other words, ``project(Q)`` is equivalent to
        ``decompress(compress(Q))``.

        Parameters
        ----------
        states : (n, ...) ndarray
            Matrix of `n`-dimensional state vectors, or a single state vector.
            The first ``full_variable_sizes[0]`` entries correspond to the
            first state variable, the next ``full_variable_sizes[1]`` entries
            correspond to the second state variable, and so on.

        Returns
        -------
        state_projected : (n, ...) ndarray
            Matrix of `n`-dimensional projected state vectors, or a single
            projected state vector. The first ``full_variable_sizes[0]``
            entries correspond to the first state variable, the next
            ``full_variable_sizes[1]`` entries correspond to the second state
            variable, and so on.
        """
        return self.decompress(self.compress(state))

    def projection_error(self, state, relative=True) -> float:
        r"""Compute the error of the basis representation of a state or states.

        This function computes :math:`\frac{\|\Q - \mathcal{P}(\Q)\|}{\|\Q\|}`,
        where :math:`\Q` is the ``state`` and :math:`\mathcal{P}` is the
        projection defined by :meth:`project()`.
        If ``state`` is one-dimensional then :math:`||\cdot||` is the vector
        2-norm. If ``state`` is two-dimensional then :math:`||\cdot||` is the
        Frobenius norm.

        Parameters
        ----------
        state : (n,) or (n, k) ndarray
            Matrix of `n`-dimensional state vectors, or a single state vector.
            The first ``full_variable_sizes[0]`` entries correspond to the
            first state variable, the next ``full_variable_sizes[1]`` entries
            correspond to the second state variable, and so on.
        relative : bool
            If ``True`` (default), return the relative projection error
            ``norm(state - project(state)) / norm(state)``.
            If ``False``, return the absolute projection error
            ``norm(state - project(state))``.

        Returns
        -------
        float
            Relative error of the projection (``relative=True``) or
            absolute error of the projection (``relative=False``).
        """
        return BasisTemplate.projection_error(self, state, relative)

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
            for i, basis in enumerate(self.bases):
                basis.save(hf.create_group(f"variable{i}"))

    @classmethod
    def load(cls, loadfile, BasisClasses):
        """Load a previously saved transformer from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            File where the transformer was stored via :meth:`save()`.
        BasisClasses : Iterable[type]
            Classes of the bases for each state variable.

        Returns
        -------
        TransformerMulti
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            num_variables = int(hf["num_variables"][0])

            if isinstance(BasisClasses, type):
                BasisClasses = [BasisClasses] * num_variables
            if (nclasses := len(BasisClasses)) != num_variables:
                raise ValueError(
                    f"file contains {num_variables:d} bases "
                    f"but {nclasses:d} classes provided"
                )

            bases = [
                BasisClasses[i].load(hf[f"variable{i}"])
                for i in range(num_variables)
            ]

            return cls(bases)

    # Verification ------------------------------------------------------------
    def verify(self):
        """Verify that :meth:`compress()` and :meth:`decompress()` are
        consistent in the sense that the range of :meth:`decompress()` is in
        the domain of :meth:`compress()` and that :meth:`project()` defines
        a projection operator, i.e., ``project(project(Q)) = project(Q)``.
        """
        for basis, name in zip(self.bases, self.variable_names):
            print(f"{name}:", end="\t")
            basis.verify()
        BasisTemplate.verify(self)

    def _verify_locs(self, states_compressed, states_projected):
        """Verify :meth:`decompress()` with ``locs != None``."""
        if len(sizes := set(self.full_variable_sizes)) != 1:
            return  # Cannot use locs unless all full variable sizes equal.
        nx = sizes.pop()
        locs = np.sort(np.random.choice(nx, size=(nx // 3), replace=False))

        states_projected_at_locs = np.concatenate(
            [Q[locs] for Q in self.split(states_projected)]
        )
        states_at_locs_projected = self.decompress(
            states_compressed,
            locs=locs,
        )

        if states_at_locs_projected.shape != states_projected_at_locs.shape:
            raise errors.VerificationError(  # pragma: no cover
                "decompress(states_compressed, locs).shape "
                "!= decompressed_states_at_locs.shape"
            )
        if not np.allclose(states_at_locs_projected, states_projected_at_locs):
            raise errors.VerificationError(  # pragma: no cover
                "decompress(states_compressed, locs) "
                "!= decompressed_states_at_locs"
            )
