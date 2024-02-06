# basis/_multi.py
"""Basis for states with multiple variables."""

__all__ = [
    "BasisMulti",
]

import numpy as np

from .. import errors, utils
from ..pre._base import _MultivarMixin
from ._base import BasisTemplate, _UnivarBasisMixin


class BasisMulti(BasisTemplate, _MultivarMixin):
    r"""Join bases together for states with multiple variables.

    This class is for states that can be written (after discretization) as

    .. math::
       \q = \left[\begin{array}{c}
       \q_0 \\ \q_1 \\ \vdots \\ \q_{n_q - 1}
       \end{array}\right]
       \in \RR^n,

    where each :math:`\q_i \in \RR^{n_x}` represents a single discretized
    state variable, and where each individual state variable is compressed
    individually. That is, the compressed state can be written as

    .. math::
       \qhat = \left[\begin{array}{c}
       \qhat_0 \\ \qhat_1 \\ \vdots \\ \qhat_{n_q - 1}
       \end{array}\right]
       \in \RR^r,

    where each :math:`\qhat_i \in \RR^{r_i}` is the compressed version of the
    state variable :math:`\q_i \in \RR^{n_x}`.
    The full state dimension is :math:`n = n_q n_x`, i.e.,
    ``full_state_dimension = num_variables * variable_size``; the reduced state
    dimension is :math:`r = \sum_{i=0}^{n_q - 1}r_i`, i.e.,
    ``reduced_state_dimension = sum(reduced_state_dimensions)``.

    Parameters
    ----------
    bases : list
        Initialized (but not necessarily trained) basis objects,
        one for each state variable.
    """

    def __init__(self, bases):
        """Initialize the bases."""
        # Extract variable names if possible.
        variable_names = []
        for i, basis in enumerate(bases):
            default = f"variable {i}"
            if isinstance(basis, _UnivarBasisMixin):
                if basis.name is None:
                    basis.name = default
                variable_names.append(basis.name)
            else:
                variable_names.append(default)

        # Store variables names and collection of bases.
        _MultivarMixin.__init__(
            self,
            num_variables=len(bases),
            variable_names=variable_names,
        )
        self.bases = bases

    # Properties --------------------------------------------------------------
    @property
    def bases(self):
        """Bases for each state variable."""
        return self.__bases

    @bases.setter
    def bases(self, bs):
        """Set the bases."""
        if len(bs) != self.num_variables:
            raise ValueError("len(bases) != num_variables")

        # Check for full_state_dimension consistency.
        dim = None
        for basis in bs:
            if (
                not hasattr(basis, "full_state_dimension")
                or (n := basis.full_state_dimension) is None
            ):
                dim = None
                break
            if dim is None:
                dim = n
            elif n != dim[0]:
                raise errors.DimensionalityError(
                    "bases have inconsistent full_state_dimension"
                )
        if dim is not None:
            self.full_state_dimension = dim

        # Set reduced_state_dimensions.
        alldims = []
        for basis in bs:
            if (
                not hasattr(basis, "reduced_state_dimension")
                or (r := basis.reduced_state_dimension) is None
            ):
                alldims = None
                break
            alldims.append(r)
        self.reduced_state_dimensions = alldims

        self.__bases = tuple(bs)

    @property
    def reduced_state_dimensions(self):
        r"""Reduced state dimensions :math:`r_0, r_1, \ldots, r_{n_q}`."""
        return self.__rs

    @reduced_state_dimensions.setter
    def reduced_state_dimensions(self, rs):
        """Set the reduced state dimensions."""
        if rs is None:
            self.__rs = None
            self.__r = None
            self.__rslices = None
            return

        if len(rs) != (nvar := self.num_variables):
            raise ValueError(
                f"reduced_state_dimensions must have length {nvar}"
            )

        self.__rs = tuple([int(r) for r in rs])
        dimsum = np.cumsum((0,) + self.__rs)
        self.__r = int(dimsum[-1])
        self.__rslices = tuple(
            [
                slice(dimsum[i], dimsum[i + 1])
                for i in range(self.num_variables)
            ]
        )

    @property
    def reduced_state_dimension(self):
        r"""Total dimension of the reduced state,
        :math:`r = \sum_{i=0}^{n_q - 1}r_i`.
        """
        return self.__r

    @property
    def shape(self):
        """Dimensions :math:`(n, r)` of the basis."""
        if (
            self.full_state_dimension is None
            or self.reduced_state_dimension is None
        ):
            return None
        return (self.full_state_dimension, self.reduced_state_dimension)

    # Magic methods -----------------------------------------------------------
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
        return all(t1 == t2 for t1, t2 in zip(self.bases, other.bases))

    def __str__(self) -> str:
        """String representation: str() of each basis."""
        out = [f"{self.num_variables}-variable {self.__class__.__name__}"]
        namelength = max(len(name) for name in self.variable_names)
        for name, basis in zip(self.variable_names, self.bases):
            out.append(f"* {{:>{namelength}}} | {basis}".format(name))
        return "\n".join(out)

    def __repr__(self) -> str:
        """Unique ID + string representation."""
        return utils.str2repr(self)

    # Convenience methods -----------------------------------------------------
    def _check_is_trained(self):
        if self.full_state_dimension or self.reduced_state_dimension is None:
            raise AttributeError("basis not trained (call fit())")

    @utils.requires("full_state_dimension")
    @utils.requires("reduced_state_dimension")
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
        if (nQ := states.shape[0]) == (n := self.full_state_dimension):
            return False
        elif nQ == (r := self.reduced_state_dimension):
            return True

        raise errors.DimensionalityError(
            f"states.shape[0] must be "
            f"full_state_dimension (n = {n:d}) or "
            f"reduced_state_dimension (r = {r:d})"
        )

    @utils.requires("reduced_state_dimension")
    def _check_shape_reduced(self, states):
        """Verify the shape of the reduced states."""
        if (nQ := states.shape[0]) != (r := self.reduced_state_dimension):
            raise errors.DimensionalityError(
                f"states.shape[0] = {nQ:d} "
                f"!= {r:d} = reduced_state_dimension"
            )

    def get_var(self, var, states):
        """Extract a single variable from the joint state.

        Parameters
        ----------
        var : int or str
            Index or name of the variable to extract.
        states : (n, ...) or (r, ...) ndarray
            Joint full or reduced state vector / snapshot matrix.

        Returns
        -------
        state_variable : (nx, ...) or (rs[i], ...) ndarray
            One full or reduced state variable, extracted from ``states``.
        """
        if self._check_is_reduced(states):
            if var in self.variable_names:
                var = self.variable_names.index(var)
            return states[self.__rslices[var]]
        return _MultivarMixin.get_var(self, var, states)

    def split(self, states):
        """Split a reduced state vector into the individual reduced variables.

        Parameters
        ----------
        states : (r,...) ndarray
            Joint full or reduced state vector / snapshot matrix to split.

        Returns
        -------
        arrs : list ``num_variables`` ndarrays
            Individual full or reduced state variables.
        """
        if self._check_is_reduced(states):
            return [states[s] for s in self.__rslices]
        return _MultivarMixin.split(self, states)

    # Main routines -----------------------------------------------------------
    def fit(self, states):
        """Construct the basis.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of `k` `n`-dimensional snapshots.

        Returns
        -------
        self
        """
        self._check_shape(states)

        n = 0
        rs = []
        for var, basis in zip(self.split(states), self.bases):
            basis.fit(var)
            n += basis.full_state_dimension
            rs.append(basis.reduced_state_dimension)

        self.full_state_dimension = n
        self.reduced_state_dimensions = rs
        return self

    def compress(self, states):
        """Map high-dimensional states to low-dimensional latent coordinates.

        Parameters
        ----------
        states : (n, ...) ndarray
            Matrix of `n`-dimensional state vectors, or a single state vector.

        Returns
        -------
        states_compressed : (r, ...) ndarray
            Matrix of `r`-dimensional latent coordinate vectors, or a single
            coordinate vector.
        """
        self._check_is_trained()
        self._check_shape(states)

        return np.concatenate(
            [
                basis.compress(Q)
                for Q, basis in zip(self.split(states), self.bases)
            ],
            axis=0,
        )

    def decompress(self, states_compressed, locs=None):
        """Map low-dimensional latent coordinates to high-dimensional states.

        Parameters
        ----------
        states_compressed : (r, ...) ndarray
            Matrix of `r`-dimensional latent coordinate vectors, or a single
            coordinate vector.
        locs : slice or (p,) ndarray of integers or None
            If given, return the decompressed state at only the `p` specified
            locations (indices) described by ``locs``.

        Returns
        -------
        states_decompressed : (n, ...) or (p, ...) ndarray
            Matrix of `n`-dimensional decompressed state vectors, or the `p`
            entries of such at the entries specified by ``locs``.
        """
        self._check_is_trained()
        self._check_shape_reduced(states_compressed)

        return np.concatenate(
            [
                basis.decompress(Q_, locs=locs)
                for Q_, basis in zip(self.split(states_compressed), self.bases)
            ]
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
            if (n := self.full_state_dimension) is not None:
                meta.attrs["full_state_dimension"] = n
            if (rs := self.reduced_state_dimensions) is not None:
                meta.attrs["reduced_state_dimensions"] = rs

            # Save individual bases.
            for i, tf in enumerate(self.bases):
                tf.save(hf.create_group(f"variable{i}"))

    # TODO: can we get rid of the BasisClasses argument?
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
            # Load metadata.
            num_variables = int(hf["meta"].attrs["num_variables"])

            # Load individual bases.
            bases = [
                BasisClasses[i].load(hf[f"variable{i}"])
                for i in range(num_variables)
            ]

            # Initialize object and (if available) set state dimension.
            obj = cls(bases)
            if "full_state_dimension" in hf["meta"].attrs:
                obj.full_state_dimension = int(
                    hf["meta"].attrs["full_state_dimension"]
                )
            if "reduced_state_dimensions" in hf["meta"].attrs:
                obj.reduced_state_dimensions = hf["meta"].attrs[
                    "reduced_state_dimensions"
                ]

            return obj

    # Verification ------------------------------------------------------------
    def _verify_locs(self, states_compressed, states_projected):
        """Verify :meth:`decompress()` with ``locs != None``."""
        nx = self.variable_size
        locs = np.sort(np.random.choice(nx, size=(nx // 3), replace=False))

        states_projected_at_locs = np.concatenate(
            [Q[locs] for Q in self.split(states_projected)]
        )
        states_at_locs_projected = self.decompress(
            states_compressed,
            locs=locs,
        )

        if states_at_locs_projected.shape != states_projected_at_locs.shape:
            raise errors.VerificationError(
                "decompress(states_compressed, locs).shape "
                "!= decompressed_states_at_locs.shape"
            )
        if not np.allclose(states_at_locs_projected, states_projected_at_locs):
            raise errors.VerificationError(
                "decompress(states_compressed, locs) "
                "!= decompressed_states_at_locs"
            )
