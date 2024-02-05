# pre/_multi.py
"""Transformer for states with multiple variables."""

__all__ = [
    "TransformerMulti",
]

import numpy as np

from .. import utils
from ._base import TransformerTemplate, _UnivarMixin, _MultivarMixin


class TransformerMulti(TransformerTemplate, _MultivarMixin):
    r"""Join transformers together for states with multiple variables.

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

    def __init__(self, transformers):
        """Initialize the transformers."""
        # Extract variable names if possible.
        variable_names = []
        for i, tf in enumerate(transformers):
            default = f"variable {i}"
            if isinstance(tf, _UnivarMixin):
                if tf.name is None:
                    tf.name = default
                variable_names.append(tf.name)
            else:
                variable_names.append(default)

        # Store variable names and collection of transformers.
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
            if (n := self.full_state_dimension) is not None:
                meta.attrs["full_state_dimension"] = n

            # Save individual transformers.
            for i, tf in enumerate(self.transformers):
                tf.save(hf.create_group(f"variable{i}"))

    # TODO: can we get rid of the TransformerClasses argument?
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
            # Load metadata.
            num_variables = int(hf["meta"].attrs["num_variables"])

            # Load individual transformers.
            transformers = [
                TransformerClasses[i].load(hf[f"variable{i}"])
                for i in range(num_variables)
            ]

            # Initialize object and (if available) set state dimension.
            obj = cls(transformers)
            if "full_state_dimension" in hf["meta"].attrs:
                obj.full_state_dimension = int(
                    hf["meta"].attrs["full_state_dimension"]
                )

            return obj
