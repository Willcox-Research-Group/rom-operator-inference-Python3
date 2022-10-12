# pre/basis/_linear.py
"""Linear basis class."""

__all__ = [
    "LinearBasis",
    "LinearBasisMulti",
]

import numpy as np
import scipy.sparse as sparse

from ...errors import LoadfileFormatError
from ...utils import hdf5_savehandle, hdf5_loadhandle
from .._multivar import _MultivarMixin
from .. import transform
from ._base import _BaseBasis


class LinearBasis(_BaseBasis):
    """Linear basis for representing the low-dimensional approximation

        q = Vr @ q_ := sum([Vr[:, j]*q_[j] for j in range(Vr.shape[1])])
        (full_state = basis * reduced_state).

    Parameters
    ----------
    transformer : Transformer or None
        Transformer for pre-processing states before dimensionality reduction.

    Attributes
    ----------
    n : int
        Dimension of the state space (size of each basis vector).
    r : int
        Dimension of the basis (number of basis vectors in the representation).
    shape : tulpe
        Dimensions (n, r).
    entries : (n, r) ndarray
        Entries of the basis matrix Vr.
    """
    def __init__(self, transformer=None):
        """Initialize an empty basis and set the transformer."""
        self.__entries = None
        _BaseBasis.__init__(self, transformer)

    # Properties --------------------------------------------------------------
    @property
    def entries(self):
        """Entries of the basis."""
        return self.__entries

    @property
    def n(self):
        """Dimension of the state, i.e., the size of each basis vector."""
        return None if self.entries is None else self.entries.shape[0]

    @property
    def r(self):
        """Dimension of the basis, i.e., the number of basis vectors."""
        return None if self.entries is None else self.entries.shape[1]

    @property
    def shape(self):
        """Dimensions of the basis (state_dimension, reduced_dimension)."""
        return None if self.entries is None else self.entries.shape

    def __getitem__(self, key):
        """self[:] --> self.entries."""
        return self.entries[key]

    def fit(self, basis):
        """Store the basis entries (without any filtering by the transformer).

        Paramters
        ---------
        basis : (n, r) ndarray
            Basis entries. These entries are NOT filtered by the transformer.

        Returns
        -------
        self
        """
        if basis is not None and (
            not hasattr(basis, "T") or not hasattr(basis, "__matmul__")
        ):
            raise TypeError("invalid basis")
        self.__entries = basis
        return self

    def __str__(self):
        """String representation: class and dimensions."""
        out = [self.__class__.__name__]
        if self.transformer is not None:
            out[0] = f"{out[0]} with {self.transformer.__class__.__name__}"
        if self.n is None:
            out[0] = f"Empty {out[0]}"
        else:
            out.append(f"Full-order dimension    n = {self.n:d}")
            out.append(f"Reduced-order dimension r = {self.r:d}")
        return "\n".join(out)

    def __repr__(self):
        """Unique ID + string representation."""
        uniqueID = f"<{self.__class__.__name__} object at {hex(id(self))}>"
        return f"{uniqueID}\n{str(self)}"

    # Encoder / decoder -------------------------------------------------------
    def encode(self, state):
        """Map high-dimensional states to low-dimensional latent coordinates.

        Parameters
        ----------
        state : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of k such vectors
            organized as the columns of a matrix.

        Returns
        -------
        state_ : (r,) or (r, k) ndarray
            Low-dimensional latent coordinate vector, or a collection of k
            such vectors organized as the columns of a matrix.
        """
        if self.transformer is not None:
            state = self.transformer.transform(state)
        return self.entries.T @ state

    def decode(self, state_):
        """Map low-dimensional latent coordinates to high-dimensional states.

        Parameters
        ----------
        state_ : (r,) or (r, k) ndarray
            Low-dimensional latent coordinate vector, or a collection of k
            such vectors organized as the columns of a matrix.

        Returns
        -------
        state : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of k such vectors
            organized as the columns of a matrix.
        """
        state = self.entries @ state_
        if self.transformer is not None:
            state = self.transformer.inverse_transform(state)
        return state

    # Persistence -------------------------------------------------------------
    def __eq__(self, other):
        """Two LinearBasis objects are equal if their type, dimensions, and
        basis entries are the same.
        """
        if not isinstance(other, self.__class__):
            return False
        if self.shape != other.shape:
            return False
        if self.transformer != other.transformer:
            return False
        return np.all(self.entries == other.entries)

    def save(self, savefile, save_transformer=True, overwrite=False):
        """Save the basis to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        save_transformer : bool
            If True, save the transformer as well as the basis entries.
            If False, only save the basis entries.
        overwrite : bool
            If True, overwrite the file if it already exists. If False
            (default), raise a FileExistsError if the file already exists.
        """
        with hdf5_savehandle(savefile, overwrite) as hf:

            if save_transformer and self.transformer is not None:
                meta = hf.create_dataset("meta", shape=(0,))
                TransformerClass = self.transformer.__class__.__name__
                meta.attrs["TransformerClass"] = TransformerClass
                self.transformer.save(hf.create_group("transformer"))

            if self.entries is not None:
                hf.create_dataset("entries", data=self.entries)

    @classmethod
    def load(cls, loadfile):
        """Load a basis from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the basis was stored (via save()).

        Returns
        -------
        _BaseTransformer
        """
        entries, transformer = None, None
        with hdf5_loadhandle(loadfile) as hf:

            if "transformer" in hf:
                if "meta" not in hf:
                    raise LoadfileFormatError("invalid save format "
                                              "(meta/ not found)")
                TransformerClassName = hf["meta"].attrs["TransformerClass"]
                TransformerClass = getattr(transform, TransformerClassName)
                transformer = TransformerClass.load(hf["transformer"])

            if "entries" in hf:
                entries = hf["entries"][:]

            return cls(transformer).fit(entries)


class LinearBasisMulti(LinearBasis, _MultivarMixin):
    r"""Block-diagonal basis grouping individual bases for each state variable.

                                                  [ Vr1         ]
        qi = Vri @ qi_      -->     LinearBasis = [     Vr2     ].
        i = 1, ..., num_variables                 [          \  ]

    The low-dimensional approximation is linear (see LinearBasis).

    Parameters
    ----------
    num_variables : int
        Number of variables represented in a single snapshot (number of
        individual bases to learn). The dimension `n` of the snapshots
        must be evenly divisible by num_variables; for example,
        num_variables=3 means the first n entries of a snapshot correspond to
        the first variable, and the next n entries correspond to the second
        variable, and the last n entries correspond to the third variable.
    transformer : Transformer or None
        Transformer for pre-processing states before dimensionality reduction.
        See SnapshotTransformerMulti for a transformer that scales state
        variables individually.
    variable_names : list of num_variables strings, optional
        Names for each of the `num_variables` variables.
        Defaults to "variable 1", "variable 2", ....

    Attributes
    ----------
    n : int
        Total dimension of the state space.
    ni : int
        Dimension of individual variables, i.e., ni = n / num_variables.
    r : int
        Total dimension of the basis (number of basis vectors).
    rs : list(int)
        Dimensions for each diagonal basis block, i.e., `r[i]` is the number
        of basis vectors in the representation for state variable `i`.
    entries : (n, r) ndarray or scipy.sparse.csc_matrix.
        Entries of the basis matrix.
    bases : list(LinearBasis)
        Individual bases for each state variable.
    """
    _BasisClass = LinearBasis

    def __init__(self, num_variables, transformer=None, variable_names=None):
        """Initialize an empty basis and set the transformer."""
        # Store dimensions and transformer.
        _MultivarMixin.__init__(self, num_variables, variable_names)
        LinearBasis.__init__(self, transformer)

        self.bases = [self._BasisClass(transformer=None)
                      for _ in range(self.num_variables)]

    # Properties -------------------------------------------------------------
    @property
    def r(self):
        """Total dimension of the basis (number of basis vectors)."""
        rs = self.rs
        return None if rs is None else sum(rs)

    @property
    def rs(self):
        """Dimensions for each diagonal basis block, i.e., `rs[i]` is the
        number of basis vectors in the representation for state variable `i`.
        """
        rs = [basis.r for basis in self.bases]
        return rs if any(rs) else None

    def _set_entries(self):
        """Stack individual basis entries as a block diagonal sparse matrix."""
        blocks = []
        for basis in self.bases:
            if basis.n is None:                 # Quit if any basis is empty.
                return
            if basis.n != self.bases[0].n:
                raise ValueError("all bases must have the same row dimension")
            blocks.append(basis.entries)
        self._LinearBasis__entries = sparse.block_diag(blocks, format="csc")

    def __eq__(self, other):
        """Test two LinearBasisMulti objects for equality."""
        if not isinstance(other, self.__class__):
            return False
        if self.num_variables != other.num_variables:
            return False
        return all(b1 == b2 for b1, b2 in zip(self.bases, other.bases))

    def __str__(self):
        """String representation: centering and scaling directives."""
        out = [f"{self.num_variables}-variable {self._BasisClass.__name__}"]
        if self.transformer is not None:
            out[0] = f"{out[0]} with {self.transformer.__class__.__name__}"
        namelength = max(len(name) for name in self.variable_names)
        sep = " " * (namelength + 5)
        for i, (name, st) in enumerate(zip(self.variable_names, self.bases)):
            ststr = str(st).replace('\n', f"\n{sep}")
            ststr = ststr.replace("n =", f"n{i+1:d} =")
            ststr = ststr.replace("r =", f"r{i+1:d} =")
            out.append(f"* {{:>{namelength}}} : {ststr}".format(name))
        if self.n is None:
            out[0] = f"Empty {out[0]}"
        else:
            out.append(f"Total full-order dimension    n = {self.n:d}")
            out.append(f"Total reduced-order dimension r = {self.r:d}")
        return '\n'.join(out)

    # Main routines -----------------------------------------------------------
    def fit(self, bases):
        """Store the basis entries (without any filtering by the transformer).

        Paramters
        ---------
        bases : list of num_entries (n, ri) ndarrays
            Basis entries. These entries are NOT filtered by the transformer.

        Returns
        -------
        self
        """
        for basis, Vi in zip(self.bases, bases):
            basis.fit(Vi)
        self._set_entries()
        return self

    # Persistence -------------------------------------------------------------
    def save(self, savefile, save_transformer=True, overwrite=False):
        """Save the basis to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        save_transformer : bool
            If True, save the transformer as well as the basis entries.
            If False, only save the basis entries.
        overwrite : bool
            If True, overwrite the file if it already exists. If False
            (default), raise a FileExistsError if the file already exists.
        """
        with hdf5_savehandle(savefile, overwrite) as hf:

            # metadata
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["num_variables"] = self.num_variables
            meta.attrs["variable_names"] = self.variable_names

            if save_transformer and self.transformer is not None:
                TransformerClass = self.transformer.__class__.__name__
                meta.attrs["TransformerClass"] = TransformerClass
                self.transformer.save(hf.create_group("transformer"))

            for i in range(self.num_variables):
                self.bases[i].save(hf.create_group(f"variable{i+1:d}"))

    @classmethod
    def load(cls, loadfile):
        """Load a basis from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the basis was stored (via save()).

        Returns
        -------
        PODBasis object
        """
        transformer = None
        with hdf5_loadhandle(loadfile) as hf:
            # Load metadata.
            if "meta" not in hf:
                raise LoadfileFormatError("invalid save format "
                                          "(meta/ not found)")
            num_variables = hf["meta"].attrs["num_variables"]
            variable_names = hf["meta"].attrs["variable_names"].tolist()

            # Load transformer if present.
            if "transformer" in hf:
                TransformerClassName = hf["meta"].attrs["TransformerClass"]
                TransformerClass = getattr(transform, TransformerClassName)
                transformer = TransformerClass.load(hf["transformer"])

            # Load individual bases.
            bases = []
            for i in range(num_variables):
                group = f"variable{i+1}"
                if group not in hf:
                    raise LoadfileFormatError("invalid save format "
                                              f"({group}/ not found)")
                bases.append(cls._BasisClass.load(hf[group]))

            # Initialize and return the basis object.
            basis = cls(num_variables,
                        transformer=transformer, variable_names=variable_names)
            basis.bases = bases
            basis._set_entries()
            return basis
