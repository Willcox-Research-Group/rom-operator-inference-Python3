# pre/basis/_linear.py
"""Linear basis class."""

__all__ = [
    "LinearBasis",
    "LinearBasisMulti",
]

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from ..errors import LoadfileFormatError
from ..utils import hdf5_savehandle, hdf5_loadhandle
from .._multivar import _MultivarMixin
from ._base import _BaseBasis


class LinearBasis(_BaseBasis):
    r"""Linear basis for representing the low-dimensional state approximation

    .. math::
        \mathbf{q}
        \approx \mathbf{V}_{r}\widehat{\mathbf{q}}
        = \sum_{i=1}^{r}\hat{q}_{i}\mathbf{v}_{i},
    where :math:`\mathbf{q}\in\mathbb{R}^{n}`,
    :math:`\mathbf{V}_{r}
    = [\mathbf{v}_{1}, \ldots, \mathbf{v}_{r}]\in \mathbb{R}^{n\times r}`, and
    :math:`\widehat{\mathbf{q}}
    = [\hat{q}_{1},\ldots,\hat{q}_{r}]\in\mathbb{R}^{r}`.

    Attributes
    ----------
    n : int
        Dimension of the state space (size of each basis vector).
    r : int
        Dimension of the basis (number of basis vectors in the representation).
    shape : tulpe
        Dimensions (n, r).
    entries : (n, r) ndarray
        Entries of the basis matrix :math:`\mathbf{V}_{r}`.
    """
    def __init__(self):
        """Initialize an empty basis."""
        self.__entries = None

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
        """Store the basis entries.

        Parameters
        ----------
        basis : (n, r) ndarray
            Basis entries.

        Returns
        -------
        self
        """
        if basis is not None and (
            not hasattr(basis, "T") or not hasattr(basis, "__matmul__")
        ):
            raise TypeError("invalid basis")
        # TODO: check for orthogonality?
        self.__entries = basis
        return self

    def __str__(self):
        """String representation: class and dimensions."""
        out = [self.__class__.__name__]
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

    # Dimension reduction -----------------------------------------------------
    def compress(self, state):
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
        return self.entries.T @ state

    def decompress(self, state_, locs=None):
        """Map low-dimensional latent coordinates to high-dimensional states.

        Parameters
        ----------
        state_ : (r,) or (r, k) ndarray
            Low-dimensional latent coordinate vector, or a collection of k
            such vectors organized as the columns of a matrix.
        locs : slice or (p,) ndarray of integers or None
            If given, return the reconstructed state at only the specified
            locations (indices).

        Returns
        -------
        state : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of k such vectors
            organized as the columns of a matrix. If `locs` is given, only
            the specified coordinates are returned.
        """
        return (self.entries if locs is None else self.entries[locs]) @ state_

    # Visualizations ----------------------------------------------------------
    def plot1D(self, x, rmax=None, ax=None, **kwargs):
        """Plot the basis vectors over a one-dimensional domain.

        Parameters
        ----------
        x : (n,) ndarray
            One-dimensional spatial domain over which to plot the vectors.
        rmax : int or None
            Number of basis vectors to plot. If None, plot all basis vectors.
        ax : plt.Axes or None
            Matplotlib Axes to plot on. If None, a new figure is created.
        kwargs : dict
            Other keyword arguments to pass to plt.plot().

        Returns
        -------
        ax : plt.Axes
            Matplotlib Axes for the plot.
        """
        if self.entries is None:
            print("no basis entries to plot")
            return
        if rmax is None:
            rmax = self.r
        if ax is None:
            ax = plt.figure().add_subplot(111)

        for j in range(rmax):
            ax.plot(x, self.entries[:, j], **kwargs)
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("Spatial domain x")
        ax.set_ylabel("Basis vectors v(x)")

        return ax

    # Persistence -------------------------------------------------------------
    def __eq__(self, other):
        """Two LinearBasis objects are equal if their type, dimensions, and
        basis entries are the same.
        """
        if not isinstance(other, self.__class__):
            return False
        if self.shape != other.shape:
            return False
        return np.all(self.entries == other.entries)

    def save(self, savefile, overwrite=False):
        """Save the basis to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        overwrite : bool
            If True, overwrite the file if it already exists. If False
            (default), raise a FileExistsError if the file already exists.
        """
        with hdf5_savehandle(savefile, overwrite) as hf:
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
        LinearBasis
        """
        entries = None
        with hdf5_loadhandle(loadfile) as hf:

            if "entries" in hf:
                entries = hf["entries"][:]

            return cls().fit(entries)


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

    def __init__(self, num_variables, variable_names=None):
        """Initialize an empty basis for each variable."""
        # Store dimensions.
        _MultivarMixin.__init__(self, num_variables, variable_names)
        LinearBasis.__init__(self)

        self.bases = [self._BasisClass() for _ in range(self.num_variables)]

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
        """Store the basis entries.

        Parameters
        ----------
        bases : list of num_entries (n, ri) ndarrays
            Basis entries.

        Returns
        -------
        self
        """
        for basis, Vi in zip(self.bases, bases):
            basis.fit(Vi)
        self._set_entries()
        return self

    # Persistence -------------------------------------------------------------
    def save(self, savefile, overwrite=False):
        """Save the basis to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        overwrite : bool
            If True, overwrite the file if it already exists. If False
            (default), raise a FileExistsError if the file already exists.
        """
        with hdf5_savehandle(savefile, overwrite) as hf:

            # metadata
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["num_variables"] = self.num_variables
            meta.attrs["variable_names"] = self.variable_names

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
        with hdf5_loadhandle(loadfile) as hf:
            # Load metadata.
            if "meta" not in hf:
                raise LoadfileFormatError("invalid save format "
                                          "(meta/ not found)")
            num_variables = hf["meta"].attrs["num_variables"]
            variable_names = hf["meta"].attrs["variable_names"].tolist()

            # Load individual bases.
            bases = []
            for i in range(num_variables):
                group = f"variable{i+1}"
                if group not in hf:
                    raise LoadfileFormatError("invalid save format "
                                              f"({group}/ not found)")
                bases.append(cls._BasisClass.load(hf[group]))

            # Initialize and return the basis object.
            basis = cls(num_variables, variable_names=variable_names)
            basis.bases = bases
            basis._set_entries()
            return basis
