# pre/basis/_linear.py
"""Linear basis class."""

__all__ = [
            "LinearBasis",
          ]

from ...errors import LoadfileFormatError
from ...utils import hdf5_savehandle, hdf5_loadhandle
from .. import transform
from ._base import _BaseBasis


class LinearBasis(_BaseBasis):
    """Linear basis for representing the low-dimensional approximation

        q = Vr @ q_ := sum([Vr[:, j]*q_[j] for j in range(Vr.shape[1])])
        (full_state = basis * reduced_state).

    Attributes
    ----------
    entries : (n, r) ndarray
        Entries of the basis matrix Vr.
    transformer : Transformer or None
        Transformer for pre-processing states before dimensionality reduction.
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
    def shape(self):
        """Dimensions of the basis."""
        return None if self.entries is None else self.entries.shape

    @property
    def r(self):
        """Dimension of the basis, i.e., the number of basis vectors."""
        return None if self.entries is None else self.entries.shape[1]

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
        if not hasattr(basis, "T") or not hasattr(basis, "__matmul__"):
            raise TypeError("invalid basis")
        self.__entries = basis
        return self

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
